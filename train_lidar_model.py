#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import tqdm
import joblib
import datetime
import seaborn as sns

# Set random seeds to ensure reproducible results
torch.manual_seed(42)
np.random.seed(42)

class LidarPointDataset(Dataset):
    """LiDAR point cloud dataset"""
    
    def __init__(self, dataset_dir, transform=None, max_files=None):
        """
        Initialize the dataset
        
        Parameters:
            dataset_dir (str): Path to the dataset root directory
            transform (callable, optional): Transform to be applied to samples
            max_files (int, optional): Maximum number of files to load, for quick testing
        """
        self.dataset_dir = dataset_dir
        self.lidar_points_dir = os.path.join(dataset_dir, 'lidar_points')
        self.transform = transform
        
        # Ensure directory exists
        if not os.path.exists(self.lidar_points_dir):
            raise ValueError(f"LiDAR point cloud directory does not exist: {self.lidar_points_dir}")
        
        # Get all JSON files
        self.json_files = [f for f in os.listdir(self.lidar_points_dir) if f.endswith('.json')]
        if not self.json_files:
            raise ValueError(f"No JSON files found in {self.lidar_points_dir}")
        
        # Limit the number of files (if specified)
        if max_files is not None:
            self.json_files = self.json_files[:min(max_files, len(self.json_files))]
            
        print(f"Found {len(self.json_files)} LiDAR point cloud data files")
        
        # Read all files and extract point data
        self.data = []
        self.labels = []
        
        print("Loading data...")
        positive_points = 0
        total_points = 0
        
        for json_file in tqdm.tqdm(self.json_files):
            file_path = os.path.join(self.lidar_points_dir, json_file)
            try:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Extract all points and their labels
                for point in json_data['points']:
                    # Extract features
                    features = [
                        point['world_position'][0],  # x
                        point['world_position'][1],  # y
                        point['world_position'][2],  # z
                        point['distance'],           # distance
                        point['intensity'],          # intensity
                        # Image position as normalized relative coordinates (from center point)
                        (point['image_position'][0] - 400) / 400.0,  # normalized x coordinate
                        (point['image_position'][1] - 300) / 300.0,  # normalized y coordinate
                    ]
                    
                    # Label: whether it hits a vehicle
                    label = 1 if point['is_hitting_vehicle'] else 0
                    
                    self.data.append(features)
                    self.labels.append(label)
                    
                    # Count positive samples
                    if label == 1:
                        positive_points += 1
                    total_points += 1
                
            except Exception as e:
                print(f"Error loading file {json_file}: {e}")
        
        # Convert to numpy arrays
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Output dataset statistics
        print(f"Dataset loaded. Total points: {len(self.data)}")
        print(f"Positive samples (hits vehicle): {positive_points} ({positive_points/total_points*100:.2f}%)")
        print(f"Negative samples (misses vehicle): {total_points-positive_points} ({(total_points-positive_points)/total_points*100:.2f}%)")
        
        # Calculate feature normalization parameters (mean and std)
        self.feature_mean = np.mean(self.data, axis=0)
        self.feature_std = np.std(self.data, axis=0)
        # Prevent division by zero
        self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)
        
        print("Feature means:", self.feature_mean)
        print("Feature standard deviations:", self.feature_std)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.labels[idx]
        
        # Normalize features
        features = (features - self.feature_mean) / self.feature_std
        
        if self.transform:
            features = self.transform(features)
        
        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return features, label
    
    def get_normalization_params(self):
        """Return feature normalization parameters"""
        return self.feature_mean, self.feature_std


class LidarPointClassifier(nn.Module):
    """Simple neural network for classifying LiDAR points"""
    
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.3):
        """
        Initialize the model
        
        Parameters:
            input_size (int): Dimension of input features
            hidden_size (int): Size of hidden layers
            dropout_rate (float): Dropout rate for preventing overfitting
        """
        super(LidarPointClassifier, self).__init__()
        
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_size, 2)  # Binary classification: 0=miss vehicle, 1=hit vehicle
        )
    
    def forward(self, x):
        return self.model(x)


def evaluate_model(model, data_loader, device):
    """Evaluate model performance"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, precision, recall, f1, conf_matrix


def plot_confusion_matrix(conf_matrix, epoch, output_dir):
    """
    Plot confusion matrix with English labels and 0.xx format values
    
    Parameters:
        conf_matrix: Confusion matrix
        epoch: Current training epoch
        output_dir: Output directory
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate percentage form of confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Use seaborn to plot heatmap
    ax = sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=["Non-Vehicle", "Vehicle"],
                    yticklabels=["Non-Vehicle", "Vehicle"])
    
    # Set title and labels
    plt.title("Normalized Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    
    # Save the image
    confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_epoch_{epoch+1}.png')
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    return confusion_matrix_path


def train_model(args):
    """Train LiDAR point cloud classification model"""
    print(f"Loading LiDAR point cloud data from {args.dataset}...")
    
    # Load dataset
    full_dataset = LidarPointDataset(args.dataset, max_files=args.max_files)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save normalization parameters
    feature_mean, feature_std = full_dataset.get_normalization_params()
    normalization_params = {
        'mean': feature_mean.tolist(),
        'std': feature_std.tolist()
    }
    with open(os.path.join(args.output_dir, 'normalization_params.json'), 'w') as f:
        json.dump(normalization_params, f)
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2,
        stratify=full_dataset.labels,
        random_state=42
    )
    
    # Create data loaders
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size,
        sampler=val_sampler
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    input_size = full_dataset.data.shape[1]
    model = LidarPointClassifier(input_size, hidden_size=args.hidden_size, dropout_rate=args.dropout_rate)
    model.to(device)
    
    # Handle class imbalance
    # Calculate weights: total_samples / (num_classes * samples_per_class)
    num_samples = len(full_dataset.labels)
    num_class_1 = np.sum(full_dataset.labels)
    num_class_0 = num_samples - num_class_1
    
    # Avoid division by zero
    if num_class_0 == 0 or num_class_1 == 0:
        class_weights = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
    else:
        weight_0 = num_samples / (2 * num_class_0)
        weight_1 = num_samples / (2 * num_class_1)
        class_weights = torch.tensor([weight_0, weight_1], device=device, dtype=torch.float32)
    
    print(f"Class weights: {class_weights}")
    
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    
    # Record training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        # Create progress bar
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for features, labels in progress_bar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate other evaluation metrics
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, val_loader, device)
        history['val_accuracy'].append(accuracy)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Validation: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved new best model with F1 score: {f1:.4f}")
        
        # Removed code to generate confusion matrix for each epoch
        # confusion_matrix_path = plot_confusion_matrix(conf_matrix, epoch, args.output_dir)
        # print(f"Confusion matrix saved to: {confusion_matrix_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    
    # Plot precision and recall
    plt.subplot(2, 2, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Precision and Recall')
    
    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot(history['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
    # Save model architecture information
    model_info = {
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'dropout_rate': args.dropout_rate,
        'features': ['x', 'y', 'z', 'distance', 'intensity', 'norm_img_x', 'norm_img_y']
    }
    with open(os.path.join(args.output_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f)
    
    print(f"Training complete. Model and training history saved to {args.output_dir}")
    
    # Final evaluation on the validation set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, val_loader, device)
    
    print("\nFinal Model Evaluation (Best Model):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Plot and save final confusion matrix
    final_confusion_matrix_path = os.path.join(args.output_dir, 'best_confusion_matrix.png')
    plt.figure(figsize=(10, 8))
    
    # Calculate percentage form of confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Use seaborn to plot heatmap
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=["Non-Vehicle", "Vehicle"],
               yticklabels=["Non-Vehicle", "Vehicle"])
    
    # Set title and labels
    plt.title("Best Model Normalized Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(final_confusion_matrix_path)
    print(f"Best model confusion matrix saved to: {final_confusion_matrix_path}")
    
    # Save evaluation results
    evaluation = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': conf_matrix.tolist(),
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(args.output_dir, 'evaluation.json'), 'w') as f:
        json.dump(evaluation, f)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LiDAR point cloud classification model')
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory to save model and results')
    parser.add_argument('--max-files', type=int, default=None, help='Maximum number of files to load, for quick testing')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA even if available')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args) 