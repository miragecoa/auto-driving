#!/usr/bin/env python

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class RadarPointCloudDataset(Dataset):
    """Radar Point Cloud Dataset Class"""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset
        Args:
            data_dir: Radar point cloud data directory
            transform: Data transformation function
        """
        self.data_dir = data_dir
        self.transform = transform
        self.json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        
        # Pre-load all data to speed up training
        self.all_points = []
        self.all_labels = []
        self.frame_indices = []  # Record the frame index for each point
        
        print(f"Found {len(self.json_files)} radar point cloud data files")
        
        # Load point data from all JSON files
        for i, json_file in enumerate(self.json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract point features and labels
            for point in data["points"]:
                # Create feature vector
                features = [
                    point["depth"],
                    point["azimuth"],
                    point["altitude"],
                    point["velocity"],
                    point["snr"],
                    point["world_position"][0],  # x
                    point["world_position"][1],  # y
                    point["world_position"][2],  # z
                ]
                
                # Get label (whether hitting vehicle)
                label = 1 if point["is_hitting_vehicle"] else 0
                
                self.all_points.append(features)
                self.all_labels.append(label)
                self.frame_indices.append(i)
        
        # Convert to NumPy arrays
        self.all_points = np.array(self.all_points, dtype=np.float32)
        self.all_labels = np.array(self.all_labels, dtype=np.int64)
        
        # Calculate the proportion of positive samples (hitting vehicles)
        positive_rate = np.mean(self.all_labels)
        print(f"Dataset size: {len(self.all_points)} points")
        print(f"Points hitting vehicles: {np.sum(self.all_labels)} ({positive_rate:.2%})")
        print(f"Points not hitting vehicles: {len(self.all_labels) - np.sum(self.all_labels)} ({1-positive_rate:.2%})")
    
    def __len__(self):
        return len(self.all_points)
    
    def __getitem__(self, idx):
        features = self.all_points[idx]
        label = self.all_labels[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return torch.FloatTensor(features), torch.tensor(label, dtype=torch.long)

class SimpleNormalization:
    """Simple feature normalization transformation"""
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def fit(self, data):
        """Calculate mean and standard deviation"""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        return self
    
    def __call__(self, features):
        """Normalize features"""
        if self.mean is None or self.std is None:
            return features
        return (features - self.mean) / self.std

class RadarPointClassifier(nn.Module):
    """Radar point classifier, predicts whether points hit vehicles"""
    
    def __init__(self, input_size=8, hidden_size=64, dropout_rate=0.3):
        super(RadarPointClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # Binary classification output
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """Train the model"""
    # Track training and validation losses
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Best model weights
    best_val_loss = float('inf')
    best_model_weights = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate average loss and accuracy on training set
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average loss and accuracy on validation set
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader.dataset)
    
    # Calculate classification report and confusion matrix
    report = classification_report(all_labels, all_preds, target_names=["Non-Vehicle", "Vehicle"])
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, report, cm, all_preds, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss graph
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy graph
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(cm, save_path=None):
    """Draw confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    # Transpose confusion matrix to have:
    # y-axis (rows) = predicted labels
    # x-axis (columns) = true labels
    cm = cm.T
    
    # Reverse the order of classes to have Vehicle first, Background second
    # This will put vehicle-vehicle in the top-left corner and background-background in the bottom-right
    cm = cm[::-1, ::-1]
    
    # Calculate normalized confusion matrix (normalize by column since columns are now true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    
    # Use imshow to display the matrix with Blues colormap
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    
    classes = ["Vehicle", "Background"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations with fixed format (2 decimal places)
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.ylabel('Predicted Label')
    plt.xlabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_model(model, normalizer, save_dir):
    """Save model and normalization parameters"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'radar_model.pth'))
    
    # Save normalization parameters
    np.savez(os.path.join(save_dir, 'normalizer_params.npz'),
             mean=normalizer.mean, std=normalizer.std)
    
    print(f"Model and normalization parameters saved to {save_dir}")

def main(data_dir, model_dir, batch_size=64, num_epochs=30, learning_rate=0.001, hidden_size=128, weight_decay=1e-4):
    """Main function"""
    # Check data directory
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    dataset = RadarPointCloudDataset(data_dir, transform=None)
    
    # Split dataset
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42,
        stratify=dataset.all_labels  # Ensure consistent class distribution in train and test sets
    )
    
    # Create normalization transformation
    normalizer = SimpleNormalization()
    normalizer.fit(dataset.all_points[train_indices])
    
    # Apply normalization and create data loaders
    train_dataset = RadarPointCloudDataset(
        data_dir,
        transform=normalizer
    )
    
    test_dataset = RadarPointCloudDataset(
        data_dir,
        transform=normalizer
    )
    
    # Use subset datasets
    train_dataset.all_points = dataset.all_points[train_indices]
    train_dataset.all_labels = dataset.all_labels[train_indices]
    train_dataset.frame_indices = [dataset.frame_indices[i] for i in train_indices]
    
    test_dataset.all_points = dataset.all_points[test_indices]
    test_dataset.all_labels = dataset.all_labels[test_indices]
    test_dataset.frame_indices = [dataset.frame_indices[i] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate class weights to handle data imbalance
    class_counts = np.bincount(dataset.all_labels)
    total_samples = len(dataset.all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class weights: {class_weights}")
    
    # Create model
    model = RadarPointClassifier(
        input_size=dataset.all_points.shape[1],
        hidden_size=hidden_size
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, num_epochs
    )
    
    # Plot training history
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(model_dir, 'training_history.png')
    )
    
    # Evaluate model
    test_loss, classification_rep, cm, _, _ = evaluate_model(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Set Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_rep)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, save_path=os.path.join(model_dir, 'confusion_matrix.png'))
    
    # Save model and normalization parameters
    save_model(model, normalizer, model_dir)
    
    print("\nTraining Completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Radar Point Cloud Vehicle Detection Model')
    parser.add_argument('--data_dir', type=str, default='radar_pointcloud_dataset/radar_points',
                        help='Radar point cloud data directory')
    parser.add_argument('--model_dir', type=str, default='radar_model',
                        help='Model save directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden layer size')
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_size=args.hidden_size
    ) 