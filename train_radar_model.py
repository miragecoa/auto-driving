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
    """雷达点云数据集类"""
    
    def __init__(self, data_dir, transform=None):
        """
        初始化数据集
        Args:
            data_dir: 雷达点云数据目录
            transform: 数据转换函数
        """
        self.data_dir = data_dir
        self.transform = transform
        self.json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        
        # 提前加载所有数据以加速训练
        self.all_points = []
        self.all_labels = []
        self.frame_indices = []  # 记录每个点所属的帧索引
        
        print(f"找到 {len(self.json_files)} 个雷达点云数据文件")
        
        # 加载所有JSON文件中的点数据
        for i, json_file in enumerate(self.json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 提取点特征和标签
            for point in data["points"]:
                # 创建特征向量
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
                
                # 获取标签 (是否击中车辆)
                label = 1 if point["is_hitting_vehicle"] else 0
                
                self.all_points.append(features)
                self.all_labels.append(label)
                self.frame_indices.append(i)
        
        # 转换为NumPy数组
        self.all_points = np.array(self.all_points, dtype=np.float32)
        self.all_labels = np.array(self.all_labels, dtype=np.int64)
        
        # 计算积极样本(击中车辆)的比例
        positive_rate = np.mean(self.all_labels)
        print(f"数据集大小: {len(self.all_points)} 个点")
        print(f"击中车辆的点: {np.sum(self.all_labels)} ({positive_rate:.2%})")
        print(f"未击中车辆的点: {len(self.all_labels) - np.sum(self.all_labels)} ({1-positive_rate:.2%})")
    
    def __len__(self):
        return len(self.all_points)
    
    def __getitem__(self, idx):
        features = self.all_points[idx]
        label = self.all_labels[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return torch.FloatTensor(features), torch.tensor(label, dtype=torch.long)

class SimpleNormalization:
    """简单的特征归一化转换"""
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def fit(self, data):
        """计算均值和标准差"""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # 避免除零错误
        self.std[self.std == 0] = 1.0
        return self
    
    def __call__(self, features):
        """将特征归一化"""
        if self.mean is None or self.std is None:
            return features
        return (features - self.mean) / self.std

class RadarPointClassifier(nn.Module):
    """雷达点分类器，预测点是否击中车辆"""
    
    def __init__(self, input_size=8, hidden_size=64, dropout_rate=0.3):
        super(RadarPointClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # 二分类输出
        
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
    """训练模型"""
    # 跟踪训练和验证损失
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 最佳模型权重
    best_val_loss = float('inf')
    best_model_weights = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度归零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 计算训练集上的平均损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算验证集上的平均损失和准确率
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, data_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader.dataset)
    
    # 计算分类报告和混淆矩阵
    report = classification_report(all_labels, all_preds, target_names=["非车辆", "车辆"])
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, report, cm, all_preds, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制损失图
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(val_losses, label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    
    # 绘制准确率图
    ax2.plot(train_accs, label='训练准确率')
    ax2.plot(val_accs, label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(cm, save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    classes = ["非车辆", "车辆"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 添加文本注释
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_model(model, normalizer, save_dir):
    """保存模型和归一化参数"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'radar_model.pth'))
    
    # 保存归一化参数
    np.savez(os.path.join(save_dir, 'normalizer_params.npz'),
             mean=normalizer.mean, std=normalizer.std)
    
    print(f"模型和归一化参数已保存到 {save_dir}")

def main(data_dir, model_dir, batch_size=64, num_epochs=30, learning_rate=0.001, hidden_size=128, weight_decay=1e-4):
    """主函数"""
    # 检查数据目录
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    # 创建输出目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载并预处理数据
    dataset = RadarPointCloudDataset(data_dir, transform=None)
    
    # 划分数据集
    train_indices, test_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42,
        stratify=dataset.all_labels  # 确保训练集和测试集中的类别分布一致
    )
    
    # 创建归一化转换
    normalizer = SimpleNormalization()
    normalizer.fit(dataset.all_points[train_indices])
    
    # 应用归一化并创建数据加载器
    train_dataset = RadarPointCloudDataset(
        data_dir,
        transform=normalizer
    )
    
    test_dataset = RadarPointCloudDataset(
        data_dir,
        transform=normalizer
    )
    
    # 使用子集数据集
    train_dataset.all_points = dataset.all_points[train_indices]
    train_dataset.all_labels = dataset.all_labels[train_indices]
    train_dataset.frame_indices = [dataset.frame_indices[i] for i in train_indices]
    
    test_dataset.all_points = dataset.all_points[test_indices]
    test_dataset.all_labels = dataset.all_labels[test_indices]
    test_dataset.frame_indices = [dataset.frame_indices[i] for i in test_indices]
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 计算类别权重以处理数据不平衡
    class_counts = np.bincount(dataset.all_labels)
    total_samples = len(dataset.all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"类别权重: {class_weights}")
    
    # 创建模型
    model = RadarPointClassifier(
        input_size=dataset.all_points.shape[1],
        hidden_size=hidden_size
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 训练模型
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, num_epochs
    )
    
    # 绘制训练历史
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(model_dir, 'training_history.png')
    )
    
    # 评估模型
    test_loss, classification_rep, cm, _, _ = evaluate_model(
        model, test_loader, criterion, device
    )
    
    print(f"\n测试集损失: {test_loss:.4f}")
    print("\n分类报告:")
    print(classification_rep)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, save_path=os.path.join(model_dir, 'confusion_matrix.png'))
    
    # 保存模型和归一化参数
    save_model(model, normalizer, model_dir)
    
    print("\n训练完成!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练雷达点云车辆检测模型')
    parser.add_argument('--data_dir', type=str, default='radar_pointcloud_dataset/radar_points',
                        help='雷达点云数据目录')
    parser.add_argument('--model_dir', type=str, default='radar_model',
                        help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='隐藏层大小')
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_size=args.hidden_size
    ) 