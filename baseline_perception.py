#!/usr/bin/env python

# Baseline Demo for Autonomous Driving Perception System in Daylight Conditions
# Demonstrating Camera, LiDAR and Radar data visualization with YOLOv5 model integration

import glob
import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import cv2  # 确保导入cv2
import math
import torch.nn as nn
import torch.nn.functional as F
import json
import copy
import queue  # 添加queue导入，用于传感器数据队列

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('Cannot import pygame, make sure pygame package is installed')


from perception_utils import (
    CustomTimer, 
    DisplayManager, 
    SensorManager, 
    clean_up_all_vehicles, 
    spawn_surrounding_vehicles, 
    set_random_seed,
    initialize_world,
    spawn_ego_vehicle
)

# 实现一个简化版的SimpleNormalization类，用于特征归一化
class SimpleNormalization:
    """简单的特征归一化转换"""
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
    
    def __call__(self, features):
        """将特征归一化"""
        if self.mean is None or self.std is None:
            return features
        return (features - self.mean) / self.std

# 实现一个简化版的RadarPointClassifier类，与训练时使用的模型结构相同
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

# 添加LidarPointClassifier类，用于处理LiDAR点云
class LidarPointClassifier(nn.Module):
    """LiDAR点分类器，预测点是否属于车辆"""
    
    def __init__(self, input_size=7, hidden_size=64, dropout_rate=0.3):
        super(LidarPointClassifier, self).__init__()
        
        # 使用Sequential结构以匹配保存的模型格式
        self.model = nn.Sequential(
            # model.0 - 线性层
            nn.Linear(input_size, hidden_size),
            # model.1 - 批归一化
            nn.BatchNorm1d(hidden_size),
            # model.2 - ReLU
            nn.ReLU(),
            # model.3 - Dropout
            nn.Dropout(dropout_rate),
            # model.4 - 线性层
            nn.Linear(hidden_size, hidden_size),
            # model.5 - 批归一化
            nn.BatchNorm1d(hidden_size),
            # model.6 - ReLU
            nn.ReLU(),
            # model.7 - Dropout
            nn.Dropout(dropout_rate),
            # model.8 - 输出层
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, x):
        return self.model(x)

# 添加适配carla_lidar_realtime.py中原始模型结构的类
class LidarModelAdapter(nn.Module):
    """兼容旧版模型结构的适配器类"""
    
    def __init__(self, input_size=7, hidden_size=64, dropout_rate=0.3):
        super(LidarModelAdapter, self).__init__()
        
        # 创建顺序模型，与原始训练模型结构保持一致
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, x):
        return self.model(x)

# 添加一个直接匹配原始模型参数名称的分类器
class LidarOldClassifier(nn.Module):
    """直接使用原始训练模型的参数名称结构"""
    
    def __init__(self, input_size=7, hidden_size=64, dropout_rate=0.3):
        super(LidarOldClassifier, self).__init__()
        
        # 定义与model.0, model.1等匹配的层
        self.fc1 = nn.Linear(input_size, hidden_size)  # model.0
        self.bn1 = nn.BatchNorm1d(hidden_size)         # model.1
        self.relu = nn.ReLU()                          # model.2/6
        self.dropout = nn.Dropout(dropout_rate)        # model.3/7
        self.fc2 = nn.Linear(hidden_size, hidden_size) # model.4
        self.bn2 = nn.BatchNorm1d(hidden_size)         # model.5
        self.fc3 = nn.Linear(hidden_size, 2)           # model.8
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def load_from_sequential(self, state_dict):
        """从Sequential模型state_dict加载权重"""
        # 创建映射
        mapping = {
            'model.0.weight': 'fc1.weight',
            'model.0.bias': 'fc1.bias',
            'model.1.weight': 'bn1.weight',
            'model.1.bias': 'bn1.bias',
            'model.1.running_mean': 'bn1.running_mean',
            'model.1.running_var': 'bn1.running_var',
            'model.1.num_batches_tracked': 'bn1.num_batches_tracked',
            'model.4.weight': 'fc2.weight',
            'model.4.bias': 'fc2.bias',
            'model.5.weight': 'bn2.weight',
            'model.5.bias': 'bn2.bias',
            'model.5.running_mean': 'bn2.running_mean',
            'model.5.running_var': 'bn2.running_var',
            'model.5.num_batches_tracked': 'bn2.num_batches_tracked',
            'model.8.weight': 'fc3.weight',
            'model.8.bias': 'fc3.bias'
        }
        
        # 创建新的state_dict
        new_state_dict = {}
        for old_key, new_key in mapping.items():
            if old_key in state_dict:
                new_state_dict[new_key] = state_dict[old_key]
        
        # 加载重映射后的state_dict
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        return len(missing_keys) == 0

# 添加一个直接复制carla_lidar_realtime.py中模型结构的类
class CarlaLidarModel(nn.Module):
    """直接复制carla_lidar_realtime.py中的模型结构"""
    
    def __init__(self, input_size=7, hidden_size=64, dropout_rate=0.3):
        super(CarlaLidarModel, self).__init__()
        
        # 完全按照carla_lidar_realtime.py中的方式定义模型
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 添加RadarInference类，用于雷达点云车辆检测
class RadarInference:
    """雷达点云推理类，用于加载训练好的模型对雷达点云进行车辆检测"""
    
    def __init__(self, model_dir):
        """
        初始化推理器
        Args:
            model_dir: 保存训练好的模型和归一化参数的目录
        """
        self.model_dir = model_dir
        self.model = None
        self.normalizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和归一化参数
        self.load_model_and_params()
    
    def load_model_and_params(self):
        """加载模型和归一化参数"""
        try:
            # 加载归一化参数
            normalizer_path = os.path.join(self.model_dir, 'normalizer_params.npz')
            if not os.path.exists(normalizer_path):
                print(f"Warning: Unable to find normalizer parameters file: {normalizer_path}")
                return False
            
            normalizer_params = np.load(normalizer_path)
            self.normalizer = SimpleNormalization(
                mean=normalizer_params['mean'],
                std=normalizer_params['std']
            )
            
            # 加载模型
            model_path = os.path.join(self.model_dir, 'radar_model.pth')
            if not os.path.exists(model_path):
                print(f"Warning: Unable to find model file: {model_path}")
                return False
            
            # 创建模型 (需要与训练时相同的结构)
            input_size = len(self.normalizer.mean)  # 输入特征数量与归一化参数一致
            self.model = RadarPointClassifier(input_size=input_size, hidden_size=128)
            
            # 加载模型权重
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            
            print(f"Radar inference model and normalizer parameters loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading radar model: {e}")
            return False
    
    def extract_features(self, point):
        """从点云数据中提取特征"""
        features = np.array([
            point["depth"],
            point["azimuth"],
            point["altitude"],
            point["velocity"],
            point["snr"],
            point["world_position"][0],  # x
            point["world_position"][1],  # y
            point["world_position"][2],  # z
        ], dtype=np.float32)
        
        # 应用归一化
        if self.normalizer:
            features = self.normalizer(features)
        
        return features
    
    def predict_point(self, point_features):
        """预测单个点是否击中车辆"""
        if self.model is None:
            return 0, 0.0
            
        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(point_features).unsqueeze(0)  # 添加批处理维度
        features_tensor = features_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence

# 添加LidarInference类，用于激光雷达点云车辆检测
class LidarInference:
    """激光雷达点云推理类，用于加载训练好的模型对点云进行车辆检测"""
    
    def __init__(self, model_dir):
        """
        初始化推理器
        Args:
            model_dir: 保存训练好的模型和归一化参数的目录
        """
        self.model_dir = model_dir
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和归一化参数
        self.load_model_and_params()
    
    def load_model_and_params(self):
        """加载模型和归一化参数"""
        try:
            # 尝试加载model_info.json
            try:
                with open(os.path.join(self.model_dir, 'model_info.json'), 'r') as f:
                    model_info = json.load(f)
                print(f"Loaded model info: {model_info}")
            except Exception as e:
                print(f"Warning: Unable to load model_info.json: {e}")
                model_info = {'input_size': 7, 'hidden_size': 64, 'dropout_rate': 0.3}
            
            # 尝试加载normalization_params.json
            try:
                with open(os.path.join(self.model_dir, 'normalization_params.json'), 'r') as f:
                    norm_params = json.load(f)
                self.feature_mean = np.array(norm_params['mean'])
                self.feature_std = np.array(norm_params['std'])
                print(f"Loaded normalization parameters")
            except Exception as e:
                print(f"Warning: Unable to load normalization_params.json: {e}")
                # 使用默认值
                self.feature_mean = np.zeros(7)
                self.feature_std = np.ones(7)
            
            # 创建模型
            input_size = model_info.get('input_size', 7)
            hidden_size = model_info.get('hidden_size', 64)
            dropout_rate = model_info.get('dropout_rate', 0.3)
            
            # 尝试多种方式加载模型权重
            model_loaded = False
            
            # 1. 尝试不同的模型结构
            model_classes = [
                # 首先尝试直接复制的carla_lidar_realtime模型结构
                CarlaLidarModel(input_size, hidden_size, dropout_rate),
                # 第一种结构：我们当前实现的结构
                LidarPointClassifier(input_size, hidden_size, dropout_rate),
                # 第二种结构：兼容旧版模型的结构
                LidarModelAdapter(input_size, hidden_size, dropout_rate),
                # 第三种结构：直接匹配原参数名
                LidarOldClassifier(input_size, hidden_size, dropout_rate)
            ]
            
            # 2. 尝试不同的文件名
            model_filenames = ['best_model.pth', 'lidar_model.pth', 'model.pth', 'checkpoint.pth', 'carla_lidar_model.pth']
            
            # 遍历每种模型结构和文件名组合
            for model_cls in model_classes:
                if model_loaded:
                    break
                    
                for filename in model_filenames:
                    model_path = os.path.join(self.model_dir, filename)
                    if not os.path.exists(model_path):
                        continue
                        
                    try:
                        # 直接加载权重
                        model_cls.load_state_dict(torch.load(model_path, map_location=self.device))
                        model_cls.to(self.device)
                        model_cls.eval()
                        self.model = model_cls
                        print(f"LiDAR model loaded successfully with {model_cls.__class__.__name__} from: {filename}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"Loading {model_cls.__class__.__name__} from {filename} failed: {e}")
                        
                        try:
                            # 尝试加载整个模型或state_dict
                            loaded_data = torch.load(model_path, map_location=self.device)
                            
                            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                                model_cls.load_state_dict(loaded_data['model_state_dict'])
                                model_cls.to(self.device)
                                model_cls.eval()
                                self.model = model_cls
                                print(f"LiDAR model loaded using model_state_dict from: {filename}")
                                model_loaded = True
                                break
                            elif isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
                                model_cls.load_state_dict(loaded_data['state_dict'])
                                model_cls.to(self.device)
                                model_cls.eval()
                                self.model = model_cls
                                print(f"LiDAR model loaded using state_dict from: {filename}")
                                model_loaded = True
                                break
                            elif isinstance(loaded_data, nn.Module):
                                # 直接使用加载的模型
                                self.model = loaded_data.to(self.device)
                                self.model.eval()
                                print(f"Loaded complete model object from: {filename}")
                                model_loaded = True
                                break
                            elif isinstance(loaded_data, dict):
                                # 尝试直接使用state_dict并转换参数名
                                if isinstance(model_cls, LidarOldClassifier):
                                    success = model_cls.load_from_sequential(loaded_data)
                                    if success:
                                        model_cls.to(self.device)
                                        model_cls.eval()
                                        self.model = model_cls
                                        print(f"LiDAR model loaded using parameter remapping from: {filename}")
                                        model_loaded = True
                                        break
                        except Exception as e2:
                            print(f"Alternative loading methods failed: {e2}")
            
            # 3. 最后尝试直接使用加载的任何.pth文件
            if not model_loaded:
                # 获取目录中的所有.pth文件
                pth_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
                for pth_file in pth_files:
                    model_path = os.path.join(self.model_dir, pth_file)
                    try:
                        loaded_data = torch.load(model_path, map_location=self.device)
                        if isinstance(loaded_data, nn.Module):
                            self.model = loaded_data.to(self.device)
                            self.model.eval()
                            print(f"Loaded model as direct object from: {pth_file}")
                            model_loaded = True
                            break
                    except Exception as e:
                        print(f"Failed to load {pth_file} as model object: {e}")
            
            # 4. 尝试创建一个简单的默认模型（如果所有加载尝试都失败）
            if not model_loaded:
                print(f"Warning: Unable to load any model from: {self.model_dir}, using default model")
                self.model = LidarPointClassifier(input_size, hidden_size, dropout_rate)
                self.model.to(self.device)
                self.model.eval()
            
            return True
        except Exception as e:
            print(f"Error loading LiDAR model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_features(self, point):
        """从点云数据中提取特征"""
        x, y, z = point.point.x, point.point.y, point.point.z
        intensity = getattr(point, 'intensity', 1.0)
        
        # 计算到原点的距离
        distance = math.sqrt(x**2 + y**2 + z**2)
        
        # 模拟图像坐标计算（修改映射逻辑）
        # 假设图像中心点坐标是(400, 300)
        center_x, center_y = 400, 300
        scale = 5.0  # 缩放因子，根据显示需要调整
        
        # 并基于(0,0)点进行翻转（x和y取反）
        px = center_x - int(x * scale)  # x轴映射到水平方向，但翻转
        py = center_y + int(y * scale)  # y轴映射到垂直方向，但翻转
        
        # 归一化图像坐标
        norm_img_x = (px - center_x) / center_x  # 归一化的x坐标
        norm_img_y = (py - center_y) / center_y  # 归一化的y坐标
        
        # 检查是否有速度补偿信息
        has_compensation = hasattr(point, 'velocity_compensation')
        
        # 如果有速度补偿信息，将其合并到现有特征中，而不是添加新维度
        if has_compensation:
            v_proj_x, v_proj_y, v_proj_z = point.velocity_compensation
            
            # 计算补偿速度的大小
            compensation_mag = math.sqrt(v_proj_x**2 + v_proj_y**2 + v_proj_z**2)
            
            # 将速度补偿信息合并到intensity中
            # 根据补偿速度调整intensity (小幅度调整，避免过度影响)
            adjusted_intensity = intensity * (1.0 - min(0.3, compensation_mag / 10.0))
            
            # 准备标准7维特征向量，但使用调整后的intensity
            features = np.array([
                x, y, z, distance, adjusted_intensity, norm_img_x, norm_img_y
            ], dtype=np.float32)
        else:
            # 准备标准特征向量
            features = np.array([
                x, y, z, distance, intensity, norm_img_x, norm_img_y
            ], dtype=np.float32)
        
        return features, (x, y, z, distance)
    
    def predict_points(self, points_data):
        """
        预测多个点是否属于车辆
        Args:
            points_data: 激光雷达点云数据列表
        Returns:
            predictions: 预测结果列表
        """
        if self.model is None or len(points_data) == 0:
            return []
        
        features_list = []
        positions_list = []
        
        # 提取特征
        for point in points_data:
            try:
                features, position_info = self.extract_features(point)
                features_list.append(features)
                positions_list.append(position_info)
            except Exception as e:
                print(f"Error extracting features from point: {e}")
                continue
        
        if len(features_list) == 0:
            return []
        
        # 归一化特征
        features_array = np.array(features_list)
        normalized_features = (features_array - self.feature_mean) / self.feature_std
        
        # 转换为PyTorch张量
        features_tensor = torch.FloatTensor(normalized_features).to(self.device)
        
        # 批量推理
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = probabilities[:, 1].cpu().numpy()  # 获取正类(车辆)的概率
        
        # 整合预测结果
        predictions = []
        for i in range(len(features_list)):
            x, y, z, distance = positions_list[i]
            predictions.append({
                'position': (x, y, z),
                'distance': distance,
                'confidence': float(confidences[i]),
                'predicted_class': int(predicted_classes[i])
            })
        
        return predictions
    
    def identify_vehicle_regions(self, predictions, min_points=5, min_confidence=0.6, max_distance=30):
        """识别可能包含车辆的区域"""
        # 过滤掉低置信度和远距离的点
        vehicle_points = [p for p in predictions if p['predicted_class'] == 1 
                           and p['confidence'] >= min_confidence
                           and p['distance'] <= max_distance]
        
        if len(vehicle_points) < min_points:
            return []
        
        # 使用简单的基于欧几里得距离的聚类方法
        regions = []
        processed = set()
        
        for i, point in enumerate(vehicle_points):
            if i in processed:
                continue
            
            # 开始一个新的聚类
            cluster = [point]
            processed.add(i)
            
            # 查找所有距离当前点足够近的点
            for j, other_point in enumerate(vehicle_points):
                if j in processed:
                    continue
                
                # 计算3D空间中的距离
                p1 = point['position']
                p2 = other_point['position']
                dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                
                if dist < 2.0:  # 小于2米距离认为是同一车辆的点
                    cluster.append(other_point)
                    processed.add(j)
            
            # 只有当聚类包含足够多的点时才认为是车辆
            if len(cluster) >= min_points:
                # 计算平均位置
                avg_x = sum(p['position'][0] for p in cluster) / len(cluster)
                avg_y = sum(p['position'][1] for p in cluster) / len(cluster)
                avg_z = sum(p['position'][2] for p in cluster) / len(cluster)
                
                # 计算平均距离和置信度
                avg_distance = sum(p['distance'] for p in cluster) / len(cluster)
                avg_confidence = sum(p['confidence'] for p in cluster) / len(cluster)
                
                # 计算聚类的半径（最远点到中心的距离）
                center = (avg_x, avg_y, avg_z)
                radius = max(math.sqrt((p['position'][0]-center[0])**2 + 
                                       (p['position'][1]-center[1])**2 + 
                                       (p['position'][2]-center[2])**2) for p in cluster)
                
                regions.append({
                    'center': center,
                    'radius': radius,
                    'points': len(cluster),
                    'distance': avg_distance,
                    'confidence': avg_confidence
                })
        
        return regions

# 添加letterbox函数，从preview_predictions.py中借鉴
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """从YOLOv5代码中提取的letterbox函数"""
    # 检查图像是否为None
    if img is None:
        raise ValueError("Input image is None")
    
    # 检查图像形状
    if not isinstance(img, np.ndarray) or img.ndim != 3:
        raise ValueError(f"Input image must be a 3D numpy array, current: {type(img)}")
    
    # 确保形状是合适的
    shape = img.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 确保新形状是有效的
    if not all(s > 0 for s in new_shape):
        raise ValueError(f"New shape must be positive, current: {new_shape}")
    
    try:
        # 尺度比例 (新 / 旧)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)

        # 计算填充
        ratio = r, r  # 宽度、高度比例
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # 最小矩形
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # 拉伸
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度、高度比例

        dw /= 2  # 分为两部分
        dh /= 2

        if shape[::-1] != new_unpad:  # 调整大小
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 加边框
        
        return img, ratio, (dw, dh)
    
    except Exception as e:
        print(f"letterbox processing error: {e}")
        # 简单回退：直接调整大小并返回
        img_resized = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        return img_resized, (1.0, 1.0), (0.0, 0.0)

# 导入预训练的YOLOv5模型
def load_yolo_model(weights_path, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    """加载预训练的YOLOv5模型，采用多重备选方案确保成功加载"""
    try:
        # 检查YOLOv5目录是否存在
        yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
        if not os.path.exists(yolov5_dir):
            print(f"Warning: Unable to find YOLOv5 directory: {yolov5_dir}")
            print("Please ensure you have cloned the YOLOv5 repository: git clone https://github.com/ultralytics/yolov5.git")
            return None
            
        # 添加YOLOv5目录到路径
        sys.path.append(yolov5_dir)
        
        # 导入所需的YOLOv5函数
        try:
            # 如果可能，导入scale_coords和non_max_suppression
            from utils.general import non_max_suppression, scale_coords
            print("Successfully imported YOLOv5 auxiliary functions")
        except ImportError:
            try:
                # 新版本的替代路径
                from utils.ops import non_max_suppression
                from utils.augmentations import scale_coords
                print("Imported YOLOv5 auxiliary functions from alternative path")
            except ImportError:
                print("Unable to import YOLOv5 auxiliary functions, using basic implementation")
        
        # 尝试多种方法加载模型
        try:
            # 方法1: 使用YOLOv5的model.py
            from models.experimental import attempt_load
            try:
                # 新版本可能使用device参数
                model = attempt_load(weights_path, device=device)
            except TypeError:
                # 旧版本可能使用map_location参数
                model = attempt_load(weights_path, map_location=device)
            print(f"Successfully used attempt_load to load YOLO model: {weights_path}")
            return model
        except Exception as e1:
            print(f"Failed to use attempt_load to load model: {e1}")
            try:
                # 方法2: 使用torch.hub
                model = torch.hub.load(yolov5_dir, 'custom', path=weights_path, source='local')
                print(f"Successfully used torch.hub to load YOLO model: {weights_path}")
                return model
            except Exception as e2:
                print(f"Failed to use torch.hub to load model: {e2}")
                try:
                    # 方法3: 直接使用PyTorch加载
                    model = torch.load(weights_path, map_location=device)
                    if isinstance(model, dict) and 'model' in model:
                        model = model['model']
                    model.to(device)
                    model.eval()
                    print(f"Successfully used torch.load to load YOLO model: {weights_path}")
                    return model
                except Exception as e3:
                    print(f"All loading attempts failed: {e3}")
                    return None
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

# 扩展SensorManager类以支持YOLOv5检测
class YOLOSensorManager(SensorManager):
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, model, conf_thres=0.5, iou_thres=0.45, img_size=640, sensor_name=""):
        # 存储YOLO模型相关信息
        self.model = model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.device = next(model.parameters()).device if model else 'cpu'
        self.timer = CustomTimer()
        self.inference_time = 0.0
        self.tics_processed = 0
        
        # 获取模型步长(stride)
        try:
            self.stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
        except:
            self.stride = 32
            print(f"Unable to get model stride, using default value: {self.stride}")
        
        # 确保导入所需的YOLOv5函数
        self.yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
        sys.path.append(self.yolov5_dir)
        
        # 调用父类构造函数
        super().__init__(world, display_man, sensor_type, transform, attached, sensor_options, display_pos, sensor_name)
    
    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))
            
            # 设置相机刷新率为10 FPS (0.1秒间隔)
            camera_bp.set_attribute('sensor_tick', '0.1')

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
            
            # 设置LiDAR刷新率为10 FPS
            lidar_bp.set_attribute('sensor_tick', '0.1')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
            lidar.listen(self.save_lidar_image)

            return lidar
        
        elif sensor_type == 'Radar':
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            
            # Set default radar parameters
            radar_bp.set_attribute('horizontal_fov', '120')
            radar_bp.set_attribute('vertical_fov', '10')
            
            # 设置雷达刷新率为10 FPS
            radar_bp.set_attribute('sensor_tick', '0.2')
            
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)

            return radar
        
        else:
            return None
    
    def save_rgb_image(self, image):
        """使用YOLO模型处理相机图像并显示检测结果"""
        t_start = self.timer.time()
        
        try:
            # 转换CARLA图像为NumPy数组
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]  # BGR -> RGB
            
            # 创建可写的副本，解决OpenCV绘图错误
            array = array.copy()
            
            if self.model:
                try:
                    # 创建图像副本进行处理
                    img0 = array.copy()
                    
                    # 使用letterbox调整图像大小，保持纵横比
                    try:
                        letterboxed_img, ratio, pad = letterbox(img0, (self.img_size, self.img_size), stride=self.stride)
                        
                        # 转换为PyTorch模型输入格式
                        img = letterboxed_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                        img = np.ascontiguousarray(img)
                        img = torch.from_numpy(img).to(self.device)
                        img = img.float() / 255.0  # 0 - 255 转为 0.0 - 1.0
                        
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)
                    except Exception as e:
                        print(f"Image preprocessing error: {e}")
                        raise
                    
                    # 使用YOLOv5模型进行检测
                    t_inference_start = self.timer.time()
                    
                    # 推理
                    with torch.no_grad():
                        try:
                            # 尝试不同的模型输出格式
                            output = self.model(img)
                            
                            # 根据输出类型的不同进行处理
                            if isinstance(output, list):
                                pred = output[0]  # 取第一个输出
                            elif isinstance(output, dict):
                                pred = output['out']  # 一些模型可能使用字典输出
                            elif isinstance(output, tuple):
                                pred = output[0]  # 通常第一个元素包含检测结果
                            elif isinstance(output, torch.Tensor):
                                pred = output
                            else:
                                pred = torch.zeros((1, 0, 6), device=self.device)
                                print(f"Unknown model output format: {type(output)}")
                                
                        except Exception as inference_error:
                            print(f"Model inference error: {inference_error}")
                            # 在图像上显示错误信息
                            cv2.putText(array, f"YOLO inference error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # 创建空预测结果
                            pred = torch.zeros((1, 0, 6), device=self.device)
                    
                    t_inference_end = self.timer.time()
                    self.inference_time += (t_inference_end - t_inference_start)
                    self.tics_processed += 1
                    
                    # 非极大值抑制
                    try:
                        # 尝试从YOLOv5导入NMS函数
                        try:
                            from utils.general import non_max_suppression
                            detections = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=[0])[0]
                        except ImportError:
                            try:
                                from utils.ops import non_max_suppression
                                detections = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=[0])[0]
                            except ImportError:
                                # 如果无法导入，使用简单过滤
                                mask = pred[0, :, 4] > self.conf_thres
                                detections = pred[0, mask]
                        
                        # 处理中心点坐标到角点坐标的转换
                        if len(detections) > 0 and detections.shape[1] > 6:
                            # 检查是否需要转换 - 如果是中心点+宽高格式
                            # 通常检测结果是 [x1, y1, x2, y2, conf, cls_id]
                            # 但有时可能是 [cx, cy, w, h, conf, cls_id]
                            if torch.any(detections[:, 2] < detections[:, 0]) or torch.any(detections[:, 3] < detections[:, 1]):
                                print("Detected center point + width and height format, converting")
                                # 转换为角点坐标
                                converted = torch.zeros_like(detections[:, :4])
                                converted[:, 0] = detections[:, 0] - detections[:, 2] / 2  # x1 = cx - w/2
                                converted[:, 1] = detections[:, 1] - detections[:, 3] / 2  # y1 = cy - h/2
                                converted[:, 2] = detections[:, 0] + detections[:, 2] / 2  # x2 = cx + w/2
                                converted[:, 3] = detections[:, 1] + detections[:, 3] / 2  # y2 = cy + h/2
                                detections[:, :4] = converted
                    
                    except Exception as nms_error:
                        print(f"NMS processing error: {nms_error}")
                        detections = torch.zeros((0, 6), device=self.device)
                    
                    # 坐标缩放回原始图像大小
                    try:
                        if len(detections) > 0:
                            # 尝试从YOLOv5导入scale_coords函数
                            try:
                                from utils.general import scale_coords
                                detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], img0.shape).round()
                            except ImportError:
                                try:
                                    from utils.augmentations import scale_coords
                                    detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], img0.shape).round()
                                except ImportError:
                                    # 如果无法导入，根据ratio和pad手动调整
                                    # 去除填充
                                    detections_copy = detections.clone()
                                    detections_copy[:, [0, 2]] -= pad[0]  # x padding
                                    detections_copy[:, [1, 3]] -= pad[1]  # y padding
                                    # 根据缩放比例调整坐标
                                    detections_copy[:, :4] /= ratio[0] if isinstance(ratio, tuple) else ratio
                                    # 限制在边界内
                                    detections_copy[:, [0, 2]] = detections_copy[:, [0, 2]].clamp(0, img0.shape[1])
                                    detections_copy[:, [1, 3]] = detections_copy[:, [1, 3]].clamp(0, img0.shape[0])
                                    detections = detections_copy
                    except Exception as scale_error:
                        print(f"Coordinate scaling error: {scale_error}")
                    
                    # 绘制检测框
                    for det in detections:
                        # 获取边界框坐标和置信度
                        if len(det) >= 6:
                            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                            # 转换为整数坐标
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # 如果模型有类别名称
                            try:
                                names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
                                cls_name = names[int(cls)]
                            except:
                                cls_name = "vehicle"
                            
                            # 绘制边界框和标签
                            label = f"{cls_name} {conf:.2f}"
                            color = (0, 255, 0)  # 绿色
                            
                            cv2.rectangle(array, (x1, y1), (x2, y2), color, 2)
                            # 标签背景
                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            cv2.rectangle(array, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 5), color, -1)
                            cv2.putText(array, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
                    # 添加性能信息
                    avg_inference_time = self.inference_time / max(1, self.tics_processed)
                    fps_text = f"YOLOv5 FPS: {1.0/avg_inference_time:.1f}"
                    cv2.putText(array, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 添加检测信息
                    detection_text = f"Objects: {len(detections)}"
                    cv2.putText(array, detection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"YOLO detection error: {e}")
                    import traceback
                    traceback.print_exc()
                    # 在图像上显示错误信息
                    cv2.putText(array, f"YOLO Error: {str(e)[:30]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 将处理后的图像转换为pygame surface
            if self.display_man.render_enabled():
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            
            t_end = self.timer.time()
            self.time_processing += (t_end-t_start)
            self.tics_processing += 1
            
        except Exception as main_error:
            print(f"Main loop error in image processing: {main_error}")
            # 确保有surface可用于渲染
            if self.display_man.render_enabled() and hasattr(self, 'surface') and self.surface is None:
                # 创建一个黑色图像作为备用
                black_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
                cv2.putText(black_img, "Error: " + str(main_error)[:50], (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                self.surface = pygame.surfarray.make_surface(black_img.swapaxes(0, 1))

# 添加RadarSensorManager类，用于雷达点云车辆检测
class RadarSensorManager(SensorManager):
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, inference_model=None, conf_thres=0.7, sensor_name=""):
        """
        初始化RadarSensorManager
        Args:
            world: Carla世界对象
            display_man: 显示管理器
            sensor_type: 传感器类型
            transform: 传感器变换
            attached: 附加到的对象
            sensor_options: 传感器选项
            display_pos: 显示位置
            inference_model: 雷达推理模型
            conf_thres: 检测置信度阈值
            sensor_name: 传感器名称
        """
        # 存储雷达模型相关信息
        self.inference_model = inference_model
        self.conf_thres = conf_thres
        self.timer = CustomTimer()
        self.inference_time = 0.0
        self.tics_processed = 0
        self.detected_vehicles = 0
        self.vehicle = None  # 附加的车辆
        
        # 调用父类构造函数
        super().__init__(world, display_man, sensor_type, transform, attached, sensor_options, display_pos, sensor_name)
        
        # 存储附加的车辆对象，用于获取速度
        if attached and hasattr(attached, 'get_velocity'):
            self.vehicle = attached
    
    def get_vehicle_velocity(self):
        """获取车辆的速度向量"""
        if self.vehicle is None:
            return (0.0, 0.0, 0.0)
        
        try:
            velocity = self.vehicle.get_velocity()
            return (velocity.x, velocity.y, velocity.z)
        except:
            return (0.0, 0.0, 0.0)
    
    def get_velocity_projection(self, azimuth):
        """
        计算车辆速度在雷达探测方向上的投影
        Args:
            azimuth: 雷达探测的方位角
        Returns:
            投影速度（标量）
        """
        # 获取车辆速度
        vx, vy, vz = self.get_vehicle_velocity()
        
        # 计算雷达方向向量
        radar_dir_x = math.cos(azimuth)  
        radar_dir_y = math.sin(azimuth)
        
        # 计算速度在雷达方向上的投影
        # 使用点积: v_proj = v·radar_dir
        # 注意: 投影方向与雷达探测方向相反，因此要取负
        v_proj = -(vx * radar_dir_x + vy * radar_dir_y)
        
        return v_proj
    
    def save_radar_image(self, radar_data):
        """处理雷达数据并进行车辆检测"""
        t_start = self.timer.time()
        
        disp_size = self.display_man.get_display_size()
        
        # 创建雷达图像画布
        radar_img = np.zeros((disp_size[1], disp_size[0], 3), dtype=np.uint8)
        
        # 计算中心点
        center_x, center_y = int(disp_size[0] / 2), int(disp_size[1] / 2)
        
        # 确定视图旋转和方向
        view_rotation = 0  # 默认无旋转
        need_vertical_flip = False  # 是否需要垂直翻转
        need_horizontal_flip = False  # 是否需要水平翻转
        
        if self.sensor_name:
            if "Left Radar" in self.sensor_name:
                view_rotation = 90  # 逆时针旋转90度
                need_vertical_flip = True
            elif "Front Radar" in self.sensor_name:
                view_rotation = -90  # 顺时针旋转90度
                need_horizontal_flip = True
            elif "Right Radar" in self.sensor_name:
                view_rotation = 90  # 逆时针旋转90度
                need_vertical_flip = True
            elif "Rear Radar" in self.sensor_name:
                view_rotation = -90  # 顺时针旋转90度
                need_horizontal_flip = True
        
        # 绘制雷达背景
        cv2.circle(radar_img, (center_x, center_y), 5, (0, 0, 255), -1)  # 中心点（车辆位置）
        
        # 绘制同心圆，表示距离
        for r in range(1, 6):
            radius = int(min(disp_size) / 12 * r)
            cv2.circle(radar_img, (center_x, center_y), radius, (50, 50, 50), 1)
        
        # 绘制坐标轴
        view_rotation_rad = math.radians(view_rotation)
        
        # 第一个轴
        axis1_start_x = center_x + int(disp_size[0] * 0.5 * math.sin(view_rotation_rad))
        axis1_start_y = center_y - int(disp_size[1] * 0.5 * math.cos(view_rotation_rad))
        axis1_end_x = center_x - int(disp_size[0] * 0.5 * math.sin(view_rotation_rad))
        axis1_end_y = center_y + int(disp_size[1] * 0.5 * math.cos(view_rotation_rad))
        
        # 第二个轴（与第一个轴垂直）
        axis2_start_x = center_x + int(disp_size[0] * 0.5 * math.sin(view_rotation_rad + math.pi/2))
        axis2_start_y = center_y - int(disp_size[1] * 0.5 * math.cos(view_rotation_rad + math.pi/2))
        axis2_end_x = center_x - int(disp_size[0] * 0.5 * math.sin(view_rotation_rad + math.pi/2))
        axis2_end_y = center_y + int(disp_size[1] * 0.5 * math.cos(view_rotation_rad + math.pi/2))
        
        cv2.line(radar_img, (int(axis1_start_x), int(axis1_start_y)), 
                 (int(axis1_end_x), int(axis1_end_y)), (50, 50, 50), 1)
        cv2.line(radar_img, (int(axis2_start_x), int(axis2_start_y)), 
                 (int(axis2_end_x), int(axis2_end_y)), (50, 50, 50), 1)
        
        # 确定雷达方向偏移
        orientation_offset = 0.0
        if self.sensor_name:
            if "Left Radar" in self.sensor_name:
                orientation_offset = -math.pi/2  # 左雷达是90度，需要-90度偏移适配显示
            elif "Right Radar" in self.sensor_name:
                orientation_offset = math.pi/2   # 右雷达是270度，需要+90度偏移适配显示 
            elif "Rear Radar" in self.sensor_name:
                orientation_offset = math.pi     # 180度
        
        # 获取最大雷达范围
        max_range = float(self.sensor_options.get('range', '50'))
        
        # 处理雷达点并提取特征
        radar_points = []
        total_points = 0
        
        # 添加检测结果计数
        detected_vehicle_count = 0
        
        # 显示车辆速度信息
        vx, vy, vz = self.get_vehicle_velocity()
        vehicle_speed = math.sqrt(vx**2 + vy**2 + vz**2) * 3.6  # 转换为km/h
        cv2.putText(radar_img, f"Vehicle: {vehicle_speed:.1f} km/h", (10, disp_size[1]-10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 处理每个雷达点
        for i, detect in enumerate(radar_data):
            total_points += 1
            
            # 获取点信息
            distance = detect.depth
            azimuth = detect.azimuth
            altitude = detect.altitude
            velocity = detect.velocity
            
            # 计算并补偿车辆速度
            v_proj = self.get_velocity_projection(azimuth)
            compensated_velocity = velocity - v_proj
            
            # 应用方向偏移调整可视化方向
            adjusted_azimuth = azimuth + orientation_offset
            
            # 应用视图旋转
            final_azimuth = adjusted_azimuth - view_rotation_rad
            
            # 计算点坐标
            scale = min(disp_size) / (2 * max_range)
            x = center_x + int(distance * math.sin(final_azimuth) * scale)
            y = center_y - int(distance * math.cos(final_azimuth) * scale)
            
            # 处理翻转
            if need_vertical_flip:
                y = center_y + (center_y - y)
            
            if need_horizontal_flip:
                x = center_x + (center_x - x)
            
            # 确保点在图像范围内
            if 0 <= x < disp_size[0] and 0 <= y < disp_size[1]:
                # 计算SNR属性（如果存在）
                try:
                    snr = detect.get_snr() if hasattr(detect, 'get_snr') else 0.0
                except:
                    snr = 0.0
                
                # 计算三维世界坐标
                world_x = distance * math.cos(altitude) * math.cos(azimuth)
                world_y = distance * math.cos(altitude) * math.sin(azimuth)
                world_z = distance * math.sin(altitude)
                
                # 创建点特征
                point_data = {
                    "id": i,
                    "depth": distance,
                    "azimuth": azimuth,
                    "altitude": altitude,
                    "velocity": compensated_velocity,  # 使用补偿后的速度
                    "snr": snr,
                    "world_position": [world_x, world_y, world_z],
                    "image_position": [x, y]
                }
                
                radar_points.append(point_data)
                
                # 默认点颜色 - 基于速度（接近为蓝色，远离为红色）
                if compensated_velocity > 0:  # 接近中的物体
                    color = (0, int(255 - min(255, abs(compensated_velocity) * 10)), min(255, abs(compensated_velocity) * 25))  # 绿色到蓝色
                else:  # 远离的物体
                    color = (min(255, abs(compensated_velocity) * 25), int(255 - min(255, abs(compensated_velocity) * 10)), 0)  # 绿色到红色
                
                # 默认绘制点
                point_size = min(5, max(3, int(abs(compensated_velocity) / 5) + 2))
                
                # 如果有推理模型，进行车辆检测
                if self.inference_model:
                    t_inference_start = self.timer.time()
                    
                    # 提取特征
                    try:
                        features = self.inference_model.extract_features(point_data)
                        
                        # 预测
                        is_vehicle, confidence = self.inference_model.predict_point(features)
                        
                        # 如果预测为车辆且置信度超过阈值
                        if is_vehicle == 1 and confidence >= self.conf_thres:
                            # 使用红色叉号标记检测到的车辆点
                            cv2.drawMarker(radar_img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                            
                            # 显示置信度数值和速度
                            cv2.putText(radar_img, f"{confidence:.2f}", (x+5, y-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                            cv2.putText(radar_img, f"{compensated_velocity:.1f}", (x+5, y+15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            
                            detected_vehicle_count += 1
                        else:
                            # 绘制普通点
                            cv2.circle(radar_img, (x, y), point_size, color, -1)
                    except Exception as e:
                        print(f"Point inference error: {e}")
                        # 绘制普通点
                        cv2.circle(radar_img, (x, y), point_size, color, -1)
                    
                    t_inference_end = self.timer.time()
                    self.inference_time += (t_inference_end - t_inference_start)
                    self.tics_processed += 1
                else:
                    # 无推理模型，直接绘制点
                    cv2.circle(radar_img, (x, y), point_size, color, -1)
                
                # 对于速度较大的点，绘制一条表示方向和速度的线
                if abs(compensated_velocity) > 5.0:
                    line_length = min(15, max(5, int(abs(compensated_velocity))))
                    end_x = x
                    end_y = y
                    
                    if need_vertical_flip and need_horizontal_flip:
                        # 垂直和水平都翻转
                        end_x = x - int(line_length * math.sin(final_azimuth))
                        end_y = y + int(line_length * math.cos(final_azimuth))
                    elif need_vertical_flip:
                        # 只有垂直翻转
                        end_x = x + int(line_length * math.sin(final_azimuth))
                        end_y = y + int(line_length * math.cos(final_azimuth))
                    elif need_horizontal_flip:
                        # 只有水平翻转
                        end_x = x - int(line_length * math.sin(final_azimuth))
                        end_y = y - int(line_length * math.cos(final_azimuth))
                    else:
                        # 无翻转
                        end_x = x + int(line_length * math.sin(final_azimuth))
                        end_y = y - int(line_length * math.cos(final_azimuth))
                        
                    cv2.line(radar_img, (x, y), (end_x, end_y), color, 1)
        
        # 保存检测到的车辆数量
        self.detected_vehicles = detected_vehicle_count
        
        # 在雷达图像中不显示任何额外文字信息，保持干净的显示界面
        
        if self.display_man.render_enabled():
            # 转换OpenCV图像（BGR）为RGB，用于pygame
            radar_img_rgb = cv2.cvtColor(radar_img, cv2.COLOR_BGR2RGB)
            self.surface = pygame.surfarray.make_surface(radar_img_rgb)
            
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

# 添加LidarSensorManager类，用于激光雷达点云车辆检测
class LidarSensorManager(SensorManager):
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, inference_model=None, conf_thres=0.6, sensor_name=""):
        """
        初始化LidarSensorManager
        Args:
            world: Carla世界对象
            display_man: 显示管理器
            sensor_type: 传感器类型
            transform: 传感器变换
            attached: 附加到的对象
            sensor_options: 传感器选项
            display_pos: 显示位置
            inference_model: 激光雷达推理模型
            conf_thres: 检测置信度阈值
            sensor_name: 传感器名称
        """
        self.inference_model = inference_model
        self.conf_thres = conf_thres
        self.timer = CustomTimer()
        self.time_processing = 0.0
        self.tics_processing = 0
        self.inference_time = 0.0
        self.tics_processed = 0
        self.detected_vehicles = 0
        self.vehicle_regions = []
        self.vehicle = None  # 附加的车辆
        
        super().__init__(world, display_man, sensor_type, transform, attached, sensor_options, display_pos, sensor_name)
        
        # 存储附加的车辆对象，用于获取速度
        if attached and hasattr(attached, 'get_velocity'):
            self.vehicle = attached
    
    def get_vehicle_velocity(self):
        """获取车辆的速度向量"""
        if self.vehicle is None:
            return (0.0, 0.0, 0.0)
        
        try:
            velocity = self.vehicle.get_velocity()
            return (velocity.x, velocity.y, velocity.z)
        except:
            return (0.0, 0.0, 0.0)
    
    def get_velocity_compensation_for_lidar_point(self, point_x, point_y, point_z):
        """
        计算车辆速度在LiDAR点方向上的补偿值
        Args:
            point_x, point_y, point_z: 点的3D坐标
        Returns:
            补偿后的点和速度向量
        """
        # 获取车辆速度
        vx, vy, vz = self.get_vehicle_velocity()
        
        # 计算点到原点的方向向量
        distance = math.sqrt(point_x**2 + point_y**2 + point_z**2)
        if distance < 0.001:  # 避免除以零
            return point_x, point_y, point_z, (0.0, 0.0, 0.0)
        
        # 计算单位方向向量
        dir_x = point_x / distance
        dir_y = point_y / distance
        dir_z = point_z / distance
        
        # 计算车辆速度在这个方向上的投影
        v_proj = dir_x * vx + dir_y * vy + dir_z * vz
        
        # 计算投影速度向量
        v_proj_x = v_proj * dir_x
        v_proj_y = v_proj * dir_y
        v_proj_z = v_proj * dir_z
        
        # 返回补偿后的速度向量
        return point_x, point_y, point_z, (v_proj_x, v_proj_y, v_proj_z)
    
    def save_lidar_image(self, lidar_data):
        """处理激光雷达数据并进行车辆检测"""
        t_start = self.timer.time()
        
        try:
            disp_size = self.display_man.get_display_size()
            points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            
            # 准备画布
            lidar_img = np.zeros((disp_size[1], disp_size[0], 3), dtype=np.uint8)
            
            # 计算中心点
            center_x, center_y = int(disp_size[0] / 2), int(disp_size[1] / 2)
            
            # 绘制背景
            cv2.circle(lidar_img, (center_x, center_y), 5, (0, 0, 255), -1)  # 中心点（车辆位置）
            
            # 绘制同心圆，表示距离
            for r in range(1, 6):
                radius = int(min(disp_size) / 12 * r)
                cv2.circle(lidar_img, (center_x, center_y), radius, (50, 50, 50), 1)
            
            # 绘制坐标轴
            cv2.line(lidar_img, (center_x, 0), (center_x, disp_size[1]), (50, 50, 50), 1)
            cv2.line(lidar_img, (0, center_y), (disp_size[0], center_y), (50, 50, 50), 1)
            
            # 显示车辆速度信息
            vx, vy, vz = self.get_vehicle_velocity()
            vehicle_speed = math.sqrt(vx**2 + vy**2 + vz**2) * 3.6  # 转换为km/h
            cv2.putText(lidar_img, f"Vehicle: {vehicle_speed:.1f} km/h", (10, disp_size[1]-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # 获取激光雷达数据
            # 如果有推理模型，进行车辆检测
            if self.inference_model:
                t_inference_start = self.timer.time()
                
                try:
                    # 创建补偿车辆速度的点云数据
                    compensated_points = []
                    for point in lidar_data:
                        try:
                            # 获取原始点数据
                            x, y, z = point.point.x, point.point.y, point.point.z
                            intensity = getattr(point, 'intensity', 1.0)
                            
                            # 计算速度补偿
                            _, _, _, v_proj = self.get_velocity_compensation_for_lidar_point(x, y, z)
                            
                            # 创建简单的点数据包装对象，而不是复制原对象
                            # 使用Python的普通对象
                            class SimplePoint:
                                def __init__(self):
                                    self.point = None
                                    self.intensity = 0.0
                                    self.velocity_compensation = None
                            
                            # 创建新对象并设置属性
                            compensated_point = SimplePoint()
                            
                            # 使用简单类表示点位置
                            class SimpleVector:
                                def __init__(self, x, y, z):
                                    self.x = x
                                    self.y = y
                                    self.z = z
                            
                            compensated_point.point = SimpleVector(x, y, z)
                            compensated_point.intensity = intensity
                            compensated_point.velocity_compensation = v_proj
                            
                            compensated_points.append(compensated_point)
                        except Exception as e:
                            print(f"Error processing point: {e}")
                            continue
                    
                    # 在UI中显示是否启用速度补偿
                    vx, vy, vz = self.get_vehicle_velocity()
                    vehicle_speed = math.sqrt(vx**2 + vy**2 + vz**2) * 3.6  # 转换为km/h
                    if vehicle_speed > 1.0:  # 只有速度足够大时显示
                        cv2.putText(lidar_img, "Speed Compensation: Enabled", 
                                  (10, disp_size[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # 预测所有补偿后的点
                    predictions = self.inference_model.predict_points(compensated_points)
                    
                    # 识别车辆区域
                    self.vehicle_regions = self.inference_model.identify_vehicle_regions(
                        predictions, 
                        min_points=5, 
                        min_confidence=self.conf_thres,
                        max_distance=30
                    )
                    self.detected_vehicles = len(self.vehicle_regions)
                    
                    # 绘制每个点 - 预测为车辆的点用绿色，其他点用白色
                    for pred in predictions:
                        if pred['predicted_class'] == 1 and pred['confidence'] >= self.conf_thres:
                            # 计算图像上的位置
                            scale = min(disp_size) / (2 * float(self.sensor_options.get('range', '100.0')))
                            px = center_x - int(pred['position'][0] * scale)  # x轴映射到水平方向，但翻转
                            py = center_y + int(pred['position'][1] * scale)  # y轴映射到垂直方向，但翻转
                            
                            # 确保在图像范围内
                            if 0 <= px < disp_size[0] and 0 <= py < disp_size[1]:
                                cv2.circle(lidar_img, (px, py), 2, (0, 255, 0), -1)  # 绿色点表示车辆
                        else:
                            # 计算图像上的位置
                            scale = min(disp_size) / (2 * float(self.sensor_options.get('range', '100.0')))
                            px = center_x - int(pred['position'][0] * scale)  # x轴映射到水平方向，但翻转
                            py = center_y + int(pred['position'][1] * scale)  # y轴映射到垂直方向，但翻转
                            
                            # 确保在图像范围内
                            if 0 <= px < disp_size[0] and 0 <= py < disp_size[1]:
                                cv2.circle(lidar_img, (px, py), 1, (255, 255, 255), -1)  # 白色点表示非车辆
                    
                    # 绘制车辆区域
                    for region in self.vehicle_regions:
                        # 转换中心点到图像坐标
                        scale = min(disp_size) / (2 * float(self.sensor_options.get('range', '100.0')))
                        cx = center_x - int(region['center'][0] * scale)  # x轴映射到水平方向，但翻转
                        cy = center_y + int(region['center'][1] * scale)  # y轴映射到垂直方向，但翻转
                        
                        # 转换半径到图像坐标
                        radius_px = int(region['radius'] * scale)
                        
                        # 绘制圆形表示车辆区域
                        cv2.circle(lidar_img, (cx, cy), radius_px, (0, 255, 255), 2)
                        
                        # 添加距离标签
                        cv2.putText(lidar_img, f"{region['distance']:.1f}m", (cx, cy - radius_px - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # 添加置信度标签
                        cv2.putText(lidar_img, f"{region['confidence']:.2f}", (cx, cy + radius_px + 15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # 添加矩形边界框
                        x1 = cx - radius_px
                        y1 = cy - radius_px
                        x2 = cx + radius_px
                        y2 = cy + radius_px
                        cv2.rectangle(lidar_img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                
                except Exception as e:
                    print(f"LiDAR inference error: {e}")
                    import traceback
                    traceback.print_exc()
                
                t_inference_end = self.timer.time()
                self.inference_time += (t_inference_end - t_inference_start)
                self.tics_processed += 1
            else:
                # 无推理模型，直接绘制点云
                for point in points:
                    # 激光雷达点
                    x, y, z = point[0], point[1], point[2]
                    intensity = point[3]
                    
                    # 计算到原点的距离
                    distance = math.sqrt(x**2 + y**2 + z**2)
                    
                    # 计算图像上的位置
                    scale = min(disp_size) / (2 * float(self.sensor_options.get('range', '100.0')))
                    px = center_x - int(x * scale)  # x轴映射到水平方向，但翻转
                    py = center_y + int(y * scale)  # y轴映射到垂直方向，但翻转
                    
                    # 根据高度和强度计算颜色 - 越高越亮
                    # 将z归一化到[0,1]范围
                    normalized_z = (z + 2) / 4  # 假设z的范围在-2到2之间
                    normalized_z = max(0, min(1, normalized_z))  # 确保在[0,1]范围内
                    
                    # 将强度归一化到[0,1]范围
                    normalized_intensity = min(1.0, intensity)
                    
                    # 根据高度和强度生成颜色
                    color = (
                        int(255 * (1 - normalized_z)),  # 蓝色分量
                        int(255 * normalized_intensity), # 绿色分量
                        int(255 * normalized_z)          # 红色分量
                    )
                    
                    # 确保在图像范围内
                    if 0 <= px < disp_size[0] and 0 <= py < disp_size[1]:
                        cv2.circle(lidar_img, (px, py), 1, color, -1)
            
            # 添加LiDAR信息文本
            cv2.putText(lidar_img, "LiDAR", (10, 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.inference_model:
                # 计算并显示FPS
                if self.tics_processed > 0:
                    avg_inference_time = self.inference_time / self.tics_processed
                    fps = 1.0 / max(0.001, avg_inference_time)
                    cv2.putText(lidar_img, f"FPS: {fps:.1f}", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示检测到的车辆数量
                cv2.putText(lidar_img, f"车辆: {self.detected_vehicles}", (10, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 转换OpenCV图像（BGR）为RGB，用于pygame
            lidar_img_rgb = cv2.cvtColor(lidar_img, cv2.COLOR_BGR2RGB)
            
            if self.display_man.render_enabled():
                self.surface = pygame.surfarray.make_surface(lidar_img_rgb)
        
        except Exception as e:
            print(f"LiDAR processing error: {e}")
            import traceback
            traceback.print_exc()
        
        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

# 添加RadarOverviewManager类，用于显示雷达信息总览
class RadarOverviewManager:
    def __init__(self, display_man, display_pos):
        """
        初始化雷达概览管理器
        Args:
            display_man: 显示管理器
            display_pos: 显示位置
        """
        self.display_man = display_man
        self.display_pos = display_pos
        self.surface = None
        self.radar_sensors = []  # 存储所有雷达传感器的引用
        
        # 初始化surface以防止黑屏
        disp_size = self.display_man.get_display_size()
        init_img = np.zeros((disp_size[1], disp_size[0], 3), dtype=np.uint8)
        cv2.putText(init_img, "RADAR OVERVIEW", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(init_img, "Waiting for radar data...", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        init_img_rgb = cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB)
        self.surface = pygame.surfarray.make_surface(init_img_rgb)
        
    def register_radar_sensor(self, radar_sensor):
        """注册雷达传感器，以便从中获取信息"""
        if radar_sensor not in self.radar_sensors:
            self.radar_sensors.append(radar_sensor)
            print(f"Registered radar sensor: {radar_sensor.sensor_name if hasattr(radar_sensor, 'sensor_name') else 'Unknown'}")
    
    def update(self):
        """更新雷达信息总览"""
        disp_size = self.display_man.get_display_size()
        # 创建黑色背景
        overview_img = np.zeros((disp_size[1], disp_size[0], 3), dtype=np.uint8)
        
        # 添加标题
        cv2.putText(overview_img, "RADAR OVERVIEW", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 如果有注册的雷达传感器，显示它们的信息
        y_offset = 70
        total_vehicles = 0
        total_points = 0
        total_inference_time = 0
        total_tics = 0
        
        # 检查是否有注册的传感器
        if len(self.radar_sensors) == 0:
            cv2.putText(overview_img, "NO RADAR SENSORS REGISTERED", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        for i, sensor in enumerate(self.radar_sensors):
            # 获取传感器名称
            sensor_name = sensor.sensor_name if hasattr(sensor, 'sensor_name') else f"Radar {i+1}"
            sensor_name = sensor_name.replace("+Detector", "")  # 移除Detector标记
            
            # 获取检测到的车辆数量
            vehicle_count = sensor.detected_vehicles if hasattr(sensor, 'detected_vehicles') else 0
            total_vehicles += vehicle_count
            
            # 计算推理时间
            if hasattr(sensor, 'inference_time') and hasattr(sensor, 'tics_processed'):
                if sensor.tics_processed > 0:
                    total_inference_time += sensor.inference_time
                    total_tics += sensor.tics_processed
            
            # 显示每个传感器的信息
            cv2.putText(overview_img, f"{sensor_name}: {vehicle_count} vehicles", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 30
        
        # 显示总体信息
        cv2.putText(overview_img, f"Total detected vehicles: {total_vehicles}", 
                  (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示推理性能
        if total_tics > 0:
            avg_inference_time = total_inference_time / total_tics
            fps = 1.0 / max(0.001, avg_inference_time)
            cv2.putText(overview_img, f"Inference FPS: {fps:.1f}", 
                      (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 添加分隔线
        cv2.line(overview_img, (10, y_offset), (disp_size[0]-10, y_offset), (100, 100, 100), 1)
        
        # 调试信息
        debug_info = f"Debug: {len(self.radar_sensors)} sensors registered"
        cv2.putText(overview_img, debug_info, (10, disp_size[1] - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # 创建pygame surface
        if self.display_man.render_enabled():
            overview_img_rgb = cv2.cvtColor(overview_img, cv2.COLOR_BGR2RGB)
            self.surface = pygame.surfarray.make_surface(overview_img_rgb)
    
    def render(self):
        """渲染雷达信息总览"""
        self.update()  # 更新信息
        
        if self.surface is not None and self.display_man.render_enabled():
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

# 添加CarlaSyncMode类，用于同步传感器数据
class CarlaSyncMode(object):
    """
    上下文管理器，用于同步不同传感器的输出。
    只要我们在这个上下文中，同步模式就会启用。

    使用示例：
        with CarlaSyncMode(world, fps=20) as sync_mode:
            while True:
                world_snapshot = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, **kwargs):
        self.world = world
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        # 首先为世界tick事件创建队列
        make_queue(self.world.on_tick)
        
        # 不为传感器注册新的监听函数，因为它们已经在SensorManager中注册了
        # 这可以防止"Assertion failed: (_clients.find(token.get_stream_id())) == (_clients.end())"错误

        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        
        # 只获取世界tick事件的数据，我们不监听传感器事件
        world_snapshot = self._retrieve_data(self._queues[0], timeout)
        
        # 返回世界快照
        return [world_snapshot]

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

# 修改run_simulation函数，使用CarlaSyncMode进行传感器同步
def run_simulation(args, client):
    """Run simulation"""
    display_manager = None
    vehicle = None
    vehicle_list = []
    other_vehicles = []
    timer = CustomTimer()
    yolo_model = None
    radar_inference = None
    lidar_inference = None
    radar_overview = None
    vehicle_prediction = None  # 添加车辆预测管理器变量
    sensor_list = []  # 用于存储所有传感器实例，便于同步

    try:
        # 加载YOLO模型
        if args.yolo_weights:
            print(f"Loading YOLOv5 model: {args.yolo_weights}")
            yolo_model = load_yolo_model(args.yolo_weights, device=args.device)
            if yolo_model:
                print("YOLOv5 model loaded successfully, will be used for real-time detection")
            else:
                print("YOLOv5 model loading failed, detection will be disabled")
        
        # 加载雷达推理模型
        if args.radar_model:
            print(f"Loading radar inference model: {args.radar_model}")
            try:
                radar_inference = RadarInference(args.radar_model)
                print("Radar inference model loaded successfully, will be used for vehicle detection")
            except Exception as e:
                print(f"Radar inference model loading failed: {e}")
                radar_inference = None
        
        # 加载LiDAR推理模型
        if args.lidar_model:
            print(f"Loading LiDAR inference model: {args.lidar_model}")
            try:
                lidar_inference = LidarInference(args.lidar_model)
                print("LiDAR inference model loaded successfully, will be used for vehicle detection")
            except Exception as e:
                print(f"LiDAR inference model loading failed: {e}")
                lidar_inference = None
        
        # 使用模块化的世界初始化函数
        world, original_settings = initialize_world(client, args)
        
        # 生成自我驾驶车辆
        vehicle, spawn_idx, spawn_points = spawn_ego_vehicle(world, args, client)
        vehicle_list.append(vehicle)
        
        # 生成其他车辆
        if args.vehicles > 0:
            # 更新可用生成点列表，移除已被使用的点
            available_spawn_points = [p for i, p in enumerate(spawn_points) if i != spawn_idx]
            other_vehicles = spawn_surrounding_vehicles(client, world, min(args.vehicles, len(available_spawn_points)), available_spawn_points)

        # 修改显示布局为3*4
        display_manager = DisplayManager(grid_size=[3, 4], window_size=[args.width, args.height])

        # 创建传感器时使用命令行参数设置range
        range_str = str(args.range)  # 将range转换为字符串
        print(f"Sensor range set to: {range_str} meters")

        # 创建雷达概览管理器 - 确保在传感器创建之前初始化
        radar_overview = RadarOverviewManager(display_manager, display_pos=[1, 3])
        print(f"Created radar overview manager at position [1, 3]")
        
        # 创建车辆运动预测管理器（独立窗口）
        vehicle_prediction = VehicleMotionPredictionManager(world, ego_vehicle=vehicle, window_size=(800, 600), window_title="Vehicle Motion Prediction")
        print(f"Created vehicle motion prediction manager in a separate window")

        # Create sensors - First row: Cameras with YOLO detection
        if yolo_model:
            # 使用YOLOv5增强的相机传感器
            # Front camera
            front_camera = YOLOSensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(yaw=0)), 
                         vehicle, {}, display_pos=[0, 1], model=yolo_model, 
                         conf_thres=args.conf_thres, iou_thres=args.iou_thres, img_size=args.img_size,
                         sensor_name="Front Camera+YOLO")
            sensor_list.append(front_camera.sensor)
            
            # Left camera
            left_camera = YOLOSensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=-90)), 
                         vehicle, {}, display_pos=[0, 0], model=yolo_model, 
                         conf_thres=args.conf_thres, iou_thres=args.iou_thres, img_size=args.img_size,
                         sensor_name="Left Camera+YOLO")
            sensor_list.append(left_camera.sensor)
            
            # Right camera - 使用YOLO识别
            right_camera = YOLOSensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=90)), 
                         vehicle, {}, display_pos=[0, 2], model=yolo_model, 
                         conf_thres=args.conf_thres, iou_thres=args.iou_thres, img_size=args.img_size,
                         sensor_name="Right Camera+YOLO")
            sensor_list.append(right_camera.sensor)
            
            # Rear camera - 使用YOLO识别
            rear_camera = YOLOSensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=180)), 
                         vehicle, {}, display_pos=[0, 3], model=yolo_model, 
                         conf_thres=args.conf_thres, iou_thres=args.iou_thres, img_size=args.img_size,
                         sensor_name="Rear Camera+YOLO")
            sensor_list.append(rear_camera.sensor)

            # 移除45度角摄像头
        
        else:
            # 使用原始相机传感器（无YOLO）
            # Front camera
            front_camera = SensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(yaw=0)), 
                         vehicle, {}, display_pos=[0, 1], sensor_name="Front Camera")
            sensor_list.append(front_camera.sensor)
            
            # Left camera
            left_camera = SensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=-90)), 
                         vehicle, {}, display_pos=[0, 0], sensor_name="Left Camera")
            sensor_list.append(left_camera.sensor)
            
            # Right camera
            right_camera = SensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=90)), 
                         vehicle, {}, display_pos=[0, 2], sensor_name="Right Camera")
            sensor_list.append(right_camera.sensor)
            
            # Rear camera
            rear_camera = SensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=180)), 
                         vehicle, {}, display_pos=[0, 3], sensor_name="Rear Camera")
            sensor_list.append(rear_camera.sensor)
            
            # 45度角相机配置在这里也移除

        # Second row: Radars - 使用命令行参数设置range和RadarSensorManager进行车辆检测
        radar_sensor_options = {
            'horizontal_fov': '120', 
            'vertical_fov': '10', 
            'range': range_str, 
            'points_per_second': '1500'
        }
        
        # Forward radar
        front_radar = RadarSensorManager(world, display_manager, 'Radar', 
                    carla.Transform(carla.Location(x=2.0, z=1.0), carla.Rotation(yaw=0)), 
                    vehicle, radar_sensor_options, 
                    display_pos=[1, 1], inference_model=radar_inference, 
                    conf_thres=args.radar_conf_thres, sensor_name="Front Radar+Detector")
        sensor_list.append(front_radar.sensor)
        
        # Left radar
        left_radar = RadarSensorManager(world, display_manager, 'Radar', 
                    carla.Transform(carla.Location(y=1.0, z=1.0), carla.Rotation(yaw=90)), 
                    vehicle, radar_sensor_options, 
                    display_pos=[1, 0], inference_model=radar_inference, 
                    conf_thres=args.radar_conf_thres, sensor_name="Left Radar+Detector")
        sensor_list.append(left_radar.sensor)
        
        # Right radar
        right_radar = RadarSensorManager(world, display_manager, 'Radar', 
                    carla.Transform(carla.Location(y=-1.0, z=1.0), carla.Rotation(yaw=270)), 
                    vehicle, radar_sensor_options, 
                    display_pos=[1, 2], inference_model=radar_inference, 
                    conf_thres=args.radar_conf_thres, sensor_name="Right Radar+Detector")
        sensor_list.append(right_radar.sensor)
        
        # Rear radar
        rear_radar = RadarSensorManager(world, display_manager, 'Radar', 
                    carla.Transform(carla.Location(x=-2.0, z=1.0), carla.Rotation(yaw=180)), 
                    vehicle, radar_sensor_options, 
                    display_pos=[1, 3], inference_model=radar_inference, 
                    conf_thres=args.radar_conf_thres, sensor_name="Rear Radar+Detector")
        sensor_list.append(rear_radar.sensor)

        # 将雷达传感器注册到雷达概览管理器 - 确保正确注册
        print("Registering radar sensors to overview manager...")
        radar_overview.register_radar_sensor(front_radar)
        radar_overview.register_radar_sensor(left_radar)
        radar_overview.register_radar_sensor(right_radar)
        radar_overview.register_radar_sensor(rear_radar)
        
        # Third row: LiDAR and additional sensors - 使用命令行参数设置range
        # LiDAR传感器选项
        lidar_sensor_options = {
            'channels': '16', 
            'range': range_str, 
            'points_per_second': '50000',  # 减少点数量以提高性能
            'rotation_frequency': '20',
            'upper_fov': '10',
            'lower_fov': '-30',
            'sensor_tick': '0.2'  # 5Hz刷新率(限制为5fps以提高性能)
        }
        
        # 根据是否有LiDAR推理模型来使用不同的传感器管理器
        if lidar_inference:
            # 使用带推理模型的LiDAR传感器管理器
            lidar_sensor = LidarSensorManager(world, display_manager, 'LiDAR', 
                             carla.Transform(carla.Location(x=-0.2, z=2.4)), 
                             vehicle, lidar_sensor_options, 
                             display_pos=[2, 1], inference_model=lidar_inference,
                             conf_thres=args.lidar_conf_thres, sensor_name="LiDAR+Detector")
            sensor_list.append(lidar_sensor.sensor)
        else:
            # 使用原始LiDAR传感器管理器
            lidar_sensor = SensorManager(world, display_manager, 'LiDAR', 
                         carla.Transform(carla.Location(x=0, z=2.4)), 
                         vehicle, lidar_sensor_options, 
                         display_pos=[2, 1], sensor_name="LiDAR")
            sensor_list.append(lidar_sensor.sensor)
        
        # 添加鸟瞰图 (Bird's eye view)
        bird_eye_view = SensorManager(world, display_manager, 'RGBCamera', 
                     carla.Transform(carla.Location(x=0, z=30.0), carla.Rotation(pitch=-90)), 
                     vehicle, {'fov': '90'}, display_pos=[2, 0], sensor_name="Bird's Eye View")
        sensor_list.append(bird_eye_view.sensor)

        # 确保我们在同步模式下运行
        args.sync = True
        print(f"启用同步模式用于传感器同步")
        
        # 创建CarlaSyncMode实例，用于同步所有传感器
        fps = 20  # 设置模拟的FPS
        print(f"设置同步模式FPS: {fps}")
        
        # Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        print(f"基线演示启动，包含 {len(other_vehicles)} 辆额外车辆。按ESC或Q退出。")
        
        # 使用同步模式上下文
        with CarlaSyncMode(world, fps=fps) as sync_mode:
            while True:
                # 使用同步模式，等待所有传感器数据
                try:
                    # 只获取世界快照，不再获取传感器数据
                    # 因为传感器数据处理已经由各自的回调函数处理
                    sync_data = sync_mode.tick(timeout=2.0)
                    snapshot = sync_data[0]  # 第一个元素是world.tick的结果
                    
                    # 确保雷达概览在每帧都更新
                    if radar_overview:
                        radar_overview.update()
                    
                    # 确保车辆运动预测在每帧都更新（OpenCV独立窗口）
                    if vehicle_prediction and vehicle_prediction.running:
                        vehicle_prediction.update()
                    
                    # 渲染所有传感器显示
                    display_manager.render()
                    
                    # 单独渲染雷达概览
                    if radar_overview:
                        radar_overview.render()
                    
                    # 如果预测窗口关闭，重新创建
                    if vehicle_prediction and not vehicle_prediction.running:
                        print("Prediction window closed, recreating...")
                        vehicle_prediction = VehicleMotionPredictionManager(world, ego_vehicle=vehicle, window_size=(800, 600), window_title="Vehicle Motion Prediction")
                    
                    # 每次迭代打印当前帧信息，可用于调试
                    fps_sim = round(1.0 / snapshot.timestamp.delta_seconds)
                    print(f"\r模拟运行中，FPS: {fps_sim} (帧: {sync_mode.frame})", end="")
                    
                except Exception as e:
                    print(f"\n同步模式执行期间出错: {e}")
                
                # 检查退出事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        call_exit = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_q:
                            call_exit = True
                            break

                if call_exit:
                    break

    finally:
        # 清理独立窗口
        if 'vehicle_prediction' in locals() and vehicle_prediction:
            vehicle_prediction.destroy()
            
        if display_manager:
            display_manager.destroy()

        # Clean up all vehicles
        all_vehicles = vehicle_list + other_vehicles
        client.apply_batch([carla.command.DestroyActor(x) for x in all_vehicles])
        if 'original_settings' in locals():
            world.apply_settings(original_settings)

# 在RadarOverviewManager类后添加新的VehicleMotionPredictionManager类
class VehicleMotionPredictionManager:
    """
    车辆运动预测管理器类，用于在独立OpenCV窗口中显示NPC车辆位置、运动预测、速度向量和碰撞时间
    """
    def __init__(self, world, ego_vehicle, window_size=(640, 480), window_title="Vehicle Motion Prediction"):
        """
        初始化车辆运动预测管理器
        Args:
            world: Carla世界对象
            ego_vehicle: 自动驾驶车辆
            window_size: 独立窗口的大小
            window_title: 独立窗口的标题
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.window_size = window_size
        self.window_title = window_title
        
        # 使用OpenCV创建窗口
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_title, self.window_size[0], self.window_size[1])
        
        self.timer = CustomTimer()
        self.time_processing = 0.0
        self.tics_processing = 0
        self.running = True
        
        # 初始化画面
        self.init_screen()
    
    def init_screen(self):
        """初始化画面"""
        try:
            init_img = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            cv2.putText(init_img, "Vehicle Motion Prediction", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(init_img, "Initializing...", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # 显示初始画面
            cv2.imshow(self.window_title, init_img)
            cv2.waitKey(1)  # 不等待按键，只刷新显示
        except Exception as e:
            print(f"Error initializing prediction window: {e}")
            self.running = False
    
    def update(self):
        """更新车辆运动预测视图"""
        if not self.running:
            return
        
        # 检查窗口是否关闭
        try:
            if cv2.getWindowProperty(self.window_title, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
                return
        except:
            self.running = False
            return
        
        t_start = self.timer.time()
        
        try:
            # 创建黑色背景
            prediction_img = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            
            # 添加标题
            cv2.putText(prediction_img, "Vehicle Motion Prediction", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 计算中心点（自车位置）
            center_x, center_y = int(self.window_size[0] / 2), int(self.window_size[1] / 2)
            
            # 绘制自车位置
            cv2.circle(prediction_img, (center_x, center_y), 8, (0, 0, 255), -1)  # 红色圆表示自车
            cv2.putText(prediction_img, "Ego", (center_x + 10, center_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 绘制参考线
            # 绘制同心圆，表示距离
            for r in range(1, 6):
                radius = int(min(self.window_size) / 10 * r)
                cv2.circle(prediction_img, (center_x, center_y), radius, (50, 50, 50), 1)
                # 添加距离标签
                cv2.putText(prediction_img, f"{r*10}m", (center_x + radius - 20, center_y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            # 绘制坐标轴
            cv2.line(prediction_img, (center_x, 0), (center_x, self.window_size[1]), (50, 50, 50), 1)
            cv2.line(prediction_img, (0, center_y), (self.window_size[0], center_y), (50, 50, 50), 1)
            
            # 获取自车速度和位置
            ego_location = self.ego_vehicle.get_location()
            ego_velocity = self.ego_vehicle.get_velocity()
            ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)
            
            # 显示自车速度
            cv2.putText(prediction_img, f"Ego Speed: {ego_speed*3.6:.1f} km/h", (10, self.window_size[1]-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 获取所有NPC车辆
            vehicles = self.world.get_actors().filter('vehicle.*')
            
            # 计算和绘制每个NPC车辆
            npc_count = 0
            
            for vehicle in vehicles:
                # 跳过自车
                if vehicle.id == self.ego_vehicle.id:
                    continue
                
                # 获取NPC车辆位置和速度
                npc_location = vehicle.get_location()
                npc_velocity = vehicle.get_velocity()
                
                # 计算NPC车辆相对于自车的位置
                rel_x = npc_location.x - ego_location.x
                rel_y = npc_location.y - ego_location.y
                distance = math.sqrt(rel_x**2 + rel_y**2)
                
                # 只显示一定范围内的车辆（50米内）
                if distance > 50:
                    continue
                
                # 计算图像上的位置
                scale = min(self.window_size) / 100.0  # 缩放因子，100米映射到整个视图
                px = center_x + int(rel_y * scale)  # 注意：x对应于自车前进方向，即y轴
                py = center_y - int(rel_x * scale)  # y对应于自车左右方向，即x轴，需要取反
                
                # 计算NPC车辆的速度大小和方向
                npc_speed = math.sqrt(npc_velocity.x**2 + npc_velocity.y**2 + npc_velocity.z**2)
                
                # 计算相对速度向量
                rel_vx = npc_velocity.x - ego_velocity.x
                rel_vy = npc_velocity.y - ego_velocity.y
                rel_speed = math.sqrt(rel_vx**2 + rel_vy**2)
                
                # 计算碰撞时间 (TTC)
                ttc = float('inf')  # 默认无碰撞风险
                if rel_speed > 0.1:  # 避免除以接近0的速度
                    # 计算相对位置和相对速度的点积
                    rel_pos_dot_vel = rel_x * rel_vx + rel_y * rel_vy
                    if rel_pos_dot_vel < 0:  # 车辆相互靠近
                        ttc = distance / rel_speed
                
                # 根据距离和TTC选择颜色
                if ttc < 1.0:  # 1秒内可能碰撞
                    color = (0, 0, 255)  # 红色 - 危险
                elif ttc < 3.0:  # 3秒内可能碰撞
                    color = (0, 165, 255)  # 橙色 - 警告
                elif ttc < 5.0:  # 5秒内可能碰撞
                    color = (0, 255, 255)  # 黄色 - 注意
                else:
                    color = (0, 255, 0)  # 绿色 - 安全
                
                # 绘制NPC车辆
                # 车辆边界框大小随距离变化（越近越大）
                box_size = max(6, int(15 - distance * 0.2))
                cv2.rectangle(prediction_img, 
                            (px - box_size, py - box_size), 
                            (px + box_size, py + box_size), 
                            color, 2)
                
                # 绘制速度向量箭头
                if npc_speed > 0.1:  # 只绘制移动中的车辆
                    # 速度向量长度与速度成正比
                    arrow_length = int(npc_speed * 1.5)
                    # 计算速度向量方向
                    vx_norm = npc_velocity.x / npc_speed
                    vy_norm = npc_velocity.y / npc_speed
                    # 计算箭头终点
                    end_x = px + int(vy_norm * arrow_length * scale)
                    end_y = py - int(vx_norm * arrow_length * scale)
                    # 绘制箭头
                    cv2.arrowedLine(prediction_img, (px, py), (end_x, end_y), color, 2)
                
                # 显示车辆信息：距离、速度和TTC
                cv2.putText(prediction_img, f"ID:{vehicle.id}", (px + box_size + 5, py - 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(prediction_img, f"{distance:.1f}m", (px + box_size + 5, py - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(prediction_img, f"{npc_speed*3.6:.1f}km/h", (px + box_size + 5, py + 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 显示碰撞时间
                if ttc < float('inf'):
                    cv2.putText(prediction_img, f"TTC:{ttc:.1f}s", (px + box_size + 5, py + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # 根据车辆当前速度和位置预测未来路径
                if npc_speed > 0.5:  # 只对移动中的车辆进行预测
                    # 预测未来3秒，每1秒一个点
                    prev_pred_x, prev_pred_y = px, py
                    for t in range(1, 4):
                        # 简单线性预测
                        pred_x = rel_x + rel_vx * t
                        pred_y = rel_y + rel_vy * t
                        
                        # 转换为图像坐标
                        pred_px = center_x + int(pred_y * scale)
                        pred_py = center_y - int(pred_x * scale)
                        
                        # 绘制预测点
                        cv2.circle(prediction_img, (pred_px, pred_py), 3, color, -1)
                        
                        # 连接预测点
                        if t > 1:
                            cv2.line(prediction_img, (prev_pred_x, prev_pred_y), (pred_px, pred_py), color, 1, cv2.LINE_AA)
                        
                        prev_pred_x, prev_pred_y = pred_px, pred_py
                
                npc_count += 1
            
            # 显示检测到的NPC车辆数量
            cv2.putText(prediction_img, f"Detected Vehicles: {npc_count}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示OpenCV窗口
            cv2.imshow(self.window_title, prediction_img)
            
            # 处理按键但不阻塞
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):  # 按ESC或Q退出
                self.running = False
                cv2.destroyWindow(self.window_title)
                
        except Exception as e:
            print(f"Error updating motion prediction: {e}")
            import traceback
            traceback.print_exc()
        
        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1
    
    def render(self):
        """兼容原接口，但不执行操作"""
        pass
    
    def destroy(self):
        """销毁窗口"""
        try:
            if self.running:
                self.running = False
                cv2.destroyWindow(self.window_title)
        except Exception as e:
            print(f"Error closing prediction window: {e}")

def main():
    """Main function"""
    argparser = argparse.ArgumentParser(description='CARLA Autonomous Driving Perception System Baseline Demo')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='CARLA server IP (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='CARLA server port (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=True,  # 修改为默认开启
        help='Enable synchronous mode (default: True, required for sensor synchronization)')
    argparser.add_argument(
        '--width',
        metavar='W',
        default=1920,
        type=int,
        help='Window width (default: 1920)')
    argparser.add_argument(
        '--height',
        metavar='H',
        default=1080,
        type=int,
        help='Window height (default: 1080)')
    argparser.add_argument(
        '--vehicles',
        metavar='N',
        default=20,
        type=int,
        help='Number of vehicles to generate (default: 20)')
    argparser.add_argument(
        '--seed',
        metavar='S',
        default=0,
        type=int,
        help='Random seed for reproducible results (default: 0, meaning random behavior)')
    argparser.add_argument(
        '--autopilot',
        action='store_true',
        default=True,
        help='Enable vehicle autonomous driving (default: True)')
    argparser.add_argument(
        '--no-autopilot',
        dest='autopilot',
        action='store_false',
        help='Disable vehicle autonomous driving')
    argparser.add_argument(
        '--range',
        metavar='R',
        default=50,
        type=int,
        help='Range for radar and LiDAR sensors, in meters (default: 50)')
    argparser.add_argument(
        '--weather',
        choices=['default', 'badweather', 'night', 'badweather_night'],
        default='default',
        help='Weather preset: default(default sunny day), badweather(bad weather), night(night), badweather_night(bad weather at night)')
    argparser.add_argument(
        '--yolo-weights',
        type=str,
        default='',
        help='YOLOv5 model weights path, e.g. "runs/train/vehicle_detection/weights/best.pt"')
    argparser.add_argument(
        '--conf-thres',
        type=float,
        default=0.5,
        help='YOLOv5 detection confidence threshold (default: 0.5)')
    argparser.add_argument(
        '--iou-thres',
        type=float,
        default=0.45,
        help='YOLOv5 detection IoU threshold (default: 0.45)')
    argparser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='YOLOv5 model input image size (default: 640)')
    argparser.add_argument(
        '--device',
        type=str,
        default='',
        help='CUDA device, e.g. "0" or "cpu" (default is auto-selected)')
    argparser.add_argument(
        '--radar-model',
        type=str,
        default='./radar_model',
        help='Radar inference model directory, containing radar_model.pth and normalizer_params.npz (default: ./radar_model)')
    argparser.add_argument(
        '--radar-conf-thres',
        type=float,
        default=0.7,
        help='Radar detection confidence threshold (default: 0.7)')
    argparser.add_argument(
        '--lidar-model',
        type=str,
        default='',
        help='LiDAR inference model directory, containing best_model.pth and normalization_params.json')
    argparser.add_argument(
        '--lidar-conf-thres',
        type=float,
        default=0.6,
        help='LiDAR detection confidence threshold (default: 0.6)')

    args = argparser.parse_args()

    try:
        # 确保导入OpenCV
        import cv2
        
        global client
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)  # 增加超时时间，用于车辆清理
        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nUser cancelled. Goodbye!')
    except Exception as e:
        print(f'Error running simulation: {e}')

if __name__ == '__main__':
    # Import required libraries
    main() 