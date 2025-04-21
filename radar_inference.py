#!/usr/bin/env python

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_radar_model import RadarPointClassifier, SimpleNormalization

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
        # 加载归一化参数
        normalizer_path = os.path.join(self.model_dir, 'normalizer_params.npz')
        if not os.path.exists(normalizer_path):
            raise FileNotFoundError(f"找不到归一化参数文件: {normalizer_path}")
        
        normalizer_params = np.load(normalizer_path)
        self.normalizer = SimpleNormalization(
            mean=normalizer_params['mean'],
            std=normalizer_params['std']
        )
        
        # 加载模型
        model_path = os.path.join(self.model_dir, 'radar_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        # 创建模型 (需要与训练时相同的结构)
        input_size = len(self.normalizer.mean)  # 输入特征数量与归一化参数一致
        self.model = RadarPointClassifier(input_size=input_size, hidden_size=128)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        print(f"模型和归一化参数加载成功")
    
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
    
    def predict_json_file(self, json_file, confidence_threshold=0.5):
        """处理单个JSON文件中的所有点"""
        # 加载JSON文件
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        points = data["points"]
        total_points = len(points)
        predicted_vehicle_points = []
        
        # 处理每个点
        for point in points:
            # 提取特征
            features = self.extract_features(point)
            
            # 预测
            is_vehicle, confidence = self.predict_point(features)
            
            if is_vehicle == 1 and confidence >= confidence_threshold:
                point_info = {
                    "id": point["id"],
                    "image_position": point["image_position"],
                    "confidence": confidence,
                    "depth": point["depth"],
                    "world_position": point["world_position"]
                }
                predicted_vehicle_points.append(point_info)
        
        print(f"文件 {os.path.basename(json_file)} 中的点总数: {total_points}")
        print(f"预测为车辆的点数量: {len(predicted_vehicle_points)}")
        
        return {
            "frame_id": data["frame_id"],
            "timestamp": data["timestamp"],
            "total_points": total_points,
            "predicted_vehicle_points": predicted_vehicle_points
        }
    
    def visualize_predictions(self, predictions, original_json_file, output_path=None):
        """可视化预测结果"""
        # 加载原始数据
        with open(original_json_file, 'r') as f:
            original_data = json.load(f)
        
        # 创建图像
        plt.figure(figsize=(10, 8))
        
        # 绘制所有点
        for point in original_data["points"]:
            x, y = point["image_position"]
            
            if point["is_hitting_vehicle"]:
                # 实际击中车辆的点
                plt.scatter(x, y, c='green', s=30, marker='o', alpha=0.7, label='真实车辆点')
            else:
                # 非车辆点
                plt.scatter(x, y, c='blue', s=10, marker='.', alpha=0.3)
        
        # 绘制预测为车辆的点
        for point in predictions["predicted_vehicle_points"]:
            x, y = point["image_position"]
            confidence = point["confidence"]
            plt.scatter(x, y, c='red', s=50, marker='x', alpha=min(1.0, confidence + 0.3))
        
        # 添加图例和标题
        plt.title(f"雷达点云车辆检测 - 帧 {predictions['frame_id']}")
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        
        # 反转Y轴（因为图像坐标是从上到下的）
        plt.gca().invert_yaxis()
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='.', color='blue', label='非车辆点', markersize=10, linestyle='None', alpha=0.3),
            Line2D([0], [0], marker='o', color='green', label='实际车辆点', markersize=10, linestyle='None', alpha=0.7),
            Line2D([0], [0], marker='x', color='red', label='预测车辆点', markersize=10, linestyle='None')
        ]
        plt.legend(handles=legend_elements)
        
        # 保存图像
        if output_path:
            plt.savefig(output_path)
            print(f"预测可视化已保存到 {output_path}")
        
        plt.show()

def batch_process_directory(inference_obj, data_dir, output_dir=None, confidence_threshold=0.7, visualize=True):
    """批量处理目录中的所有JSON文件"""
    # 获取所有JSON文件
    json_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    
    if not json_files:
        print(f"在 {data_dir} 中没有找到JSON文件")
        return
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个文件
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        print(f"\n处理文件: {json_file}")
        
        # 预测
        predictions = inference_obj.predict_json_file(file_path, confidence_threshold)
        
        # 保存预测结果
        if output_dir:
            output_file = os.path.join(output_dir, f"pred_{json_file}")
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2)
        
        # 可视化
        if visualize and output_dir:
            output_image = os.path.join(output_dir, f"pred_{os.path.splitext(json_file)[0]}.png")
            inference_obj.visualize_predictions(predictions, file_path, output_image)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='雷达点云车辆检测推理')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='训练好的模型目录')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='雷达点云数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='检测置信度阈值 (0-1)')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.model_dir):
        raise ValueError(f"模型目录不存在: {args.model_dir}")
    
    if not os.path.exists(args.data_dir):
        raise ValueError(f"数据目录不存在: {args.data_dir}")
    
    # 创建推理器
    inference = RadarInference(args.model_dir)
    
    # 批量处理
    batch_process_directory(
        inference,
        args.data_dir,
        args.output_dir,
        args.confidence,
        args.visualize
    )

if __name__ == "__main__":
    main() 