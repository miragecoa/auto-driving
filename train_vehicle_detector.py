#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import torch
import argparse
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='训练车辆检测模型')
    parser.add_argument('--data_path', type=str, default='vehicle_dataset', 
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='训练批量大小')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='训练轮数')
    parser.add_argument('--img_size', type=int, default=640, 
                        help='输入图像尺寸')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', 
                        help='初始权重')
    parser.add_argument('--device', default='', 
                        help='cuda设备, 例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--project', default='runs/train', 
                        help='保存结果的目录')
    parser.add_argument('--name', default='vehicle_detection', 
                        help='实验名称')
    
    return parser.parse_args()

def check_yolov5_environment():
    """检查YOLOv5环境并安装必要的依赖"""
    yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
    if not os.path.exists(yolov5_dir):
        print(f"未找到YOLOv5目录: {yolov5_dir}")
        return False
    
    requirements_file = os.path.join(yolov5_dir, 'requirements.txt')
    if not os.path.exists(requirements_file):
        print(f"未找到YOLOv5依赖文件: {requirements_file}")
        return False
    
    # 检查是否已安装依赖
    try:
        # 尝试导入一些关键依赖
        import torch
        import yaml
        from tqdm import tqdm
        
        print("YOLOv5基本依赖已安装")
        return True
    except ImportError as e:
        print(f"缺少必要的依赖: {e}")
        # 询问用户是否安装依赖
        response = input("是否要安装YOLOv5必要的依赖? (y/n): ")
        if response.lower() == 'y':
            print(f"正在安装YOLOv5依赖，这可能需要几分钟...")
            import subprocess
            try:
                subprocess.check_call(['pip', 'install', '-r', requirements_file])
                print("YOLOv5依赖安装完成")
                return True
            except subprocess.CalledProcessError as e:
                print(f"安装依赖失败: {e}")
                return False
        else:
            print("用户选择不安装依赖")
            return False

def create_dataset_yaml(data_path):
    """创建数据集配置文件"""
    yaml_path = os.path.join(data_path, 'dataset.yaml')
    
    # 确保images和annotations目录存在
    img_dir = os.path.join(data_path, 'images')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"未找到图像目录: {img_dir}")
    
    # 计算图像数量
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    num_images = len(img_files)
    
    # 创建训练/验证集划分 (80% 训练, 20% 验证)
    train_size = int(num_images * 0.8)
    train_imgs = img_files[:train_size]
    val_imgs = img_files[train_size:]
    
    # 创建训练和验证图像列表文件
    with open(os.path.join(data_path, 'train.txt'), 'w') as f:
        for img in train_imgs:
            f.write(os.path.join(img_dir, img) + '\n')
    
    with open(os.path.join(data_path, 'val.txt'), 'w') as f:
        for img in val_imgs:
            f.write(os.path.join(img_dir, img) + '\n')
    
    # 创建数据集配置
    data_yaml = {
        'path': os.path.abspath(data_path),
        'train': os.path.join(os.path.abspath(data_path), 'train.txt'),
        'val': os.path.join(os.path.abspath(data_path), 'val.txt'),
        'nc': 1,  # 1个类别（车辆）
        'names': ['vehicle']
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"已创建数据集配置文件: {yaml_path}")
    print(f"训练集: {len(train_imgs)} 张图像, 验证集: {len(val_imgs)} 张图像")
    
    return yaml_path

def convert_voc_to_yolo(data_path):
    """将VOC格式的标注转换为YOLO格式"""
    import xml.etree.ElementTree as ET
    
    annotations_dir = os.path.join(data_path, 'annotations')
    images_dir = os.path.join(data_path, 'images')
    labels_dir = os.path.join(data_path, 'labels')
    
    # 创建labels目录
    os.makedirs(labels_dir, exist_ok=True)
    
    # 处理所有XML文件
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    for xml_file in xml_files:
        # 解析XML
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        
        # 创建对应的YOLO格式标签文件
        txt_file = os.path.join(labels_dir, xml_file.replace('.xml', '.txt'))
        
        with open(txt_file, 'w') as f:
            # 处理每个object
            for obj in root.findall('object'):
                # 对于车辆,类别索引为0
                class_id = 0
                
                # 获取边界框坐标
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # 转换为YOLO格式 (中心点坐标和宽高,归一化到0-1)
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                # 写入YOLO格式
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    print(f"已将 {len(xml_files)} 个VOC格式标注转换为YOLO格式")

def train_model(data_yaml, args):
    """准备YOLOv5训练命令，但不执行训练"""
    # 检查YOLOv5目录
    yolov5_dir = os.path.join(os.getcwd(), 'yolov5')
    if not os.path.exists(yolov5_dir):
        print(f"YOLOv5目录不存在: {yolov5_dir}")
        print("请先克隆YOLOv5仓库: git clone https://github.com/ultralytics/yolov5.git")
        return None
    
    # 构建训练命令
    train_cmd = f"python {os.path.join(yolov5_dir, 'train.py')} --data={data_yaml} --epochs={args.epochs} --batch-size={args.batch_size} --img={args.img_size} --weights={args.weights} --project={args.project} --name={args.name} --single-cls"
    
    if args.device:
        train_cmd += f" --device={args.device}"
    
    return train_cmd

def main():
    args = parse_arguments()
    
    print("=== 车辆检测模型训练准备 ===")
    print(f"数据集路径: {args.data_path}")
    
    # 检查YOLOv5环境
    print("\n预检查: 检查YOLOv5环境...")
    env_ok = check_yolov5_environment()
    if not env_ok:
        print("YOLOv5环境未正确设置，请先解决环境问题")
        print("1. 确保YOLOv5仓库已克隆到当前目录")
        print("2. 安装必要的依赖: pip install -r yolov5/requirements.txt")
        return
    
    # 转换VOC标注为YOLO格式
    print("\n第1步: 转换标注格式...")
    convert_voc_to_yolo(args.data_path)
    
    # 创建数据集配置
    print("\n第2步: 创建数据集配置...")
    data_yaml = create_dataset_yaml(args.data_path)
    
    # 准备训练命令但不执行
    print("\n第3步: 准备训练命令...")
    train_cmd = train_model(data_yaml, args)
    
    if train_cmd:
        print("\n数据准备完成！请手动执行以下命令开始训练:")
        print(train_cmd)

if __name__ == "__main__":
    main()