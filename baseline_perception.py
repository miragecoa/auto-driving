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
            radar_bp.set_attribute('sensor_tick', '0.1')
            
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
        
        # 调用父类构造函数
        super().__init__(world, display_man, sensor_type, transform, attached, sensor_options, display_pos, sensor_name)
    
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
        
        # 处理每个雷达点
        for i, detect in enumerate(radar_data):
            total_points += 1
            
            # 获取点信息
            distance = detect.depth
            azimuth = detect.azimuth
            altitude = detect.altitude
            velocity = detect.velocity
            
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
                    "velocity": velocity,
                    "snr": snr,
                    "world_position": [world_x, world_y, world_z],
                    "image_position": [x, y]
                }
                
                radar_points.append(point_data)
                
                # 默认点颜色 - 基于速度（接近为蓝色，远离为红色）
                if velocity > 0:  # 接近中的物体
                    color = (0, int(255 - min(255, abs(velocity) * 10)), min(255, abs(velocity) * 25))  # 绿色到蓝色
                else:  # 远离的物体
                    color = (min(255, abs(velocity) * 25), int(255 - min(255, abs(velocity) * 10)), 0)  # 绿色到红色
                
                # 默认绘制点
                point_size = min(5, max(3, int(abs(velocity) / 5) + 2))
                
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
                            
                            # 只显示置信度数值
                            cv2.putText(radar_img, f"{confidence:.2f}", (x+5, y-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                            
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
                if abs(velocity) > 5.0:
                    line_length = min(15, max(5, int(abs(velocity))))
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

def run_simulation(args, client):
    """Run simulation"""
    display_manager = None
    vehicle = None
    vehicle_list = []
    other_vehicles = []
    timer = CustomTimer()
    yolo_model = None
    radar_inference = None
    radar_overview = None

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
        
        # 使用模块化的世界初始化函数
        world, original_settings = initialize_world(client, args)
        
        # 生成自我驾驶车辆
        vehicle, spawn_idx, spawn_points = spawn_ego_vehicle(world, args)
        vehicle_list.append(vehicle)
        
        # 生成其他车辆
        if args.vehicles > 0:
            # 更新可用生成点列表，移除已被使用的点
            available_spawn_points = [p for i, p in enumerate(spawn_points) if i != spawn_idx]
            other_vehicles = spawn_surrounding_vehicles(client, world, min(args.vehicles, len(available_spawn_points)), available_spawn_points)

        # 修改显示布局为3*5
        display_manager = DisplayManager(grid_size=[3, 5], window_size=[args.width, args.height])

        # 创建传感器时使用命令行参数设置range
        range_str = str(args.range)  # 将range转换为字符串
        print(f"Sensor range set to: {range_str} meters")

        # 创建雷达概览管理器 - 确保在传感器创建之前初始化
        radar_overview = RadarOverviewManager(display_manager, display_pos=[1, 4])
        print(f"Created radar overview manager at position [1, 4]")

        # Create sensors - First row: Cameras with YOLO detection
        if yolo_model:
            # 使用YOLOv5增强的相机传感器
            # Front camera
            YOLOSensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(yaw=0)), 
                         vehicle, {}, display_pos=[0, 1], model=yolo_model, 
                         conf_thres=args.conf_thres, iou_thres=args.iou_thres, img_size=args.img_size,
                         sensor_name="Front Camera+YOLO")
            
            # Left camera
            YOLOSensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=-90)), 
                         vehicle, {}, display_pos=[0, 0], model=yolo_model, 
                         conf_thres=args.conf_thres, iou_thres=args.iou_thres, img_size=args.img_size,
                         sensor_name="Left Camera+YOLO")
            
            # Right camera
            YOLOSensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=90)), 
                         vehicle, {}, display_pos=[0, 2], model=yolo_model, 
                         conf_thres=args.conf_thres, iou_thres=args.iou_thres, img_size=args.img_size,
                         sensor_name="Right Camera+YOLO")
            
            # Rear camera
            YOLOSensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=180)), 
                         vehicle, {}, display_pos=[0, 3], model=yolo_model, 
                         conf_thres=args.conf_thres, iou_thres=args.iou_thres, img_size=args.img_size,
                         sensor_name="Rear Camera+YOLO")
            
        else:
            # 使用原始相机传感器（无YOLO）
            # Front camera
            SensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(yaw=0)), 
                         vehicle, {}, display_pos=[0, 1], sensor_name="Front Camera")
            
            # Left camera
            SensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=-90)), 
                         vehicle, {}, display_pos=[0, 0], sensor_name="Left Camera")
            
            # Right camera
            SensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=90)), 
                         vehicle, {}, display_pos=[0, 2], sensor_name="Right Camera")
            
            # Rear camera
            SensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=180)), 
                         vehicle, {}, display_pos=[0, 3], sensor_name="Rear Camera")

            # 添加45度角摄像头
            SensorManager(world, display_manager, 'RGBCamera', 
                         carla.Transform(carla.Location(x=1.5, z=2.0), carla.Rotation(yaw=45)), 
                         vehicle, {}, display_pos=[0, 4], sensor_name="Front-Right Camera")
        
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
        
        # Left radar
        left_radar = RadarSensorManager(world, display_manager, 'Radar', 
                    carla.Transform(carla.Location(y=1.0, z=1.0), carla.Rotation(yaw=90)), 
                    vehicle, radar_sensor_options, 
                    display_pos=[1, 0], inference_model=radar_inference, 
                    conf_thres=args.radar_conf_thres, sensor_name="Left Radar+Detector")
        
        # Right radar
        right_radar = RadarSensorManager(world, display_manager, 'Radar', 
                    carla.Transform(carla.Location(y=-1.0, z=1.0), carla.Rotation(yaw=270)), 
                    vehicle, radar_sensor_options, 
                    display_pos=[1, 2], inference_model=radar_inference, 
                    conf_thres=args.radar_conf_thres, sensor_name="Right Radar+Detector")
        
        # Rear radar
        rear_radar = RadarSensorManager(world, display_manager, 'Radar', 
                    carla.Transform(carla.Location(x=-2.0, z=1.0), carla.Rotation(yaw=180)), 
                    vehicle, radar_sensor_options, 
                    display_pos=[1, 3], inference_model=radar_inference, 
                    conf_thres=args.radar_conf_thres, sensor_name="Rear Radar+Detector")

        # 将雷达传感器注册到雷达概览管理器 - 确保正确注册
        print("Registering radar sensors to overview manager...")
        radar_overview.register_radar_sensor(front_radar)
        radar_overview.register_radar_sensor(left_radar)
        radar_overview.register_radar_sensor(right_radar)
        radar_overview.register_radar_sensor(rear_radar)
        
        # Third row: LiDAR and additional sensors - 使用命令行参数设置range
        # Forward LiDAR
        SensorManager(world, display_manager, 'LiDAR', 
                     carla.Transform(carla.Location(x=0, z=2.4)), 
                     vehicle, {'channels': '16', 'range': range_str, 'points_per_second': '100000', 'rotation_frequency': '20'}, 
                     display_pos=[2, 1], sensor_name="LiDAR")
        
        # 添加鸟瞰图 (Bird's eye view)
        SensorManager(world, display_manager, 'RGBCamera', 
                     carla.Transform(carla.Location(x=0, z=30.0), carla.Rotation(pitch=-90)), 
                     vehicle, {'fov': '90'}, display_pos=[2, 0], sensor_name="Bird's Eye View")

        # Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        print(f"Baseline demonstration started, including {len(other_vehicles)} additional vehicles. Press ESC or Q to exit.")
        
        while True:
            # Carla tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # 确保雷达概览在每帧都更新
            if radar_overview:
                radar_overview.update()
            
            # 先渲染普通传感器
            display_manager.render()
            
            # 最后单独渲染雷达概览，确保它不会被其他渲染覆盖
            if radar_overview:
                radar_overview.render()

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
        if display_manager:
            display_manager.destroy()

        # Clean up all vehicles
        all_vehicles = vehicle_list + other_vehicles
        client.apply_batch([carla.command.DestroyActor(x) for x in all_vehicles])
        world.apply_settings(original_settings)


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
        help='Enable synchronous mode')
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