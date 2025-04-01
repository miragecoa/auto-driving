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

# 添加letterbox函数，从preview_predictions.py中借鉴
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """从YOLOv5代码中提取的letterbox函数"""
    # 检查图像是否为None
    if img is None:
        raise ValueError("输入图像为None")
    
    # 检查图像形状
    if not isinstance(img, np.ndarray) or img.ndim != 3:
        raise ValueError(f"输入图像必须是3D numpy数组，当前: {type(img)}")
    
    # 确保形状是合适的
    shape = img.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 确保新形状是有效的
    if not all(s > 0 for s in new_shape):
        raise ValueError(f"新形状必须是正数，当前: {new_shape}")
    
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
        print(f"letterbox处理错误: {e}")
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
            print(f"警告: 未找到YOLOv5目录: {yolov5_dir}")
            print("请确保已克隆YOLOv5仓库: git clone https://github.com/ultralytics/yolov5.git")
            return None
            
        # 添加YOLOv5目录到路径
        sys.path.append(yolov5_dir)
        
        # 导入所需的YOLOv5函数
        try:
            # 如果可能，导入scale_coords和non_max_suppression
            from utils.general import non_max_suppression, scale_coords
            print("成功导入YOLOv5辅助函数")
        except ImportError:
            try:
                # 新版本的替代路径
                from utils.ops import non_max_suppression
                from utils.augmentations import scale_coords
                print("从替代路径导入YOLOv5辅助函数")
            except ImportError:
                print("无法导入YOLOv5辅助函数，将使用基本实现")
        
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
            print(f"成功使用attempt_load加载YOLO模型: {weights_path}")
            return model
        except Exception as e1:
            print(f"尝试使用attempt_load加载模型失败: {e1}")
            try:
                # 方法2: 使用torch.hub
                model = torch.hub.load(yolov5_dir, 'custom', path=weights_path, source='local')
                print(f"成功通过torch.hub加载YOLO模型: {weights_path}")
                return model
            except Exception as e2:
                print(f"尝试使用torch.hub加载模型失败: {e2}")
                try:
                    # 方法3: 直接使用PyTorch加载
                    model = torch.load(weights_path, map_location=device)
                    if isinstance(model, dict) and 'model' in model:
                        model = model['model']
                    model.to(device)
                    model.eval()
                    print(f"成功通过torch.load加载YOLO模型: {weights_path}")
                    return model
                except Exception as e3:
                    print(f"所有加载尝试都失败: {e3}")
                    return None
    except Exception as e:
        print(f"加载YOLO模型时出错: {e}")
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
            print(f"无法获取模型步长，使用默认值: {self.stride}")
        
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
            radar_bp.set_attribute('horizontal_fov', '30')
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
                        print(f"图像预处理错误: {e}")
                        raise
                    
                    # 使用YOLOv5模型进行检测
                    t_inference_start = self.timer.time()
                    
                    # 推理
                    with torch.no_grad():
                        try:
                            # 尝试不同的模型输出格式
                            output = self.model(img)
                            if isinstance(output, list):
                                pred = output[0]  # 取第一个输出
                            elif isinstance(output, dict):
                                pred = output['out']  # 一些模型可能使用字典输出
                            elif isinstance(output, torch.Tensor):
                                pred = output
                            else:
                                pred = torch.zeros((1, 0, 6), device=self.device)
                                print(f"未知的模型输出格式: {type(output)}")
                                
                        except Exception as inference_error:
                            print(f"模型推理错误: {inference_error}")
                            # 在图像上显示错误信息
                            cv2.putText(array, f"YOLO推理错误", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
                                print("检测到中心点+宽高格式，进行转换")
                                # 转换为角点坐标
                                converted = torch.zeros_like(detections[:, :4])
                                converted[:, 0] = detections[:, 0] - detections[:, 2] / 2  # x1 = cx - w/2
                                converted[:, 1] = detections[:, 1] - detections[:, 3] / 2  # y1 = cy - h/2
                                converted[:, 2] = detections[:, 0] + detections[:, 2] / 2  # x2 = cx + w/2
                                converted[:, 3] = detections[:, 1] + detections[:, 3] / 2  # y2 = cy + h/2
                                detections[:, :4] = converted
                    
                    except Exception as nms_error:
                        print(f"NMS处理错误: {nms_error}")
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
                        print(f"坐标缩放错误: {scale_error}")
                    
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
                    print(f"YOLO检测出错: {e}")
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
            print(f"图像处理主循环错误: {main_error}")
            # 确保有surface可用于渲染
            if self.display_man.render_enabled() and hasattr(self, 'surface') and self.surface is None:
                # 创建一个黑色图像作为备用
                black_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
                cv2.putText(black_img, "Error: " + str(main_error)[:50], (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                self.surface = pygame.surfarray.make_surface(black_img.swapaxes(0, 1))

def run_simulation(args, client):
    """Run simulation"""
    display_manager = None
    vehicle = None
    vehicle_list = []
    other_vehicles = []
    timer = CustomTimer()
    yolo_model = None

    try:
        # 加载YOLO模型
        if args.yolo_weights:
            print(f"正在加载YOLOv5模型: {args.yolo_weights}")
            yolo_model = load_yolo_model(args.yolo_weights, device=args.device)
            if yolo_model:
                print("YOLOv5模型加载成功，将用于实时检测")
            else:
                print("YOLOv5模型加载失败，将不使用检测功能")
        
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

        # Display manager - modified to 3x3 grid to accommodate more sensors
        display_manager = DisplayManager(grid_size=[3, 4], window_size=[args.width, args.height])

        # 创建传感器时使用命令行参数设置range
        range_str = str(args.range)  # 将range转换为字符串
        print(f"传感器范围设置为: {range_str} 米")

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
        
        # Second row: Radars - 使用命令行参数设置range
        # Forward radar
        SensorManager(world, display_manager, 'Radar', 
                     carla.Transform(carla.Location(x=2.0, z=1.0), carla.Rotation(yaw=0)), 
                     vehicle, {'horizontal_fov': '120', 'vertical_fov': '10', 'range': range_str}, 
                     display_pos=[1, 1], sensor_name="Front Radar")
        
        # Left radar
        SensorManager(world, display_manager, 'Radar', 
                     carla.Transform(carla.Location(x=-2, z=1.0), carla.Rotation(yaw=-90)), 
                     vehicle, {'horizontal_fov': '120', 'vertical_fov': '10', 'range': range_str}, 
                     display_pos=[1, 0], sensor_name="Left Radar")
        
        # Right radar
        SensorManager(world, display_manager, 'Radar', 
                     carla.Transform(carla.Location(x=-2, z=1.0), carla.Rotation(yaw=90)), 
                     vehicle, {'horizontal_fov': '120', 'vertical_fov': '10', 'range': range_str}, 
                     display_pos=[1, 2], sensor_name="Right Radar")
        
        # Rear radar
        SensorManager(world, display_manager, 'Radar', 
                     carla.Transform(carla.Location(x=-2.0, z=1.0), carla.Rotation(yaw=180)), 
                     vehicle, {'horizontal_fov': '120', 'vertical_fov': '10', 'range': range_str}, 
                     display_pos=[1, 3], sensor_name="Rear Radar")
        
        # Third row: LiDAR and additional sensors - 使用命令行参数设置range
        # Forward LiDAR
        SensorManager(world, display_manager, 'LiDAR', 
                     carla.Transform(carla.Location(x=0, z=2.4)), 
                     vehicle, {'channels': '32', 'range': range_str, 'points_per_second': '100000', 'rotation_frequency': '20'}, 
                     display_pos=[2, 1], sensor_name="LiDAR")
        
        # 添加鸟瞰图 (Bird's eye view)
        SensorManager(world, display_manager, 'RGBCamera', 
                     carla.Transform(carla.Location(x=0, z=30.0), carla.Rotation(pitch=-90)), 
                     vehicle, {'fov': '90'}, display_pos=[2, 0], sensor_name="Bird's Eye View")

        # Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        print(f"基准演示已启动，包含 {len(other_vehicles)} 辆额外车辆。按 ESC 或 Q 退出。")
        
        while True:
            # Carla tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

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
        help='CARLA服务器IP (默认: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='CARLA服务器端口 (默认: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='启用同步模式')
    argparser.add_argument(
        '--width',
        metavar='W',
        default=1920,
        type=int,
        help='窗口宽度 (默认: 1920)')
    argparser.add_argument(
        '--height',
        metavar='H',
        default=1080,
        type=int,
        help='窗口高度 (默认: 1080)')
    argparser.add_argument(
        '--vehicles',
        metavar='N',
        default=20,
        type=int,
        help='要生成的车辆数量 (默认: 20)')
    argparser.add_argument(
        '--seed',
        metavar='S',
        default=0,
        type=int,
        help='用于可重现结果的随机种子 (默认: 0，表示随机行为)')
    argparser.add_argument(
        '--autopilot',
        action='store_true',
        default=True,
        help='启用车辆自动驾驶 (默认: True)')
    argparser.add_argument(
        '--no-autopilot',
        dest='autopilot',
        action='store_false',
        help='禁用车辆自动驾驶')
    argparser.add_argument(
        '--range',
        metavar='R',
        default=50,
        type=int,
        help='雷达和激光雷达传感器的探测范围，单位为米 (默认: 50)')
    argparser.add_argument(
        '--weather',
        choices=['default', 'badweather', 'night', 'badweather_night'],
        default='default',
        help='天气预设: default(默认晴天), badweather(恶劣天气), night(夜晚), badweather_night(恶劣天气的夜晚)')
    argparser.add_argument(
        '--yolo-weights',
        type=str,
        default='',
        help='YOLOv5模型权重路径，例如"runs/train/vehicle_detection/weights/best.pt"')
    argparser.add_argument(
        '--conf-thres',
        type=float,
        default=0.5,
        help='YOLOv5检测置信度阈值 (默认: 0.5)')
    argparser.add_argument(
        '--iou-thres',
        type=float,
        default=0.45,
        help='YOLOv5检测IoU阈值 (默认: 0.45)')
    argparser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='YOLOv5模型输入图像大小 (默认: 640)')
    argparser.add_argument(
        '--device',
        type=str,
        default='',
        help='CUDA设备，如"0"或"cpu" (默认为自动选择)')

    args = argparser.parse_args()

    try:
        # 确保导入OpenCV
        import cv2
        
        global client
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)  # 增加超时时间，用于车辆清理
        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\n用户取消。再见！')

if __name__ == '__main__':
    # Import required libraries
    main() 