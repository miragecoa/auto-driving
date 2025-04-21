#!/usr/bin/env python

import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import os
import json
import datetime
import argparse
from pascal_voc_writer import Writer

# 导入天气预设功能
try:
    from perception_utils import set_weather_preset
except ImportError:
    print("警告: 无法导入perception_utils中的set_weather_preset函数，将使用基本天气设置")
    
    # 提供基本的天气预设函数作为备用
    def set_weather_preset(world, preset="default"):
        """基本的天气预设功能"""
        weather = world.get_weather()
        
        if preset == "default":
            # 晴朗的白天
            weather.sun_altitude_angle = 85.0
            weather.cloudiness = 10.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.wind_intensity = 10.0
            weather.fog_density = 0.0
            weather.fog_distance = 0.0
            weather.wetness = 0.0
        
        elif preset == "badweather":
            # 恶劣天气 - 强降雨和大风
            weather.sun_altitude_angle = 45.0
            weather.cloudiness = 90.0
            weather.precipitation = 80.0
            weather.precipitation_deposits = 60.0
            weather.wind_intensity = 70.0
            weather.fog_density = 40.0
            weather.fog_distance = 40.0
            weather.wetness = 80.0
        
        elif preset == "night":
            # 晴朗的夜晚
            weather.sun_altitude_angle = -80.0
            weather.cloudiness = 10.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.wind_intensity = 10.0
            weather.fog_density = 0.0
            weather.fog_distance = 0.0
            weather.wetness = 0.0
        
        elif preset == "badweather_night":
            # 恶劣天气的夜晚 - 雨夜
            weather.sun_altitude_angle = -80.0
            weather.cloudiness = 90.0
            weather.precipitation = 80.0
            weather.precipitation_deposits = 60.0
            weather.wind_intensity = 70.0
            weather.fog_density = 50.0
            weather.fog_distance = 25.0
            weather.wetness = 80.0
        
        else:
            print(f"未知的天气预设: {preset}，使用默认晴天")
            weather.sun_altitude_angle = 85.0
            weather.cloudiness = 10.0
            weather.precipitation = 0.0
            weather.precipitation_deposits = 0.0
            weather.wind_intensity = 10.0
            weather.fog_density = 0.0
            weather.fog_distance = 0.0
            weather.wetness = 0.0
        
        world.set_weather(weather)
        print(f"已设置天气为: {preset}")
        
        return weather

# 添加NumPy到Python原生类型的转换函数
def convert_numpy_types_for_json(obj):
    """递归地将NumPy数据类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {key: convert_numpy_types_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types_for_json(item) for item in obj)
    else:
        return obj

class CarlaDatasetGenerator:
    def __init__(self, host='localhost', port=2000, output_dir='vehicle_dataset'):
        # 连接到CARLA服务器
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        
        # 创建输出目录
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        self.annotations_dir = os.path.join(output_dir, 'annotations')
        self.preview_dir = os.path.join(output_dir, 'previews')  # 预览目录
        self.lidar_points_dir = os.path.join(output_dir, 'lidar_points')  # 激光雷达点云数据目录
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        os.makedirs(self.lidar_points_dir, exist_ok=True)  # 创建激光雷达点云数据目录
        
        # 数据集属性
        self.image_count = 0
        self.coco_images = []
        self.coco_annotations = []
        self.annotation_id = 0
        
        # 设置场景
        self.vehicle = None
        # 激光雷达传感器
        self.lidar = None
        # 鸟瞰相机传感器
        self.birdview_camera = None
        
        # 激光雷达数据队列
        self.lidar_queue = None
        # 鸟瞰相机数据队列
        self.birdview_queue = None
        
        # 激光雷达/图像属性 - 使用与baseline一致的设置
        self.image_w = 800
        self.image_h = 600
        self.lidar_range = 50.0  # 激光雷达探测范围，单位为米，与baseline一致
        self.lidar_rotation_frequency = 20.0  # 激光雷达旋转频率，与baseline一致
        self.lidar_channels = 16  # 激光雷达通道数，与baseline一致
        self.lidar_points_per_second = 100000  # 每秒点云数量，与baseline一致
        self.lidar_upper_fov = 10.0  # 上视场角
        self.lidar_lower_fov = -30.0  # 下视场角
        self.lidar_height = 2.4  # 激光雷达安装高度，与baseline一致
        # 鸟瞰相机高度和FOV设置
        self.birdview_height = 50.0  # 鸟瞰相机距离地面高度
        self.birdview_fov = 90.0    # 鸟瞰相机视场角度

        # 边界框绘制
        self.edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        
        # 调试模式默认设置
        self.debug_mode = False
        self.show_radar_preview = True
        self.vehicle_model = 'vehicle.lincoln.mkz_2020'
        self.use_tuple_format = False

    def build_projection_matrix(self, w, h, fov, is_behind_camera=False):
        """创建相机投影矩阵"""
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)

        if is_behind_camera:
            K[0, 0] = K[1, 1] = -focal
        else:
            K[0, 0] = K[1, 1] = focal

        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def get_image_point(self, loc, K, w2c):
        """将3D点投影到2D图像平面"""
        # 格式化输入坐标 (loc是一个carla.Position对象)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # 转换到相机坐标
        point_camera = np.dot(w2c, point)

        # 现在我们必须从UE4的坐标系转换到"标准"坐标系
        # (x, y, z) -> (y, -z, x)
        # 同时移除第四个分量
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # 现在使用相机矩阵投影3D->2D
        point_img = np.dot(K, point_camera)
        # 标准化
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    def point_in_canvas(self, pos, img_h, img_w):
        """检查点是否在画布内"""
        if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
            return True
        return False
    
    def dot_product(self, v1, v2):
        """计算两个向量的点积"""
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

    def is_bbox_contained(self, bbox1, bbox2):
        """检查bbox1是否完全包含在bbox2内部
        
        参数:
            bbox1: [x_min, y_min, x_max, y_max] 格式的边界框
            bbox2: [x_min, y_min, x_max, y_max] 格式的边界框
            
        返回:
            bool: 如果bbox1完全包含在bbox2内部，则返回True
        """
        # 提取坐标
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 检查bbox1是否完全在bbox2内部
        return (x1_min >= x2_min and 
                y1_min >= y2_min and 
                x1_max <= x2_max and 
                y1_max <= y2_max)
    
    def filter_contained_bboxes(self, bboxes):
        """过滤掉被其他边界框完全包含的边界框
        
        参数:
            bboxes: 边界框列表，每个边界框为字典格式，包含'bbox'键
            
        返回:
            list: 过滤后的边界框列表
        """
        if not bboxes:
            return []
            
        # 按面积从大到小排序边界框
        sorted_bboxes = sorted(bboxes, 
                              key=lambda b: (b['bbox'][2] - b['bbox'][0]) * (b['bbox'][3] - b['bbox'][1]), 
                              reverse=True)
        
        filtered_bboxes = [sorted_bboxes[0]]  # 始终保留最大的边界框
        
        # 检查除了最大边界框之外的每个边界框
        for i in range(1, len(sorted_bboxes)):
            current_bbox = sorted_bboxes[i]
            is_contained = False
            
            # 检查当前边界框是否被任何已过滤边界框包含
            for filtered_bbox in filtered_bboxes:
                if self.is_bbox_contained(current_bbox['bbox'], filtered_bbox['bbox']):
                    # 当前边界框被包含在已过滤的边界框中，跳过它
                    is_contained = True
                    print(f"过滤掉被包含的边界框: 车辆ID {current_bbox['vehicle_id']}")
                    break
            
            # 如果当前边界框不被任何已过滤边界框包含，则添加到过滤后的列表中
            if not is_contained:
                filtered_bboxes.append(current_bbox)
        
        return filtered_bboxes

    def setup_scenario(self, num_vehicles=50, weather_preset="default"):
        """设置模拟场景"""
        # 首先清除世界上的所有物体
        print("正在清除世界上的所有物体...")
        
        # 获取所有actors
        all_actors = self.world.get_actors()
        
        # 清理过程：先删除传感器、车辆和行人
        sensors = [actor for actor in all_actors if 'sensor' in actor.type_id]
        vehicles = [actor for actor in all_actors if 'vehicle' in actor.type_id]
        walkers = [actor for actor in all_actors if 'walker' in actor.type_id]
        
        print(f"发现 {len(sensors)} 个传感器，{len(vehicles)} 辆车辆，{len(walkers)} 个行人")
        
        # 按优先级顺序删除
        for actor in sensors + vehicles + walkers:
            try:
                actor.destroy()
            except Exception as e:
                print(f"删除actor {actor.id} 时出错: {e}")
                
        print("场景已清理完成")
        
        # 设置天气
        set_weather_preset(self.world, weather_preset)
        
        # 获取地图生成点
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)  # 随机排序生成点
        
        # 设置交通管理器参数
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)  # 设置前车距离
        traffic_manager.set_synchronous_mode(True)
        
        # 生成主车辆，使用指定的车辆模型
        try:
            vehicle_bp = self.bp_lib.find(self.vehicle_model)
            if vehicle_bp is None:
                print(f"警告：找不到指定的车辆模型 '{self.vehicle_model}'，使用默认车辆")
                vehicle_bp = self.bp_lib.find('vehicle.lincoln.mkz_2020')
        except:
            print(f"警告：使用指定车辆模型出错，使用默认车辆")
            vehicle_bp = self.bp_lib.find('vehicle.lincoln.mkz_2020')
        
        spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            # 如果生成失败，尝试其他点位
            print("在随机点位生成主车失败，尝试其他点位...")
            for sp in spawn_points:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle is not None:
                    break
        
        if self.vehicle is None:
            raise Exception("无法生成主车辆，请检查CARLA环境")
        
        print(f"成功生成主车，车辆模型: {vehicle_bp.id}, ID: {self.vehicle.id}")
        
        # 创建激光雷达传感器蓝图
        lidar_bp = self.bp_lib.find('sensor.lidar.ray_cast')
        # 设置基本参数 - 降低通道数和每秒点数以减轻卡顿
        lidar_bp.set_attribute('range', str(self.lidar_range))
        lidar_bp.set_attribute('rotation_frequency', str(self.lidar_rotation_frequency))
        lidar_bp.set_attribute('channels', str(self.lidar_channels))
        lidar_bp.set_attribute('points_per_second', str(self.lidar_points_per_second))
        lidar_bp.set_attribute('upper_fov', str(self.lidar_upper_fov))
        lidar_bp.set_attribute('lower_fov', str(self.lidar_lower_fov))
        
        # 与baseline一致的设置
        # 设置sensor_tick为0.05秒，即20Hz刷新率
        lidar_bp.set_attribute('sensor_tick', '0.05')
        
        # 设置dropoff参数为推荐值
        try:
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
            lidar_bp.set_attribute('noise_stddev', '0.0')  # 设置噪声为0，提高点云精度
        except Exception as e:
            print(f"注意: 设置dropoff参数失败，可能是CARLA版本问题: {e}")
        
        # 额外的LiDAR参数配置 - 使用try/except处理不同版本的CARLA API
        try:
            # 尝试设置额外的LiDAR参数
            if hasattr(lidar_bp, 'has_attribute'):
                if lidar_bp.has_attribute('atmosphere_attenuation_rate'):
                    lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.004')  # 大气衰减率
        except Exception as e:
            print(f"注意: 设置额外的激光雷达参数失败，这在某些CARLA版本中是正常的: {e}")
        
        # 在车顶安装激光雷达，使用与baseline一致的安装位置
        lidar_transform = carla.Transform(carla.Location(x=-0.2, z=self.lidar_height))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        
        print(f"成功安装激光雷达，ID: {self.lidar.id}, 安装高度: {self.lidar_height}米, 向后偏移: 1.0米")
        
        # 创建鸟瞰相机传感器
        print("正在创建鸟瞰相机传感器...")
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        # 设置相机属性
        camera_bp.set_attribute('image_size_x', str(self.image_w))
        camera_bp.set_attribute('image_size_y', str(self.image_h))
        camera_bp.set_attribute('fov', str(self.birdview_fov))
        camera_bp.set_attribute('sensor_tick', '0.05')  # 20Hz刷新率
        
        # 设置鸟瞰相机位置 - 在车辆正上方
        camera_transform = carla.Transform(
            carla.Location(x=0.0, z=self.birdview_height),
            carla.Rotation(pitch=-90.0)  # 垂直向下看
        )
        self.birdview_camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        print(f"成功安装鸟瞰相机，ID: {self.birdview_camera.id}, 安装高度: {self.birdview_height}米")
        
        # 设置自动驾驶
        try:
            self.vehicle.set_autopilot(True, traffic_manager.get_port())
        except:
            # 旧版本CARLA可能不接受端口参数
            self.vehicle.set_autopilot(True)
        
        # 设置主车辆行为 - 设置为更慢的速度
        try:
            traffic_manager.ignore_lights_percentage(self.vehicle, 30)  # 30%概率忽略红绿灯，保持适当移动
            traffic_manager.distance_to_leading_vehicle(self.vehicle, 1.0)  # 正常前车距离
            
            # 主车速度设置为正常速度的30%
            print("设置主车速度为正常速度的30%")
            traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 70.0)
            
        except Exception as e:
            print(f"注意: 设置高级交通管理器参数失败，这在某些CARLA版本中是正常的: {e}")
        
        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # 启用同步模式
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # 创建队列存储激光雷达数据
        self.lidar_queue = queue.Queue()
        # 创建队列存储鸟瞰相机数据
        self.birdview_queue = queue.Queue()
        
        # 设置激光雷达监听
        self.lidar.listen(self.lidar_queue.put)
        # 设置鸟瞰相机监听
        self.birdview_camera.listen(self.birdview_queue.put)
        
        # 生成其他车辆，避免相互卡死
        vehicles_list = []
        remaining_spawn_points = spawn_points.copy()
        
        # 移除主车所在的生成点
        if self.vehicle is not None:
            vehicle_transform = self.vehicle.get_transform()
            for i, sp in enumerate(remaining_spawn_points):
                if abs(sp.location.x - vehicle_transform.location.x) < 5 and \
                   abs(sp.location.y - vehicle_transform.location.y) < 5:
                    remaining_spawn_points.pop(i)
                    break
        
        # 限制车辆数量，不要超过生成点数量
        num_vehicles = min(num_vehicles, len(remaining_spawn_points))
        
        # ===== 使用与baseline_perception相同的方法过滤车辆蓝图 =====
        # 获取所有车辆蓝图
        all_vehicles = self.bp_lib.filter('vehicle.*')
        
        # 直接通过车轮数过滤，确保只有4轮车辆
        car_blueprints = [bp for bp in all_vehicles if int(bp.get_attribute('number_of_wheels')) == 4]
        
        print(f"CARLA中所有可用车辆类型: {len(all_vehicles)}种")
        print(f"已筛选出 {len(car_blueprints)} 种四轮汽车类型")
        
        if len(car_blueprints) == 0:
            print("警告：没有找到符合条件的四轮汽车蓝图，将使用默认汽车蓝图")
            # 使用一些已知的四轮车，而不是所有车辆
            safe_cars = ['vehicle.audi.a2', 'vehicle.audi.tt', 'vehicle.lincoln.mkz2017']
            car_blueprints = [self.bp_lib.find(car) for car in safe_cars if self.bp_lib.find(car)]
            if not car_blueprints:
                raise Exception("无法找到任何可用的四轮车蓝图，请检查CARLA环境")
        
        # 生成NPC车辆 - 减少车辆数量，避免场景过于复杂导致卡顿
        batch = []
        for i in range(min(num_vehicles, 20)):  # 最多20辆NPC车辆
            if i >= len(remaining_spawn_points):
                break
                
            vehicle_bp = random.choice(car_blueprints)
            # 设置车辆不会不稳定（避免物理效应导致翻车）
            vehicle_bp.set_attribute('role_name', 'autopilot')
            
            # 在特定的生成点生成车辆，避免碰撞
            npc = self.world.try_spawn_actor(vehicle_bp, remaining_spawn_points[i])
            if npc:
                try:
                    npc.set_autopilot(True, traffic_manager.get_port())
                except:
                    # 旧版本CARLA可能不接受端口参数
                    npc.set_autopilot(True)
                    
                # 为NPC车辆设置正常默认行为
                try:
                    # 车辆有20%概率可能忽略红绿灯，确保交通不会完全堵塞
                    traffic_manager.ignore_lights_percentage(npc, 20)
                    # 设置正常的车距
                    traffic_manager.distance_to_leading_vehicle(npc, 1.5)
                    
                    # 设置为正常速度的30%
                    # 在CARLA中，值越大表示越慢，70表示只有30%的原速度
                    traffic_manager.vehicle_percentage_speed_difference(npc, 70.0)
                    
                    # 默认车道变更行为
                    traffic_manager.auto_lane_change(npc, True)
                    try:
                        # 启用碰撞检测，保障安全驾驶
                        traffic_manager.collision_detection(npc, True)
                    except:
                        # 忽略老版本API不支持的错误
                        pass
                except Exception as e:
                    print(f"注意: 设置NPC车辆行为参数失败: {e}")
                
                vehicles_list.append(npc)

        print(f"成功生成 {len(vehicles_list)} 辆NPC车辆")
        
        # 等待一段时间，让车辆开始移动
        print("等待车辆开始移动...")
        for _ in range(30):  # 等待1.5秒(30 ticks * 0.05秒)
            self.world.tick()
            
        if self.debug_mode:
            print(f"LiDAR参数配置:")
            print(f"- 通道数: {self.lidar_channels}")
            print(f"- 探测范围: {self.lidar_range}米")
            print(f"- 旋转频率: {self.lidar_rotation_frequency}Hz")
            print(f"- 点云密度: {self.lidar_points_per_second}点/秒")
            print(f"- 视场角: {self.lidar_upper_fov}°(上) ~ {self.lidar_lower_fov}°(下)")
            print(f"- 数据处理模式: {'元组格式' if self.use_tuple_format else '自动检测'}")
            
        # 等待第一帧数据，确保激光雷达工作正常
        print("等待首帧激光雷达数据...")
        for _ in range(10):  # 最多等待10帧
            self.world.tick()
        print("场景设置完成，开始数据采集...")

    def process_lidar_data(self, lidar_data):
        """处理激光雷达点云数据，创建鸟瞰图（BEV）可视化，完全照搬雷达实现的样式"""
        # 创建鸟瞰图画布
        lidar_img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        
        # 计算中心点，表示本车位置
        center_x, center_y = self.image_w // 2, self.image_h // 2
        
        # 绘制背景网格和坐标轴
        # 绘制中心点（车辆位置）
        cv2.circle(lidar_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 绘制同心圆，表示距离
        for r in range(1, 6):
            radius = int(min(self.image_w, self.image_h) / 12 * r)
            cv2.circle(lidar_img, (center_x, center_y), radius, (50, 50, 50), 1)
        
        # 绘制坐标轴
        cv2.line(lidar_img, (center_x, 0), (center_x, self.image_h), (50, 50, 50), 1)  # 垂直轴
        cv2.line(lidar_img, (0, center_y), (self.image_w, center_y), (50, 50, 50), 1)  # 水平轴
        
        # 添加距离标签
        max_range_shown = self.lidar_range
        for r in range(1, 6):
            distance = max_range_shown * r / 5
            # 在同心圆右侧标注距离
            label_x = center_x + int(min(self.image_w, self.image_h) / 12 * r)
            label_y = center_y
            cv2.putText(lidar_img, f"{distance:.0f}m", (label_x + 5, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # 处理点云数据 - 完全照搬雷达实现
        lidar_points = []
        point_id_counter = 0
        
        # 检测CARLA版本并适配不同的LiDAR数据格式
        using_carla_tuple_format = True
        sample_point = lidar_data[0] if len(lidar_data) > 0 else None
        
        # 确定LiDAR数据格式
        if sample_point is not None:
            if hasattr(sample_point, 'point') or hasattr(sample_point, 'get_point'):
                using_carla_tuple_format = False
            
            # 调试信息
            if self.debug_mode:
                print(f"LiDAR数据格式检测: {'对象格式' if not using_carla_tuple_format else '元组格式'}")
                print(f"样本点类型: {type(sample_point)}")
                if hasattr(sample_point, '__dir__'):
                    print(f"样本点属性: {dir(sample_point)}")
        
        # 遍历点云中的每个点，提取坐标并存储
        for point in lidar_data:
            try:
                x, y, z, intensity = None, None, None, None
                
                if using_carla_tuple_format:
                    # 尝试直接解包（元组格式）
                    if hasattr(point, '__getitem__') and len(point) >= 3:
                        x, y, z = point[0], point[1], point[2]
                        intensity = point[3] if len(point) > 3 else 1.0
                    else:
                        # 如果不能解包但有适当的属性，尝试直接访问
                        if hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
                            x, y, z = point.x, point.y, point.z
                            intensity = getattr(point, 'intensity', 1.0)
                        else:
                            # 跳过无法处理的点
                            continue
                else:
                    # 处理LidarDetection对象
                    if hasattr(point, 'point'):
                        # 尝试通过point属性访问
                        if hasattr(point.point, 'x'):
                            # 如果point是一个有x,y,z属性的对象
                            x, y, z = point.point.x, point.point.y, point.point.z
                        else:
                            # 如果point是一个可迭代对象
                            try:
                                if hasattr(point.point, '__getitem__'):
                                    x, y, z = point.point[0], point.point[1], point.point[2]
                                else:
                                    # 可能是其他格式，尝试直接解包
                                    x, y, z = point.point
                            except:
                                # 如果上述方法都失败，尝试其他可能的格式
                                if self.debug_mode:
                                    print(f"警告: 无法解析点云数据点，将尝试其他方法")
                                continue
                    elif hasattr(point, 'get_point'):
                        # 有些版本可能使用get_point()方法
                        point_data = point.get_point()
                        if hasattr(point_data, 'x'):
                            x, y, z = point_data.x, point_data.y, point_data.z
                        else:
                            try:
                                x, y, z = point_data
                            except:
                                continue
                    else:
                        # 尝试直接解包
                        try:
                            if hasattr(point, 'x') and hasattr(point, 'y') and hasattr(point, 'z'):
                                x, y, z = point.x, point.y, point.z
                            else:
                                # 最后尝试，如果是可转换为列表的对象
                                coords = list(point)
                                x, y, z = coords[0], coords[1], coords[2]
                        except:
                            # 如果所有方法都失败，跳过这个点
                            if self.debug_mode and point_id_counter < 5:  # 只打印前几个点的错误以避免刷屏
                                print(f"无法解析点云数据格式，跳过一个点。点数据类型: {type(point)}")
                            continue
                    
                    # 获取强度
                    if hasattr(point, 'intensity'):
                        intensity = point.intensity
                    elif hasattr(point, 'get_intensity'):
                        intensity = point.get_intensity()
                    else:
                        # 如果没有强度信息，使用默认值
                        intensity = 1.0
                
                # 如果仍然没有有效的坐标，跳过这个点
                if x is None or y is None or z is None:
                    continue
                
                # 计算到原点的距离
                distance = math.sqrt(x**2 + y**2 + z**2)
                
                # 计算方位角和仰角
                azimuth = math.atan2(y, x)
                altitude = math.atan2(z, math.sqrt(x**2 + y**2))
                
                # 完全照搬雷达实现 - 直接根据极坐标计算图像上的位置
                scale = min(self.image_w, self.image_h) / (2 * self.lidar_range)
                px = center_x + int(distance * math.sin(azimuth) * scale)
                py = center_y - int(distance * math.cos(azimuth) * scale)  # 减法是因为屏幕坐标y轴向下
                
                # 确保点在图像范围内
                if 0 <= px < self.image_w and 0 <= py < self.image_h:
                    # 使用白色绘制点
                    cv2.circle(lidar_img, (px, py), 2, (255, 255, 255), -1)
                    
                    # 存储点信息供后续处理
                    lidar_points.append({
                        'id': point_id_counter,  # 点ID
                        'x': px, 'y': py,  # 图像坐标
                        'world_x': x, 'world_y': y, 'world_z': z,  # 3D世界坐标
                        'distance': distance,  # 到原点距离
                        'azimuth': azimuth,  # 方位角
                        'altitude': altitude,  # 仰角
                        'intensity': intensity,  # 点云强度
                        'hit_vehicle_id': None,  # 击中的车辆ID，初始为None
                        'is_hitting_vehicle': False,  # 是否击中车辆
                    })
                    point_id_counter += 1
                
            except Exception as e:
                # 如果解析失败，记录错误并跳过这个点
                if self.debug_mode and point_id_counter < 5:  # 只打印前几个点的错误
                    print(f"解析点云数据点时出错: {e}, 点数据类型: {type(point)}")
                continue
                
        return lidar_img, lidar_points

    def get_lidar_data(self):
        """从激光雷达获取点云数据"""
        try:
            # 获取激光雷达数据
            lidar_data = self.lidar_queue.get(timeout=2.0)
            
            # 调试：打印第一个点的详细信息
            if len(lidar_data) > 0 and self.debug_mode:
                first_point = lidar_data[0]
                try:
                    print(f"调试信息 - 第一个激光雷达点的类型: {type(first_point)}")
                    print(f"调试信息 - 第一个点的属性和方法: {dir(first_point)}")
                    # 尝试不同的方式访问点的属性
                    if hasattr(first_point, 'point'):
                        print(f"点的position: {first_point.point.x}, {first_point.point.y}, {first_point.point.z}")
                        if hasattr(first_point, 'intensity'):
                            print(f"点的intensity: {first_point.intensity}")
                    elif hasattr(first_point, '__getitem__'):
                        val3 = first_point[3] if len(first_point) > 3 else 'N/A'
                        print(f"点作为数组: {first_point[0]}, {first_point[1]}, {first_point[2]}, {val3}")
                except Exception as e:
                    print(f"调试点数据时出错: {e}")
                
                # 只打印一次调试信息
                self.get_lidar_data = self._get_lidar_data_no_debug
            
            # 处理激光雷达数据
            lidar_img, lidar_points = self.process_lidar_data(lidar_data)
            
            return lidar_img, lidar_points
            
        except queue.Empty:
            print("激光雷达数据获取超时")
            return None, None
            
    def _get_lidar_data_no_debug(self):
        """不带调试信息的获取点云数据方法"""
        try:
            # 获取激光雷达数据
            lidar_data = self.lidar_queue.get(timeout=2.0)
            
            # 处理激光雷达数据
            lidar_img, lidar_points = self.process_lidar_data(lidar_data)
            
            return lidar_img, lidar_points
            
        except queue.Empty:
            print("激光雷达数据获取超时")
            return None, None

    def generate_dataset(self, num_frames=100, capture_interval=5, show_3d_bbox=False, detection_range=50, skip_empty_frames=False, hit_tolerance=5, save_lidar_points=False):
        """生成激光雷达数据集"""
        print(f"开始生成激光雷达数据集：目标帧数 {num_frames}，采集间隔 {capture_interval} 秒")
        print(f"激光雷达范围: {self.lidar_range}米，检测范围: {detection_range}米")
        print(f"注意: 只有被激光雷达点云实际击中的车辆才会被标记（容忍范围: {hit_tolerance}像素）")
        
        if save_lidar_points:
            print("已启用激光雷达点云数据保存: 将记录每个点及其击中的目标")
        
        if skip_empty_frames:
            print("已启用空帧过滤: 只保存包含车辆标记的帧")
        else:
            print("未启用空帧过滤: 所有捕获的帧都将被保存")
        
        # 确保输出目录已创建
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        os.makedirs(self.lidar_points_dir, exist_ok=True)
        
        # 检查已存在的图像文件，确定起始的image_count
        existing_images = [f for f in os.listdir(self.images_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if existing_images:
            # 提取文件名中的数字部分，找出最大值
            max_id = -1
            for img_name in existing_images:
                try:
                    id_str = os.path.splitext(img_name)[0]  # 去除扩展名
                    img_id = int(id_str)
                    max_id = max(max_id, img_id)
                except ValueError:
                    continue
            
            if max_id >= 0:
                self.image_count = max_id + 1
                print(f"根据已存在的图像文件，起始图像ID设置为: {self.image_count}")
        
        # 确保车辆在开始采集前已经开始移动
        print("等待车辆移动到适当位置...")
        start_wait_time = time.time()
        # 额外等待5秒，让车辆移动到更好的位置
        while time.time() - start_wait_time < 5.0:
            self.world.tick()
            time.sleep(0.05)
        
        last_capture_time = time.time()
        frames_processed = 0
        frames_skipped = 0
        start_time = time.time()
        
        # 清空队列，确保同步起点
        while not self.lidar_queue.empty():
            self.lidar_queue.get()
            
        while not self.birdview_queue.empty():
            self.birdview_queue.get()
            
        print("开始数据采集，已清空所有数据队列以确保同步...")
        
        while frames_processed < num_frames:
            try:
                # 刷新世界 - 先触发一次世界更新，然后才获取数据
                self.world.tick()
                
                # 等待所有传感器数据都准备好，确保同一时间点的数据同步
                time.sleep(0.05)  # 稍微等待以确保所有传感器数据都已生成
                
                # 同步LiDAR和相机数据的获取
                timestamp_start = time.time()
                max_wait_time = 0.2  # 最长等待时间200ms
                
                while True:
                    # 检查是否超时
                    if time.time() - timestamp_start > max_wait_time:
                        print("数据同步超时，跳过当前帧")
                        break
                    
                    # 检查两个队列是否都有数据
                    if not self.lidar_queue.empty() and not self.birdview_queue.empty():
                        # 如果两个队列都有数据，就获取并处理
                        lidar_img, lidar_points = self.get_lidar_data()
                        birdview_img = self.get_birdview_image()
                        
                        if lidar_img is not None and birdview_img is not None:
                            break  # 成功获取到两个数据，退出循环
                    
                    # 如果队列中没有足够数据，等待一小段时间
                    time.sleep(0.01)
                
                # 如果没有成功获取到数据，继续下一次循环
                if lidar_img is None or lidar_points is None or birdview_img is None:
                    continue
                
                # 创建一个工作副本用于显示
                display_img = lidar_img.copy()
                birdview_display = birdview_img.copy()
                
                # 获取场景中的所有车辆，用于在激光雷达图上标注
                detected_vehicles = []
                
                # 获取主车位置和方向
                ego_transform = self.vehicle.get_transform()
                ego_location = ego_transform.location
                ego_rotation = ego_transform.rotation
                ego_forward = ego_transform.get_forward_vector()
                
                # 计算激光雷达图中心点（本车位置）
                center_x, center_y = self.image_w // 2, self.image_h // 2
                
                # 处理场景中的所有车辆
                for npc in self.world.get_actors().filter('*vehicle*'):
                    # 跳过主车
                    if npc.id != self.vehicle.id:
                        try:
                            # 获取NPC车辆位置和距离
                            npc_location = npc.get_transform().location
                            dist = npc_location.distance(ego_location)
                            
                            # 只处理检测范围内的车辆
                            if dist < detection_range:
                                # 计算相对位置向量
                                rel_vector = npc_location - ego_location
                                
                                # 计算方位角（相对于本车前方向量）
                                # 首先得到平面上的相对向量（忽略z轴）
                                flat_rel_vector = carla.Vector3D(rel_vector.x, rel_vector.y, 0)
                                flat_forward = carla.Vector3D(ego_forward.x, ego_forward.y, 0)
                                
                                # 计算方位角（右侧为正，左侧为负）
                                # 使用点积和叉积计算夹角
                                forward_length = math.sqrt(flat_forward.x**2 + flat_forward.y**2)
                                rel_length = math.sqrt(flat_rel_vector.x**2 + flat_rel_vector.y**2)
                                
                                if forward_length > 0 and rel_length > 0:
                                    # 点积计算cos
                                    dot_product = (flat_forward.x * flat_rel_vector.x + 
                                                flat_forward.y * flat_rel_vector.y) / (forward_length * rel_length)
                                    # 叉积方向确定正负（右侧为正，左侧为负）
                                    cross_product = (flat_forward.x * flat_rel_vector.y - 
                                                    flat_forward.y * flat_rel_vector.x)
                                    
                                    # 计算角度
                                    angle = math.acos(max(-1, min(1, dot_product)))  # 防止浮点误差
                                    if cross_product < 0:
                                        angle = -angle  # 左侧为负角度
                                    
                                    # 完全照搬雷达实现，移除额外的角度旋转逻辑
                                    scale = min(self.image_w, self.image_h) / (2 * self.lidar_range)
                                    x = center_x + int(dist * math.sin(angle) * scale)
                                    y = center_y - int(dist * math.cos(angle) * scale)
                                    
                                    # 获取车辆尺寸信息（用于计算边界框大小）
                                    bb = npc.bounding_box
                                    vehicle_length = bb.extent.x * 2
                                    vehicle_width = bb.extent.y * 2
                                    
                                    # 计算边界框大小（根据距离缩放）
                                    # 使用更平滑的距离缩放函数，防止近距离车辆边界框过大
                                    # 距离越远，边界框越小
                                    min_dist = 10.0  # 增加最小距离阈值，防止极小距离导致边界框过大
                                    # 使用对数函数使缩放更平滑
                                    adjusted_dist = max(min_dist, dist)
                                    # 降低基础系数(从15到8)和幂指数(从0.7到0.5)，使远距离和近距离的边界框都更小
                                    box_size_factor = 8 * math.pow(10.0 / adjusted_dist, 0.5)
                                    
                                    # 限制最大缩放因子
                                    box_size_factor = min(10.0, box_size_factor)
                                    
                                    # 计算边界框尺寸，限制最大尺寸为50像素
                                    box_width = max(8, min(50, int(vehicle_width * box_size_factor)))
                                    box_height = max(8, min(50, int(vehicle_length * box_size_factor)))
                                    
                                    # 额外处理：确保近距离车辆边界框不会太大
                                    if dist < 20:
                                        # 对近距离车辆额外限制尺寸
                                        max_size = int(35 - (dist < 10) * 10)  # 10m内最大25, 10-20m最大35
                                        box_width = min(box_width, max_size)
                                        box_height = min(box_height, max_size)
                                    
                                    # 计算边界框坐标
                                    x1 = max(0, x - box_width // 2)
                                    y1 = max(0, y - box_height // 2)
                                    x2 = min(self.image_w - 1, x + box_width // 2)
                                    y2 = min(self.image_h - 1, y + box_height // 2)
                                    
                                    # 检查是否有激光雷达点与当前车辆的边界框重叠
                                    lidar_hit = False
                                    hitting_points = []  # 记录击中当前车辆的激光雷达点
                                    for point in lidar_points:
                                        # 检查激光雷达点是否在边界框内或边界框外的容忍范围内
                                        if ((x1-hit_tolerance) <= point['x'] <= (x2+hit_tolerance) and 
                                            (y1-hit_tolerance) <= point['y'] <= (y2+hit_tolerance)):
                                            lidar_hit = True
                                            
                                            # 记录该点击中了哪个车辆
                                            point['hit_vehicle_id'] = npc.id
                                            point['is_hitting_vehicle'] = True
                                            hitting_points.append(point)
                                    
                                    # 只有当车辆被激光雷达点击中时，才标记为检测到的车辆
                                    if lidar_hit:
                                        # 存储检测到的车辆信息，增加点击车辆的激光雷达点信息
                                        detected_vehicles.append({
                                            "vehicle_id": npc.id,
                                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                            "distance": dist,
                                            "angle": angle,  # 使用原始角度，不再使用旋转后的角度
                                            "position": (x, y),
                                            "real_position": (dist, angle),  # 保存极坐标，使用原始角度
                                            "box_size_factor": box_size_factor,  # 保存缩放因子以便调试
                                            "hitting_points_count": len(hitting_points),  # 击中该车辆的激光雷达点数量
                                            "hitting_points_ids": [p['id'] for p in hitting_points]  # 击中该车辆的激光雷达点ID列表
                                        })
                                        
                                        # 在激光雷达图上绘制边界框
                                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                                        # 添加距离和缩放因子信息
                                        cv2.putText(display_img, f"{dist:.1f}m", (x1, y1-5), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                        # 在边界框底部添加缩放因子信息
                                        cv2.putText(display_img, f"S:{box_size_factor:.1f}", (x1, y2+12), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
                                        
                                        # 添加中心点标记
                                        center_x_box = (x1 + x2) // 2
                                        center_y_box = (y1 + y2) // 2
                                        cv2.circle(display_img, (center_x_box, center_y_box), 3, (255, 0, 0), -1)  # 蓝色实心圆
                                        
                                    else:
                                        # 对于未被激光雷达击中但在检测范围内的车辆，用虚线框标记（便于调试）
                                        # OpenCV没有内置的虚线矩形绘制方法，手动绘制虚线矩形
                                        # 绘制四条边，每条边都是断断续续的线段
                                        color = (100, 100, 100)  # 灰色
                                        dash_length = 5
                                        
                                        # 上边
                                        for i in range(x1, x2, dash_length*2):
                                            x_end = min(i + dash_length, x2)
                                            cv2.line(display_img, (i, y1), (x_end, y1), color, 1)
                                        
                                        # 右边
                                        for i in range(y1, y2, dash_length*2):
                                            y_end = min(i + dash_length, y2)
                                            cv2.line(display_img, (x2, i), (x2, y_end), color, 1)
                                        
                                        # 下边
                                        for i in range(x1, x2, dash_length*2):
                                            x_end = min(i + dash_length, x2)
                                            cv2.line(display_img, (i, y2), (x_end, y2), color, 1)
                                        
                                        # 左边
                                        for i in range(y1, y2, dash_length*2):
                                            y_end = min(i + dash_length, y2)
                                            cv2.line(display_img, (x1, i), (x1, y_end), color, 1)
                        except Exception as e:
                            print(f"处理车辆 {npc.id} 时出错: {e}")
                            continue
                
                # 显示激光雷达图
                cv2.putText(display_img, f"Frames: {frames_processed}/{num_frames}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 显示下次采集倒计时
                time_to_next = max(0, capture_interval - (time.time() - last_capture_time))
                cv2.putText(display_img, f"Next capture: {time_to_next:.1f}s", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 显示检测到的车辆数量
                cv2.putText(display_img, f"Vehicles detected: {len(detected_vehicles)}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 在鸟瞰图上绘制检测到的车辆边界框，使用与LiDAR图相同的计算方式
                for vehicle_info in detected_vehicles:
                    x1, y1, x2, y2 = vehicle_info["bbox"]
                    # 在鸟瞰图上绘制边界框 - 使用不同颜色和更明显的线条
                    cv2.rectangle(birdview_display, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色，更粗的线条
                    # 添加距离信息
                    cv2.putText(birdview_display, f"{vehicle_info['distance']:.1f}m", (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # 红色粗字体
                    
                    # 添加中心点标记，更容易看出是否对齐
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    cv2.circle(birdview_display, (center_x, center_y), 3, (255, 0, 0), -1)  # 蓝色实心圆
                
                # 显示窗口
                cv2.putText(display_img, "LiDAR View", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('CARLA LiDAR Dataset - BEV', display_img)
                
                # 显示鸟瞰图像窗口，添加标题
                cv2.putText(birdview_display, "Camera BirdView", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('CARLA BirdView Camera', birdview_display)
                
                # 如果启用了雷达样式预览，显示额外的窗口
                if self.show_radar_preview:
                    # 创建雷达样式预览
                    radar_preview = self.create_radar_style_preview(lidar_points, detected_vehicles)
                    # 显示雷达样式预览窗口
                    cv2.imshow('CARLA LiDAR Dataset - Radar Style', radar_preview)
                
                # 每隔capture_interval秒采集一次数据
                current_time = time.time()
                if current_time - last_capture_time >= capture_interval:
                    try:
                        # 首先检查是否检测到了车辆，如果启用了空帧过滤，则跳过无车辆帧
                        if skip_empty_frames and not detected_vehicles:
                            frames_skipped += 1
                            print(f"当前帧未检测到车辆，跳过保存 (已跳过: {frames_skipped} 帧)")
                            last_capture_time = current_time
                            # 继续处理下一帧，不计入已处理帧数
                            continue
                        
                        # 获取可用的图像ID
                        frame_id = self.image_count
                        img_filename = f'{frame_id:06d}.png'
                        img_path = os.path.join(self.images_dir, img_filename)
                        
                        # 检查文件是否已存在
                        while os.path.exists(img_path):
                            frame_id += 1
                            img_filename = f'{frame_id:06d}.png'
                            img_path = os.path.join(self.images_dir, img_filename)
                        
                        if frame_id != self.image_count:
                            print(f"跳过已存在的文件名，使用新ID: {frame_id}")
                            self.image_count = frame_id
                        
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 保存激光雷达图像
                        cv2.imwrite(img_path, lidar_img)
                        
                        # 保存带有边界框标记的预览图像
                        preview_img = display_img.copy()
                        # 添加信息到预览图像
                        cv2.putText(preview_img, f"Vehicles: {len(detected_vehicles)}", (10, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # 为每个边界框添加ID标签
                        for i, vehicle_info in enumerate(detected_vehicles):
                            x1, y1, x2, y2 = vehicle_info["bbox"]
                            # 在边界框上方显示ID
                            cv2.putText(preview_img, f"V{i+1}", (int(x1), int(y1) - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # 保存预览图像
                        preview_path = os.path.join(self.preview_dir, img_filename)
                        cv2.imwrite(preview_path, preview_img)
                        
                        # 保存激光雷达点云数据（如果启用）
                        if save_lidar_points:
                            # 创建激光雷达点云数据字典
                            lidar_data = {
                                "frame_id": frame_id,
                                "timestamp": timestamp,
                                "total_points": len(lidar_points),
                                "vehicle_count": len(detected_vehicles),
                                "points": []
                            }
                            
                            # 将每个激光雷达点的信息添加到数据中
                            for point in lidar_points:
                                # 确保所有NumPy类型被转换为Python原生类型
                                point_data = {
                                    "id": int(point["id"]),
                                    "world_position": [float(point["world_x"]), float(point["world_y"]), float(point["world_z"])],
                                    "image_position": [int(point["x"]), int(point["y"])],
                                    "distance": float(point["distance"]),
                                    "intensity": float(point["intensity"]),
                                    "hit_vehicle_id": int(point["hit_vehicle_id"]) if point["hit_vehicle_id"] is not None else None,
                                    "is_hitting_vehicle": bool(point["is_hitting_vehicle"])
                                }
                                lidar_data["points"].append(point_data)
                            
                            # 保存激光雷达点云数据为JSON文件
                            lidar_path = os.path.join(self.lidar_points_dir, f'{frame_id:06d}.json')
                            with open(lidar_path, 'w') as f:
                                # 使用convert_numpy_types_for_json函数处理数据
                                json_data = convert_numpy_types_for_json(lidar_data)
                                json.dump(json_data, f, indent=2)
                            
                            print(f"已保存激光雷达点云数据: {len(lidar_points)} 个点，其中 {sum(1 for p in lidar_points if p['is_hitting_vehicle'])} 个点击中车辆")
                        
                        # 创建VOC标注writer
                        voc_writer = Writer(img_path, self.image_w, self.image_h)
                        
                        # 添加到COCO图像列表
                        self.coco_images.append({
                            "license": 1,
                            "file_name": img_filename,
                            "height": self.image_h,
                            "width": self.image_w,
                            "date_captured": timestamp,
                            "id": frame_id
                        })
                        
                        # 处理边界框
                        for vehicle_info in detected_vehicles:
                            x_min, y_min, x_max, y_max = vehicle_info["bbox"]
                            
                            # 添加到VOC标注
                            voc_writer.addObject('vehicle', x_min, y_min, x_max, y_max)
                            
                            # 添加到COCO标注
                            self.coco_annotations.append({
                                "id": self.annotation_id,
                                "image_id": frame_id,
                                "category_id": 1,  # 1 表示车辆
                                "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                                "area": float((x_max - x_min) * (y_max - y_min)),
                                "segmentation": [],
                                "iscrowd": 0,
                                "distance": vehicle_info["distance"],
                                "angle": vehicle_info["angle"]
                            })
                            self.annotation_id += 1
                        
                        # 保存VOC格式标注
                        voc_path = os.path.join(self.annotations_dir, f'{frame_id:06d}.xml')
                        voc_writer.save(voc_path)
                        
                        self.image_count += 1  # 更新图像计数
                        frames_processed += 1
                        last_capture_time = current_time
                        
                        elapsed_time = time.time() - start_time
                        remaining_time = (elapsed_time / frames_processed) * (num_frames - frames_processed) if frames_processed > 0 else 0
                        status_msg = f'已采集: {frames_processed}/{num_frames} 帧 ({frames_processed/num_frames*100:.1f}%), '
                        if skip_empty_frames:
                            status_msg += f'跳过无车辆帧: {frames_skipped} 帧, '
                        status_msg += f'下一帧将在 {capture_interval} 秒后采集'
                        print(status_msg)
                        print(f'预计剩余时间: {int(remaining_time//60)}分{int(remaining_time%60)}秒')
                    except Exception as e:
                        print(f"保存数据帧时出错: {e}")
                        # 继续尝试下一帧，而不是直接中断数据集生成
                        last_capture_time = current_time
                
                # 按'q'键退出
                if cv2.waitKey(1) == ord('q'):
                    print("用户按下q键，提前结束数据集生成")
                    break
            except Exception as e:
                print(f"数据处理循环中发生错误: {e}")
                # 短暂等待后继续尝试
                time.sleep(0.1)
        
        # 关闭显示窗口
        cv2.destroyAllWindows()
        
        # 保存COCO格式数据集
        print("数据采集完成，正在保存COCO格式数据集...")
        self.save_coco_dataset()
        
        # 如果启用了激光雷达点云数据保存，则保存元数据
        if save_lidar_points:
            print("正在保存激光雷达点云数据集元数据...")
            self.save_lidar_dataset_metadata()
        
        # 打印数据集统计信息
        print(f"\n数据集生成统计:")
        print(f"- 总共采集激光雷达图像: {self.image_count} 帧")
        print(f"- 总共标注对象: {self.annotation_id} 个")
        print(f"- 数据集保存路径: {os.path.abspath(self.output_dir)}")
        
        elapsed_time = time.time() - start_time
        print(f"总耗时: {int(elapsed_time//60)}分{int(elapsed_time%60)}秒")

    def save_coco_dataset(self):
        """保存COCO格式的数据集"""
        coco_data = {
            "info": {
                "description": "CARLA Vehicle Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "CARLA Simulator",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License"
                }
            ],
            "images": self.coco_images,
            "annotations": self.coco_annotations,
            "categories": [
                {
                    "supercategory": "vehicle",
                    "id": 1,
                    "name": "vehicle"
                }
            ]
        }
        
        coco_path = os.path.join(self.output_dir, 'coco_annotations.json')
        with open(coco_path, 'w') as f:
            # 使用convert_numpy_types_for_json函数处理COCO数据
            json_data = convert_numpy_types_for_json(coco_data)
            json.dump(json_data, f, indent=4)
        
        print(f'COCO格式数据集已保存至: {coco_path}')

    def cleanup(self):
        """清理资源"""
        # 恢复异步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        # 销毁激光雷达传感器
        if self.lidar:
            self.lidar.destroy()
        
        # 销毁鸟瞰相机传感器
        if self.birdview_camera:
            self.birdview_camera.destroy()
        
        if self.vehicle:
            self.vehicle.destroy()
        
        print('已清理资源')

    def save_lidar_dataset_metadata(self):
        """保存激光雷达点云数据集的元数据文件"""
        metadata = {
            "info": {
                "description": "CARLA LiDAR Point Cloud Dataset",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "CARLA Simulator",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "lidar_parameters": {
                "channels": self.lidar_channels,
                "rotation_frequency": self.lidar_rotation_frequency,
                "points_per_second": self.lidar_points_per_second,
                "range": self.lidar_range,
                "upper_fov": self.lidar_upper_fov,
                "lower_fov": self.lidar_lower_fov,
                "image_width": self.image_w,
                "image_height": self.image_h
            },
            "lidar_sensor": {
                "position": {"x": 0.0, "y": 0.0, "z": 2.0}
            },
            "files": {
                "lidar_points_dir": "lidar_points",
                "images_dir": "images",
                "annotations_dir": "annotations",
                "previews_dir": "previews"
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'lidar_dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            # 使用convert_numpy_types_for_json函数处理元数据
            json_data = convert_numpy_types_for_json(metadata)
            json.dump(json_data, f, indent=4)
        
        print(f'激光雷达数据集元数据已保存至: {metadata_path}')

    def create_radar_style_preview(self, lidar_points, detected_vehicles):
        """创建一个类似雷达的预览图，显示击中车辆的标记，保持与baseline一致的样式"""
        # 创建画布
        radar_preview = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        
        # 计算中心点，表示本车位置
        center_x, center_y = self.image_w // 2, self.image_h // 2
        
        # 绘制背景网格和坐标轴
        # 绘制中心点（车辆位置）
        cv2.circle(radar_preview, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 绘制同心圆，表示距离
        for r in range(1, 6):
            radius = int(min(self.image_w, self.image_h) / 12 * r)
            cv2.circle(radar_preview, (center_x, center_y), radius, (30, 30, 30), 1)
        
        # 绘制坐标轴
        cv2.line(radar_preview, (center_x, 0), (center_x, self.image_h), (30, 30, 30), 1)  # 垂直轴
        cv2.line(radar_preview, (0, center_y), (self.image_w, center_y), (30, 30, 30), 1)  # 水平轴
        
        # 添加距离标签
        max_range_shown = self.lidar_range
        for r in range(1, 6):
            distance = max_range_shown * r / 5
            # 在同心圆右侧标注距离
            label_x = center_x + int(min(self.image_w, self.image_h) / 12 * r)
            label_y = center_y
            cv2.putText(radar_preview, f"{distance:.0f}m", (label_x + 5, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # 简化点云渲染，与baseline保持一致
        # 对于有x,y坐标的点，直接绘制为白色
        for point in lidar_points:
            if 'x' in point and 'y' in point:
                x, y = point['x'], point['y']
                if 0 <= x < self.image_w and 0 <= y < self.image_h:
                    # 检查该点是否击中车辆
                    if point['is_hitting_vehicle']:
                        # 击中车辆的点用亮一点的颜色
                        radar_preview[y, x] = (255, 255, 255)  # 白色
                    else:
                        # 未击中车辆的点也用白色，但亮度稍低
                        radar_preview[y, x] = (180, 180, 180)
        
        # 绘制检测到的车辆边界框
        for vehicle in detected_vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]
            # 绘制车辆边界框
            cv2.rectangle(radar_preview, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            # 显示车辆ID和距离
            cv2.putText(radar_preview, f"{vehicle['distance']:.1f}m", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 添加标题
        cv2.putText(radar_preview, "LiDAR Radar View", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
        
        return radar_preview

    def get_birdview_image(self):
        """获取鸟瞰相机图像"""
        try:
            # 获取鸟瞰相机数据
            camera_data = self.birdview_queue.get(timeout=2.0)
            
            # 转换为OpenCV格式
            array = np.frombuffer(camera_data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (camera_data.height, camera_data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]  # BGR -> RGB
            
            # 创建一个副本
            birdview_img = array.copy()
            
            # 绘制中心点（表示车辆位置）
            center_x, center_y = self.image_w // 2, self.image_h // 2
            cv2.circle(birdview_img, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 绘制同心圆，表示距离 - 使用半透明覆盖以保持原图可见
            overlay = birdview_img.copy()
            
            # 绘制同心圆，表示距离
            for r in range(1, 6):
                radius = int(min(self.image_w, self.image_h) / 12 * r)
                cv2.circle(overlay, (center_x, center_y), radius, (0, 0, 0), 1)
            
            # 绘制坐标轴
            cv2.line(overlay, (center_x, 0), (center_x, self.image_h), (0, 0, 0), 1)  # 垂直轴
            cv2.line(overlay, (0, center_y), (self.image_w, center_y), (0, 0, 0), 1)  # 水平轴
            
            # 添加距离标签
            max_range_shown = self.lidar_range
            for r in range(1, 6):
                distance = max_range_shown * r / 5
                # 在同心圆右侧标注距离
                label_x = center_x + int(min(self.image_w, self.image_h) / 12 * r)
                label_y = center_y
                cv2.putText(overlay, f"{distance:.0f}m", (label_x + 5, label_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # 合并原图和覆盖层
            alpha = 0.7  # 透明度
            cv2.addWeighted(overlay, alpha, birdview_img, 1 - alpha, 0, birdview_img)
            
            return birdview_img
            
        except queue.Empty:
            print("鸟瞰相机数据获取超时")
            return None


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARLA激光雷达数据集生成器 - 使用LiDAR传感器收集数据并生成鸟瞰图数据集')
    
    parser.add_argument('--host', default='localhost', help='CARLA服务器主机名 (默认: localhost)')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口 (默认: 2000)')
    parser.add_argument('--output-dir', default='lidar_dataset', help='数据集输出目录 (默认: lidar_dataset)')
    parser.add_argument('--num-frames', type=int, default=100, help='要生成的帧数 (默认: 100)')
    parser.add_argument('--num-vehicles', type=int, default=20, help='场景中的车辆数量 (默认: 20)')
    parser.add_argument('--capture-interval', type=float, default=3.0, help='帧捕获间隔，单位为秒 (默认: 3.0)')
    parser.add_argument('--image-width', type=int, default=800, help='激光雷达图像宽度 (默认: 800)')
    parser.add_argument('--image-height', type=int, default=600, help='激光雷达图像高度 (默认: 600)')
    parser.add_argument('--lidar-range', type=float, default=50.0, help='激光雷达探测范围，单位为米 (默认: 50.0)')
    parser.add_argument('--lidar-rotation-frequency', type=float, default=20.0, help='激光雷达旋转频率 (默认: 20.0)')
    parser.add_argument('--lidar-channels', type=int, default=16, help='激光雷达通道数 (默认: 16)')
    parser.add_argument('--lidar-points-per-second', type=int, default=100000, help='激光雷达每秒点数 (默认: 100000)')
    parser.add_argument('--lidar-upper-fov', type=float, default=10.0, help='激光雷达上视场角度 (默认: 10.0)')
    parser.add_argument('--lidar-lower-fov', type=float, default=-30.0, help='激光雷达下视场角度 (默认: -30.0)')
    parser.add_argument('--detection-range', type=float, default=50.0, help='车辆检测范围，单位为米 (默认: 50.0)')
    parser.add_argument('--hit-tolerance', type=int, default=5, help='激光雷达点击中判定的容忍值范围，单位为像素 (默认: 5)')
    parser.add_argument('--show-3d-bbox', action='store_true', help='在预览中显示3D边界框（已弃用）')
    parser.add_argument('--weather', choices=['default', 'badweather', 'night', 'badweather_night'], 
                        default='default', help='天气预设: default(默认晴天), badweather(恶劣天气), night(夜晚), badweather_night(恶劣天气的夜晚)')
    parser.add_argument('--skip-empty', action='store_true', help='跳过无车辆帧，只保存包含车辆的图像 (默认: 不跳过)')
    parser.add_argument('--save-lidar-points', action='store_true', help='保存激光雷达点云数据，用于训练激光雷达识别模型 (默认: 不保存)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，显示更多的调试信息 (默认: 不启用)')
    parser.add_argument('--use-tuple-format', action='store_true', help='强制使用元组格式解析LiDAR数据 (默认: 自动检测)')
    parser.add_argument('--vehicle-model', default='vehicle.lincoln.mkz_2020', help='主车辆模型 (默认: vehicle.lincoln.mkz_2020)')
    parser.add_argument('--lidar-height', type=float, default=2.4, help='激光雷达传感器安装高度，单位为米 (默认: 2.4)')
    parser.add_argument('--no-radar-preview', action='store_true', help='禁用雷达样式的预览窗口 (默认: 启用)')
    
    return parser.parse_args()


def main():
    try:
        print("=" * 80)
        print("CARLA激光雷达数据集生成器 - 与Baseline一致版本")
        print("注意: 所有LiDAR参数已调整为与baseline完全一致")
        print("- 激光雷达探测范围: 50米")
        print("- 激光雷达通道数: 16")
        print("- 激光雷达旋转频率: 20Hz")
        print("- 每秒点数: 10万")
        print("- 安装高度: 2.4米")
        print("- 刷新率: 20Hz (sensor_tick=0.05)")
        print("=" * 80)
        
        # 解析命令行参数
        args = parse_arguments()
        
        # 创建数据集生成器
        generator = CarlaDatasetGenerator(
            host=args.host, 
            port=args.port, 
            output_dir=args.output_dir
        )
        
        # 设置相机和激光雷达属性
        generator.image_w = args.image_width
        generator.image_h = args.image_height
        generator.lidar_range = args.lidar_range
        generator.lidar_rotation_frequency = args.lidar_rotation_frequency
        generator.lidar_channels = args.lidar_channels
        generator.lidar_points_per_second = args.lidar_points_per_second
        generator.lidar_upper_fov = args.lidar_upper_fov
        generator.lidar_lower_fov = args.lidar_lower_fov
        
        # 添加调试标志
        generator.debug_mode = args.debug
        generator.use_tuple_format = args.use_tuple_format
        generator.vehicle_model = args.vehicle_model
        generator.lidar_height = args.lidar_height
        generator.show_radar_preview = not args.no_radar_preview
        
        print(f"使用参数：")
        print(f"- 输出目录: {args.output_dir}")
        print(f"- 帧数: {args.num_frames}")
        print(f"- 激光雷达范围: {args.lidar_range}米")
        print(f"- 激光雷达通道数: {args.lidar_channels}")
        print(f"- 激光雷达旋转频率: {args.lidar_rotation_frequency}Hz")
        print(f"- 激光雷达每秒点数: {args.lidar_points_per_second}")
        print(f"- 车辆模型: {args.vehicle_model}")
        print(f"- 气象条件: {args.weather}")
        print(f"- 调试模式: {'开启' if args.debug else '关闭'}")
        
        # 设置场景，传入天气预设参数
        generator.setup_scenario(num_vehicles=args.num_vehicles, weather_preset=args.weather)
        
        # 生成数据集
        generator.generate_dataset(
            num_frames=args.num_frames,
            capture_interval=args.capture_interval,
            show_3d_bbox=args.show_3d_bbox,
            detection_range=args.detection_range,
            skip_empty_frames=args.skip_empty,
            hit_tolerance=args.hit_tolerance,
            save_lidar_points=args.save_lidar_points
        )
        
    except KeyboardInterrupt:
        print('\n中断由用户触发...')
    except Exception as e:
        print(f'发生错误: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # 确保清理资源
        if 'generator' in locals():
            generator.cleanup()
        print('程序结束')


if __name__ == '__main__':
    main() 