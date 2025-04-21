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
        self.radar_points_dir = os.path.join(output_dir, 'radar_points')  # 雷达点云数据目录
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        os.makedirs(self.radar_points_dir, exist_ok=True)  # 创建雷达点云数据目录
        
        # 数据集属性
        self.image_count = 0
        self.coco_images = []
        self.coco_annotations = []
        self.annotation_id = 0
        
        # 设置场景
        self.vehicle = None
        # 改为四个方向的雷达传感器
        self.radar_front = None
        self.radar_back = None
        self.radar_left = None
        self.radar_right = None
        
        # 四个方向的雷达数据队列
        self.radar_front_queue = None
        self.radar_back_queue = None
        self.radar_left_queue = None
        self.radar_right_queue = None
        
        # 雷达/图像属性
        self.image_w = 800
        self.image_h = 600
        self.radar_range = 100.0  # 雷达探测范围，单位为米
        self.radar_fov = 30.0    # 雷达视场角度

        # 边界框绘制
        self.edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

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
        
        # 生成主车辆
        vehicle_bp = self.bp_lib.find('vehicle.lincoln.mkz_2020')
        spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            # 如果生成失败，尝试其他点位
            for sp in spawn_points:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle is not None:
                    break
        
        if self.vehicle is None:
            raise Exception("无法生成主车辆")
        
        # 创建雷达传感器蓝图
        radar_bp = self.bp_lib.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', str(self.radar_fov))
        radar_bp.set_attribute('vertical_fov', '10.0')  # 垂直视场角度
        radar_bp.set_attribute('range', str(self.radar_range))
        # 设置雷达点云密度
        radar_bp.set_attribute('points_per_second', '1500')
        
        # 创建四个方向的雷达传感器

        # 1. 前向雷达
        front_transform = carla.Transform(carla.Location(x=2.0, z=1.0), carla.Rotation(yaw=0))
        self.radar_front = self.world.spawn_actor(radar_bp, front_transform, attach_to=self.vehicle)
        
        # 2. 后向雷达
        back_transform = carla.Transform(carla.Location(x=-2.0, z=1.0), carla.Rotation(yaw=180))
        self.radar_back = self.world.spawn_actor(radar_bp, back_transform, attach_to=self.vehicle)
        
        # 3. 左向雷达
        left_transform = carla.Transform(carla.Location(y=1.0, z=1.0), carla.Rotation(yaw=90))
        self.radar_left = self.world.spawn_actor(radar_bp, left_transform, attach_to=self.vehicle)
        
        # 4. 右向雷达
        right_transform = carla.Transform(carla.Location(y=-1.0, z=1.0), carla.Rotation(yaw=270))
        self.radar_right = self.world.spawn_actor(radar_bp, right_transform, attach_to=self.vehicle)
        
        # 设置自动驾驶
        try:
            self.vehicle.set_autopilot(True, traffic_manager.get_port())
        except:
            # 旧版本CARLA可能不接受端口参数
            self.vehicle.set_autopilot(True)
        
        # 设置主车辆行为 - 设置为原来的1/3速度
        try:
            traffic_manager.ignore_lights_percentage(self.vehicle, 30)  # 30%概率忽略红绿灯，保持适当移动
            traffic_manager.distance_to_leading_vehicle(self.vehicle, 1.0)  # 正常前车距离
            
            # 主车速度设置为原来的1/3 
            print("设置主车速度为正常速度的1/3")
            traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 66.7)
            
        except Exception as e:
            print(f"注意: 设置高级交通管理器参数失败，这在某些CARLA版本中是正常的: {e}")
        
        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # 启用同步模式
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # 创建队列存储四个方向的雷达数据
        self.radar_front_queue = queue.Queue()
        self.radar_back_queue = queue.Queue()
        self.radar_left_queue = queue.Queue()
        self.radar_right_queue = queue.Queue()
        
        # 设置雷达监听
        self.radar_front.listen(self.radar_front_queue.put)
        self.radar_back.listen(self.radar_back_queue.put)
        self.radar_left.listen(self.radar_left_queue.put)
        self.radar_right.listen(self.radar_right_queue.put)
        
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
        
        # 生成NPC车辆
        batch = []
        for i in range(num_vehicles):
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
                    
                    # 设置为原来的1/3速度
                    # 在CARLA中，值越大表示越慢，66.7表示只有33.3%的原速度
                    traffic_manager.vehicle_percentage_speed_difference(npc, 66.7)
                    print(f"设置NPC车辆(ID:{npc.id})速度为正常速度的1/3")
                    
                    # 默认车道变更行为
                    traffic_manager.auto_lane_change(npc, True)
                    try:
                        # 启用碰撞检测，保障安全驾驶
                        traffic_manager.collision_detection(npc, True)
                    except:
                        pass
                    
                except Exception as e:
                    print(f"注意: 设置车辆 {npc.id} 的高级交通管理器参数失败: {e}")
                    
                vehicles_list.append(npc)
        
        print(f"成功初始化场景: 已生成 {len(vehicles_list)} 辆动态NPC车辆，主车ID: {self.vehicle.id}")

    def merge_radar_data(self):
        """从四个雷达获取数据并合并"""
        try:
            # 获取四个方向的雷达数据
            radar_front_data = self.radar_front_queue.get(timeout=2.0)
            radar_back_data = self.radar_back_queue.get(timeout=2.0)
            radar_left_data = self.radar_left_queue.get(timeout=2.0)
            radar_right_data = self.radar_right_queue.get(timeout=2.0)
            
            # 处理并合并四个方向的雷达数据
            radar_img, radar_points = self.process_radar_data([
                ("front", radar_front_data),
                ("back", radar_back_data),
                ("left", radar_left_data),
                ("right", radar_right_data)
            ])
            
            return radar_img, radar_points
            
        except queue.Empty:
            print("雷达数据获取超时")
            return None, None

    def process_radar_data(self, radar_data_list):
        """处理多个方向的雷达数据，创建雷达可视化图像"""
        # 创建雷达图像画布
        radar_img = np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        
        # 计算中心点，表示本车位置
        center_x, center_y = self.image_w // 2, self.image_h // 2
        
        # 绘制雷达背景网格和坐标轴
        # 绘制中心点（车辆位置）
        cv2.circle(radar_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # 绘制同心圆，表示距离
        for r in range(1, 6):
            radius = int(min(self.image_w, self.image_h) / 12 * r)
            cv2.circle(radar_img, (center_x, center_y), radius, (50, 50, 50), 1)
        
        # 绘制坐标轴
        cv2.line(radar_img, (center_x, 0), (center_x, self.image_h), (50, 50, 50), 1)  # 垂直轴
        cv2.line(radar_img, (0, center_y), (self.image_w, center_y), (50, 50, 50), 1)  # 水平轴
        
        # 添加距离标签
        max_range_shown = self.radar_range
        for r in range(1, 6):
            distance = max_range_shown * r / 5
            # 在同心圆右侧标注距离
            label_x = center_x + int(min(self.image_w, self.image_h) / 12 * r)
            label_y = center_y
            cv2.putText(radar_img, f"{distance:.0f}m", (label_x + 5, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # 处理所有方向的雷达探测点
        radar_points = []
        point_id_counter = 0
        
        # 方向到颜色的映射，用于在预览图像中区分不同方向的点
        direction_colors = {
            "front": (0, 180, 0),   # 深绿色
            "back": (0, 120, 180),  # 橙色
            "left": (180, 0, 180),  # 紫色
            "right": (180, 180, 0)  # 青色
        }
        
        for direction, radar_data in radar_data_list:
            for detect in radar_data:
                # 获取极坐标信息
                azimuth = detect.azimuth
                altitude = detect.altitude
                depth = detect.depth  # 距离
                velocity = detect.velocity  # 相对速度
                
                # 调整方位角度，考虑雷达的朝向
                adjusted_azimuth = azimuth
                if direction == "back":
                    # 后向雷达，需要旋转180度
                    adjusted_azimuth = azimuth + math.pi
                elif direction == "left":
                    # 左向雷达，需要旋转90度
                    adjusted_azimuth = azimuth + math.pi/2
                elif direction == "right":
                    # 右向雷达，需要旋转270度
                    adjusted_azimuth = azimuth - math.pi/2
                
                # 确保方位角在-π到π范围内
                while adjusted_azimuth > math.pi:
                    adjusted_azimuth -= 2 * math.pi
                while adjusted_azimuth < -math.pi:
                    adjusted_azimuth += 2 * math.pi
                
                # 根据极坐标计算直角坐标 - 注意y轴向下
                # 计算点在雷达图像上的位置，从中心点出发
                scale = min(self.image_w, self.image_h) / (2 * self.radar_range)
                x = center_x + int(depth * math.sin(adjusted_azimuth) * scale)
                y = center_y - int(depth * math.cos(adjusted_azimuth) * scale)  # 减法是因为屏幕坐标y轴向下
                
                # 确保点在图像范围内
                if 0 <= x < self.image_w and 0 <= y < self.image_h:
                    # 根据方向选择颜色
                    color = direction_colors.get(direction, (0, 100, 0))
                    
                    # 点的大小仍然基于速度，使移动较快的点更明显
                    point_size = min(5, max(2, int(abs(velocity)) + 2))
                    cv2.circle(radar_img, (x, y), point_size, color, -1)
                    
                    # 获取更多雷达属性，如信号强度等
                    try:
                        # 在CARLA 0.9.10+中可用
                        snr = detect.get_snr() if hasattr(detect, 'get_snr') else 0.0
                    except:
                        snr = 0.0
                    
                    # 计算三维世界坐标 (相对于雷达传感器)
                    world_x = depth * math.cos(altitude) * math.cos(azimuth)
                    world_y = depth * math.cos(altitude) * math.sin(azimuth)
                    world_z = depth * math.sin(altitude)
                    
                    # 根据雷达方向调整世界坐标
                    if direction == "back":
                        world_x = -world_x
                        world_y = -world_y
                    elif direction == "left":
                        temp_x = world_x
                        world_x = -world_y
                        world_y = temp_x
                    elif direction == "right":
                        temp_x = world_x
                        world_x = world_y
                        world_y = -temp_x
                    
                    # 存储点信息供后续处理
                    radar_points.append({
                        'id': point_id_counter,  # 雷达点ID（全局唯一）
                        'direction': direction,  # 雷达方向
                        'x': x, 'y': y,  # 图像坐标
                        'depth': depth,  # 实际距离
                        'azimuth': adjusted_azimuth,  # 调整后的方位角
                        'original_azimuth': azimuth,  # 原始方位角
                        'altitude': altitude,  # 俯仰角
                        'velocity': velocity,  # 相对速度
                        'snr': snr,  # 信噪比
                        'raw_position': (detect.depth, detect.azimuth, detect.altitude),  # 原始极坐标
                        'world_position': (world_x, world_y, world_z),  # 3D世界坐标 (相对于车辆)
                        'hit_vehicle_id': None,  # 击中的车辆ID，初始为None
                        'is_hitting_vehicle': False,  # 是否击中车辆
                    })
                    point_id_counter += 1
                    
                    # 为明显移动的物体绘制速度矢量线
                    if abs(velocity) > 3.0:
                        line_length = min(20, max(5, int(abs(velocity) * 2)))
                        end_x = x + int(line_length * math.sin(adjusted_azimuth))
                        end_y = y - int(line_length * math.cos(adjusted_azimuth))
                        cv2.line(radar_img, (x, y), (end_x, end_y), color, 1)
        
        return radar_img, radar_points

    def generate_dataset(self, num_frames=100, capture_interval=5, show_3d_bbox=False, detection_range=50, skip_empty_frames=False, hit_tolerance=5, save_radar_points=False):
        """生成雷达数据集"""
        print(f"开始生成雷达数据集：目标帧数 {num_frames}，采集间隔 {capture_interval} 秒")
        print(f"雷达范围: {self.radar_range}米，检测范围: {detection_range}米")
        print(f"使用前、后、左、右四个方向的雷达传感器采集数据")
        print(f"注意: 只有被雷达点云实际击中的车辆才会被标记（容忍范围: {hit_tolerance}像素）")
        
        if save_radar_points:
            print("已启用雷达点云数据保存: 将记录每个雷达点及其击中的目标")
        
        if skip_empty_frames:
            print("已启用空帧过滤: 只保存包含车辆标记的帧")
        else:
            print("未启用空帧过滤: 所有捕获的帧都将被保存")
        
        # 确保输出目录已创建
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        
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
        
        last_capture_time = time.time()
        frames_processed = 0
        frames_skipped = 0
        start_time = time.time()
        
        while frames_processed < num_frames:
            try:
                # 刷新世界
                self.world.tick()
                
                # 从四个雷达获取数据并合并
                radar_img, radar_points = self.merge_radar_data()
                
                if radar_img is None or radar_points is None:
                    continue
                
                # 创建一个工作副本用于显示
                display_img = radar_img.copy()
                
                # 获取场景中的所有车辆，用于在雷达图上标注
                detected_vehicles = []
                
                # 获取主车位置和方向
                ego_transform = self.vehicle.get_transform()
                ego_location = ego_transform.location
                ego_rotation = ego_transform.rotation
                ego_forward = ego_transform.get_forward_vector()
                
                # 计算雷达图中心点（本车位置）
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
                                    
                                    # 计算在雷达图上的坐标
                                    scale = min(self.image_w, self.image_h) / (2 * self.radar_range)
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
                                    
                                    # 检查是否有雷达点与当前车辆的边界框重叠
                                    radar_hit = False
                                    hitting_points = []  # 记录击中当前车辆的雷达点
                                    for point in radar_points:
                                        # 检查雷达点是否在边界框内或边界框外的容忍范围内
                                        if ((x1-hit_tolerance) <= point['x'] <= (x2+hit_tolerance) and 
                                            (y1-hit_tolerance) <= point['y'] <= (y2+hit_tolerance)):
                                            radar_hit = True
                                            
                                            # 记录该点击中了哪个车辆
                                            point['hit_vehicle_id'] = npc.id
                                            point['is_hitting_vehicle'] = True
                                            hitting_points.append(point)
                                    
                                    # 只有当车辆被雷达点击中时，才标记为检测到的车辆
                                    if radar_hit:
                                        # 存储检测到的车辆信息，增加点击车辆的雷达点信息
                                        detected_vehicles.append({
                                            "vehicle_id": npc.id,
                                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                            "distance": dist,
                                            "angle": angle,
                                            "position": (x, y),
                                            "real_position": (dist, angle),  # 保存极坐标
                                            "box_size_factor": box_size_factor,  # 保存缩放因子以便调试
                                            "hitting_points_count": len(hitting_points),  # 击中该车辆的雷达点数量
                                            "hitting_points_ids": [p['id'] for p in hitting_points]  # 击中该车辆的雷达点ID列表
                                        })
                                        
                                        # 在雷达图上绘制边界框
                                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                                        # 添加距离和缩放因子信息
                                        cv2.putText(display_img, f"{dist:.1f}m", (x1, y1-5), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                        # 在边界框底部添加缩放因子信息
                                        cv2.putText(display_img, f"S:{box_size_factor:.1f}", (x1, y2+12), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 200), 1)
                                    else:
                                        # 对于未被雷达击中但在检测范围内的车辆，用虚线框标记（便于调试）
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
                
                # 显示雷达图
                cv2.putText(display_img, f"Frames: {frames_processed}/{num_frames}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 显示下次采集倒计时
                time_to_next = max(0, capture_interval - (time.time() - last_capture_time))
                cv2.putText(display_img, f"Next capture: {time_to_next:.1f}s", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 显示检测到的车辆数量
                cv2.putText(display_img, f"Vehicles detected: {len(detected_vehicles)}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('CARLA Radar Dataset', display_img)
                
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
                        
                        # 保存雷达图像
                        cv2.imwrite(img_path, radar_img)
                        
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
                        
                        # 保存雷达点云数据（如果启用）
                        if save_radar_points:
                            # 创建雷达点云数据字典
                            radar_data = {
                                "frame_id": frame_id,
                                "timestamp": timestamp,
                                "total_points": len(radar_points),
                                "vehicle_count": len(detected_vehicles),
                                "points": []
                            }
                            
                            # 将每个雷达点的信息添加到数据中
                            for point in radar_points:
                                radar_data["points"].append({
                                    "id": point["id"],
                                    "direction": point["direction"],  # 雷达方向
                                    "depth": point["depth"],
                                    "azimuth": point["azimuth"],
                                    "altitude": point["altitude"],
                                    "velocity": point["velocity"],
                                    "snr": point["snr"],
                                    "world_position": point["world_position"],
                                    "image_position": [point["x"], point["y"]],
                                    "hit_vehicle_id": point["hit_vehicle_id"],
                                    "is_hitting_vehicle": point["is_hitting_vehicle"]
                                })
                            
                            # 保存雷达点云数据为JSON文件
                            radar_path = os.path.join(self.radar_points_dir, f'{frame_id:06d}.json')
                            with open(radar_path, 'w') as f:
                                json.dump(radar_data, f, indent=2)
                            
                            print(f"已保存雷达点云数据: {len(radar_points)} 个点，其中 {sum(1 for p in radar_points if p['is_hitting_vehicle'])} 个点击中车辆")
                        
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
        
        # 如果启用了雷达点云数据保存，则保存元数据
        if save_radar_points:
            print("正在保存雷达点云数据集元数据...")
            self.save_radar_dataset_metadata()
        
        # 打印数据集统计信息
        print(f"\n数据集生成统计:")
        print(f"- 总共采集雷达图像: {self.image_count} 帧")
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
            json.dump(coco_data, f, indent=4)
        
        print(f'COCO格式数据集已保存至: {coco_path}')

    def cleanup(self):
        """清理资源"""
        # 恢复异步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        # 销毁四个方向的雷达传感器
        for radar in [self.radar_front, self.radar_back, self.radar_left, self.radar_right]:
            if radar:
                radar.destroy()
        
        if self.vehicle:
            self.vehicle.destroy()
        
        print('已清理资源')

    def save_radar_dataset_metadata(self):
        """保存雷达点云数据集的元数据文件"""
        metadata = {
            "info": {
                "description": "CARLA Radar Point Cloud Dataset with Multi-directional Radar",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "CARLA Simulator",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "radar_parameters": {
                "horizontal_fov": self.radar_fov,
                "vertical_fov": 10.0,  # 默认垂直FOV
                "range": self.radar_range,
                "image_width": self.image_w,
                "image_height": self.image_h
            },
            "radar_sensors": {
                "directions": ["front", "back", "left", "right"],
                "positions": {
                    "front": {"x": 2.0, "y": 0.0, "z": 1.0, "yaw": 0},
                    "back": {"x": -2.0, "y": 0.0, "z": 1.0, "yaw": 180},
                    "left": {"x": 0.0, "y": 1.0, "z": 1.0, "yaw": 90},
                    "right": {"x": 0.0, "y": -1.0, "z": 1.0, "yaw": 270}
                }
            },
            "files": {
                "radar_points_dir": "radar_points",
                "images_dir": "images",
                "annotations_dir": "annotations",
                "previews_dir": "previews"
            }
        }
        
        metadata_path = os.path.join(self.output_dir, 'radar_dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f'雷达点云数据集元数据已保存至: {metadata_path}')


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARLA雷达数据集生成器 - 使用前后左右四个方向的雷达传感器收集数据')
    
    parser.add_argument('--host', default='localhost', help='CARLA服务器主机名 (默认: localhost)')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口 (默认: 2000)')
    parser.add_argument('--output-dir', default='radar_dataset', help='数据集输出目录 (默认: radar_dataset)')
    parser.add_argument('--num-frames', type=int, default=200, help='要生成的帧数 (默认: 200)')
    parser.add_argument('--num-vehicles', type=int, default=50, help='场景中的车辆数量 (默认: 50)')
    parser.add_argument('--capture-interval', type=float, default=5.0, help='帧捕获间隔，单位为秒 (默认: 5.0)')
    parser.add_argument('--image-width', type=int, default=800, help='雷达图像宽度 (默认: 800)')
    parser.add_argument('--image-height', type=int, default=600, help='雷达图像高度 (默认: 600)')
    parser.add_argument('--fov', type=float, default=120.0, help='每个雷达的水平视场角度 (默认: 120.0)')
    parser.add_argument('--detection-range', type=float, default=70.0, help='雷达探测范围，单位为米 (默认: 70.0)')
    parser.add_argument('--hit-tolerance', type=int, default=5, help='雷达点击中判定的容忍值范围，单位为像素 (默认: 5)')
    parser.add_argument('--show-3d-bbox', action='store_true', help='在预览中显示3D边界框（已弃用）')
    parser.add_argument('--weather', choices=['default', 'badweather', 'night', 'badweather_night'], 
                        default='default', help='天气预设: default(默认晴天), badweather(恶劣天气), night(夜晚), badweather_night(恶劣天气的夜晚)')
    parser.add_argument('--skip-empty', action='store_true', help='跳过无车辆帧，只保存包含车辆的图像 (默认: 不跳过)')
    parser.add_argument('--save-radar-points', action='store_true', help='保存雷达点云数据，用于训练雷达识别模型 (默认: 不保存)')
    
    return parser.parse_args()


def main():
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 创建数据集生成器
        generator = CarlaDatasetGenerator(
            host=args.host, 
            port=args.port, 
            output_dir=args.output_dir
        )
        
        # 设置相机属性
        generator.image_w = args.image_width
        generator.image_h = args.image_height
        generator.radar_range = args.detection_range
        generator.radar_fov = args.fov
        
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
            save_radar_points=args.save_radar_points
        )
        
    except KeyboardInterrupt:
        print('\n中断由用户触发...')
    except Exception as e:
        print(f'发生错误: {e}')
    finally:
        # 确保清理资源
        if 'generator' in locals():
            generator.cleanup()
        print('程序结束')


if __name__ == '__main__':
    main() 