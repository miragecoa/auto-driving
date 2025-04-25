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

# 全局变量声明
client = None

# 导入perception_utils中的函数
try:
    from perception_utils import (
        set_weather_preset,
        set_random_seed,
        initialize_world,
        spawn_ego_vehicle,
        spawn_surrounding_vehicles,
        clean_up_all_vehicles
    )
except ImportError:
    print("警告: 无法导入perception_utils中的函数，将使用基本功能")
    
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
        
    # 基础的随机种子设置函数
    def set_random_seed(seed_value):
        if seed_value > 0:
            random.seed(seed_value)
            np.random.seed(seed_value)
            print(f"随机种子设置为 {seed_value}")

class CarlaDatasetGenerator:
    def __init__(self, host='localhost', port=2000, output_dir='vehicle_dataset'):
        # 连接到CARLA服务器
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = None  # 将在setup_scenario中初始化
        self.original_settings = None  # 存储原始世界设置
        
        # 创建输出目录
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        self.annotations_dir = os.path.join(output_dir, 'annotations')
        self.preview_dir = os.path.join(output_dir, 'previews')  # 新增预览目录
        self.radar_points_dir = os.path.join(output_dir, 'radar_points')  # 雷达点云数据目录
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)  # 创建预览目录
        os.makedirs(self.radar_points_dir, exist_ok=True)  # 创建雷达点云目录
        
        # 数据集属性
        self.image_count = 0
        self.coco_images = []
        self.coco_annotations = []
        self.annotation_id = 0
        
        # 设置场景
        self.vehicle = None
        
        # 多方向相机
        self.camera_front = None
        self.camera_back = None
        self.camera_left = None
        self.camera_right = None
        
        # 多方向相机数据队列
        self.camera_front_queue = None
        self.camera_back_queue = None
        self.camera_left_queue = None
        self.camera_right_queue = None
        
        # 雷达传感器 (四个方向)
        self.radar_front = None
        self.radar_back = None
        self.radar_left = None
        self.radar_right = None
        
        # 雷达数据队列
        self.radar_front_queue = None
        self.radar_back_queue = None
        self.radar_left_queue = None
        self.radar_right_queue = None
        
        # 传感器属性
        self.image_w = 800
        self.image_h = 600
        self.fov = 90
        self.radar_range = 100.0  # 雷达探测范围，单位为米
        self.radar_fov = 30.0    # 雷达视场角度
        
        # 边界框绘制
        self.edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        
        # 其他车辆列表
        self.other_vehicles = []
        
        # 雷达车辆命中记录
        self.vehicle_hit_time = {}  # 记录每个车辆最近被雷达命中的时间

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

    def setup_scenario(self, args):
        """设置模拟场景，使用与baseline_perception相同的初始化逻辑
        
        参数:
            args: 命令行参数，包含num_vehicles, weather, seed等
        """
        print("开始初始化场景，使用与baseline_perception相同的逻辑...")
        
        # 确保args包含baseline_perception所需的所有参数
        # perception_utils中初始化世界和生成车辆需要的参数
        if not hasattr(args, 'sync'):
            print("添加默认sync参数")
            args.sync = True
            
        if not hasattr(args, 'autopilot'):
            print("添加默认autopilot参数")
            args.autopilot = True
            
        # 初始化世界（使用perception_utils中的函数）
        print(f"使用seed={args.seed}初始化世界")
        self.world, self.original_settings = initialize_world(self.client, args)
        self.bp_lib = self.world.get_blueprint_library()
        
        # 显式调用set_random_seed确保种子设置正确
        set_random_seed(args.seed)
        
        # 生成主车辆（使用perception_utils中的函数）
        print(f"使用seed={args.seed}生成主车辆")
        main_vehicle, spawn_idx, spawn_points = spawn_ego_vehicle(self.world, args)
        self.vehicle = main_vehicle
        print(f"主车已生成，车辆ID: {self.vehicle.id}, 生成点索引: {spawn_idx}")
        
        # 生成NPC车辆（使用perception_utils中的函数）
        # 过滤掉已使用的生成点
        available_spawn_points = [p for i, p in enumerate(spawn_points) if i != spawn_idx]
        npc_count = min(args.num_vehicles, len(available_spawn_points))
        
        if npc_count > 0:
            print(f"尝试生成{npc_count}辆NPC车辆")
            self.other_vehicles = spawn_surrounding_vehicles(
                self.client, 
                self.world, 
                number_of_vehicles=npc_count, 
                spawn_points=available_spawn_points
            )
            print(f"已生成 {len(self.other_vehicles)} 辆NPC车辆")
        
        # 生成多方向相机
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_w))
        camera_bp.set_attribute('image_size_y', str(self.image_h))
        camera_bp.set_attribute('fov', str(self.fov))
        
        # 创建四个方向的相机
        # 1. 前向相机
        front_cam_transform = carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(yaw=0))
        self.camera_front = self.world.spawn_actor(camera_bp, front_cam_transform, attach_to=self.vehicle)
        
        # 2. 后向相机
        back_cam_transform = carla.Transform(carla.Location(x=-2.0, z=2.0), carla.Rotation(yaw=180))
        self.camera_back = self.world.spawn_actor(camera_bp, back_cam_transform, attach_to=self.vehicle)
        
        # 3. 左向相机
        left_cam_transform = carla.Transform(carla.Location(y=1.0, z=2.0), carla.Rotation(yaw=90))
        self.camera_left = self.world.spawn_actor(camera_bp, left_cam_transform, attach_to=self.vehicle)
        
        # 4. 右向相机
        right_cam_transform = carla.Transform(carla.Location(y=-1.0, z=2.0), carla.Rotation(yaw=270))
        self.camera_right = self.world.spawn_actor(camera_bp, right_cam_transform, attach_to=self.vehicle)
        
        # 创建队列存储各个方向的相机数据
        self.camera_front_queue = queue.Queue()
        self.camera_back_queue = queue.Queue()
        self.camera_left_queue = queue.Queue()
        self.camera_right_queue = queue.Queue()
        
        # 设置相机数据监听
        self.camera_front.listen(self.camera_front_queue.put)
        self.camera_back.listen(self.camera_back_queue.put)
        self.camera_left.listen(self.camera_left_queue.put)
        self.camera_right.listen(self.camera_right_queue.put)
        
        # 初始化雷达传感器 - 与generate_radar_dataset完全一致
        radar_bp = self.bp_lib.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', str(self.radar_fov))
        radar_bp.set_attribute('vertical_fov', '10.0')  # 垂直视场角度
        radar_bp.set_attribute('range', str(self.radar_range))
        # 设置雷达点云密度
        radar_bp.set_attribute('points_per_second', '1500')
        
        # 创建四个方向的雷达传感器
        # 1. 前向雷达
        front_radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0), carla.Rotation(yaw=0))
        self.radar_front = self.world.spawn_actor(radar_bp, front_radar_transform, attach_to=self.vehicle)
        
        # 2. 后向雷达
        back_radar_transform = carla.Transform(carla.Location(x=-2.0, z=1.0), carla.Rotation(yaw=180))
        self.radar_back = self.world.spawn_actor(radar_bp, back_radar_transform, attach_to=self.vehicle)
        
        # 3. 左向雷达
        left_radar_transform = carla.Transform(carla.Location(y=1.0, z=1.0), carla.Rotation(yaw=90))
        self.radar_left = self.world.spawn_actor(radar_bp, left_radar_transform, attach_to=self.vehicle)
        
        # 4. 右向雷达
        right_radar_transform = carla.Transform(carla.Location(y=-1.0, z=1.0), carla.Rotation(yaw=270))
        self.radar_right = self.world.spawn_actor(radar_bp, right_radar_transform, attach_to=self.vehicle)
        
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
        
        # 获取交通管理器以进行额外的车辆设置
        traffic_manager = self.client.get_trafficmanager(8000)
        
        # 为主车设置额外参数
        try:
            # 主车速度设置为原来的1/3 
            traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 66.7)
            print("设置主车速度为正常速度的1/3")
        except Exception as e:
            print(f"设置主车参数时出错: {e}")
            
        # 为其他车辆设置额外参数
        for npc in self.other_vehicles:
            try:
                # 设置为原来的1/3速度
                traffic_manager.vehicle_percentage_speed_difference(npc, 66.7)
            except Exception as e:
                print(f"设置NPC车辆参数时出错: {e}")
                
        print("场景初始化完成")

    def generate_dataset(self, num_frames=100, capture_interval=5, show_3d_bbox=False, detection_range=50, skip_empty_frames=False, hit_tolerance=5, radar_hit_timeout=2.0):
        """生成数据集
        
        参数:
            num_frames: 要生成的帧数
            capture_interval: 帧捕获间隔，单位为秒
            show_3d_bbox: 是否显示3D边界框
            detection_range: 车辆检测范围，单位为米
            skip_empty_frames: 是否跳过无车辆的帧
            hit_tolerance: 雷达命中检测的容忍像素范围
            radar_hit_timeout: 雷达命中超时时间，单位为秒，超过此时间未命中则不标记
        """
        print(f"开始生成数据集：目标帧数 {num_frames}，采集间隔 {capture_interval} 秒")
        print(f"图像尺寸: {self.image_w}x{self.image_h}，检测范围: {detection_range}米")
        print(f"显示3D边界框: {'是' if show_3d_bbox else '否'}")
        print(f"雷达命中检测: 命中容忍范围 {hit_tolerance} 像素，命中超时 {radar_hit_timeout} 秒")
        print("已启用边界框重叠过滤: 当一个边界框完全包含另一个时，只保留较大的边界框")
        
        if skip_empty_frames:
            print("已启用空帧过滤: 只保存包含车辆标记的帧")
        else:
            print("未启用空帧过滤: 所有捕获的帧都将被保存")
        
        # 确保输出目录已创建
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        os.makedirs(self.radar_points_dir, exist_ok=True)
        
        # 检查已存在的图像文件，确定起始的image_count
        existing_images = [f for f in os.listdir(self.images_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if existing_images:
            # 提取文件名中的数字部分，找出最大值
            max_id = -1
            for img_name in existing_images:
                try:
                    # 尝试从文件名中提取数字ID
                    id_str = os.path.splitext(img_name)[0]  # 去除扩展名
                    img_id = int(id_str)
                    max_id = max(max_id, img_id)
                except ValueError:
                    # 如果文件名不是纯数字，则忽略
                    continue
            
            # 确保image_count大于已存在的最大ID
            if max_id >= 0:
                self.image_count = max_id + 1
                print(f"根据已存在的图像文件，起始图像ID设置为: {self.image_count}")
                print(f"这将防止覆盖现有的图像和标注文件")
        
        # 获取投影矩阵
        K = self.build_projection_matrix(self.image_w, self.image_h, self.fov)
        K_b = self.build_projection_matrix(self.image_w, self.image_h, self.fov, is_behind_camera=True)
        
        last_capture_time = time.time()
        frames_processed = 0
        frames_skipped = 0  # 新增：记录因无车辆而跳过的帧数
        start_time = time.time()
        
        # 创建四个预览窗口
        cv2.namedWindow('Front View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Back View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Left View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Right View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Radar View', cv2.WINDOW_NORMAL)
        
        # 调整预览窗口大小
        cv2.resizeWindow('Front View', int(self.image_w * 0.7), int(self.image_h * 0.7))
        cv2.resizeWindow('Back View', int(self.image_w * 0.7), int(self.image_h * 0.7))
        cv2.resizeWindow('Left View', int(self.image_w * 0.7), int(self.image_h * 0.7))
        cv2.resizeWindow('Right View', int(self.image_w * 0.7), int(self.image_h * 0.7))
        cv2.resizeWindow('Radar View', int(self.image_w * 0.7), int(self.image_h * 0.7))
        
        # 调整窗口位置以便同时显示所有窗口
        cv2.moveWindow('Front View', 0, 0)
        cv2.moveWindow('Back View', self.image_w, 0)
        cv2.moveWindow('Left View', 0, self.image_h)
        cv2.moveWindow('Right View', self.image_w, self.image_h)
        cv2.moveWindow('Radar View', int(self.image_w * 1.5), 0)
        
        while frames_processed < num_frames:
            try:
                # 刷新世界
                self.world.tick()
                
                # 获取雷达数据
                radar_img, radar_points = self.merge_radar_data()
                
                if radar_img is None or radar_points is None:
                    print("无法获取雷达数据，跳过当前帧")
                    continue
                
                # 检索四个方向的相机图像
                try:
                    camera_front_img = self.camera_front_queue.get(timeout=2.0)
                    camera_back_img = self.camera_back_queue.get(timeout=2.0)
                    camera_left_img = self.camera_left_queue.get(timeout=2.0)
                    camera_right_img = self.camera_right_queue.get(timeout=2.0)
                except queue.Empty:
                    print("相机数据获取超时，跳过当前帧")
                    continue
                
                # 获取各个相机的转换矩阵
                world_2_camera_front = np.array(self.camera_front.get_transform().get_inverse_matrix())
                world_2_camera_back = np.array(self.camera_back.get_transform().get_inverse_matrix())
                world_2_camera_left = np.array(self.camera_left.get_transform().get_inverse_matrix())
                world_2_camera_right = np.array(self.camera_right.get_transform().get_inverse_matrix())
                
                # 转换图像为NumPy数组用于显示
                img_front = np.reshape(np.copy(camera_front_img.raw_data), (camera_front_img.height, camera_front_img.width, 4))
                img_back = np.reshape(np.copy(camera_back_img.raw_data), (camera_back_img.height, camera_back_img.width, 4))
                img_left = np.reshape(np.copy(camera_left_img.raw_data), (camera_left_img.height, camera_left_img.width, 4))
                img_right = np.reshape(np.copy(camera_right_img.raw_data), (camera_right_img.height, camera_right_img.width, 4))
                
                # 创建工作副本用于显示和处理
                img_front_display = img_front.copy()
                img_back_display = img_back.copy()
                img_left_display = img_left.copy()
                img_right_display = img_right.copy()
                radar_display = radar_img.copy()
                
                # 创建要渲染的边界框列表（四个相机方向）
                bboxes_front = []
                bboxes_back = []
                bboxes_left = []
                bboxes_right = []
                
                # 获取当前时间，用于雷达命中检测
                current_time = time.time()
                
                # 处理场景中的所有车辆
                for npc in self.world.get_actors().filter('*vehicle*'):
                    # 过滤出主车辆
                    if npc.id != self.vehicle.id:
                        try:
                            bb = npc.bounding_box
                            dist = npc.get_transform().location.distance(self.vehicle.get_transform().location)
                            
                            # 过滤出指定范围内的车辆
                            if dist < detection_range:
                                # 检查雷达是否命中该车辆
                                vehicle_hit_by_radar = False
                                
                                # 处理前方相机 - 判断是否在前方视野中
                                forward_vec = self.camera_front.get_transform().get_forward_vector()
                                ray = npc.get_transform().location - self.camera_front.get_transform().location
                                
                                if self.dot_product(forward_vec, ray) > 0:
                                    # 渲染3D边界框
                                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                    
                                    # 找出2D边界框的极值
                                    x_max = -10000
                                    x_min = 10000
                                    y_max = -10000
                                    y_min = 10000
                                    
                                    for vert in verts:
                                        p = self.get_image_point(vert, K, world_2_camera_front)
                                        # 找出最右侧的顶点
                                        if p[0] > x_max:
                                            x_max = p[0]
                                        # 找出最左侧的顶点
                                        if p[0] < x_min:
                                            x_min = p[0]
                                        # 找出最高的顶点
                                        if p[1] > y_max:
                                            y_max = p[1]
                                        # 找出最低的顶点
                                        if p[1] < y_min:
                                            y_min = p[1]
                                    
                                    # 确保边界框在图像内部
                                    if x_min > 0 and x_max < 1.5*self.image_w and y_min > 0 and y_max < 1.5*self.image_h:
                                        # 检查雷达是否命中该车辆
                                        for point in radar_points:
                                            # 更新雷达命中状态
                                            if ((x_min-hit_tolerance) <= point['x'] <= (x_max+hit_tolerance) and 
                                                (y_min-hit_tolerance) <= point['y'] <= (y_max+hit_tolerance)):
                                                vehicle_hit_by_radar = True
                                                point['hit_vehicle_id'] = npc.id
                                                point['is_hitting_vehicle'] = True
                                                # 更新最近命中时间
                                                self.vehicle_hit_time[npc.id] = current_time
                                                break
                                                
                                        # 检查时间窗口内是否有雷达命中
                                        radar_hit_valid = npc.id in self.vehicle_hit_time and (current_time - self.vehicle_hit_time[npc.id]) <= radar_hit_timeout
                                        
                                        # 只有在雷达命中有效时才存储边界框信息
                                        if radar_hit_valid:
                                            bboxes_front.append({
                                                "vehicle_id": npc.id,
                                                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                                            })
                                            
                                            # 在这里绘制3D边界框（如果需要）
                                            if show_3d_bbox:
                                                for edge in self.edges:
                                                    p1 = self.get_image_point(verts[edge[0]], K, world_2_camera_front)
                                                    p2 = self.get_image_point(verts[edge[1]], K, world_2_camera_front)
                                                    if self.point_in_canvas(p1, self.image_h, self.image_w) and self.point_in_canvas(p2, self.image_h, self.image_w):
                                                        cv2.line(img_front_display, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)
                                
                                # 处理后方相机 - 判断是否在后方视野中
                                backward_vec = self.camera_back.get_transform().get_forward_vector()
                                ray = npc.get_transform().location - self.camera_back.get_transform().location
                                
                                if self.dot_product(backward_vec, ray) > 0:
                                    # 处理后方相机逻辑，类似于前方相机
                                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                    
                                    # 找出2D边界框的极值
                                    x_max = -10000
                                    x_min = 10000
                                    y_max = -10000
                                    y_min = 10000
                                    
                                    for vert in verts:
                                        p = self.get_image_point(vert, K, world_2_camera_back)
                                        if p[0] > x_max: x_max = p[0]
                                        if p[0] < x_min: x_min = p[0]
                                        if p[1] > y_max: y_max = p[1]
                                        if p[1] < y_min: y_min = p[1]
                                    
                                    # 确保边界框在图像内部
                                    if x_min > 0 and x_max < 1.5*self.image_w and y_min > 0 and y_max < 1.5*self.image_h:
                                        # 检查雷达是否命中该车辆
                                        for point in radar_points:
                                            # 更新雷达命中状态
                                            if ((x_min-hit_tolerance) <= point['x'] <= (x_max+hit_tolerance) and 
                                                (y_min-hit_tolerance) <= point['y'] <= (y_max+hit_tolerance)):
                                                vehicle_hit_by_radar = True
                                                point['hit_vehicle_id'] = npc.id
                                                point['is_hitting_vehicle'] = True
                                                # 更新最近命中时间
                                                self.vehicle_hit_time[npc.id] = current_time
                                                break
                                        
                                        # 检查时间窗口内是否有雷达命中
                                        radar_hit_valid = npc.id in self.vehicle_hit_time and (current_time - self.vehicle_hit_time[npc.id]) <= radar_hit_timeout
                                        
                                        # 只有在雷达命中有效时才存储边界框信息
                                        if radar_hit_valid:
                                            bboxes_back.append({
                                                "vehicle_id": npc.id,
                                                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                                            })
                                            
                                            # 在这里绘制3D边界框（如果需要）
                                            if show_3d_bbox:
                                                for edge in self.edges:
                                                    p1 = self.get_image_point(verts[edge[0]], K, world_2_camera_back)
                                                    p2 = self.get_image_point(verts[edge[1]], K, world_2_camera_back)
                                                    if self.point_in_canvas(p1, self.image_h, self.image_w) and self.point_in_canvas(p2, self.image_h, self.image_w):
                                                        cv2.line(img_back_display, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)
                                
                                # 处理左方相机 - 判断是否在左方视野中
                                left_vec = self.camera_left.get_transform().get_forward_vector()
                                ray = npc.get_transform().location - self.camera_left.get_transform().location
                                
                                if self.dot_product(left_vec, ray) > 0:
                                    # 处理左方相机逻辑
                                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                    
                                    # 找出2D边界框的极值
                                    x_max = -10000
                                    x_min = 10000
                                    y_max = -10000
                                    y_min = 10000
                                    
                                    for vert in verts:
                                        p = self.get_image_point(vert, K, world_2_camera_left)
                                        if p[0] > x_max: x_max = p[0]
                                        if p[0] < x_min: x_min = p[0]
                                        if p[1] > y_max: y_max = p[1]
                                        if p[1] < y_min: y_min = p[1]
                                    
                                    # 确保边界框在图像内部
                                    if x_min > 0 and x_max < 1.5*self.image_w and y_min > 0 and y_max < 1.5*self.image_h:
                                        # 检查雷达是否命中该车辆
                                        for point in radar_points:
                                            # 更新雷达命中状态
                                            if ((x_min-hit_tolerance) <= point['x'] <= (x_max+hit_tolerance) and 
                                                (y_min-hit_tolerance) <= point['y'] <= (y_max+hit_tolerance)):
                                                vehicle_hit_by_radar = True
                                                point['hit_vehicle_id'] = npc.id
                                                point['is_hitting_vehicle'] = True
                                                # 更新最近命中时间
                                                self.vehicle_hit_time[npc.id] = current_time
                                                break
                                        
                                        # 检查时间窗口内是否有雷达命中
                                        radar_hit_valid = npc.id in self.vehicle_hit_time and (current_time - self.vehicle_hit_time[npc.id]) <= radar_hit_timeout
                                        
                                        # 只有在雷达命中有效时才存储边界框信息
                                        if radar_hit_valid:
                                            bboxes_left.append({
                                                "vehicle_id": npc.id,
                                                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                                            })
                                            
                                            # 在这里绘制3D边界框（如果需要）
                                            if show_3d_bbox:
                                                for edge in self.edges:
                                                    p1 = self.get_image_point(verts[edge[0]], K, world_2_camera_left)
                                                    p2 = self.get_image_point(verts[edge[1]], K, world_2_camera_left)
                                                    if self.point_in_canvas(p1, self.image_h, self.image_w) and self.point_in_canvas(p2, self.image_h, self.image_w):
                                                        cv2.line(img_left_display, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)
                                
                                # 处理右方相机 - 判断是否在右方视野中
                                right_vec = self.camera_right.get_transform().get_forward_vector()
                                ray = npc.get_transform().location - self.camera_right.get_transform().location
                                
                                if self.dot_product(right_vec, ray) > 0:
                                    # 处理右方相机逻辑
                                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                    
                                    # 找出2D边界框的极值
                                    x_max = -10000
                                    x_min = 10000
                                    y_max = -10000
                                    y_min = 10000
                                    
                                    for vert in verts:
                                        p = self.get_image_point(vert, K, world_2_camera_right)
                                        if p[0] > x_max: x_max = p[0]
                                        if p[0] < x_min: x_min = p[0]
                                        if p[1] > y_max: y_max = p[1]
                                        if p[1] < y_min: y_min = p[1]
                                    
                                    # 确保边界框在图像内部
                                    if x_min > 0 and x_max < 1.5*self.image_w and y_min > 0 and y_max < 1.5*self.image_h:
                                        # 检查雷达是否命中该车辆
                                        for point in radar_points:
                                            # 更新雷达命中状态
                                            if ((x_min-hit_tolerance) <= point['x'] <= (x_max+hit_tolerance) and 
                                                (y_min-hit_tolerance) <= point['y'] <= (y_max+hit_tolerance)):
                                                vehicle_hit_by_radar = True
                                                point['hit_vehicle_id'] = npc.id
                                                point['is_hitting_vehicle'] = True
                                                # 更新最近命中时间
                                                self.vehicle_hit_time[npc.id] = current_time
                                                break
                                        
                                        # 检查时间窗口内是否有雷达命中
                                        radar_hit_valid = npc.id in self.vehicle_hit_time and (current_time - self.vehicle_hit_time[npc.id]) <= radar_hit_timeout
                                        
                                        # 只有在雷达命中有效时才存储边界框信息
                                        if radar_hit_valid:
                                            bboxes_right.append({
                                                "vehicle_id": npc.id,
                                                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                                            })
                                            
                                            # 在这里绘制3D边界框（如果需要）
                                            if show_3d_bbox:
                                                for edge in self.edges:
                                                    p1 = self.get_image_point(verts[edge[0]], K, world_2_camera_right)
                                                    p2 = self.get_image_point(verts[edge[1]], K, world_2_camera_right)
                                                    if self.point_in_canvas(p1, self.image_h, self.image_w) and self.point_in_canvas(p2, self.image_h, self.image_w):
                                                        cv2.line(img_right_display, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)
                                        
                        except Exception as e:
                            print(f"处理车辆 {npc.id} 时出错: {e}")
                            continue
                
                # 过滤掉被完全包含的边界框
                filtered_bboxes_front = self.filter_contained_bboxes(bboxes_front)
                filtered_bboxes_back = self.filter_contained_bboxes(bboxes_back)
                filtered_bboxes_left = self.filter_contained_bboxes(bboxes_left)
                filtered_bboxes_right = self.filter_contained_bboxes(bboxes_right)
                
                # 绘制雷达点
                for point in radar_points:
                    # 如果是击中车辆的点，使用红色
                    if point['is_hitting_vehicle']:
                        cv2.circle(radar_display, (point['x'], point['y']), 5, (0, 0, 255), -1)
                
                # 绘制四个方向的边界框
                for bbox_info in filtered_bboxes_front:
                    x_min, y_min, x_max, y_max = bbox_info["bbox"]
                    cv2.rectangle(img_front_display, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
                
                for bbox_info in filtered_bboxes_back:
                    x_min, y_min, x_max, y_max = bbox_info["bbox"]
                    cv2.rectangle(img_back_display, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
                
                for bbox_info in filtered_bboxes_left:
                    x_min, y_min, x_max, y_max = bbox_info["bbox"]
                    cv2.rectangle(img_left_display, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
                
                for bbox_info in filtered_bboxes_right:
                    x_min, y_min, x_max, y_max = bbox_info["bbox"]
                    cv2.rectangle(img_right_display, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
                
                # 显示图像
                # 转换BGRA到BGR用于显示
                display_front = cv2.cvtColor(img_front_display, cv2.COLOR_BGRA2BGR)
                display_back = cv2.cvtColor(img_back_display, cv2.COLOR_BGRA2BGR)
                display_left = cv2.cvtColor(img_left_display, cv2.COLOR_BGRA2BGR)
                display_right = cv2.cvtColor(img_right_display, cv2.COLOR_BGRA2BGR)
                
                # 添加状态信息
                cv2.putText(display_front, f"Front View: {len(filtered_bboxes_front)} vehicles", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_back, f"Back View: {len(filtered_bboxes_back)} vehicles", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_left, f"Left View: {len(filtered_bboxes_left)} vehicles", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_right, f"Right View: {len(filtered_bboxes_right)} vehicles", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示倒计时
                time_to_next = max(0, capture_interval - (time.time() - last_capture_time))
                cv2.putText(radar_display, f"Next Capture: {time_to_next:.1f}s", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(radar_display, f"Radar Points: {len(radar_points)}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(radar_display, f"Vehicles Hit: {sum(1 for p in radar_points if p['is_hitting_vehicle'])}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示所有窗口
                cv2.imshow('Front View', display_front)
                cv2.imshow('Back View', display_back)
                cv2.imshow('Left View', display_left)
                cv2.imshow('Right View', display_right)
                cv2.imshow('Radar View', radar_display)
                
                # 每隔capture_interval秒采集一次数据
                current_capture_time = time.time()
                if current_capture_time - last_capture_time >= capture_interval:
                    try:
                        # 首先检查是否所有方向都检测到了车辆，如果启用了空帧过滤，则跳过无车辆帧
                        total_vehicles = len(filtered_bboxes_front) + len(filtered_bboxes_back) + len(filtered_bboxes_left) + len(filtered_bboxes_right)
                        
                        if skip_empty_frames and total_vehicles == 0:
                            frames_skipped += 1
                            print(f"当前帧未检测到雷达命中的车辆，跳过保存 (已跳过: {frames_skipped} 帧)")
                            last_capture_time = current_capture_time
                            # 继续处理下一帧，不计入已处理帧数
                            continue
                        
                        # 获取可用的图像ID，确保不覆盖现有文件
                        frame_id = self.image_count
                        
                        # 获取当前时间戳
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 为四个方向分别保存图像和标注
                        directions = ["front", "back", "left", "right"]
                        cameras = [self.camera_front, self.camera_back, self.camera_left, self.camera_right]
                        images = [camera_front_img, camera_back_img, camera_left_img, camera_right_img]
                        bboxes_list = [filtered_bboxes_front, filtered_bboxes_back, filtered_bboxes_left, filtered_bboxes_right]
                        display_images = [display_front, display_back, display_left, display_right]
                        
                        for i, direction in enumerate(directions):
                            img_filename = f'{frame_id:06d}_{direction}.png'
                            img_path = os.path.join(self.images_dir, img_filename)
                            
                            # 保存图像
                            images[i].save_to_disk(img_path)
                            
                            # 保存带有边界框标记的预览图像
                            preview_path = os.path.join(self.preview_dir, img_filename)
                            cv2.imwrite(preview_path, display_images[i])
                            
                            # 创建VOC标注writer
                            voc_writer = Writer(img_path, self.image_w, self.image_h)
                            
                            # 添加到COCO图像列表
                            self.coco_images.append({
                                "license": 1,
                                "file_name": img_filename,
                                "height": self.image_h,
                                "width": self.image_w,
                                "date_captured": timestamp,
                                "id": frame_id * 10 + i,
                                "direction": direction
                            })
                            
                            # 处理边界框
                            for bbox_info in bboxes_list[i]:
                                x_min, y_min, x_max, y_max = bbox_info["bbox"]
                                
                                # 添加到VOC标注
                                voc_writer.addObject('vehicle', x_min, y_min, x_max, y_max)
                                
                                # 添加到COCO标注
                                self.coco_annotations.append({
                                    "id": self.annotation_id,
                                    "image_id": frame_id * 10 + i,
                                    "category_id": 1,  # 1 表示车辆
                                    "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                                    "area": float((x_max - x_min) * (y_max - y_min)),
                                    "segmentation": [],
                                    "iscrowd": 0,
                                    "direction": direction,
                                    "vehicle_id": bbox_info["vehicle_id"]
                                })
                                self.annotation_id += 1
                            
                            # 保存VOC格式标注
                            voc_path = os.path.join(self.annotations_dir, f'{frame_id:06d}_{direction}.xml')
                            voc_writer.save(voc_path)
                        
                        # 保存雷达点云数据
                        radar_data = {
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "total_points": len(radar_points),
                            "vehicle_count": total_vehicles,
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
                        
                        # 更新图像计数和计时器
                        self.image_count += 1
                        frames_processed += 1
                        last_capture_time = current_capture_time
                        
                        # 计算剩余时间并显示进度
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
                        last_capture_time = current_capture_time
                
                # 按'q'键退出
                if cv2.waitKey(1) == ord('q'):
                    print("用户按下q键，提前结束数据集生成")
                    break
            except Exception as e:
                print(f"数据处理循环中发生错误: {e}")
                # 短暂等待后继续尝试
                time.sleep(0.1)
        
        # 关闭所有显示窗口
        cv2.destroyAllWindows()
        
        # 保存COCO格式数据集
        print("数据采集完成，正在保存COCO格式数据集...")
        self.save_coco_dataset()
        
        # 打印数据集统计信息
        print(f"\n数据集生成统计:")
        print(f"- 总共采集图像: {self.image_count * 4} 帧 (4个方向)")
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
        # 恢复原始世界设置
        if self.original_settings:
            self.world.apply_settings(self.original_settings)
        
        # 销毁所有传感器
        # 相机
        if self.camera_front:
            self.camera_front.destroy()
        if self.camera_back:
            self.camera_back.destroy()
        if self.camera_left:
            self.camera_left.destroy()
        if self.camera_right:
            self.camera_right.destroy()
            
        # 雷达
        if self.radar_front:
            self.radar_front.destroy()
        if self.radar_back:
            self.radar_back.destroy()
        if self.radar_left:
            self.radar_left.destroy()
        if self.radar_right:
            self.radar_right.destroy()
            
        # 销毁所有生成的车辆
        for vehicle in self.other_vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
                
        # 销毁主车
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
        
        print('已清理所有资源')

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
                y = center_y - int(depth * math.cos(adjusted_azimuth) * scale)
                
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


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARLA车辆数据集生成器 - 多相机和雷达版本')
    
    parser.add_argument('--host', default='localhost', help='CARLA服务器主机名 (默认: localhost)')
    parser.add_argument('--port', type=int, default=2000, help='CARLA服务器端口 (默认: 2000)')
    parser.add_argument('--output-dir', default='vehicle_dataset', help='数据集输出目录 (默认: vehicle_dataset)')
    parser.add_argument('--num-frames', type=int, default=200, help='要生成的帧数 (默认: 200)')
    parser.add_argument('--num-vehicles', type=int, default=50, help='场景中的车辆数量 (默认: 50)')
    parser.add_argument('--capture-interval', type=float, default=5.0, help='帧捕获间隔，单位为秒 (默认: 5.0)')
    parser.add_argument('--image-width', type=int, default=800, help='图像宽度 (默认: 800)')
    parser.add_argument('--image-height', type=int, default=600, help='图像高度 (默认: 600)')
    parser.add_argument('--fov', type=float, default=90.0, help='相机视场角度 (默认: 90.0)')
    parser.add_argument('--detection-range', type=float, default=70.0, help='车辆检测范围，单位为米 (默认: 70.0)')
    parser.add_argument('--show-3d-bbox', action='store_true', help='显示3D边界框 (默认不显示)')
    parser.add_argument('--weather', choices=['default', 'badweather', 'night', 'badweather_night'], 
                        default='default', help='天气预设: default(默认晴天), badweather(恶劣天气), night(夜晚), badweather_night(恶劣天气的夜晚)')
    parser.add_argument('--skip-empty', action='store_true', help='跳过无车辆帧，只保存包含车辆的图像 (默认: 不跳过)')
    parser.add_argument('--seed', metavar='S', default=0, type=int, 
                      help='用于可重现结果的随机种子 (默认: 0，表示随机行为)')
    parser.add_argument('--sync', action='store_true', default=True,
                      help='启用同步模式 (默认: 启用)')
    parser.add_argument('--autopilot', action='store_true', default=True,
                      help='启用车辆自动驾驶 (默认: 启用)')
    parser.add_argument('--radar-fov', type=float, default=120.0,
                      help='雷达视场角度 (默认: 120.0)')
    parser.add_argument('--hit-tolerance', type=int, default=5,
                      help='雷达命中检测的容忍像素范围 (默认: 5)')
    parser.add_argument('--radar-hit-timeout', type=float, default=2.0,
                      help='雷达命中超时时间，单位为秒，超过此时间未命中则不标记 (默认: 2.0)')
    
    return parser.parse_args()


def main():
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 确保命令行参数的一致性
        global client  # 确保与baseline_perception相同的全局变量定义
        
        # 创建数据集生成器
        generator = CarlaDatasetGenerator(
            host=args.host, 
            port=args.port, 
            output_dir=args.output_dir
        )
        
        # 设置相机和雷达属性
        generator.image_w = args.image_width
        generator.image_h = args.image_height
        generator.fov = args.fov
        generator.radar_range = args.detection_range
        generator.radar_fov = args.radar_fov
        
        # 使用与baseline_perception完全相同的方式连接客户端
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)  # 与baseline_perception相同的超时设置
        
        # 确保由客户端正确初始化
        generator.client = client
        
        # 设置场景 - 现在直接传递args
        generator.setup_scenario(args)
        
        # 生成数据集
        generator.generate_dataset(
            num_frames=args.num_frames,
            capture_interval=args.capture_interval,
            show_3d_bbox=args.show_3d_bbox,
            detection_range=args.detection_range,
            skip_empty_frames=args.skip_empty,
            hit_tolerance=args.hit_tolerance,
            radar_hit_timeout=args.radar_hit_timeout
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