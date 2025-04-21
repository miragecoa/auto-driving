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
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)  # 创建预览目录
        
        # 数据集属性
        self.image_count = 0
        self.coco_images = []
        self.coco_annotations = []
        self.annotation_id = 0
        
        # 设置场景
        self.vehicle = None
        self.camera = None
        self.image_queue = None
        
        # 相机属性
        self.image_w = 800
        self.image_h = 600
        self.fov = 90
        
        # 边界框绘制
        self.edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        
        # 其他车辆列表
        self.other_vehicles = []

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
        
        # 生成相机
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_w))
        camera_bp.set_attribute('image_size_y', str(self.image_h))
        camera_bp.set_attribute('fov', str(self.fov))
        
        camera_init_trans = carla.Transform(carla.Location(z=2))
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)
        
        # 创建队列存储传感器数据
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)
        
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

    def generate_dataset(self, num_frames=100, capture_interval=5, show_3d_bbox=False, detection_range=50, skip_empty_frames=False):
        """生成数据集"""
        print(f"开始生成数据集：目标帧数 {num_frames}，采集间隔 {capture_interval} 秒")
        print(f"图像尺寸: {self.image_w}x{self.image_h}，检测范围: {detection_range}米")
        print(f"显示3D边界框: {'是' if show_3d_bbox else '否'}")
        print("已启用边界框重叠过滤: 当一个边界框完全包含另一个时，只保留较大的边界框")
        
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
        
        while frames_processed < num_frames:
            try:
                # 检索图像
                self.world.tick()
                carla_image = self.image_queue.get()
                
                # 获取相机到世界的矩阵
                world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())
                
                # 转换图像为NumPy数组用于显示 - 保留一个干净的副本
                img_clean = np.reshape(np.copy(carla_image.raw_data), (carla_image.height, carla_image.width, 4))
                img_display = img_clean.copy()  # 用于显示和处理的副本
                
                # 要渲染的边界框列表
                bboxes_to_render = []
                
                # 处理场景中的所有车辆
                for npc in self.world.get_actors().filter('*vehicle*'):
                    # 过滤出主车辆
                    if npc.id != self.vehicle.id:
                        try:
                            bb = npc.bounding_box
                            dist = npc.get_transform().location.distance(self.vehicle.get_transform().location)
                            
                            # 过滤出指定范围内的车辆
                            if dist < detection_range:
                                # 计算主车前向向量与主车到其他车辆的向量的点积
                                # 我们将这个点积限制为仅在相机前方绘制边界框
                                forward_vec = self.vehicle.get_transform().get_forward_vector()
                                ray = npc.get_transform().location - self.vehicle.get_transform().location
                                
                                # 使用自定义点积函数而不是.dot()方法
                                if self.dot_product(forward_vec, ray) > 0:
                                    # 渲染3D边界框
                                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                    
                                    # 找出2D边界框的极值
                                    x_max = -10000
                                    x_min = 10000
                                    y_max = -10000
                                    y_min = 10000
                                    
                                    for vert in verts:
                                        p = self.get_image_point(vert, K, world_2_camera)
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
                                        # 存储边界框信息供保存时使用
                                        bboxes_to_render.append({
                                            "vehicle_id": npc.id,
                                            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
                                        })
                                        
                                        # 在这里绘制3D边界框（如果需要）
                                        if show_3d_bbox:
                                            for edge in self.edges:
                                                p1 = self.get_image_point(verts[edge[0]], K, world_2_camera)
                                                p2 = self.get_image_point(verts[edge[1]], K, world_2_camera)
                                                if self.point_in_canvas(p1, self.image_h, self.image_w) and self.point_in_canvas(p2, self.image_h, self.image_w):
                                                    cv2.line(img_display, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)
                        except Exception as e:
                            print(f"处理车辆 {npc.id} 时出错: {e}")
                            continue
                
                # 过滤掉被完全包含的边界框
                filtered_bboxes = self.filter_contained_bboxes(bboxes_to_render)
                
                # 只绘制过滤后的边界框
                for bbox_info in filtered_bboxes:
                    x_min, y_min, x_max, y_max = bbox_info["bbox"]
                    # 绘制2D边界框
                    cv2.rectangle(img_display, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
                
                # 显示图像
                display_img = cv2.cvtColor(img_display, cv2.COLOR_BGRA2BGR)
                cv2.putText(display_img, f"Frames: {frames_processed}/{num_frames}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 显示下次采集倒计时
                time_to_next = max(0, capture_interval - (time.time() - last_capture_time))
                cv2.putText(display_img, f"Next capture: {time_to_next:.1f}s", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # 显示检测到的车辆数量 - 使用过滤后的边界框数量
                cv2.putText(display_img, f"Vehicles detected: {len(filtered_bboxes)}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('CARLA Vehicle Dataset', display_img)
                
                # 每隔capture_interval秒采集一次数据
                current_time = time.time()
                if current_time - last_capture_time >= capture_interval:
                    try:
                        # 首先检查是否检测到了车辆，如果启用了空帧过滤，则跳过无车辆帧
                        if skip_empty_frames and not filtered_bboxes:
                            frames_skipped += 1
                            print(f"当前帧未检测到车辆，跳过保存 (已跳过: {frames_skipped} 帧)")
                            last_capture_time = current_time
                            # 继续处理下一帧，不计入已处理帧数
                            continue
                        
                        # 获取可用的图像ID，确保不覆盖现有文件
                        frame_id = self.image_count
                        img_filename = f'{frame_id:06d}.png'
                        img_path = os.path.join(self.images_dir, img_filename)
                        
                        # 检查文件是否已存在，如果存在则增加ID直到找到未使用的
                        while os.path.exists(img_path):
                            frame_id += 1
                            img_filename = f'{frame_id:06d}.png'
                            img_path = os.path.join(self.images_dir, img_filename)
                        
                        # 如果frame_id与原始self.image_count不同，更新并提示用户
                        if frame_id != self.image_count:
                            print(f"跳过已存在的文件名，使用新ID: {frame_id}")
                            self.image_count = frame_id
                        
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 保存图像
                        carla_image.save_to_disk(img_path)
                        
                        # 保存带有边界框标记的预览图像
                        preview_img = display_img.copy()  # 使用已处理好的显示图像
                        # 添加额外信息到预览图像
                        cv2.putText(preview_img, f"Vehicles: {len(filtered_bboxes)}", (10, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # 在预览图像中为每个边界框添加ID标签 - 使用过滤后的边界框
                        for i, bbox_info in enumerate(filtered_bboxes):
                            x_min, y_min, x_max, y_max = bbox_info["bbox"]
                            # 在边界框上方显示编号
                            cv2.putText(preview_img, f"V{i+1}", (int(x_min), int(y_min) - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # 保存预览图像，使用与原始图像相同的文件名
                        preview_path = os.path.join(self.preview_dir, img_filename)
                        
                        # 直接使用相同的文件名保存预览图，不需要检查是否存在
                        # 因为图像ID已经确保唯一性，所以预览图也会使用唯一的文件名
                        cv2.imwrite(preview_path, preview_img)
                        
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
                        
                        # 处理边界框 - 使用过滤后的边界框
                        for bbox_info in filtered_bboxes:
                            x_min, y_min, x_max, y_max = bbox_info["bbox"]
                            
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
                                "iscrowd": 0
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
        
        # 打印数据集统计信息
        print(f"\n数据集生成统计:")
        print(f"- 总共采集图像: {self.image_count} 帧")
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
        
        # 销毁传感器
        if self.camera:
            self.camera.destroy()
            
        # 销毁所有生成的车辆
        for vehicle in self.other_vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
                
        # 销毁主车
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
        
        print('已清理所有资源')


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARLA车辆数据集生成器')
    
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
        
        # 设置相机属性
        generator.image_w = args.image_width
        generator.image_h = args.image_height
        generator.fov = args.fov
        
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
            skip_empty_frames=args.skip_empty
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