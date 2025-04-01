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
        
        # 生成相机
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.image_w))
        camera_bp.set_attribute('image_size_y', str(self.image_h))
        camera_bp.set_attribute('fov', str(self.fov))
        
        camera_init_trans = carla.Transform(carla.Location(z=2))
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)
        
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
            # 在CARLA中，值越大表示越慢，66.7表示只有33.3%的原速度
            # 该值表示"比正常速度慢多少百分比"
            print("设置主车速度为正常速度的1/3")
            traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 66.7)
            
        except Exception as e:
            print(f"注意: 设置高级交通管理器参数失败，这在某些CARLA版本中是正常的: {e}")
        
        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True  # 启用同步模式
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # 创建队列存储传感器数据
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)
        
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

    def generate_dataset(self, num_frames=100, capture_interval=5, show_3d_bbox=False, detection_range=50):
        """生成数据集"""
        print(f"开始生成数据集：目标帧数 {num_frames}，采集间隔 {capture_interval} 秒")
        print(f"图像尺寸: {self.image_w}x{self.image_h}，检测范围: {detection_range}米")
        print(f"显示3D边界框: {'是' if show_3d_bbox else '否'}")
        print("已启用边界框重叠过滤: 当一个边界框完全包含另一个时，只保留较大的边界框")
        
        # 确保输出目录已创建
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)
        
        # 获取投影矩阵
        K = self.build_projection_matrix(self.image_w, self.image_h, self.fov)
        K_b = self.build_projection_matrix(self.image_w, self.image_h, self.fov, is_behind_camera=True)
        
        last_capture_time = time.time()
        frames_processed = 0
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
                print(f"原始检测边界框: {len(bboxes_to_render)}, 过滤后边界框: {len(filtered_bboxes)}")
                
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
                        # 保存图像
                        frame_id = self.image_count
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        img_filename = f'{frame_id:06d}.png'
                        img_path = os.path.join(self.images_dir, img_filename)
                        
                        # 保存原始图像
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
                        
                        # 保存预览图像
                        preview_path = os.path.join(self.preview_dir, img_filename)
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
                        
                        self.image_count += 1
                        frames_processed += 1
                        last_capture_time = current_time
                        
                        elapsed_time = time.time() - start_time
                        remaining_time = (elapsed_time / frames_processed) * (num_frames - frames_processed) if frames_processed > 0 else 0
                        print(f'已采集: {frames_processed}/{num_frames} 帧 ({frames_processed/num_frames*100:.1f}%)，'
                              f'下一帧将在 {capture_interval} 秒后采集')
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
        # 恢复异步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        # 销毁演员
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        
        print('已清理资源')


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
    parser.add_argument('--detection-range', type=float, default=75.0, help='车辆检测范围，单位为米 (默认: 75.0)')
    parser.add_argument('--show-3d-bbox', action='store_true', help='显示3D边界框 (默认不显示)')
    parser.add_argument('--weather', choices=['default', 'badweather', 'night', 'badweather_night'], 
                        default='default', help='天气预设: default(默认晴天), badweather(恶劣天气), night(夜晚), badweather_night(恶劣天气的夜晚)')
    
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
        generator.fov = args.fov
        
        # 设置场景，传入天气预设参数
        generator.setup_scenario(num_vehicles=args.num_vehicles, weather_preset=args.weather)
        
        # 生成数据集
        generator.generate_dataset(
            num_frames=args.num_frames,
            capture_interval=args.capture_interval,
            show_3d_bbox=args.show_3d_bbox,
            detection_range=args.detection_range
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