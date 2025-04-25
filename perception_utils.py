import glob
import os
import sys
import argparse
import random
import time
import numpy as np
import math
import cv2

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

# 天气预设定义
WEATHER_PRESETS = {
    'ClearNoon': {
        'cloudiness': 0.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 0.0,
        'sun_azimuth_angle': 180.0,
        'sun_altitude_angle': 50.0,
        'fog_density': 0.0,
        'fog_distance': 0.0,
        'wetness': 0.0
    },
    'ClearSunset': {
        'cloudiness': 0.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 0.0,
        'sun_azimuth_angle': 280.0,
        'sun_altitude_angle': 5.0,
        'fog_density': 0.0,
        'fog_distance': 0.0,
        'wetness': 0.0
    },
    'CloudyNoon': {
        'cloudiness': 80.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 20.0,
        'sun_azimuth_angle': 180.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 0.0,
        'fog_distance': 0.0,
        'wetness': 0.0
    },
    'WetNoon': {
        'cloudiness': 30.0,
        'precipitation': 0.0,
        'precipitation_deposits': 50.0,
        'wind_intensity': 20.0,
        'sun_azimuth_angle': 180.0,
        'sun_altitude_angle': 45.0,
        'fog_density': 0.0,
        'fog_distance': 0.0,
        'wetness': 100.0
    },
    'HardRainNoon': {
        'cloudiness': 90.0,
        'precipitation': 80.0,
        'precipitation_deposits': 90.0,
        'wind_intensity': 60.0,
        'sun_azimuth_angle': 180.0,
        'sun_altitude_angle': 40.0,
        'fog_density': 5.0,
        'fog_distance': 10.0,
        'wetness': 100.0
    },
    'SoftFogSunset': {
        'cloudiness': 50.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 15.0,
        'sun_azimuth_angle': 270.0,
        'sun_altitude_angle': 5.0,
        'fog_density': 25.0,
        'fog_distance': 50.0,
        'wetness': 20.0
    },
    'ClearNight': {
        'cloudiness': 0.0,
        'precipitation': 0.0,
        'precipitation_deposits': 0.0,
        'wind_intensity': 10.0,
        'sun_azimuth_angle': 0.0,
        'sun_altitude_angle': -80.0,
        'fog_density': 0.0,
        'fog_distance': 0.0,
        'wetness': 0.0
    },
    'Storm': {
        'cloudiness': 100.0,
        'precipitation': 100.0,
        'precipitation_deposits': 100.0,
        'wind_intensity': 100.0,
        'sun_azimuth_angle': 180.0,
        'sun_altitude_angle': 15.0,
        'fog_density': 10.0,
        'fog_distance': 25.0,
        'wetness': 100.0
    }
}

def set_weather_preset(world, preset="default"):
    """Set weather based on predefined presets"""
    weather = world.get_weather()
    
    # 根据预设名称设置不同的天气条件
    if preset == "default":
        # 晴朗的白天
        weather.sun_altitude_angle = 85.0  # 太阳高度角（正午）
        weather.cloudiness = 10.0  # 云量 (0-100)
        weather.precipitation = 0.0  # 降水量 (0-100)
        weather.precipitation_deposits = 0.0  # 地面积水 (0-100)
        weather.wind_intensity = 10.0  # 风强度 (0-100)
        weather.fog_density = 0.0  # 雾密度 (0-100)
        weather.fog_distance = 0.0  # 雾可见距离
        weather.wetness = 0.0  # 湿度 (0-100)
        weather.sun_azimuth_angle = 45.0  # 太阳方位角
        weather.fog_falloff = 0.0  # 雾衰减
    
    elif preset == "badweather":
        # 恶劣天气 - 强降雨和大风
        weather.sun_altitude_angle = 45.0  # 较低的太阳角度
        weather.cloudiness = 90.0  # 高云量
        weather.precipitation = 80.0  # 大雨
        weather.precipitation_deposits = 60.0  # 积水
        weather.wind_intensity = 70.0  # 大风
        weather.fog_density = 40.0  # 轻雾
        weather.fog_distance = 40.0  
        weather.wetness = 80.0  # 很湿
        weather.sun_azimuth_angle = 45.0  
        weather.fog_falloff = 1.0  
    
    elif preset == "night":
        # 晴朗的夜晚
        weather.sun_altitude_angle = -80.0  # 太阳在地平线以下（夜晚）
        weather.cloudiness = 10.0  # 少云
        weather.precipitation = 0.0  # 无雨
        weather.precipitation_deposits = 0.0  # 无积水
        weather.wind_intensity = 10.0  # 微风
        weather.fog_density = 0.0  # 无雾
        weather.fog_distance = 0.0  
        weather.wetness = 0.0  # 干燥
        weather.sun_azimuth_angle = 225.0  # 太阳方位角（夜晚）
        weather.fog_falloff = 0.0  
    
    elif preset == "badweather_night":
        # 恶劣天气的夜晚 - 雨夜
        weather.sun_altitude_angle = -80.0  # 夜晚
        weather.cloudiness = 90.0  # 多云
        weather.precipitation = 80.0  # 大雨
        weather.precipitation_deposits = 60.0  # 积水
        weather.wind_intensity = 70.0  # 大风
        weather.fog_density = 50.0  # 大雾
        weather.fog_distance = 25.0  # 雾气浓重，可见度低
        weather.wetness = 80.0  # 很湿
        weather.sun_azimuth_angle = 225.0  
        weather.fog_falloff = 1.0  
    
    else:
        print(f"未知的天气预设: {preset}，使用默认晴天")
        # 使用默认晴天设置
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
    
class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Autonomous Driving Perception System - Daylight Baseline Demo")

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        # Calculate the size of each grid cell with proper padding
        width = int(self.window_size[0]/self.grid_size[1])
        height = int(self.window_size[1]/self.grid_size[0])
        return [width, height]

    def get_display_offset(self, gridPos):
        # Calculate the offset for each grid cell to ensure proper alignment
        dis_size = self.get_display_size()
        x_offset = int(gridPos[1] * dis_size[0])
        y_offset = int(gridPos[0] * dis_size[1])
        return [x_offset, y_offset]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        self.display.fill((0, 0, 0))
        
        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, sensor_name=""):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor_name = sensor_name
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

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

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        
        # 逆时针旋转LiDAR点云90度
        # 创建旋转矩阵: [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
        # 对于逆时针旋转90度，theta = 90°，cos(90°) = 0, sin(90°) = 1
        # 旋转矩阵变为: [[0, -1], [1, 0]]
        # rotation_matrix = np.array([[0, -1], [1, 0]])
        # 顺时针旋转90度的旋转矩阵
        rotation_matrix = np.array([[0, 1], [-1, 0]])
        
        # 应用旋转
        lidar_data = np.dot(lidar_data, rotation_matrix.T)
        
        # 继续处理点云数据
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros(lidar_img_size)
        
        # Draw LiDAR points
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        
        disp_size = self.display_man.get_display_size()
        
        # Draw radar data with proper sizing to ensure it fits the display area
        # Use exact display dimensions to avoid any offset
        radar_img = np.zeros((disp_size[1], disp_size[0], 3), dtype=np.uint8)
        
        # Calculate center point for this specific display cell
        center_x, center_y = int(disp_size[0] / 2), int(disp_size[1] / 2)
        
        # Determine view rotation based on sensor name
        view_rotation = 0  # Default no rotation
        # Flag to indicate if we need to flip the image vertically (mirror)
        need_vertical_flip = False
        # Flag to indicate if we need to flip the image horizontally (mirror)
        need_horizontal_flip = False
        
        if self.sensor_name:
            if "Left Radar" in self.sensor_name:
                view_rotation = 90  # 逆时针旋转90度
                need_vertical_flip = True  # 为左侧雷达添加上下镜像
            elif "Front Radar" in self.sensor_name:
                view_rotation = -90   # 顺时针旋转90度
                need_horizontal_flip = True  # 为前雷达添加左右镜像
            elif "Right Radar" in self.sensor_name:
                view_rotation = 90  # 逆时针旋转90度
                need_vertical_flip = True  # 为右侧雷达添加上下镜像
            elif "Rear Radar" in self.sensor_name:
                view_rotation = -90   # 顺时针旋转90度
                need_horizontal_flip = True  # 为后雷达添加左右镜像
        
        # Draw radar background with grid and axes
        # Center point (vehicle position)
        cv2.circle(radar_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw direction indicator
        direction_indicator_length = 20
        indicator_color = (150, 150, 0)  # Yellow-ish
        
        # Determine radar orientation based on sensor_name for direction indicator
        heading_angle = 0.0  # Default forward
        if self.sensor_name:
            if "Left Radar" in self.sensor_name:
                heading_angle = -math.pi/2  # -90 degrees
            elif "Right Radar" in self.sensor_name:
                heading_angle = math.pi/2   # 90 degrees
            elif "Rear Radar" in self.sensor_name:
                heading_angle = math.pi     # 180 degrees
        
        # Apply view rotation to heading angle (in radians)
        view_rotation_rad = math.radians(view_rotation)
        heading_angle_adjusted = heading_angle - view_rotation_rad
                        

        # Draw rotated coordinate grid
        # Draw concentric circles
        for r in range(1, 6):
            radius = int(min(disp_size) / 12 * r)
            cv2.circle(radar_img, (center_x, center_y), radius, (50, 50, 50), 1)
        
        # Draw rotated coordinate axes
        # First axis
        axis1_start_x = center_x + int(disp_size[0] * 0.5 * math.sin(view_rotation_rad))
        axis1_start_y = center_y - int(disp_size[1] * 0.5 * math.cos(view_rotation_rad))
        axis1_end_x = center_x - int(disp_size[0] * 0.5 * math.sin(view_rotation_rad))
        axis1_end_y = center_y + int(disp_size[1] * 0.5 * math.cos(view_rotation_rad))
        
        # Second axis (perpendicular to first)
        axis2_start_x = center_x + int(disp_size[0] * 0.5 * math.sin(view_rotation_rad + math.pi/2))
        axis2_start_y = center_y - int(disp_size[1] * 0.5 * math.cos(view_rotation_rad + math.pi/2))
        axis2_end_x = center_x - int(disp_size[0] * 0.5 * math.sin(view_rotation_rad + math.pi/2))
        axis2_end_y = center_y + int(disp_size[1] * 0.5 * math.cos(view_rotation_rad + math.pi/2))
        
        cv2.line(radar_img, (int(axis1_start_x), int(axis1_start_y)), 
                 (int(axis1_end_x), int(axis1_end_y)), (50, 50, 50), 1)
        cv2.line(radar_img, (int(axis2_start_x), int(axis2_start_y)), 
                 (int(axis2_end_x), int(axis2_end_y)), (50, 50, 50), 1)
        
        # Determine radar orientation based on sensor_name for detected points
        orientation_offset = 0.0
        if self.sensor_name:
            if "Left Radar" in self.sensor_name:
                orientation_offset = -math.pi/2  # 左雷达是90度，需要-90度偏移适配显示
            elif "Right Radar" in self.sensor_name:
                orientation_offset = math.pi/2   # 右雷达是270度，需要+90度偏移适配显示
            elif "Rear Radar" in self.sensor_name:
                orientation_offset = math.pi     # 180度
        
        # Draw range indicator text
        max_range = float(self.sensor_options.get('range', '50'))
        
        for detect in radar_data:
            # Calculate point position
            distance = detect.depth
            # Radar range limited to configuration, scaled to display area
            scale = min(disp_size) / (2 * max_range)
            
            # Apply orientation offset to adjust the visualization direction
            adjusted_azimuth = detect.azimuth + orientation_offset
            
            # Apply view rotation to the point position
            final_azimuth = adjusted_azimuth - view_rotation_rad
            
            # Calculate point coordinates with adjusted orientation and view rotation
            x = center_x + int(distance * math.sin(final_azimuth) * scale)
            y = center_y - int(distance * math.cos(final_azimuth) * scale)
            
            # For left and right radar, flip points vertically if needed
            if need_vertical_flip:
                # Mirror the y-coordinate around the center
                y = center_y + (center_y - y)
            
            # For front and rear radar, flip points horizontally if needed
            if need_horizontal_flip:
                # Mirror the x-coordinate around the center
                x = center_x + (center_x - x)
            
            if 0 <= x < disp_size[0] and 0 <= y < disp_size[1]:
                # Color based on velocity (faster is redder)
                velocity = detect.velocity
                if velocity > 0:  # Approaching objects
                    color = (0, int(255 - min(255, abs(velocity) * 10)), min(255, abs(velocity) * 25))  # Green to blue
                else:  # Receding objects
                    color = (min(255, abs(velocity) * 25), int(255 - min(255, abs(velocity) * 10)), 0)  # Green to red
                
                # Draw point with size based on velocity
                point_size = min(5, max(3, int(abs(velocity) / 5) + 2))
                cv2.circle(radar_img, (x, y), point_size, color, -1)
                
                # For significant velocities, draw a line indicating direction and speed
                if abs(velocity) > 5.0:
                    line_length = min(15, max(5, int(abs(velocity))))
                    # Also rotate the velocity direction indicator
                    # For flipped radar, also adjust the direction line
                    end_x = x
                    end_y = y
                    
                    if need_vertical_flip and need_horizontal_flip:
                        # Both vertical and horizontal flip
                        end_x = x - int(line_length * math.sin(final_azimuth))
                        end_y = y + int(line_length * math.cos(final_azimuth))
                    elif need_vertical_flip:
                        # Only vertical flip
                        end_x = x + int(line_length * math.sin(final_azimuth))
                        end_y = y + int(line_length * math.cos(final_azimuth))
                    elif need_horizontal_flip:
                        # Only horizontal flip
                        end_x = x - int(line_length * math.sin(final_azimuth))
                        end_y = y - int(line_length * math.cos(final_azimuth))
                    else:
                        # No flip
                        end_x = x + int(line_length * math.sin(final_azimuth))
                        end_y = y - int(line_length * math.cos(final_azimuth))
                        
                    cv2.line(radar_img, (x, y), (end_x, end_y), color, 1)
        
        # If we need to flip the image vertically (for left and right radar)
        if need_vertical_flip:
            # Note: We don't flip here anymore since we're flipping the individual points instead
            # This gives us more control over direction indicators
            pass
            
        if self.display_man.render_enabled():
            # Convert OpenCV image (BGR) to RGB for pygame
            radar_img_rgb = cv2.cvtColor(radar_img, cv2.COLOR_BGR2RGB)
            self.surface = pygame.surfarray.make_surface(radar_img_rgb)
            
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)
            
            # Add sensor name label
            if self.sensor_name:
                font = pygame.font.Font(None, 24)
                text = font.render(self.sensor_name, True, (255, 255, 255))
                self.display_man.display.blit(text, (offset[0] + 5, offset[1] + 5))

    def destroy(self):
        self.sensor.destroy()

def spawn_surrounding_vehicles(client, world, number_of_vehicles=20, spawn_points=None):
    """Spawn other vehicles in the simulation"""
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
    # Filter out bicycles and motorcycles
    vehicle_blueprints = [blueprint for blueprint in vehicle_blueprints if int(blueprint.get_attribute('number_of_wheels')) == 4]
    
    if spawn_points is None:
        spawn_points = world.get_map().get_spawn_points()
        
    number_of_spawn_points = len(spawn_points)

    if number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif number_of_vehicles > number_of_spawn_points:
        number_of_vehicles = number_of_spawn_points
        print(f"警告: 请求生成 {number_of_vehicles} 辆车，但只有 {number_of_spawn_points} 个可用生成点。")

    # 使用批处理命令来提高效率
    batch = []
    vehicle_list = []
    
    # 创建生成命令批处理
    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(vehicle_blueprints)
        # Set attributes if available
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        
        # Set the blueprint as autopilot
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', 'autopilot')

        # 添加到批处理
        batch.append(carla.command.SpawnActor(blueprint, transform)
                    .then(carla.command.SetAutopilot(carla.command.FutureActor, True)))
    
    # 批量执行生成命令
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(f"警告: 车辆生成失败: {response.error}")
        else:
            vehicle_list.append(response.actor_id)
    
    # 获取实际生成的车辆Actor引用
    actual_vehicles = []
    for actor_id in vehicle_list:
        actor = world.get_actor(actor_id)
        if actor:
            actual_vehicles.append(actor)
    
    # 为所有生成的车辆设置速度
    try:
        traffic_manager = client.get_trafficmanager(8000)
        for vehicle in actual_vehicles:
            # 设置与主车相同的速度（正常速度的1/3）
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 66.7)
            # 其他设置保持默认
            traffic_manager.distance_to_leading_vehicle(vehicle, 1.0)
            traffic_manager.ignore_lights_percentage(vehicle, 30)
        print("已为所有生成的车辆设置速度为正常速度的1/3")
    except Exception as e:
        print(f"设置车辆行为参数失败: {e}")
    
    print(f"成功生成 {len(actual_vehicles)} 辆车 (请求数量: {number_of_vehicles})")
    return actual_vehicles

# Add fixed seed initialization
def set_random_seed(seed_value):
    """Set fixed random seed for reproducible results"""
    if seed_value > 0:
        random.seed(seed_value)
        np.random.seed(seed_value)
        print(f"Random seed set to {seed_value} for reproducible results")

def clean_up_all_vehicles(world):
    """Remove all vehicles from the world before spawning new ones"""
    print("Cleaning up all existing vehicles...")
    # Get all actors in the world
    actor_list = world.get_actors()
    # Filter just the vehicles
    vehicle_list = [actor for actor in actor_list if 'vehicle' in actor.type_id]
    
    if vehicle_list:
        print(f"Removing {len(vehicle_list)} existing vehicles")
        # Use batch command to efficiently destroy all vehicles
        client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in vehicle_list])
        # Allow some time for the destruction to take effect
        time.sleep(0.5)
    else:
        print("No existing vehicles to remove")
    
    return len(vehicle_list)

def initialize_world(client, args):
    """初始化模拟世界，设置环境和车辆"""
    
    # 获取世界和设置
    world = client.get_world()
    original_settings = world.get_settings()
    
    # 设置固定随机种子（如果指定）
    set_random_seed(args.seed)

    # 同步模式设置
    if args.sync:
        traffic_manager = client.get_trafficmanager(8000)
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # 如果提供了种子，设置交通管理器种子
        if args.seed > 0:
            traffic_manager.set_random_device_seed(args.seed)
            print(f"交通管理器种子设置为 {args.seed}")
    
    # 设置天气 - 使用命令行参数提供的天气预设
    weather_preset = getattr(args, 'weather', 'default')
    set_weather_preset(world, weather_preset)
    
    # 清理现有车辆
    removed_count = clean_up_all_vehicles(world)
    time.sleep(1.0)  # 等待片刻确保车辆完全移除
    
    return world, original_settings

def spawn_ego_vehicle(world, args, client=None):
    """生成自我驾驶车辆
    
    Args:
        world: CARLA世界对象
        args: 命令行参数
        client: CARLA客户端对象，用于获取交通管理器
    """
    # 创建自我车辆
    bp = world.get_blueprint_library().filter('model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    
    # 尝试不同的生成点，直到找到一个没有碰撞的位置
    vehicle = None
    if args.seed > 0:
        # 使用固定种子时，从第一个点开始尝试
        spawn_indices = list(range(len(spawn_points)))
    else:
        # 随机情况下，打乱生成点顺序
        spawn_indices = list(range(len(spawn_points)))
        random.shuffle(spawn_indices)
    
    spawn_idx = None
    for idx in spawn_indices:
        try:
            vehicle = world.spawn_actor(bp, spawn_points[idx])
            spawn_idx = idx
            print(f"在生成点索引 {idx} 成功生成自我车辆")
            break
        except RuntimeError as e:
            if "collision" in str(e).lower():
                continue  # 尝试下一个生成点
            else:
                raise  # 如果是其他错误，则抛出
    
    if vehicle is None:
        raise RuntimeError("在尝试所有可用点后，无法找到适合自我车辆的生成点")
    
    # 设置自动驾驶
    vehicle.set_autopilot(args.autopilot)
    autopilot_status = "已启用" if args.autopilot else "已禁用"
    print(f"自动驾驶{autopilot_status}")
    
    # 设置车辆速度为正常的1/3，保持与generate_radar_dataset.py一致
    if args.autopilot and client is not None:
        try:
            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.ignore_lights_percentage(vehicle, 30)  # 30%概率忽略红绿灯，保持适当移动
            traffic_manager.distance_to_leading_vehicle(vehicle, 1.0)  # 正常前车距离
            
            # 主车速度设置为原来的1/3 
            print("设置主车速度为正常速度的1/3")
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 66.7)
        except Exception as e:
            print(f"注意: 设置高级交通管理器参数失败: {e}")
    
    return vehicle, spawn_idx, spawn_points

