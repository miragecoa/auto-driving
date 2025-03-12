#!/usr/bin/env python

# Baseline Demo for Autonomous Driving Perception System in Daylight Conditions
# Demonstrating Camera, LiDAR and Radar data visualization

import glob
import os
import sys
import argparse
import random
import time
import numpy as np

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
                orientation_offset = -math.pi/2  # -90 degrees
            elif "Right Radar" in self.sensor_name:
                orientation_offset = math.pi/2   # 90 degrees
            elif "Rear Radar" in self.sensor_name:
                orientation_offset = math.pi     # 180 degrees
        
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

def set_weather_to_clear_noon(world):
    """Set weather to clear noon"""
    weather = world.get_weather()
    
    # Set to clear day
    weather.sun_altitude_angle = 85.0  # Sun height angle (noon)
    weather.cloudiness = 10.0  # Cloudiness (0-100)
    weather.precipitation = 0.0  # Precipitation (0-100)
    weather.precipitation_deposits = 0.0  # Ground water (0-100)
    weather.wind_intensity = 10.0  # Wind intensity (0-100)
    weather.fog_density = 0.0  # Fog density (0-100)
    weather.fog_distance = 0.0  # Fog visibility distance
    weather.wetness = 0.0  # Wetness (0-100)
    
    world.set_weather(weather)
    
    return weather

def spawn_surrounding_vehicles(world, number_of_vehicles=20, spawn_points=None):
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

def run_simulation(args, client):
    """Run simulation"""
    display_manager = None
    vehicle = None
    vehicle_list = []
    other_vehicles = []
    timer = CustomTimer()

    try:
        # Set fixed random seed if specified
        set_random_seed(args.seed)
        
        # Get world and settings
        world = client.get_world()
        original_settings = world.get_settings()

        # Setup synchronous mode
        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            
            # Set traffic manager seed if seed is provided
            if args.seed > 0:
                traffic_manager.set_random_device_seed(args.seed)
                print(f"Traffic Manager seed set to {args.seed}")

        # Set to clear day
        set_weather_to_clear_noon(world)
        
        # Clean up existing vehicles before spawning new ones
        removed_count = clean_up_all_vehicles(world)
        
        # Wait a short moment to ensure vehicles are fully removed
        time.sleep(1.0)
        
        # Create ego vehicle FIRST (before spawning other vehicles)
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
        
        for spawn_idx in spawn_indices:
            try:
                vehicle = world.spawn_actor(bp, spawn_points[spawn_idx])
                print(f"Spawned ego vehicle at spawn point index: {spawn_idx}")
                break
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    continue  # 尝试下一个生成点
                else:
                    raise  # 如果是其他错误，则抛出
        
        if vehicle is None:
            raise RuntimeError("Could not find a suitable spawn point for ego vehicle after trying all available points")
        
        vehicle_list.append(vehicle)
        
        # Set autopilot based on command line argument
        vehicle.set_autopilot(args.autopilot)
        autopilot_status = "enabled" if args.autopilot else "disabled"
        print(f"Autopilot is {autopilot_status}")
        
        # NOW spawn other vehicles in the simulation
        if args.vehicles > 0:
            # 更新可用生成点列表，移除已被使用的点
            available_spawn_points = [p for i, p in enumerate(spawn_points) if i not in [spawn_idx]]
            other_vehicles = spawn_surrounding_vehicles(world, min(args.vehicles, len(available_spawn_points)))

        # Display manager - modified to 3x3 grid to accommodate more sensors
        display_manager = DisplayManager(grid_size=[3, 4], window_size=[args.width, args.height])

        # 创建传感器时使用命令行参数设置range
        range_str = str(args.range)  # 将range转换为字符串
        print(f"Sensor range set to: {range_str} meters")

        # Create sensors - First row: Cameras
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
        print(f"Daylight baseline demo started with {len(other_vehicles)} additional vehicles. Press ESC or Q to exit.")
        
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
        default=1280,
        type=int,
        help='Window width (default: 1280)')
    argparser.add_argument(
        '--height',
        metavar='H',
        default=720,
        type=int,
        help='Window height (default: 720)')
    argparser.add_argument(
        '--vehicles',
        metavar='N',
        default=20,
        type=int,
        help='Number of vehicles to spawn (default: 20)')
    argparser.add_argument(
        '--seed',
        metavar='S',
        default=0,
        type=int,
        help='Random seed for reproducible results (default: 0, which means random behavior)')
    argparser.add_argument(
        '--autopilot',
        action='store_true',
        default=True,
        help='Enable vehicle autopilot (default: True)')
    argparser.add_argument(
        '--no-autopilot',
        dest='autopilot',
        action='store_false',
        help='Disable vehicle autopilot')
    argparser.add_argument(
        '--range',
        metavar='R',
        default=50,
        type=int,
        help='Detection range for radar and lidar sensors in meters (default: 50)')

    args = argparser.parse_args()

    try:
        global client
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)  # Increase timeout for vehicle cleanup
        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    # Import required libraries
    import math
    import cv2
    main() 