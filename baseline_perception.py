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
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

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
            lidar_bp.set_attribute('range', '100')
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
            radar_bp.set_attribute('range', '100')
            
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
        radar_img_size = (disp_size[0], disp_size[1], 3)
        radar_img = np.zeros(radar_img_size)
        
        # Draw radar data
        radar_img = np.zeros((disp_size[1], disp_size[0], 3), dtype=np.uint8)
        
        # Draw origin point in the center of the radar image
        center_x, center_y = int(disp_size[0] / 2), int(disp_size[1] / 2)
        cv2.circle(radar_img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw concentric circles
        for r in range(1, 6):
            radius = int(min(disp_size) / 12 * r)
            cv2.circle(radar_img, (center_x, center_y), radius, (50, 50, 50), 1)
        
        # Draw coordinate axes
        cv2.line(radar_img, (center_x, 0), (center_x, disp_size[1]), (50, 50, 50), 1)
        cv2.line(radar_img, (0, center_y), (disp_size[0], center_y), (50, 50, 50), 1)
        
        for detect in radar_data:
            # Calculate point position
            distance = detect.depth
            # Radar range limited to 50m, scaled to display area
            scale = min(disp_size) / (2 * float(self.sensor_options.get('range', '50')))
            
            # Calculate point coordinates (forward is positive y-axis, right is positive x-axis)
            x = center_x + int(distance * math.sin(detect.azimuth) * scale)
            y = center_y - int(distance * math.cos(detect.azimuth) * scale)
            
            if 0 <= x < disp_size[0] and 0 <= y < disp_size[1]:
                # Color based on velocity (faster is redder)
                velocity = detect.velocity
                if velocity > 0:  # Approaching objects
                    color = (0, int(255 - min(255, abs(velocity) * 10)), min(255, abs(velocity) * 25))  # Green to blue
                else:  # Receding objects
                    color = (min(255, abs(velocity) * 25), int(255 - min(255, abs(velocity) * 10)), 0)  # Green to red
                
                # Draw point
                cv2.circle(radar_img, (x, y), 3, color, -1)
        
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(radar_img)
            
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

def spawn_surrounding_vehicles(world, number_of_vehicles=20):
    """Spawn other vehicles in the simulation"""
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
    # Filter out bicycles and motorcycles
    vehicle_blueprints = [blueprint for blueprint in vehicle_blueprints if int(blueprint.get_attribute('number_of_wheels')) == 4]
    
    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if number_of_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif number_of_vehicles > number_of_spawn_points:
        number_of_vehicles = number_of_spawn_points

    # Use command batch to spawn vehicles more efficiently
    spawn_actors = []
    vehicle_list = []

    for n, transform in enumerate(spawn_points):
        if n >= number_of_vehicles:
            break
        blueprint = random.choice(vehicle_blueprints)
        # Set autopilot attribute if it exists
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        
        # Set the blueprint as not invincible
        if blueprint.has_attribute('role_name'):
            blueprint.set_attribute('role_name', 'autopilot')

        # Spawn the vehicle
        vehicle = world.spawn_actor(blueprint, transform)
        vehicle_list.append(vehicle)
        
    # Set autopilot for all vehicles
    for vehicle in vehicle_list:
        vehicle.set_autopilot(True)
        
    print(f"Spawned {len(vehicle_list)} vehicles")
    return vehicle_list

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
        
        # Spawn other vehicles in the simulation
        if args.vehicles > 0:
            other_vehicles = spawn_surrounding_vehicles(world, args.vehicles)

        # Create ego vehicle
        bp = world.get_blueprint_library().filter('model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        # Always use the same spawn point if seed is set
        spawn_point_index = 0 if args.seed > 0 else random.randint(0, len(spawn_points) - 1)
        vehicle = world.spawn_actor(bp, spawn_points[spawn_point_index])
        vehicle_list.append(vehicle)
        vehicle.set_autopilot(True)
        
        if args.seed > 0:
            print(f"Spawned ego vehicle at fixed spawn point index: {spawn_point_index}")

        # Display manager - 2x3 grid
        display_manager = DisplayManager(grid_size=[2, 3], window_size=[args.width, args.height])

        # Create sensors
        # Front camera
        SensorManager(world, display_manager, 'RGBCamera', 
                     carla.Transform(carla.Location(x=2.0, z=2.0), carla.Rotation(yaw=0)), 
                     vehicle, {}, display_pos=[0, 0], sensor_name="Front Camera")
        
        # Left camera
        SensorManager(world, display_manager, 'RGBCamera', 
                     carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=-90)), 
                     vehicle, {}, display_pos=[0, 1], sensor_name="Left Camera")
        
        # Right camera
        SensorManager(world, display_manager, 'RGBCamera', 
                     carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=90)), 
                     vehicle, {}, display_pos=[0, 2], sensor_name="Right Camera")
        
        # Forward LiDAR
        SensorManager(world, display_manager, 'LiDAR', 
                     carla.Transform(carla.Location(x=0, z=2.4)), 
                     vehicle, {'channels': '32', 'range': '50', 'points_per_second': '100000', 'rotation_frequency': '20'}, 
                     display_pos=[1, 0], sensor_name="LiDAR")
        
        # Forward Radar
        SensorManager(world, display_manager, 'Radar', 
                     carla.Transform(carla.Location(x=2.0, z=1.0), carla.Rotation(yaw=0)), 
                     vehicle, {'horizontal_fov': '60', 'vertical_fov': '10', 'range': '50'}, 
                     display_pos=[1, 1], sensor_name="Radar")
        
        # Rear camera
        SensorManager(world, display_manager, 'RGBCamera', 
                     carla.Transform(carla.Location(x=0, z=2.0), carla.Rotation(yaw=180)), 
                     vehicle, {}, display_pos=[1, 2], sensor_name="Rear Camera")

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