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

def run_simulation(args, client):
    """Run simulation"""
    display_manager = None
    vehicle = None
    vehicle_list = []
    other_vehicles = []
    timer = CustomTimer()

    try:
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
        default=1280,
        type=int,
        help='窗口宽度 (默认: 1280)')
    argparser.add_argument(
        '--height',
        metavar='H',
        default=720,
        type=int,
        help='窗口高度 (默认: 720)')
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

    args = argparser.parse_args()

    try:
        global client
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)  # 增加超时时间，用于车辆清理
        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\n用户取消。再见！')

if __name__ == '__main__':
    # Import required libraries
    main() 