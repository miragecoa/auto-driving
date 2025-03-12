# Autonomous Driving Perception System - Baseline Demo

This project demonstrates a baseline implementation of an autonomous driving perception system based on the CARLA simulator. It showcases how to use cameras, LiDAR, and radar for environmental perception, laying the foundation for further development of perception systems under night and adverse weather conditions.

## Project Goals

- Create a baseline daylight scenario for comparison with night and adverse weather conditions
- Display multiple camera views (front, rear, left, right)
- Visualize LiDAR point cloud data
- Visualize radar target data
- Provide a foundation for future object detection algorithm implementation

## Environment Requirements

- CARLA 0.9.12
- Python 3.7+
- PyGame
- NumPy
- OpenCV
- Conda environment management recommended

## Installation and Setup

1. Ensure CARLA 0.9.12 is installed and configured
2. Use Conda environment:
   ```bash
   conda activate carla-perception
   ```

## Running the Baseline Demo

1. First, start the CARLA server:
   ```bash
   # In the CARLA directory
   ./CarlaUE4.exe
   ```

2. Then run the baseline demo script:
   ```bash
   python baseline_perception.py
   ```

3. Optional parameters:
   - `--host`: CARLA server IP (default: 127.0.0.1)
   - `--port`: CARLA server port (default: 2000)
   - `--sync`: Enable synchronous mode (recommended)
   - `--width`: Window width (default: 1280)
   - `--height`: Window height (default: 720)
   - `--vehicles`: Number of additional vehicles to spawn (default: 20)
   - `--seed`: Random seed for reproducible results (default: 0, which means random behavior)

   Example:
   ```bash
   python baseline_perception.py --sync --width 1600 --height 900 --vehicles 30 --seed 42
   ```

## Interface Description

The demo interface contains 6 windows, arranged in a 2x3 grid:

- Top row: Front camera, Left camera, Right camera
- Bottom row: LiDAR point cloud, Radar targets, Rear camera

## Control

- Press `ESC` or `Q` to exit the demo
- The vehicle drives automatically in the city (using CARLA's autopilot mode)

## Testing with Other Vehicles

The demo now supports spawning additional vehicles to test sensor perception. By default, 20 vehicles will be spawned and set to autopilot mode. You can adjust this number using the `--vehicles` parameter:

- To spawn 30 vehicles: `--vehicles 30`
- To disable additional vehicles: `--vehicles 0`

These additional vehicles help test the sensors' ability to detect other objects in the environment.

## Reproducible Results

To get reproducible results across different runs, you can use the `--seed` parameter to set a fixed random seed. This ensures:

- The same vehicle models are spawned
- They appear at the same locations
- The ego vehicle (your car) starts from the same position
- The traffic manager uses the same patterns for vehicle movement

Example:
```bash
python baseline_perception.py --sync --vehicles 15 --seed 42
```

When you set a seed, the simulation will first clean up any existing vehicles from previous runs to ensure a consistent environment.

## Future Development

Building on this baseline, future extensions will include:
- Night scene simulation
- Various adverse weather conditions (rain, fog, snow, etc.)
- Object detection algorithms
- Sensor fusion algorithms 