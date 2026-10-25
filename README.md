# Multi-Sensor Active Perception, Latency Benchmarking & Dynamic Tuning Suite

### Intel RealSense D456 · LDLiDAR STL27L (2D) · Livox Mid-360 (3D)

A unified hardware benchmarking, latency profiling, and real-time interactive parameter tuning suite for robotic active sensing systems:

| Sensor | Type | Interface | ROS 2 Topic |
|---|---|---|---|
| **Intel RealSense D456** | Industrial RGB-D & NIR stereo camera | USB 3.2 | — |
| **LDLiDAR STL27L** | 360° 2D planar LiDAR | UART serial | `/scan` |
| **Livox Mid-360** | High-density 3D solid-state LiDAR | UDP Ethernet | `/livox/lidar` |

---

## Table of Contents

- [1. Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
- [2. Repository Architecture](#2-repository-architecture)
- [3. Execution Guide & Measured Parameters](#3-execution-guide--measured-parameters)
  - [Section A: Intel RealSense D456](#section-a-intel-realsense-d456)
  - [Section B: LDLiDAR STL27L (2D Planar LiDAR)](#section-b-lidar2d-stl27l-2d-planar-lidar)
  - [Section C: Livox Mid-360 (3D Solid-State LiDAR)](#section-c-livox-mid-360-3d-solid-state-lidar)
  - [Section D: Integrated Isaac Lab Multi-Sensor Demo](#section-d-integrated-isaac-lab-multi-sensor-demo)
- [4. Demo Videos](#4-demo-videos)
- [5. Benchmark Artifacts & Reports](#5-benchmark-artifacts--reports)

---

## 1. Prerequisites & Environment Setup

### System & Python Dependencies

- **Operating System:** Ubuntu 24.04 LTS / 22.04 LTS
- **ROS 2 Distribution:** ROS 2 Jazzy Jalisco / Humble Hawksbill

**Python virtual environment setup:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install pyrealsense2 opencv-python numpy
```

### Network & Ethernet Configuration (Livox Mid-360)

The Livox Mid-360 communicates via UDP broadcasts on the `192.168.1.0/24` subnet. Configure the host network interface with a static IP:

```bash
sudo ip addr flush dev <ETH_INTERFACE>
sudo ip addr add 192.168.1.50/24 dev <ETH_INTERFACE>
sudo ip link set <ETH_INTERFACE> up
```

### ROS 2 Workspace Sourcing

Ensure your active workspace and ROS 2 installation are sourced in every terminal session:

```bash
source /opt/ros/jazzy/setup.bash
source ~/livox_ros2_ws/install/setup.bash
```

### Isaac Lab (for Section D only)

`Section D` runs inside **Isaac Lab 3.0** (which wraps NVIDIA Isaac Sim). Install Isaac Lab per the [official guide](https://isaac-sim.github.io/IsaacLab/), then confirm:

```bash
python -c "import isaaclab; print(isaaclab.__version__)"
```

`isaaclab_visualizers` (Kit) and `isaaclab.sensors` (Camera, RayCaster) must be importable.

---

## 2. Repository Architecture

```text
├── benchmark_realsence.py                    # RealSense D456 latency & metadata benchmark suite
├── realsense_interactive_tuner.py            # RealSense live CLI tuner + OpenCV telemetry HUD
├── benchmark_lidar2d.py                      # LDLiDAR STL27L dual-layer lifecycle & RAM benchmark
├── lidar2d_interactive_tuner.py              # LDLiDAR 2D live interactive CLI tuner + auto RViz2
├── benchmark_livox_mid360_dual_lifecycle.py  # Livox Mid-360 3D master lifecycle & FOV benchmark
├── livox_mid360_interactive_tuner.py         # Livox Mid-360 3D interactive CLI tuner + decay buffer
├── sensor_suite_demo.py                      # Isaac Lab 3.0 wheeled robot + D456 + 2D/3D LiDAR dashboard
├── test_d456_params.py                       # RealSense D456 parameter sanity checks
├── usb_test.py                               # USB link / enumeration diagnostics
├── benchmark_report.json                     # RealSense benchmark output
├── benchmark_lidar2d_dual_report.json        # LDLiDAR STL27L benchmark output
├── benchmark_livox_mid360_master_report.json # Livox Mid-360 benchmark output
├── lidar2D_rviz/                             # RViz2 config for LDLiDAR STL27L
├── lidar3D_rviz/                             # RViz2 config for Livox Mid-360
└── videos/
    └── lidar3D-compressed.mp4                # Livox Mid-360 3D point cloud demo
```

---

## 3. Execution Guide & Measured Parameters

### Section A: Intel RealSense D456

#### 1. Comprehensive Benchmark — `benchmark_realsence.py`

Executes hardware profiling across 4 testing phases, validating hardware registers against frame metadata and exporting `benchmark_report.json`.

**Command:**

```bash
python3 benchmark_realsence.py
```

**Measured Parameters & Metrics:**

- **Host-Side USB Command Latency (T_cmd):** USB register write dispatch duration in milliseconds (5-run mean).
- **Hardware Settling Latency (T_settling):** Frame count and elapsed time (ms) required for physical optic/sensor register latching, verified via `rs.frame_metadata_value` with an Optical Delta Safety Net fallback.
- **Frequency Stress Testing:** Frame drop profiling under dynamic parameter toggling at 2, 5, 10, 15, and 30 Hz.
- **Individual Dynamic Parameters:**
  - Depth Laser Power (0–240 mW)
  - Depth Exposure (6,000–25,000 µs)
  - Depth Gain (16–64)
  - RGB Exposure (80–300 UVC units)
  - RGB Gain (16–64)
  - RGB White Balance (3,000–5,500 K)
- **Bundled Dynamic Transitions:**
  - **Bundle 1 (Lighting Adaptation):** Synchronous Depth Exposure + Depth Gain
  - **Bundle 2 (Active/Passive Switching):** Synchronous Depth Emitter + Laser Power + Exposure
  - **Bundle 3 (Synced Exposure):** Synchronous Depth Exposure + RGB Exposure
  - **Bundle 4 (RGB Tuning):** Synchronous RGB Exposure + RGB Gain + RGB White Balance
- **Just Noticeable Difference (JND):** Minimum parameter step threshold (`min_perceptible_step`) required to produce an optical photon shift.
- **Vectorized NIR Filtering Latency:** In-memory array filtering latency (ms) and pixel reduction ratios for low-intensity noise rejection, high-reflectivity targets, and bandpass surface masking.
- **Active vs. Passive Optical Contrast Ratio:** Optical NIR magnification factor comparing emitter enabled vs. disabled.
- **Photometric Luminance vs. NIR Correlation:** Ratio and distribution dynamics of visible ITU-R BT.601 luminance against infrared intensity.
- **Photonic Step Settling Time:** Transient time (ms) and frame interval required for photon flux stabilization under a 0 → 240 mW step excitation.

#### 2. Interactive Tuner — `realsense_interactive_tuner.py`

Provides live parameter tuning from an interactive terminal with a synchronized OpenCV HUD stream.

**Command:**

```bash
python3 realsense_interactive_tuner.py
```

**Configurable Controls:**

- **Hardware Registers:** Laser Power, Depth Exposure, Depth Gain, Emitter State, RGB Exposure, RGB Gain, RGB White Balance.
- **RAM Filtering Layer:** Depth Range Min/Max, NIR Intensity Cutoffs, RGB Photometric Luminance Gating.
- **Static Lifecycle Profiles:** Cold pipeline restart downtime tracking across 848×480 @ 30 FPS, 1280×720 @ 15 FPS, and 640×480 @ 30 FPS.

---

### Section B: LDLiDAR STL27L (2D Planar LiDAR)

#### 1. Dual-Layer Benchmark — `benchmark_lidar2d.py`

Profiles zero-downtime vectorized RAM array processing versus driver teardown cycles, exporting `benchmark_lidar2d_dual_report.json`.

**Command:**

```bash
python3 benchmark_lidar2d.py
```

**Measured Parameters & Metrics:**

- **Dynamic Sector & Reflectivity Filters:**
  - 90° Front Sector Crop (0° → 90°)
  - 180° Side Corridor Mask (45° → 225°)
  - Intensity thresholding for dust rejection (≥ 50.0) and retro-reflector extraction (≥ 180.0)
  - Composite filter combining 90° front crop and intensity threshold (≥ 100.0)
- **Dynamic Performance Metrics:** Algorithmic RAM filter latency (ms), beam reduction ratio (%), hardware settling boundary (100 ms at 10 Hz), and dropped scans during 2–30 Hz stress tests.
- **Static Driver Lifecycle Reconfiguration:**
  - Coordinate rotational inversion (`laser_scan_dir`: CW vs. CCW)
  - C++ driver-level angle crop daemon activation
  - Corridor geometry reconfiguration (45° → 135°)
- **Static Lifecycle Metrics:** Serial port teardown latency (T_shutdown), time to first valid scan packet (T_first_scan), total perception blackout downtime (T_total_downtime), and robot safety blind distance calculated at 0.5, 1.0, and 2.0 m/s.

#### 2. Interactive Tuner — `lidar2d_interactive_tuner.py`

Features live CLI reconfiguration, static TF broadcasting, software decay accumulation, and an automated RViz2 instance.

**Command:**

```bash
python3 lidar2d_interactive_tuner.py
```

**Configurable Controls:**

- **Dynamic Parameters:** Crop Angle Range (0° → 360°), Distance Cutoffs (0.05 → 25.0 m), Intensity Range (0 → 255), and Temporal Decay Window (`decay_time_sec`: 0.0 to 5.0 s).
- **Static Daemon Parameters:** Rotation Direction, Driver Hardware Crop, Target Frame ID, Serial Port Path, and Baudrate (921600).

**RViz2 Mouse Navigation:**

| Action | Input |
|---|---|
| Orbit / Rotate | Left-Click + Drag |
| Pan | Shift + Left-Click + Drag |
| Zoom | Scroll Wheel |

---

### Section C: Livox Mid-360 (3D Solid-State LiDAR)

#### 1. Master Lifecycle & FOV Benchmark — `benchmark_livox_mid360_dual_lifecycle.py`

Executes comprehensive evaluation of PointCloud2 processing, custom 3D geometric filters, and UDP socket rebind latencies, exporting `benchmark_livox_mid360_master_report.json`.

**Command:**

```bash
python3 benchmark_livox_mid360_dual_lifecycle.py
```

**Measured Parameters & Metrics:**

- **Dynamic FOV & Geometric Trimming:**
  - Native 360° Omnidirectional Baseline
  - 180° Forward Semi-Sphere FOV (−90° → +90°)
  - 90° Forward Quarter Sector FOV (−45° → +45°)
  - 45° Narrow Corridor FOV (−22.5° → +22.5°)
  - Vertical Pitch Crop (−5° → +25° across −7° to +52° range)
  - Spherical Range Gating (0.2 → 15.0 m)
  - Reflectivity / Intensity Thresholding (≥ 25.0)
  - Cartesian 3D Bounding Box Envelope (X ∈ [−2, 2], Y ∈ [−1, 5], Z ∈ [−0.5, 1.5] m)
- **Dynamic Composite Multi-Filter Pipelines:**
  - **High-Speed Transit:** 180° FOV + Pitch Window [−5°..25°] + Range 20 m
  - **Precision Docking:** Front Cartesian Box (2.0 × 1.2 × 1.0 m) + Intensity > 40
  - **Safety Bubble:** 360° Sphere (r ≤ 1.5 m) + Ground Plane Cut (Z ≥ −0.2 m)
- **Static Lifecycle Benchmarks (UDP Socket Rebind):**
  - Scan Pattern Switch (`pattern_mode`: Non-repetitive flower vs. Repetitive concentric)
  - Publish Frequency Switch (`publish_freq`: 10 Hz vs. 20 Hz vs. 50 Hz)
  - Mounting Extrinsics Inversion (180° Roll)
  - ROS 2 TF Frame Reconfiguration (`livox_frame` → `base_link`)
- **Multi-Parameter Static State Transitions:**
  - **State 1 → State 2 (Exploration to Transit):** Pattern Mode 0 → 1 + Frequency 10 → 20 Hz + Frame `base_link`
  - **State 2 → State 3 (Transit to Inverted Inspection):** Roll 180° + Translation Z = 500 mm + Frame `roof_lidar`
  - **State 3 → State 1 (Inverted to Full 360 Reset):** Pattern Mode 1 → 0 + Frequency 20 → 10 Hz + Extrinsics Reset
- **Static Lifecycle Metrics:** Driver teardown and UDP ports 56100–56501 release latency (T_shutdown), time to first 3D cloud lock (T_first_scan), total perception blackout downtime (T_total_downtime), and blind travel distances.

#### 2. Interactive Tuner — `livox_mid360_interactive_tuner.py`

Interactive CLI controller featuring automated RViz2 view generation, static TF broadcasting, and temporal point cloud accumulation.

**Command:**

```bash
python3 livox_mid360_interactive_tuner.py
```

**Configurable Controls:**

- **Dynamic Controls:** Azimuth Range, Pitch Range, Temporal Decay Buffer Window (`decay_time_sec`: 0.0 to 5.0 s), Distance Cutoffs, Intensity Thresholds, and 3D Bounding Box ROI.
- **Dynamic Composite Presets:** Full 360 Passthrough, High-Speed Transit, Precision Docking, and Safety Bubble.
- **Static Daemon Controls:** Pattern Mode, Publish Frequency, Frame ID, 6-DOF Extrinsic Calibration (RPY degrees and XYZ mm), and Static State Transitions.

**RViz2 Mouse Navigation:**

| Action | Input |
|---|---|
| Orbit / Rotate | Left-Click + Drag |
| Pan | Shift + Left-Click + Drag |
| Zoom | Scroll Wheel |

#### 3. Demo Video — Livox Mid-360 3D Point Cloud

`videos/lidar3D-compressed.mp4` — recorded 3D point cloud from the Livox Mid-360 running through `livox_mid360_interactive_tuner.py`, showing FOV trimming, range gating, intensity thresholding, and composite presets live in RViz2.

https://github.com/USER/REPO/raw/main/videos/lidar3D-compressed.mp4

> **Note:** Replace `USER/REPO` with your GitHub username and repository name. GitHub does not render `<video>` tags pointing to repo-relative paths, so the most reliable option is to **drag-and-drop the MP4 into a GitHub issue or PR comment**, copy the generated `https://github.com/user-attachments/assets/...` URL, and paste it here. GitHub will then render an inline playable video player.

<video src="https://github.com/USER/REPO/raw/main/videos/lidar3D-compressed.mp4" controls width="100%"></video>

---

### Section D: Integrated Isaac Lab Multi-Sensor Demo

#### `sensor_suite_demo.py`

A self-contained **Isaac Lab 3.0** simulation (powered by **NVIDIA Isaac Sim**) that places all three sensors on a **2-wheeled differential-drive robot** patrolling an enclosed 8×8 m room with six colored obstacles. The robot drives autonomously along a 12-waypoint patrol route (or is teleoperated manually), and a floating **Omni.UI dashboard** renders a live 2×2 telemetry grid: **D456 RGB · D456 Depth · 2D LiDAR polar view · 3D LiDAR perspective view**. A translucent overlay draws the **2D LiDAR rays as 2 mm-thin green curves** directly in the 3D viewport (togglable).

This script integrates directly with Isaac Sim's simulation pipeline to provide pre-hardware validation—allowing developers to verify filter pipelines, extrinsic mounting, and sensor FOV alignment before deploying code onto physical hardware.

**Prerequisites:**

- **Isaac Lab 3.0** (bundles Isaac Sim — install per the [Isaac Lab guide](https://isaac-sim.github.io/IsaacLab/))
- `isaaclab`, `isaaclab_visualizers`, `isaaclab.sensors` importable
- OpenCV (`opencv-python`) and NumPy in the active Python env

**Command:**

```bash
python3 sensor_suite_demo.py
```

The script auto-generates two temporary USD assets on first launch (`/tmp/isaaclab_d456_wheeled_robot.usda` and `/tmp/isaaclab_room_world_mesh.usda`) — no manual asset files are required.

**Scene:**

| Element | Description |
|---|---|
| **Room** | 10 × 10 m floor, four 8.20 m perimeter walls (1.5 m high) |
| **Obstacles** | 3 cylinders (Red 0.35 m Ø, Yellow 0.35 m Ø, Cyan 0.32 m Ø) + 3 boxes (Blue 0.75×0.75×0.90 m, Green 0.65×0.95×0.80 m, Purple 0.95×0.45×1.00 m) |
| **Robot** | 2-wheel differential drive · chassis 5 kg · wheels 0.8 kg each · wheel radius 0.10 m · wheelbase 0.38 m · axle height `z = 0.102 m` (wheels rest exactly on the floor, no sinking) |
| **Wheel visuals** | Dark rubber tire + yellow cross-spokes that rotate with the physics wheel-joint states |

**Sensors mounted on the robot:**

| Sensor | Configuration | Mount (chassis-local) | Dashboard View |
|---|---|---|---|
| **Intel RealSense D456** | `focal_length=1.93 mm`, horizontal aperture `3.663 mm` (87° H FOV), vertical aperture `2.140 mm` (58° V FOV), clip `0.05–12 m` | Front visor, `x≈0.21 m`, pitch −3° | RGB (top-left) + Depth colormap with center-distance crosshair (top-right) |
| **2D LiDAR** | `1` channel, `1.0°` horizontal resolution, `360°` sweep, `max_distance=12 m` | `(0.02, 0.0, 0.14)` — red turret puck | Polar scan view with range rings, min-distance readout, and FOV wedge (bottom-left) |
| **3D LiDAR** | `16` channels, `−15° … +15°` vertical FOV, `2.0°` horizontal resolution, `max_distance=12 m` | `(−0.09, 0.0, 0.23)` — gold dome | Perspective projection with Turbo colormap by range (bottom-right) |

Both LiDARs raycast against a **single combined world-space collider mesh** (`/World/RoomCollider`) that includes the floor, four walls, three cylinders, and three boxes, so all beam hits are consistent frame-to-frame.

**Robot behaviour:**

- **AUTO PATROL** — 12-waypoint loop with smooth yaw tracking; speed auto-reduces during sharp turns.
- **Hard SDF collision resolver** — never penetrates walls, cylinders, or boxes; slides tangentially around obstacles and advances the waypoint if a stuck condition is detected, so the patrol never deadlocks.
- **MANUAL TELEOP** — keyboard or UI buttons; linear velocity clamp ±1.2/1.5 m/s, angular ±2.2 rad/s.
- **Wheel joint states** are synchronized to the commanded velocity (`ω = v / R`) so wheel spokes visually spin at the correct rate.

**Keyboard Controls (active only in GUI mode):**

| Key | Action |
|---|---|
| `↑` / `I` | Increase forward speed (+0.15 m/s) |
| `↓` / `K` | Decrease forward speed (−0.15 m/s) |
| `←` / `J` | Turn left (+0.35 rad/s) |
| `→` / `L` | Turn right (−0.35 rad/s) |
| `Space` | Emergency stop (linear + angular = 0) |
| `P` | Toggle AUTO PATROL ↔ MANUAL TELEOP |
| `B` | Toggle 2D LiDAR ray overlay (2 mm green curves in 3D viewport) |

**Dashboard UI buttons & sliders:**

| Control | Function |
|---|---|
| **Mode: AUTO PATROL / MANUAL TELEOP** | Toggle autonomous waypoint following |
| **3D Beams: ON (2mm) / OFF** | Toggle the thin laser-ray overlay |
| **+Speed / −Speed / Left / Right / STOP** | Button equivalents of the keyboard teleop |
| **Preset: D456 Aligned Depth (87×58°)** | Apply depth-stream intrinsics |
| **Preset: D456 Native RGB (90×65°)** | Apply RGB-stream intrinsics |
| **D456 Focal Length (mm)** | Slider 1.00 – 4.50 mm (live USD camera update) |
| **D456 Max Depth (m)** | Slider 1.0 – 10.0 m (depth colormap normalization) |
| **2D LiDAR Range (m)** | Slider 1.5 – 10.0 m |
| **2D LiDAR FOV (deg)** | Slider 30 – 360° |
| **3D LiDAR Range (m)** | Slider 1.5 – 10.0 m |
| **3D LiDAR VFOV (deg)** | Slider ±2 – ±15° |

> **Note:** This demo is simulation-only — it does not export a benchmark JSON report. Use it to sanity-check filter behaviour and sensor alignment before running the hardware benchmark scripts in Sections A–C.

---

## 4. Demo Videos

| Video | Sensor | Description |
|---|---|---|
| [`videos/lidar3D-compressed.mp4`](videos/lidar3D-compressed.mp4) | Livox Mid-360 (3D) | Live 3D point cloud with FOV trimming, range gating, intensity thresholds, and composite presets via `livox_mid360_interactive_tuner.py` |

---

## 5. Benchmark Artifacts & Reports

Every benchmark script exports a machine-readable JSON report upon test completion.

| Benchmark Script | Generated Report File | Core Contents |
|---|---|---|
| `benchmark_realsence.py` | `benchmark_report.json` | USB latencies, hardware settling frame offsets, stress drop rates, JND thresholds, and optical contrast ratios |
| `benchmark_lidar2d.py` | `benchmark_lidar2d_dual_report.json` | RAM filter execution times, beam reduction ratios, serial port shutdown latencies, and safety blind margins |
| `benchmark_livox_mid360_dual_lifecycle.py` | `benchmark_livox_mid360_master_report.json` | 3D sector filter benchmarks, composite pipelines, UDP socket unbind times, and state transition blackouts |

> `sensor_suite_demo.py` is a simulation-only demo and does not export a benchmark report.

---

## License

Add your license here (e.g. MIT, Apache-2.0).
