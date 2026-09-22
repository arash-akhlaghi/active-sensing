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
  - [Section B: LDLiDAR STL27L (2D Planar LiDAR)](#section-b-ldlidar-stl27l-2d-planar-lidar)
  - [Section C: Livox Mid-360 (3D Solid-State LiDAR)](#section-c-livox-mid-360-3d-solid-state-lidar)
- [4. Benchmark Artifacts & Reports](#4-benchmark-artifacts--reports)

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

---

## 2. Repository Architecture

```text
├── benchmark_realsence.py                    # RealSense D456 latency & metadata benchmark suite
├── realsense_interactive_tuner.py            # RealSense live CLI tuner + OpenCV telemetry HUD
├── benchmark_lidar2d.py                      # LDLiDAR STL27L dual-layer lifecycle & RAM benchmark
├── lidar2d_interactive_tuner.py              # LDLiDAR 2D live interactive CLI tuner + auto RViz2
├── benchmark_livox_mid360_dual_lifecycle.py  # Livox Mid-360 3D master lifecycle & FOV benchmark
└── livox_mid360_interactive_tuner.py         # Livox Mid-360 3D interactive CLI tuner + decay buffer
```

---

## 3. Execution Guide & Measured Parameters

### Section A: Intel RealSense D456

#### 1. Comprehensive Benchmark — `benchmark_realsense.py`

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

---

## 4. Benchmark Artifacts & Reports

Every benchmark script exports a machine-readable JSON report upon test completion.

| Benchmark Script | Generated Report File | Core Contents |
|---|---|---|
| `benchmark_realsence.py` | `benchmark_report.json` | USB latencies, hardware settling frame offsets, stress drop rates, JND thresholds, and optical contrast ratios |
| `benchmark_lidar2d.py` | `benchmark_lidar2d_dual_report.json` | RAM filter execution times, beam reduction ratios, serial port shutdown latencies, and safety blind margins |
| `benchmark_livox_mid360_dual_lifecycle.py` | `benchmark_livox_mid360_master_report.json` | 3D sector filter benchmarks, composite pipelines, UDP socket unbind times, and state transition blackouts |

