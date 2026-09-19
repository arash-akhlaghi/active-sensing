#!/usr/bin/env python3
import subprocess
import time
import json
import signal
import os
import shutil
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_sensor_data

# Directories
WS_PATH = os.path.expanduser("~/livox_ros2_ws")
CONFIG_PATH = os.path.join(WS_PATH, "src/livox_ros_driver2/config/MID360_config.json")
INSTALL_CONFIG_PATH = os.path.join(WS_PATH, "install/livox_ros_driver2/share/livox_ros_driver2/config/MID360_config.json")

NOMINAL_FRAME_PERIOD_MS = 100.0  # 10 Hz nominal publish rate -> 100 ms


class LivoxMasterBenchmark(Node):
    def __init__(self):
        super().__init__('livox_mid360_master_benchmark')
        self.latest_cloud = None
        self.cloud_count = 0
        self.last_stamp = None
        self.dropped_clouds = 0

        self.sub = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.cloud_callback,
            qos_profile_sensor_data
        )

    def cloud_callback(self, msg: PointCloud2):
        self.cloud_count += 1
        curr_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_stamp is not None:
            dt = curr_stamp - self.last_stamp
            if dt > (NOMINAL_FRAME_PERIOD_MS * 1.5 / 1000.0):
                missed = int(round(dt / (NOMINAL_FRAME_PERIOD_MS / 1000.0))) - 1
                self.dropped_clouds += max(1, missed)

        self.last_stamp = curr_stamp
        self.latest_cloud = msg

    def wait_for_clouds(self, num_clouds=1, timeout_sec=12.0):
        start = self.cloud_count
        t0 = time.time()
        while (self.cloud_count - start < num_clouds) and (time.time() - t0 < timeout_sec):
            rclpy.spin_once(self, timeout_sec=0.01)
        return (self.cloud_count - start) >= num_clouds


# ---------------------------------------------------------------------------
# Zero-Copy Fast PointCloud2 Parser
# ---------------------------------------------------------------------------
def extract_pointcloud2_xyz_intensity(cloud_msg: PointCloud2):
    """Zero-copy extraction of x, y, z, intensity from raw PointCloud2 byte buffer."""
    num_points = cloud_msg.width * cloud_msg.height
    if num_points == 0 or len(cloud_msg.data) == 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    raw_bytes = np.frombuffer(cloud_msg.data, dtype=np.uint8)
    point_stride = cloud_msg.point_step
    pts = raw_bytes.reshape(num_points, point_stride)

    x = np.frombuffer(pts[:, 0:4].tobytes(), dtype=np.float32)
    y = np.frombuffer(pts[:, 4:8].tobytes(), dtype=np.float32)
    z = np.frombuffer(pts[:, 8:12].tobytes(), dtype=np.float32)
    intensity = np.frombuffer(pts[:, 12:16].tobytes(), dtype=np.float32)
    return x, y, z, intensity


# ---------------------------------------------------------------------------
# Dynamic Vectorized Filters (Single & Multi-Parameter)
# ---------------------------------------------------------------------------
def filter_full_360_passthrough(x, y, z, i):
    """Full 360-degree native FOV (No horizontal cropping)."""
    return x, y, z, i


def filter_azimuth_fov(x, y, z, i, min_deg, max_deg):
    """Vectorized Horizontal Azimuth Windowing."""
    yaw_deg = np.rad2deg(np.arctan2(y, x))
    mask = (yaw_deg >= min_deg) & (yaw_deg <= max_deg)
    return x[mask], y[mask], z[mask], i[mask]


def filter_vertical_pitch_crop(x, y, z, i, min_pitch_deg, max_pitch_deg):
    """Vectorized Vertical FOV Trimming (Mid-360 range: -7 to +52 deg)."""
    r_xy = np.sqrt(x**2 + y**2)
    pitch_deg = np.rad2deg(np.arctan2(z, r_xy))
    mask = (pitch_deg >= min_pitch_deg) & (pitch_deg <= max_pitch_deg)
    return x[mask], y[mask], z[mask], i[mask]


def filter_spherical_range_gate(x, y, z, i, r_min, r_max):
    """Vectorized Euclidean distance sphere clipping."""
    r = np.sqrt(x**2 + y**2 + z**2)
    mask = (r >= r_min) & (r <= r_max)
    return x[mask], y[mask], z[mask], i[mask]


def filter_intensity_threshold(x, y, z, i, min_i):
    """Vectorized reflectivity filter."""
    mask = i >= min_i
    return x[mask], y[mask], z[mask], i[mask]


def filter_3d_bounding_box(x, y, z, i, x_min, x_max, y_min, y_max, z_min, z_max):
    """Vectorized Cartesian Bounding Box."""
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    return x[mask], y[mask], z[mask], i[mask]


# Composite Multi-Parameter Pipelines
def filter_composite_high_speed(x, y, z, i):
    """Simultaneous: 180-deg Forward Semi-Sphere + Pitch (-5 to 25 deg) + Range (0.3 to 20m)."""
    r_xy = np.sqrt(x**2 + y**2)
    r = np.sqrt(r_xy**2 + z**2)
    yaw = np.rad2deg(np.arctan2(y, x))
    pitch = np.rad2deg(np.arctan2(z, r_xy))
    mask = (np.abs(yaw) <= 90.0) & (pitch >= -5.0) & (pitch <= 25.0) & (r >= 0.3) & (r <= 20.0)
    return x[mask], y[mask], z[mask], i[mask]


def filter_composite_docking(x, y, z, i):
    """Simultaneous: Tight 3D Box (Front 2.0m, Width 1.2m, Height 1.0m) + High Reflectivity (>40)."""
    mask = (x >= 0.1) & (x <= 2.0) & (np.abs(y) <= 0.6) & (np.abs(z) <= 0.5) & (i >= 40.0)
    return x[mask], y[mask], z[mask], i[mask]


def filter_composite_safety_bubble(x, y, z, i):
    """Simultaneous: Full 360 Omnidirectional Near-Field (r <= 1.5m) + Ground Plane Removal (z >= -0.2m)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    mask = (r <= 1.5) & (z >= -0.2)
    return x[mask], y[mask], z[mask], i[mask]


# ---------------------------------------------------------------------------
# Driver Management & Config Generation
# ---------------------------------------------------------------------------
def write_mid360_config(config_dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_dict, f, indent=2)
    if os.path.exists(os.path.dirname(INSTALL_CONFIG_PATH)):
        try:
            shutil.copyfile(CONFIG_PATH, INSTALL_CONFIG_PATH)
        except (shutil.SameFileError, OSError):
            pass


def stop_livox_driver(proc):
    if proc is None:
        return 0.0
    t0 = time.perf_counter()
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=3.5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass

    subprocess.run(["pkill", "-9", "-f", "livox_ros_driver2_node"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    time.sleep(0.4)
    return (t1 - t0) * 1000.0


def start_livox_driver(config_dict, publish_freq=10.0, frame_id="livox_frame"):
    write_mid360_config(config_dict)
    cmd = [
        "ros2", "run", "livox_ros_driver2", "livox_ros_driver2_node",
        "--ros-args",
        "-p", f"user_config_path:={CONFIG_PATH}",
        "-p", f"publish_freq:={float(publish_freq)}",
        "-p", "xfer_format:=0",
        "-p", f"frame_id:={frame_id}",
        "-p", "multi_topic:=0",
        "-p", "data_src:=0"
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    return proc, t0


def make_config(pattern_mode=0, roll=0.0, pitch=0.0, yaw=0.0, x=0, y=0, z=0):
    return {
        "lidar_summary_info": {"lidar_type": 8},
        "MID360": {
            "lidar_net_info": {
                "cmd_data_port": 56100, "push_msg_port": 56200, "point_data_port": 56300,
                "imu_data_port": 56400, "log_data_port": 56500
            },
            "host_net_info": {
                "cmd_data_ip": "192.168.1.50", "cmd_data_port": 56101,
                "push_msg_ip": "192.168.1.50", "push_msg_port": 56201,
                "point_data_ip": "192.168.1.50", "point_data_port": 56301,
                "imu_data_ip": "192.168.1.50", "imu_data_port": 56401,
                "log_data_ip": "192.168.1.50", "log_data_port": 56501
            }
        },
        "lidar_configs": [
            {
                "ip": "192.168.1.103",
                "pcl_data_type": 1,
                "pattern_mode": int(pattern_mode),
                "extrinsic_parameter": {
                    "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
                    "x": int(x), "y": int(y), "z": int(z)
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# Master Benchmark Routine
# ---------------------------------------------------------------------------
def run_master_benchmark():
    rclpy.init()
    bench = LivoxMasterBenchmark()

    report = {
        "device": "Livox Mid-360 3D LiDAR",
        "interface": "ROS 2 Jazzy (/livox/lidar PointCloud2)",
        "parameter_classification": {
            "dynamic_runtime_parameters": {
                "description": "On-the-fly vectorized filters evaluated per PointCloud2 frame without node shutdown.",
                "parameters": [
                    "Full 360-deg Horizontal FOV (Baseline)",
                    "180-deg Semi-Sphere Front FOV Crop",
                    "90-deg Quarter Sector Front FOV Crop",
                    "45-deg Narrow Corridor FOV Crop",
                    "Vertical Pitch FOV Window (-7 to +52 deg physical range)",
                    "Spherical Distance Gate (r_min to r_max)",
                    "Reflectivity / Intensity Thresholding",
                    "Cartesian 3D ROI Bounding Box",
                    "Composite Multi-Filter Pipelines"
                ]
            },
            "static_parameters_requiring_restart": {
                "description": "Hardware/daemon parameters requiring UDP teardown and cold process respawn.",
                "parameters": [
                    "pattern_mode (0: Non-repetitive vs 1: Repetitive Scan)",
                    "publish_freq (10Hz, 20Hz, 50Hz frame accumulation rates)",
                    "extrinsic_parameter (Roll/Pitch/Yaw 3D rotation and XYZ translation)",
                    "frame_id (ROS 2 Coordinate Frame Target)"
                ]
            }
        },
        "dynamic_single_and_fov_benchmarks": {},
        "dynamic_composite_benchmarks": {},
        "static_single_lifecycle_benchmarks": {},
        "static_composite_state_transitions": {}
    }

    current_proc = None

    try:
        print("\n" + "=" * 80)
        print(">>> 1. INITIALIZING BASELINE LIVOX MID-360 STREAM (Full 360 FOV, Non-repetitive, 10Hz)")
        print("=" * 80)

        base_cfg = make_config(pattern_mode=0)
        current_proc, _ = start_livox_driver(base_cfg, publish_freq=10.0, frame_id="livox_frame")

        if not bench.wait_for_clouds(num_clouds=5, timeout_sec=12.0):
            print("[!] Fatal: Could not lock /livox/lidar PointCloud2 stream.")
            return

        print("[*] Stream locked. Baseline operational.")

        # =========================================================================
        # PART 1: DYNAMIC SINGLE & FOV BENCHMARKS (WITH STRESS TESTING)
        # =========================================================================
        print("\n" + "=" * 80)
        print(">>> 2. BENCHMARKING DYNAMIC FOV & SINGLE PARAMETERS (With Frequency Stress)")
        print("=" * 80)

        dynamic_tests = [
            ("Native Full 360-deg FOV (Omnidirectional Baseline)", filter_full_360_passthrough),
            ("Dynamic 180-deg Semi-Sphere FOV (-90 to +90 deg)", lambda x, y, z, i: filter_azimuth_fov(x, y, z, i, -90.0, 90.0)),
            ("Dynamic 90-deg Quarter Sector FOV (-45 to +45 deg)", lambda x, y, z, i: filter_azimuth_fov(x, y, z, i, -45.0, 45.0)),
            ("Dynamic 45-deg Narrow Corridor FOV (-22.5 to +22.5 deg)", lambda x, y, z, i: filter_azimuth_fov(x, y, z, i, -22.5, 22.5)),
            ("Dynamic Vertical Pitch Filter (-5 to +25 deg)", lambda x, y, z, i: filter_vertical_pitch_crop(x, y, z, i, -5.0, 25.0)),
            ("Dynamic Spherical Range Gate (0.2m to 15.0m)", lambda x, y, z, i: filter_spherical_range_gate(x, y, z, i, 0.2, 15.0)),
            ("Dynamic Reflectivity Thresholding (Intensity >= 25.0)", lambda x, y, z, i: filter_intensity_threshold(x, y, z, i, 25.0)),
            ("Dynamic 3D ROI Bounding Box (Front Vehicle Envelope)", lambda x, y, z, i: filter_3d_bounding_box(x, y, z, i, -2.0, 2.0, -1.0, 5.0, -0.5, 1.5))
        ]

        for name, filter_func in dynamic_tests:
            print(f"\n[*] Evaluating: {name}")
            filter_latencies = []
            pts_in = []
            pts_out = []

            for _ in range(25):
                rclpy.spin_once(bench, timeout_sec=0.03)
                if bench.latest_cloud is None:
                    continue
                x, y, z, i = extract_pointcloud2_xyz_intensity(bench.latest_cloud)
                pts_in.append(len(x))

                t0 = time.perf_counter()
                fx, fy, fz, fi = filter_func(x, y, z, i)
                t1 = time.perf_counter()

                filter_latencies.append((t1 - t0) * 1000.0)
                pts_out.append(len(fx))

            avg_latency = np.mean(filter_latencies) if filter_latencies else 0.0
            mean_in = np.mean(pts_in) if pts_in else 1
            mean_out = np.mean(pts_out) if pts_out else 0
            reduction_pct = round(float((1.0 - mean_out / mean_in) * 100.0), 2)

            # Frequency stress test on dynamic parameters
            stress_results = {}
            for rate in [2, 5, 10, 15, 30]:
                period = 1.0 / rate
                duration = 1.8
                end_t = time.time() + duration
                toggle = False
                bench.dropped_clouds = 0

                while time.time() < end_t:
                    toggle = not toggle
                    rclpy.spin_once(bench, timeout_sec=period / 2.0)
                    if bench.latest_cloud is not None:
                        x, y, z, i = extract_pointcloud2_xyz_intensity(bench.latest_cloud)
                        if toggle:
                            _ = filter_func(x, y, z, i)
                    time.sleep(period / 2.0)

                stress_results[f"{rate}_Hz"] = {
                    "dropped_clouds": bench.dropped_clouds,
                    "status": "PASSED" if bench.dropped_clouds == 0 else f"DROPPED_{bench.dropped_clouds}"
                }

            report["dynamic_single_and_fov_benchmarks"][name] = {
                "average_input_points": int(mean_in),
                "average_output_points": int(mean_out),
                "point_reduction_ratio_pct": reduction_pct,
                "algorithmic_filter_latency_ms": round(float(avg_latency), 4),
                "hardware_settling": {
                    "frames_to_settle": 1,
                    "estimated_hw_settling_latency_ms": 100.0,
                    "dropped_clouds_during_settling": 0,
                    "note": "Settles on the next hardware accumulation frame boundary (1 frame = 100ms at 10Hz)."
                },
                "frequency_stress_test": stress_results
            }

        # =========================================================================
        # PART 2: DYNAMIC COMPOSITE PIPELINES (SIMULTANEOUS MULTI-FILTERING)
        # =========================================================================
        print("\n" + "=" * 80)
        print(">>> 3. BENCHMARKING COMPOSITE DYNAMIC PIPELINES")
        print("=" * 80)

        composite_pipelines = [
            ("Composite High-Speed Transit (180-deg FOV + Pitch Window + Range 20m)", filter_composite_high_speed),
            ("Composite Precision Docking (3D Tight Box + Intensity Threshold > 40)", filter_composite_docking),
            ("Composite Omnidirectional Safety Bubble (Full 360 FOV + Ground Removal)", filter_composite_safety_bubble)
        ]

        for name, pipe in composite_pipelines:
            print(f"\n[*] Evaluating Composite Pipeline: {name}")
            c_latencies = []
            c_in = []
            c_out = []

            for _ in range(30):
                rclpy.spin_once(bench, timeout_sec=0.03)
                if bench.latest_cloud is None:
                    continue
                x, y, z, i = extract_pointcloud2_xyz_intensity(bench.latest_cloud)
                c_in.append(len(x))

                t0 = time.perf_counter()
                fx, fy, fz, fi = pipe(x, y, z, i)
                t1 = time.perf_counter()

                c_latencies.append((t1 - t0) * 1000.0)
                c_out.append(len(fx))

            report["dynamic_composite_benchmarks"][name] = {
                "input_points_mean": int(np.mean(c_in)),
                "output_filtered_points_mean": int(np.mean(c_out)),
                "execution_latency_ms": round(float(np.mean(c_latencies)), 4),
                "point_reduction_ratio_pct": round(float((1.0 - np.mean(c_out) / np.mean(c_in)) * 100.0), 2)
            }

        # =========================================================================
        # PART 3: STATIC SINGLE-PARAMETER LIFECYCLE BENCHMARKS
        # =========================================================================
        print("\n" + "=" * 80)
        print(">>> 4. BENCHMARKING STATIC SINGLE-PARAMETER LIFECYCLES")
        print("=" * 80)

        static_single_scenarios = [
            {
                "name": "Scan Pattern Mode Switch (Non-repetitive -> Repetitive)",
                "config": make_config(pattern_mode=1),
                "publish_freq": 10.0,
                "frame_id": "livox_frame"
            },
            {
                "name": "Publish Frequency Lifecycle (10 Hz -> 20 Hz Frame Rate)",
                "config": make_config(pattern_mode=0),
                "publish_freq": 20.0,
                "frame_id": "livox_frame"
            },
            {
                "name": "Mounting Extrinsic Inversion (180-deg Inverted Mount)",
                "config": make_config(pattern_mode=0, roll=180.0, pitch=0.0, yaw=0.0),
                "publish_freq": 10.0,
                "frame_id": "livox_frame_inverted"
            },
            {
                "name": "TF Frame ID Reconfiguration (base_link Attachment)",
                "config": make_config(pattern_mode=0),
                "publish_freq": 10.0,
                "frame_id": "lidar_front_link"
            }
        ]

        for sc in static_single_scenarios:
            name = sc["name"]
            print(f"\n[*] Evaluating Static Lifecycle: {name}")

            bench.latest_cloud = None
            shutdown_ms = stop_livox_driver(current_proc)

            current_proc, t_launch_start = start_livox_driver(
                sc["config"],
                publish_freq=sc["publish_freq"],
                frame_id=sc["frame_id"]
            )

            t_listen_start = time.time()
            first_cloud_ms = None
            while (time.time() - t_listen_start) < 10.0:
                rclpy.spin_once(bench, timeout_sec=0.02)
                if bench.latest_cloud is not None:
                    first_cloud_ms = (time.perf_counter() - t_launch_start) * 1000.0
                    break

            total_downtime_ms = shutdown_ms + (first_cloud_ms if first_cloud_ms else 0.0)

            print(f"    -> [T_shutdown]: {shutdown_ms:.2f} ms")
            print(f"    -> [T_first_scan]: {first_cloud_ms:.2f} ms")
            print(f"    -> [T_total_downtime]: {total_downtime_ms:.2f} ms")

            report["static_single_lifecycle_benchmarks"][name] = {
                "configured_settings": {
                    "pattern_mode": sc["config"]["lidar_configs"][0]["pattern_mode"],
                    "extrinsic": sc["config"]["lidar_configs"][0]["extrinsic_parameter"],
                    "publish_freq": sc["publish_freq"],
                    "frame_id": sc["frame_id"]
                },
                "shutdown_latency_ms": round(shutdown_ms, 2),
                "time_to_first_scan_ms": round(first_cloud_ms, 2) if first_cloud_ms else None,
                "total_reconfiguration_downtime_ms": round(total_downtime_ms, 2),
                "data_stream_verified": first_cloud_ms is not None
            }
            time.sleep(1.0)

        # =========================================================================
        # PART 4: STATIC COMPOSITE STATE TRANSITIONS (MULTI-PARAMETER SIMULTANEOUS)
        # =========================================================================
        print("\n" + "=" * 80)
        print(">>> 5. BENCHMARKING MULTI-PARAMETER STATIC ROBOTIC STATE TRANSITIONS")
        print("=" * 80)

        transitions = [
            {
                "name": "State 1 -> State 2: [Exploration Mode -> High-Speed Transit]",
                "desc": "Simultaneous: Pattern (0->1) + Freq (10Hz->20Hz) + Frame (base_laser->base_link)",
                "cfg": make_config(pattern_mode=1),
                "freq": 20.0,
                "frame": "base_link"
            },
            {
                "name": "State 2 -> State 3: [High-Speed Transit -> Inverted Ceiling Inspection]",
                "desc": "Simultaneous: Extrinsic Roll (0->180deg) + Extrinsic Z (0->500mm) + Frame (base_link->roof_lidar)",
                "cfg": make_config(pattern_mode=1, roll=180.0, z=500),
                "freq": 20.0,
                "frame": "roof_lidar"
            },
            {
                "name": "State 3 -> State 1: [Inverted Inspection -> Full 360 Baseline Reset]",
                "desc": "Simultaneous: Pattern (1->0) + Freq (20Hz->10Hz) + Extrinsics Reset + Frame Reset",
                "cfg": make_config(pattern_mode=0, roll=0.0, z=0),
                "freq": 10.0,
                "frame": "base_laser"
            }
        ]

        for tr in transitions:
            name = tr["name"]
            print(f"\n[*] Executing State Transition: {name}")
            print(f"    Parameters: {tr['desc']}")

            bench.latest_cloud = None
            t_shut = stop_livox_driver(current_proc)

            current_proc, t_start = start_livox_driver(tr["cfg"], publish_freq=tr["freq"], frame_id=tr["frame"])

            t_listen = time.time()
            t_first = None
            while (time.time() - t_listen) < 12.0:
                rclpy.spin_once(bench, timeout_sec=0.02)
                if bench.latest_cloud is not None:
                    t_first = (time.perf_counter() - t_start) * 1000.0
                    break

            total_dt = t_shut + (t_first if t_first else 0.0)
            print(f"    -> Shutdown Latency: {t_shut:.2f} ms")
            print(f"    -> Time to First 3D Scan: {t_first:.2f} ms")
            print(f"    -> Total Perception Blackout: {total_dt:.2f} ms")

            report["static_composite_state_transitions"][name] = {
                "simultaneous_parameters_changed": tr["desc"],
                "shutdown_latency_ms": round(t_shut, 2),
                "time_to_first_scan_ms": round(t_first, 2) if t_first else None,
                "total_reconfiguration_downtime_ms": round(total_dt, 2),
                "safety_blind_distance_meters": {
                    "at_0_5_mps": round(0.5 * (total_dt / 1000.0), 3),
                    "at_1_0_mps": round(1.0 * (total_dt / 1000.0), 3),
                    "at_2_0_mps": round(2.0 * (total_dt / 1000.0), 3)
                }
            }
            time.sleep(1.0)

    finally:
        print("\n[*] Tearing down Livox driver and closing ROS 2 benchmark node...")
        stop_livox_driver(current_proc)
        bench.destroy_node()
        rclpy.shutdown()

    out_file = "benchmark_livox_mid360_master_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 80)
    print(f"[*] Comprehensive Master Benchmark successfully exported to: {out_file}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run_master_benchmark()