#!/usr/bin/env python3
import os
import sys
import time
import json
import shutil
import signal
import threading
import subprocess
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

# Workspace & Config Paths
WS_PATH = os.path.expanduser("~/livox_ros2_ws")
CONFIG_PATH = os.path.join(WS_PATH, "src/livox_ros_driver2/config/MID360_config.json")
INSTALL_CONFIG_PATH = os.path.join(WS_PATH, "install/livox_ros_driver2/share/livox_ros_driver2/config/MID360_config.json")
RVIZ_CONFIG_PATH = "/tmp/livox3d_tuner.rviz"


def cleanup_udp_ports():
    """Kill lingering processes holding Mid-360 UDP network sockets."""
    ports = [56100, 56101, 56200, 56201, 56300, 56301, 56400, 56401, 56500, 56501]
    for p in ports:
        subprocess.run(["fuser", "-k", f"{p}/udp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "livox_ros_driver2_node"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)


def generate_rviz_config(frame_id="livox_frame"):
    """Generates an RViz2 configuration ensuring full mouse interactivity and proper views."""
    rviz_content = f"""Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Filtered PointCloud1
      Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Plane Cell Count: 20
      Cell Size: 1
    - Class: rviz_default_plugins/PointCloud2
      Name: Filtered PointCloud
      Enabled: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /livox/lidar_filtered
      Size (m): 0.04
      Style: Points
      Color Transformer: Intensity
      Invert Rainbow: false
      Autocompute Intensity Bounds: true
      Use Fixed Frame: true
      Decay Time: 1.0
  Global Options:
    Background Color: 25; 25; 25
    Fixed Frame: map
    Frame Rate: 30
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 4.5
      Focal Point:
        X: 0.0
        Y: 0.0
        Z: 0.0
      Focal Shape Size: 0.05
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.01
      Pitch: 0.45
      Target Frame: map
      Value: Orbit (rviz_default_plugins)
      Yaw: 0.78
"""
    with open(RVIZ_CONFIG_PATH, "w") as f:
        f.write(rviz_content)


def extract_pointcloud2_xyz_intensity(cloud_msg: PointCloud2):
    """Zero-copy extraction of x, y, z, intensity from PointCloud2 buffer."""
    num_points = cloud_msg.width * cloud_msg.height
    if num_points == 0 or len(cloud_msg.data) == 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    point_stride = cloud_msg.point_step
    raw_bytes = np.frombuffer(cloud_msg.data, dtype=np.uint8)
    actual_points = min(num_points, len(raw_bytes) // point_stride)
    if actual_points == 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    pts = raw_bytes[:actual_points * point_stride].reshape(actual_points, point_stride)

    x = np.frombuffer(pts[:, 0:4].tobytes(), dtype=np.float32)
    y = np.frombuffer(pts[:, 4:8].tobytes(), dtype=np.float32)
    z = np.frombuffer(pts[:, 8:12].tobytes(), dtype=np.float32)
    intensity = np.frombuffer(pts[:, 12:16].tobytes(), dtype=np.float32)
    return x, y, z, intensity


class LivoxInteractiveNode(Node):
    """ROS 2 Node handling dynamic 3D filtering, temporal accumulation, and scan republishing."""
    def __init__(self, tuner):
        super().__init__('livox_mid360_interactive_tuner_node')
        self.tuner = tuner

        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_static_tf()

        self.sub = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.cloud_callback,
            qos_profile_sensor_data
        )

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pub = self.create_publisher(PointCloud2, '/livox/lidar_filtered', pub_qos)

        self.last_in_points = 0
        self.last_out_points = 0
        self.first_cloud_event = threading.Event()
        self.pending_dynamic_metric = None

        # Sliding window buffer for real-time temporal decay accumulation
        self.frame_buffer = deque()

    def broadcast_static_tf(self):
        """Broadcasts static transform from map to target frame_id."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = self.tuner.static_params['frame_id']
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def request_dynamic_measurement(self, label, new_val):
        self.pending_dynamic_metric = {
            "label": label,
            "val": new_val,
            "t_req": time.perf_counter()
        }

    def cloud_callback(self, msg: PointCloud2):
        t_frame_start = time.perf_counter()
        self.first_cloud_event.set()

        x, y, z, i = extract_pointcloud2_xyz_intensity(msg)
        self.last_in_points = len(x)
        if len(x) == 0:
            return

        dp = self.tuner.dynamic_params
        mask = np.ones(len(x), dtype=bool)

        # 1. Azimuth Horizontal Windowing
        if dp["azimuth_min_deg"] != -180.0 or dp["azimuth_max_deg"] != 180.0:
            yaw_deg = np.rad2deg(np.arctan2(y, x))
            mask &= (yaw_deg >= dp["azimuth_min_deg"]) & (yaw_deg <= dp["azimuth_max_deg"])

        # 2. Vertical Pitch Windowing
        if dp["pitch_min_deg"] != -7.0 or dp["pitch_max_deg"] != 52.0:
            r_xy = np.sqrt(x**2 + y**2)
            pitch_deg = np.rad2deg(np.arctan2(z, r_xy))
            mask &= (pitch_deg >= dp["pitch_min_deg"]) & (pitch_deg <= dp["pitch_max_deg"])

        # 3. Spherical Range Gating
        r_spherical = np.sqrt(x**2 + y**2 + z**2)
        mask &= (r_spherical >= dp["range_min_m"]) & (r_spherical <= dp["range_max_m"])

        # 4. Intensity Filtering
        mask &= (i >= dp["min_intensity"]) & (i <= dp["max_intensity"])

        # 5. Cartesian 3D Bounding Box
        if dp["use_bbox"]:
            mask &= (x >= dp["bbox_x_min"]) & (x <= dp["bbox_x_max"]) & \
                    (y >= dp["bbox_y_min"]) & (y <= dp["bbox_y_max"]) & \
                    (z >= dp["bbox_z_min"]) & (z <= dp["bbox_z_max"])

        fx = x[mask]
        fy = y[mask]
        fz = z[mask]
        fi = i[mask]

        # -------------------------------------------------------------
        # Temporal Point Accumulation (Software-level Decay Time)
        # -------------------------------------------------------------
        now_ts = time.time()
        decay_sec = dp["decay_time_sec"]

        if decay_sec > 0.0:
            self.frame_buffer.append((now_ts, fx, fy, fz, fi))
            # Purge frames older than decay_time_sec
            while self.frame_buffer and (now_ts - self.frame_buffer[0][0] > decay_sec):
                self.frame_buffer.popleft()

            # Merge all points in sliding window
            fx = np.concatenate([f[1] for f in self.frame_buffer])
            fy = np.concatenate([f[2] for f in self.frame_buffer])
            fz = np.concatenate([f[3] for f in self.frame_buffer])
            fi = np.concatenate([f[4] for f in self.frame_buffer])
        else:
            self.frame_buffer.clear()

        self.last_out_points = len(fx)

        # Build PointCloud2 message
        filtered_msg = PointCloud2()
        filtered_msg.header = msg.header
        filtered_msg.header.frame_id = self.tuner.static_params['frame_id']
        filtered_msg.height = 1
        filtered_msg.width = len(fx)
        filtered_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        filtered_msg.is_bigendian = False
        filtered_msg.point_step = 16
        filtered_msg.row_step = 16 * len(fx)
        filtered_msg.is_dense = True

        out_pts = np.empty((len(fx), 4), dtype=np.float32)
        out_pts[:, 0] = fx
        out_pts[:, 1] = fy
        out_pts[:, 2] = fz
        out_pts[:, 3] = fi
        filtered_msg.data = out_pts.tobytes()
        self.pub.publish(filtered_msg)

        t_filter_proc_ms = (time.perf_counter() - t_frame_start) * 1000.0

        if self.pending_dynamic_metric is not None:
            t_settle_ms = (time.perf_counter() - self.pending_dynamic_metric["t_req"]) * 1000.0
            lbl = self.pending_dynamic_metric["label"]
            val = self.pending_dynamic_metric["val"]
            frame_boundary_ms = 1000.0 / float(self.tuner.static_params["publish_freq"])

            banner = (
                f"[DYNAMIC FILTER] [{lbl} -> {val}] Applied! | "
                f"RAM Processing: {t_filter_proc_ms:.4f} ms | "
                f"Settle on Frame Boundary: {t_settle_ms:.2f} ms (Target Frame: {frame_boundary_ms:.1f} ms)"
            )
            self.tuner.last_telemetry_banner = banner
            self.tuner.telemetry_history.append(banner)
            if len(self.tuner.telemetry_history) > 4:
                self.tuner.telemetry_history.pop(0)
            self.pending_dynamic_metric = None


class LivoxTunerApp:
    def __init__(self):
        cleanup_udp_ports()

        # Dynamic Parameters (Vectorized in RAM)
        self.dynamic_params = {
            "azimuth_min_deg": -180.0,
            "azimuth_max_deg": 180.0,
            "pitch_min_deg": -7.0,
            "pitch_max_deg": 52.0,
            "range_min_m": 0.15,
            "range_max_m": 40.0,
            "min_intensity": 0.0,
            "max_intensity": 255.0,
            "decay_time_sec": 1.0,       # Temporal accumulation window (0.0 to 5.0 s)
            "use_bbox": False,
            "bbox_x_min": -2.0,
            "bbox_x_max": 2.0,
            "bbox_y_min": -1.0,
            "bbox_y_max": 5.0,
            "bbox_z_min": -0.5,
            "bbox_z_max": 1.5
        }

        # Static Parameters (Driver Level)
        self.static_params = {
            "pattern_mode": 0,           # 0: Non-repetitive, 1: Repetitive
            "publish_freq": 10.0,
            "frame_id": "livox_frame",
            "extrinsic_roll": 0.0,
            "extrinsic_pitch": 0.0,
            "extrinsic_yaw": 0.0,
            "extrinsic_x": 0,
            "extrinsic_y": 0,
            "extrinsic_z": 0
        }

        self.driver_process = None
        self.rviz_process = None
        self.node = None
        self.executor_thread = None
        self.running = True

        self.last_telemetry_banner = "Livox Mid-360 Interactive Tuner initialized. Ready."
        self.telemetry_history = []

    def make_config_dict(self):
        sp = self.static_params
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
                    "ip": "192.168.1.3",
                    "pcl_data_type": 1,
                    "pattern_mode": int(sp["pattern_mode"]),
                    "extrinsic_parameter": {
                        "roll": float(sp["extrinsic_roll"]),
                        "pitch": float(sp["extrinsic_pitch"]),
                        "yaw": float(sp["extrinsic_yaw"]),
                        "x": int(sp["extrinsic_x"]),
                        "y": int(sp["extrinsic_y"]),
                        "z": int(sp["extrinsic_z"])
                    }
                }
            ]
        }

    def write_config(self):
        cfg = self.make_config_dict()
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)
        if os.path.exists(os.path.dirname(INSTALL_CONFIG_PATH)):
            try:
                shutil.copyfile(CONFIG_PATH, INSTALL_CONFIG_PATH)
            except (shutil.SameFileError, OSError):
                pass

    def start_driver(self):
        self.write_config()
        cmd = [
            "ros2", "run", "livox_ros_driver2", "livox_ros_driver2_node",
            "--ros-args",
            "-p", f"user_config_path:={CONFIG_PATH}",
            "-p", f"publish_freq:={float(self.static_params['publish_freq'])}",
            "-p", "xfer_format:=0",
            "-p", f"frame_id:={self.static_params['frame_id']}",
            "-p", "multi_topic:=0",
            "-p", "data_src:=0"
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        return proc

    def stop_driver(self):
        if self.driver_process is None:
            return 0.0
        t0 = time.perf_counter()
        try:
            os.killpg(os.getpgid(self.driver_process.pid), signal.SIGINT)
            self.driver_process.wait(timeout=3.0)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(self.driver_process.pid), signal.SIGKILL)
                self.driver_process.wait()
            except ProcessLookupError:
                pass
        self.driver_process = None
        cleanup_udp_ports()
        return (time.perf_counter() - t0) * 1000.0

    def start_rviz(self):
        generate_rviz_config(self.static_params["frame_id"])
        cmd = ["rviz2", "-d", RVIZ_CONFIG_PATH]
        self.rviz_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )

    def restart_hardware_lifecycle(self, reason_label):
        print(f"\n[*] Cold Restarting Livox Driver [{reason_label}]...")
        self.node.first_cloud_event.clear()

        t_shut = self.stop_driver()
        t0_launch = time.perf_counter()
        self.driver_process = self.start_driver()

        locked = self.node.first_cloud_event.wait(timeout=10.0)
        t_first = (time.perf_counter() - t0_launch) * 1000.0 if locked else 0.0
        t_total = t_shut + t_first

        self.node.broadcast_static_tf()

        banner = (
            f"[STATIC RECONFIG] [{reason_label}] Settled! -> "
            f"T_shutdown: {t_shut:.2f} ms | T_first_scan: {t_first:.2f} ms | "
            f"Total Downtime: {t_total:.2f} ms"
        )
        self.last_telemetry_banner = banner
        self.telemetry_history.append(banner)
        if len(self.telemetry_history) > 4:
            self.telemetry_history.pop(0)

    def print_menu(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print("=" * 96)
        print("          LIVOX MID-360 3D LiDAR - INTERACTIVE BENCHMARK & PARAMETER TUNER")
        print("=" * 96)
        print(f" [*] Topics: In [/livox/lidar] -> Out [/livox/lidar_filtered] | Frame: [{self.static_params['frame_id']}]")
        if self.node:
            in_pts = self.node.last_in_points
            out_pts = self.node.last_out_points
            print(f" [*] Stream: Raw Per-Frame: {in_pts:,} pts | Active Filtered Buffer: {out_pts:,} pts")

        # Persistent Telemetry Banner
        print("-" * 96)
        print(" \033[92m[LATEST TELEMETRY REPORT]\033[0m")
        print(f" \033[93m>>> {self.last_telemetry_banner}\033[0m")
        if len(self.telemetry_history) > 1:
            print(" [Previous Telemetry Events]:")
            for h in self.telemetry_history[-3:-1]:
                print(f"   * {h}")
        print("-" * 96)

        dp = self.dynamic_params
        sp = self.static_params

        print(" [A] DYNAMIC HORIZONTAL FOV, PITCH & TEMPORAL ACCUMULATION (RAM Layer)")
        print(f"   1. azimuth_min_deg : {dp['azimuth_min_deg']:<8} [Deg: -180.0 to 180.0]")
        print(f"   2. azimuth_max_deg : {dp['azimuth_max_deg']:<8} [Deg: -180.0 to 180.0]")
        print(f"   3. pitch_min_deg   : {dp['pitch_min_deg']:<8} [Deg: -7.0 to 52.0 - Ground Trimming]")
        print(f"   4. pitch_max_deg   : {dp['pitch_max_deg']:<8} [Deg: -7.0 to 52.0 - Ceiling Trimming]")
        print(f"   5. decay_time_sec  : {dp['decay_time_sec']:<8} [Sec: 0.0 to 5.0 - Sliding Decay Window]")
        print("-" * 96)
        print(" [B] DYNAMIC RANGE & INTENSITY / REFLECTIVITY FILTERING")
        print(f"   6. range_min_m     : {dp['range_min_m']:<8} [Meters: Min Distance Cutoff]")
        print(f"   7. range_max_m     : {dp['range_max_m']:<8} [Meters: Max Distance Cutoff]")
        print(f"   8. min_intensity   : {dp['min_intensity']:<8} [0.0 to 255.0 - Dust / Noise Reject]")
        print(f"   9. max_intensity   : {dp['max_intensity']:<8} [0.0 to 255.0 - Reflector Select]")
        print("-" * 96)
        print(" [C] DYNAMIC CARTESIAN 3D BOUNDING BOX (ROI Envelope)")
        print(f"  10. use_bbox        : {str(dp['use_bbox']):<8} [True/False - Toggle 3D ROI Box]")
        print(f"  11. bbox_x (min/max): [{dp['bbox_x_min']}, {dp['bbox_x_max']}] m")
        print(f"  12. bbox_y (min/max): [{dp['bbox_y_min']}, {dp['bbox_y_max']}] m")
        print(f"  13. bbox_z (min/max): [{dp['bbox_z_min']}, {dp['bbox_z_max']}] m")
        print("-" * 96)
        print(" [D] DYNAMIC COMPOSITE PRESETS (Instant Multi-Parameter Macros)")
        print("  14. preset_full360  : Reset to Full 360-deg Omnidirectional Passthrough")
        print("  15. preset_transit  : High-Speed Transit (180-deg Forward + Pitch [-5..25] + Range 20m)")
        print("  16. preset_docking  : Precision Docking (Tight Front 3D Box + Intensity >= 40.0)")
        print("  17. preset_safety   : Omnidirectional Safety Bubble (Radius <= 1.5m + Ground Removal)")
        print("-" * 96)
        print(" [E] STATIC HARDWARE DAEMON PARAMETERS (Driver Restart & UDP Unbind)")
        print(f"  18. pattern_mode    : {sp['pattern_mode']:<8} [0: Non-repetitive vs 1: Repetitive Scan]")
        print(f"  19. publish_freq    : {sp['publish_freq']:<8} [Hz: 10.0, 20.0, or 50.0]")
        print(f"  20. frame_id        : {sp['frame_id']:<8} [TF Transform Target Name]")
        print("-" * 96)
        print(" [F] STATIC 6-DOF EXTRINSIC CALIBRATION (Roll/Pitch/Yaw Deg, X/Y/Z mm)")
        print(f"  21. extrinsic_rpy   : [{sp['extrinsic_roll']}, {sp['extrinsic_pitch']}, {sp['extrinsic_yaw']}] deg")
        print(f"  22. extrinsic_xyz   : [{sp['extrinsic_x']}, {sp['extrinsic_y']}, {sp['extrinsic_z']}] mm")
        print("-" * 96)
        print(" [G] STATIC MULTI-PARAMETER ROBOTIC STATE TRANSITIONS")
        print("  23. trans_transit   : State 1->2 [Exploration -> Transit]: Repetitive + 20Hz + base_link")
        print("  24. trans_inverted  : State 2->3 [Transit -> Inverted]: Roll 180deg + Z 500mm + roof_lidar")
        print("  25. trans_reset     : State 3->1 [Inverted -> Exploration]: Non-repetitive + 10Hz + base_laser")
        print("=" * 96)
        print(" Controls: Left-Click = Rotate | Shift+Left-Click = Pan | Scroll = Zoom")
        print(" Commands: Enter parameter number or name | 'reset' | 'refresh' | 'exit'")
        print("=" * 96)

    def run(self):
        print("[*] Launching Livox Mid-360 C++ driver node...")
        self.driver_process = self.start_driver()

        self.node = LivoxInteractiveNode(self)
        self.executor_thread = threading.Thread(
            target=lambda: rclpy.spin(self.node),
            daemon=True
        )
        self.executor_thread.start()

        print("[*] Waiting for incoming PointCloud2 stream lock...")
        if self.node.first_cloud_event.wait(timeout=10.0):
            print("[*] PointCloud2 stream locked. Launching RViz2...")
        else:
            print("[!] Warning: PointCloud2 stream delayed, launching RViz2 anyway...")

        self.start_rviz()

        param_map = {
            "1": "azimuth_min_deg", "2": "azimuth_max_deg",
            "3": "pitch_min_deg", "4": "pitch_max_deg", "5": "decay_time_sec",
            "6": "range_min_m", "7": "range_max_m",
            "8": "min_intensity", "9": "max_intensity",
            "10": "use_bbox", "11": "bbox_x", "12": "bbox_y", "13": "bbox_z",
            "14": "preset_full360", "15": "preset_transit", "16": "preset_docking", "17": "preset_safety",
            "18": "pattern_mode", "19": "publish_freq", "20": "frame_id",
            "21": "extrinsic_rpy", "22": "extrinsic_xyz",
            "23": "trans_transit", "24": "trans_inverted", "25": "trans_reset"
        }

        while self.running:
            self.print_menu()
            try:
                user_cmd = input("\nEnter parameter to tune (or command): ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_cmd:
                continue

            cmd_lower = user_cmd.lower()
            if cmd_lower in ["exit", "quit", "q"]:
                break
            elif cmd_lower in ["refresh", "r"]:
                continue
            elif cmd_lower == "reset":
                self.dynamic_params = {
                    "azimuth_min_deg": -180.0, "azimuth_max_deg": 180.0,
                    "pitch_min_deg": -7.0, "pitch_max_deg": 52.0,
                    "decay_time_sec": 1.0,
                    "range_min_m": 0.15, "range_max_m": 40.0,
                    "min_intensity": 0.0, "max_intensity": 255.0,
                    "use_bbox": False,
                    "bbox_x_min": -2.0, "bbox_x_max": 2.0,
                    "bbox_y_min": -1.0, "bbox_y_max": 5.0,
                    "bbox_z_min": -0.5, "bbox_z_max": 1.5
                }
                self.last_telemetry_banner = "Dynamic filters reset to factory defaults."
                continue

            resolved_key = param_map.get(user_cmd, user_cmd)

            # --- [A & B] Dynamic Angular, Decay, Range & Intensity Filters ---
            if resolved_key in [
                "azimuth_min_deg", "azimuth_max_deg", "pitch_min_deg", "pitch_max_deg",
                "decay_time_sec", "range_min_m", "range_max_m", "min_intensity", "max_intensity"
            ]:
                val_str = input(f"Enter new value for [{resolved_key}] (current: {self.dynamic_params[resolved_key]}): ").strip()
                try:
                    val_flt = float(val_str)
                    self.node.request_dynamic_measurement(resolved_key, val_flt)
                    self.dynamic_params[resolved_key] = val_flt
                    time.sleep(0.15)
                except ValueError:
                    print("[!] Error: Invalid numeric value.")
                    time.sleep(0.8)

            # --- [C] 3D Bounding Box Controls ---
            elif resolved_key == "use_bbox":
                self.dynamic_params["use_bbox"] = not self.dynamic_params["use_bbox"]
                self.last_telemetry_banner = f"3D Bounding Box set to: {self.dynamic_params['use_bbox']}"
            elif resolved_key in ["bbox_x", "bbox_y", "bbox_z"]:
                axis = resolved_key.split("_")[1]
                val_min_str = input(f"Enter {axis.upper()}_MIN (m): ").strip()
                val_max_str = input(f"Enter {axis.upper()}_MAX (m): ").strip()
                try:
                    self.dynamic_params[f"bbox_{axis}_min"] = float(val_min_str)
                    self.dynamic_params[f"bbox_{axis}_max"] = float(val_max_str)
                    self.dynamic_params["use_bbox"] = True
                    self.last_telemetry_banner = f"Updated 3D ROI Box on {axis.upper()} axis: [{val_min_str} .. {val_max_str}] m"
                except ValueError:
                    print("[!] Error: Invalid coordinate value.")
                    time.sleep(0.8)

            # --- [D] Dynamic Preset Macros ---
            elif resolved_key == "preset_full360":
                self.dynamic_params.update({
                    "azimuth_min_deg": -180.0, "azimuth_max_deg": 180.0,
                    "pitch_min_deg": -7.0, "pitch_max_deg": 52.0,
                    "decay_time_sec": 1.0,
                    "range_min_m": 0.15, "range_max_m": 40.0,
                    "min_intensity": 0.0, "max_intensity": 255.0,
                    "use_bbox": False
                })
                self.last_telemetry_banner = "Preset: Full 360 Omnidirectional Loaded."
            elif resolved_key == "preset_transit":
                self.dynamic_params.update({
                    "azimuth_min_deg": -90.0, "azimuth_max_deg": 90.0,
                    "pitch_min_deg": -5.0, "pitch_max_deg": 25.0,
                    "decay_time_sec": 0.5,
                    "range_min_m": 0.3, "range_max_m": 20.0,
                    "use_bbox": False
                })
                self.last_telemetry_banner = "Preset: High-Speed Transit (180-deg Front + Trimming) Loaded."
            elif resolved_key == "preset_docking":
                self.dynamic_params.update({
                    "use_bbox": True,
                    "bbox_x_min": 0.1, "bbox_x_max": 2.0,
                    "bbox_y_min": -0.6, "bbox_y_max": 0.6,
                    "bbox_z_min": -0.5, "bbox_z_max": 0.5,
                    "decay_time_sec": 1.5,
                    "min_intensity": 40.0
                })
                self.last_telemetry_banner = "Preset: Precision Docking (3D Box + High Reflectivity) Loaded."
            elif resolved_key == "preset_safety":
                self.dynamic_params.update({
                    "range_min_m": 0.1, "range_max_m": 1.5,
                    "pitch_min_deg": -2.0, "pitch_max_deg": 52.0,
                    "decay_time_sec": 0.3,
                    "use_bbox": False
                })
                self.last_telemetry_banner = "Preset: Safety Bubble (r <= 1.5m + Ground Removal) Loaded."

            # --- [E] Static Daemon Parameters ---
            elif resolved_key == "pattern_mode":
                val_str = input("Enter Pattern Mode (0 = Non-repetitive flower, 1 = Repetitive concentric): ").strip()
                if val_str in ["0", "1"]:
                    self.static_params["pattern_mode"] = int(val_str)
                    self.restart_hardware_lifecycle(f"pattern_mode -> {val_str}")
            elif resolved_key == "publish_freq":
                val_str = input("Enter Publish Frequency (10.0, 20.0, 50.0 Hz): ").strip()
                try:
                    val_flt = float(val_str)
                    self.static_params["publish_freq"] = val_flt
                    self.restart_hardware_lifecycle(f"publish_freq -> {val_flt} Hz")
                except ValueError:
                    print("[!] Error: Frequency must be a valid float.")
                    time.sleep(0.8)
            elif resolved_key == "frame_id":
                val_str = input("Enter ROS 2 Frame ID (e.g., base_link, livox_frame): ").strip()
                if val_str:
                    self.static_params["frame_id"] = val_str
                    self.node.broadcast_static_tf()
                    self.restart_hardware_lifecycle(f"frame_id -> {val_str}")

            # --- [F] Static Extrinsic Calibration ---
            elif resolved_key == "extrinsic_rpy":
                r_str = input("Enter Roll (deg): ").strip()
                p_str = input("Enter Pitch (deg): ").strip()
                y_str = input("Enter Yaw (deg): ").strip()
                try:
                    self.static_params["extrinsic_roll"] = float(r_str)
                    self.static_params["extrinsic_pitch"] = float(p_str)
                    self.static_params["extrinsic_yaw"] = float(y_str)
                    self.restart_hardware_lifecycle(f"Extrinsics RPY -> [{r_str}, {p_str}, {y_str}] deg")
                except ValueError:
                    print("[!] Error: Invalid angle value.")
                    time.sleep(0.8)
            elif resolved_key == "extrinsic_xyz":
                x_str = input("Enter X Translation (mm): ").strip()
                y_str = input("Enter Y Translation (mm): ").strip()
                z_str = input("Enter Z Translation (mm): ").strip()
                try:
                    self.static_params["extrinsic_x"] = int(x_str)
                    self.static_params["extrinsic_y"] = int(y_str)
                    self.static_params["extrinsic_z"] = int(z_str)
                    self.restart_hardware_lifecycle(f"Extrinsics XYZ -> [{x_str}, {y_str}, {z_str}] mm")
                except ValueError:
                    print("[!] Error: Invalid translation value.")
                    time.sleep(0.8)

            # --- [G] Static State Transitions ---
            elif resolved_key == "trans_transit":
                self.static_params["pattern_mode"] = 1
                self.static_params["publish_freq"] = 20.0
                self.static_params["frame_id"] = "base_link"
                self.restart_hardware_lifecycle("State 1->2: Exploration -> Transit")
            elif resolved_key == "trans_inverted":
                self.static_params["extrinsic_roll"] = 180.0
                self.static_params["extrinsic_z"] = 500
                self.static_params["frame_id"] = "roof_lidar"
                self.restart_hardware_lifecycle("State 2->3: Transit -> Inverted Ceiling")
            elif resolved_key == "trans_reset":
                self.static_params["pattern_mode"] = 0
                self.static_params["publish_freq"] = 10.0
                self.static_params["extrinsic_roll"] = 0.0
                self.static_params["extrinsic_z"] = 0
                self.static_params["frame_id"] = "livox_frame"
                self.restart_hardware_lifecycle("State 3->1: Inverted -> Full 360 Reset")
            else:
                print(f"[!] Unknown parameter: '{user_cmd}'")
                time.sleep(0.8)

    def stop_all(self):
        self.running = False
        print("\n[*] Shutting down Livox Mid-360 driver and RViz2...")
        self.stop_driver()
        if self.rviz_process is not None:
            try:
                os.killpg(os.getpgid(self.rviz_process.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
        cleanup_udp_ports()


def main():
    rclpy.init()
    app = LivoxTunerApp()

    def signal_handler(sig, frame):
        app.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        app.run()
    finally:
        app.stop_all()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()