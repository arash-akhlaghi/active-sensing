#!/usr/bin/env python3
import os
import sys
import time
import signal
import threading
import subprocess
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

RVIZ_CONFIG_PATH = "/tmp/lidar2d_tuner.rviz"


def generate_rviz_config():
    """Generates an optimized RViz2 configuration file matching QoS and Fixed Frame."""
    rviz_content = """Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /Filtered LaserScan1
      Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Name: Grid
      Enabled: true
      Plane Cell Count: 20
      Cell Size: 1
    - Class: rviz_default_plugins/LaserScan
      Name: Filtered LaserScan
      Enabled: true
      Topic:
        Value: /scan_filtered
        Depth: 10
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
      Size (m): 0.04
      Style: Points
      Color Transformer: Intensity
      Autocompute Intensity Bounds: true
      Use Fixed Frame: true
  Global Options:
    Background Color: 25; 25; 25
    Fixed Frame: base_laser
    Frame Rate: 30
"""
    with open(RVIZ_CONFIG_PATH, "w") as f:
        f.write(rviz_content)


class LidarInteractiveNode(Node):
    """ROS 2 Node handling real-time dynamic filtering, scan republishing, and settling telemetry."""
    def __init__(self, tuner):
        super().__init__('lidar2d_interactive_tuner_node')
        self.tuner = tuner

        # Subscriber to raw LiDAR stream with SensorData QoS
        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # Publisher for filtered visualization stream with RELIABLE QoS (Guarantees RViz compatibility)
        scan_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pub = self.create_publisher(LaserScan, '/scan_filtered', scan_pub_qos)

        self.last_received_beams = 0
        self.last_filtered_beams = 0
        self.first_scan_event = threading.Event()

        # Dynamic parameter settling latency tracking
        self.pending_dynamic_metric = None
        self.last_telemetry_msg = ""

    def request_dynamic_measurement(self, param_name, new_val):
        """Arm the telemetry tracker to capture settling time on the next frame arrival."""
        self.pending_dynamic_metric = {
            "name": param_name,
            "val": new_val,
            "t_req": time.perf_counter()
        }

    def scan_callback(self, msg: LaserScan):
        t_frame_start = time.perf_counter()
        self.first_scan_event.set()

        filtered_msg = LaserScan()
        filtered_msg.header = msg.header
        filtered_msg.angle_min = msg.angle_min
        filtered_msg.angle_max = msg.angle_max
        filtered_msg.angle_increment = msg.angle_increment
        filtered_msg.time_increment = msg.time_increment
        filtered_msg.scan_time = msg.scan_time
        filtered_msg.range_min = msg.range_min
        filtered_msg.range_max = msg.range_max

        ranges = np.array(msg.ranges, dtype=np.float32)
        num_points = len(ranges)
        self.last_received_beams = num_points

        # Calculate exact per-beam angles in degrees
        angles_deg = np.rad2deg(
            msg.angle_min + np.arange(num_points) * msg.angle_increment
        ) % 360.0

        # 1. Dynamic Angular Sector Window Filter
        c_min = self.tuner.dynamic_params["crop_min_angle"]
        c_max = self.tuner.dynamic_params["crop_max_angle"]
        if c_min != c_max:
            if c_min < c_max:
                angle_mask = (angles_deg >= c_min) & (angles_deg <= c_max)
            else:
                angle_mask = (angles_deg >= c_min) | (angles_deg <= c_max)
            ranges[angle_mask] = np.nan

        # 2. Dynamic Radial Distance Gate Filter
        r_min = self.tuner.dynamic_params["min_range"]
        r_max = self.tuner.dynamic_params["max_range"]
        range_mask = (ranges < r_min) | (ranges > r_max)
        ranges[range_mask] = np.nan

        # 3. Dynamic Intensity / Reflectivity Threshold Filter
        if len(msg.intensities) > 0:
            intensities = np.array(msg.intensities, dtype=np.float32)
            i_min = self.tuner.dynamic_params["min_intensity"]
            i_max = self.tuner.dynamic_params["max_intensity"]
            intensity_mask = (intensities < i_min) | (intensities > i_max)
            ranges[intensity_mask] = np.nan
            filtered_msg.intensities = msg.intensities
        else:
            filtered_msg.intensities = []

        self.last_filtered_beams = int(np.count_nonzero(~np.isnan(ranges)))
        filtered_msg.ranges = ranges.tolist()
        self.pub.publish(filtered_msg)

        t_filter_proc_ms = (time.perf_counter() - t_frame_start) * 1000.0

        # Check and record dynamic parameter settling latency
        if self.pending_dynamic_metric is not None:
            t_settle_ms = (time.perf_counter() - self.pending_dynamic_metric["t_req"]) * 1000.0
            p_name = self.pending_dynamic_metric["name"]
            p_val = self.pending_dynamic_metric["val"]
            self.last_telemetry_msg = (
                f"[TELEMETRY] Param [{p_name} -> {p_val}] SETTLED in: {t_settle_ms:.2f} ms "
                f"(Algorithmic Filter Time: {t_filter_proc_ms:.4f} ms)"
            )
            self.pending_dynamic_metric = None


class LidarTunerApp:
    """CLI orchestrator managing driver lifecycle, RViz2, and interactive inputs."""
    def __init__(self):
        # Dynamic Parameters (Vectorized RAM processing, zero stream downtime)
        self.dynamic_params = {
            "crop_min_angle": 0.0,
            "crop_max_angle": 0.0,
            "min_intensity": 0.0,
            "max_intensity": 255.0,
            "min_range": 0.05,
            "max_range": 25.0
        }

        # Static Parameters (Driver level, requires background process restart)
        self.static_params = {
            "laser_scan_dir": "true",
            "enable_angle_crop_func": "false",
            "angle_crop_min": 0.0,
            "angle_crop_max": 0.0,
            "frame_id": "base_laser",
            "port_name": "/dev/ttyUSB0",
            "port_baudrate": 921600
        }

        self.driver_process = None
        self.rviz_process = None
        self.node = None
        self.executor_thread = None
        self.running = True

    def start_driver(self):
        """Starts or restarts the C++ LDLiDAR vendor driver in the background."""
        if self.driver_process is not None:
            self.stop_driver()

        cmd = [
            "ros2", "run", "ldlidar_stl_ros2", "ldlidar_stl_ros2_node",
            "--ros-args",
            "-r", "__node:=STL27L",
            "-p", "product_name:=LDLiDAR_STL27L",
            "-p", "topic_name:=scan",
            "-p", f"frame_id:={self.static_params['frame_id']}",
            "-p", f"port_name:={self.static_params['port_name']}",
            "-p", f"port_baudrate:={self.static_params['port_baudrate']}",
            "-p", f"laser_scan_dir:={self.static_params['laser_scan_dir']}",
            "-p", f"enable_angle_crop_func:={self.static_params['enable_angle_crop_func']}",
            "-p", f"angle_crop_min:={self.static_params['angle_crop_min']}",
            "-p", f"angle_crop_max:={self.static_params['angle_crop_max']}"
        ]

        self.driver_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )

    def stop_driver(self):
        """Terminates driver cleanly and clears serial locks."""
        if self.driver_process is None:
            return 0.0
        t0 = time.perf_counter()
        try:
            os.killpg(os.getpgid(self.driver_process.pid), signal.SIGINT)
            self.driver_process.wait(timeout=2.0)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(os.getpgid(self.driver_process.pid), signal.SIGKILL)
                self.driver_process.wait()
            except ProcessLookupError:
                pass
        self.driver_process = None
        subprocess.run(["pkill", "-9", "-f", "ldlidar_stl_ros2_node"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.3)
        return (time.perf_counter() - t0) * 1000.0

    def start_rviz(self):
        """Spawns RViz2 with pre-configured configuration file."""
        generate_rviz_config()
        cmd = ["rviz2", "-d", RVIZ_CONFIG_PATH]
        self.rviz_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )

    def stop_all(self):
        """Shuts down all child processes and ROS 2 communication."""
        self.running = False
        print("\n[*] Shutting down LiDAR driver and RViz2...")
        self.stop_driver()
        if self.rviz_process is not None:
            try:
                os.killpg(os.getpgid(self.rviz_process.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
        subprocess.run(["pkill", "-9", "-f", "ldlidar_stl_ros2_node"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def print_menu(self):
        """Displays formatted parameter tables and operational metrics in terminal."""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("=" * 85)
        print("         LDLiDAR STL27L 2D - INTERACTIVE PARAMETER TUNING SYSTEM")
        print("=" * 85)
        print(" [*] Topics: Input [/scan] -> Filtered [/scan_filtered] (Visualized in RViz2)")
        if self.node:
            active = self.node.last_filtered_beams
            total = self.node.last_received_beams
            pct = (1.0 - (active / total)) * 100.0 if total > 0 else 0.0
            print(f" [*] Stream Health: Total Beams: {total} | Active Beams: {active} | Filtered: {pct:.1f}%")
            if self.node.last_telemetry_msg:
                print(f" \033[92m{self.node.last_telemetry_msg}\033[0m")
        print("-" * 85)
        print(" [A] DYNAMIC PARAMETERS (Zero-Downtime, Instant RAM Vectorized Processing)")
        print(f"   1. crop_min_angle  : {self.dynamic_params['crop_min_angle']:<8} [Deg: 0.0 to 360.0]")
        print(f"   2. crop_max_angle  : {self.dynamic_params['crop_max_angle']:<8} [Deg: 0.0 to 360.0]")
        print(f"   3. min_intensity   : {self.dynamic_params['min_intensity']:<8} [0.0 to 255.0 - Noise/Dust Reject]")
        print(f"   4. max_intensity   : {self.dynamic_params['max_intensity']:<8} [0.0 to 255.0 - Reflector Select]")
        print(f"   5. min_range       : {self.dynamic_params['min_range']:<8} [Meters: Min distance cutoff]")
        print(f"   6. max_range       : {self.dynamic_params['max_range']:<8} [Meters: Max distance cutoff]")
        print("-" * 85)
        print(" [B] STATIC PARAMETERS (Driver Level, Automatically Measures Restart Downtime)")
        print(f"   7. laser_scan_dir          : {self.static_params['laser_scan_dir']:<8} ['true'=CW, 'false'=CCW]")
        print(f"   8. enable_angle_crop_func  : {self.static_params['enable_angle_crop_func']:<8} ['true' or 'false']")
        print(f"   9. angle_crop_min          : {self.static_params['angle_crop_min']:<8} [Deg: 0.0 to 360.0]")
        print(f"  10. angle_crop_max          : {self.static_params['angle_crop_max']:<8} [Deg: 0.0 to 360.0]")
        print(f"  11. frame_id                : {self.static_params['frame_id']:<8} [TF Transform Target Name]")
        print(f"  12. port_name               : {self.static_params['port_name']:<8} [UART Serial Device Path]")
        print(f"  13. port_baudrate           : {self.static_params['port_baudrate']:<8} [Default: 921600]")
        print("=" * 85)
        print(" Commands: Type parameter name/number to edit | 'reset' | 'refresh' | 'exit'")
        print("=" * 85)

    def run(self):
        print("[*] Initializing LiDAR driver...")
        self.start_driver()

        # Start ROS 2 background spinning
        self.node = LidarInteractiveNode(self)
        self.executor_thread = threading.Thread(
            target=lambda: rclpy.spin(self.node),
            daemon=True
        )
        self.executor_thread.start()

        # Wait for hardware packets to lock before popping RViz
        print("[*] Waiting for incoming LiDAR stream lock...")
        if self.node.first_scan_event.wait(timeout=6.0):
            print("[*] LiDAR stream locked. Spawning auto-configured RViz2...")
        else:
            print("[!] Warning: LiDAR stream delayed, launching RViz2 anyway...")

        self.start_rviz()

        param_index_map = {
            "1": "crop_min_angle", "2": "crop_max_angle",
            "3": "min_intensity", "4": "max_intensity",
            "5": "min_range", "6": "max_range",
            "7": "laser_scan_dir", "8": "enable_angle_crop_func",
            "9": "angle_crop_min", "10": "angle_crop_max",
            "11": "frame_id", "12": "port_name", "13": "port_baudrate"
        }

        while self.running:
            self.print_menu()
            try:
                raw_input_cmd = input("\nEnter parameter to tune (or command): ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not raw_input_cmd:
                continue

            cmd_lower = raw_input_cmd.lower()
            if cmd_lower in ["exit", "quit", "q"]:
                break
            elif cmd_lower in ["refresh", "r"]:
                continue
            elif cmd_lower == "reset":
                self.dynamic_params = {
                    "crop_min_angle": 0.0, "crop_max_angle": 0.0,
                    "min_intensity": 0.0, "max_intensity": 255.0,
                    "min_range": 0.05, "max_range": 25.0
                }
                print("[*] All dynamic filters reset to factory defaults.")
                time.sleep(0.8)
                continue

            param_key = param_index_map.get(raw_input_cmd, raw_input_cmd)

            if param_key in self.dynamic_params:
                val_str = input(f"Enter new value for [{param_key}] (current: {self.dynamic_params[param_key]}): ").strip()
                try:
                    val_float = float(val_str)
                    if "angle" in param_key and not (0.0 <= val_float <= 360.0):
                        print("[!] Error: Angle must be between 0.0 and 360.0 degrees.")
                    elif "intensity" in param_key and not (0.0 <= val_float <= 255.0):
                        print("[!] Error: Intensity must be between 0.0 and 255.0.")
                    elif "range" in param_key and val_float < 0.0:
                        print("[!] Error: Range cannot be negative.")
                    else:
                        # Arm latency telemetry tracker and update parameter
                        self.node.request_dynamic_measurement(param_key, val_float)
                        self.dynamic_params[param_key] = val_float
                        time.sleep(0.15)  # Allow next frame boundary to register telemetry
                except ValueError:
                    print("[!] Error: Invalid numeric value entered.")
                time.sleep(0.7)

            elif param_key in self.static_params:
                val_str = input(f"Enter new value for [{param_key}] (current: {self.static_params[param_key]}): ").strip()
                if not val_str:
                    continue

                if param_key in ["laser_scan_dir", "enable_angle_crop_func"]:
                    clean_bool = "true" if val_str.lower() in ["true", "1", "t", "yes"] else "false"
                    self.static_params[param_key] = clean_bool
                elif param_key in ["angle_crop_min", "angle_crop_max"]:
                    try:
                        self.static_params[param_key] = float(val_str)
                    except ValueError:
                        print("[!] Error: Angle must be a valid float.")
                        time.sleep(0.8)
                        continue
                elif param_key == "port_baudrate":
                    try:
                        self.static_params[param_key] = int(val_str)
                    except ValueError:
                        print("[!] Error: Baudrate must be an integer.")
                        time.sleep(0.8)
                        continue
                else:
                    self.static_params[param_key] = val_str

                print(f"[*] Stopping LiDAR driver and unbinding serial port...")
                self.node.first_scan_event.clear()
                t_shutdown_ms = self.stop_driver()

                print(f"[*] Spawning driver with new parameter [{param_key} = {self.static_params[param_key]}]...")
                t_respawn_start = time.perf_counter()
                self.start_driver()

                # Measure time to first valid scan packet
                locked = self.node.first_scan_event.wait(timeout=6.0)
                t_first_scan_ms = (time.perf_counter() - t_respawn_start) * 1000.0 if locked else 0.0
                t_total_downtime_ms = t_shutdown_ms + t_first_scan_ms

                self.node.last_telemetry_msg = (
                    f"[TELEMETRY] Static Param [{param_key}] Settled! "
                    f"T_shutdown: {t_shutdown_ms:.2f} ms | T_first_scan: {t_first_scan_ms:.2f} ms | "
                    f"Total Downtime: {t_total_downtime_ms:.2f} ms"
                )
                time.sleep(1.2)

            else:
                print(f"[!] Unknown parameter: '{raw_input_cmd}'. Enter a valid name or number.")
                time.sleep(1.0)


def main():
    rclpy.init()
    app = LidarTunerApp()

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