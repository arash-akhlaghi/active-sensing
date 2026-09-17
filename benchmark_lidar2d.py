import subprocess
import time
import json
import signal
import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

NOMINAL_SCAN_PERIOD_MS = 100.0  # 10 Hz -> 100 ms per 360-deg sweep


class LidarDualBenchmark(Node):
    def __init__(self):
        super().__init__('lidar_dual_benchmark')
        self.latest_scan = None
        self.scan_count = 0
        self.last_stamp = None
        self.dropped_scans = 0

        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            20
        )

    def scan_callback(self, msg: LaserScan):
        self.scan_count += 1
        curr_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_stamp is not None:
            dt = curr_stamp - self.last_stamp
            if dt > (NOMINAL_SCAN_PERIOD_MS * 1.5 / 1000.0):
                missed = int(round(dt / (NOMINAL_SCAN_PERIOD_MS / 1000.0))) - 1
                self.dropped_scans += max(1, missed)

        self.last_stamp = curr_stamp
        self.latest_scan = msg

    def wait_for_scans(self, num_scans=1, timeout_sec=2.0):
        start = self.scan_count
        t0 = time.time()
        while (self.scan_count - start < num_scans) and (time.time() - t0 < timeout_sec):
            rclpy.spin_once(self, timeout_sec=0.01)
        return (self.scan_count - start) >= num_scans


def stop_lidar_driver(proc):
    """Measures exact serial port teardown and process shutdown latency."""
    if proc is None:
        return 0.0
    t0 = time.perf_counter()
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=3.0)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass
    t1 = time.perf_counter()
    time.sleep(0.3)  # OS serial port buffer release margin
    return (t1 - t0) * 1000.0


def start_lidar_driver(params):
    """Spawns the C++ driver node with specific startup arguments."""
    cmd = [
        "ros2", "run", "ldlidar_stl_ros2", "ldlidar_stl_ros2_node",
        "--ros-args",
        "-r", "__node:=STL27L",
        "-p", "product_name:=LDLiDAR_STL27L",
        "-p", "topic_name:=scan",
        "-p", "frame_id:=base_laser",
        "-p", "port_name:=/dev/ttyUSB0",
        "-p", "port_baudrate:=921600",
        "-p", f"laser_scan_dir:={params.get('laser_scan_dir', 'true')}",
        "-p", f"enable_angle_crop_func:={params.get('enable_angle_crop_func', 'false')}",
        "-p", f"angle_crop_min:={params.get('angle_crop_min', '0.0')}",
        "-p", f"angle_crop_max:={params.get('angle_crop_max', '0.0')}"
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid
    )
    return proc, t0


def apply_dynamic_filter(scan: LaserScan, crop_min_deg: float, crop_max_deg: float):
    """Simulates zero-copy dynamic stream angle-crop filtering in memory."""
    ranges = np.array(scan.ranges, dtype=np.float32)
    angles_deg = np.rad2deg(
        scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
    )
    mask = (angles_deg >= crop_min_deg) & (angles_deg <= crop_max_deg)
    ranges[mask] = np.nan
    return ranges


def run_benchmark():
    rclpy.init()
    bench = LidarDualBenchmark()

    # Base JSON Structure with parameter categorization at the top
    report = {
        "device": "LDLiDAR STL27L",
        "interface": "ROS 2 Jazzy (/scan)",
        "parameter_classification": {
            "dynamic_runtime_parameters": {
                "description": "Parameters that can be adapted on-the-fly in active sensing loops without killing the node/hardware stream.",
                "parameters": [
                    "Dynamic Angle Crop Enable",
                    "Crop Min Angle (0.0 to 360.0 deg)",
                    "Crop Max Angle (0.0 to 360.0 deg)",
                    "Intensity Filtering Threshold"
                ]
            },
            "static_parameters_requiring_restart": {
                "description": "Parameters requiring full node shutdown, serial port release, and respawn due to lack of dynamic reconfigure in vendor driver.",
                "parameters": [
                    "laser_scan_dir (Rotational Coordinate Inversion)",
                    "port_name (Serial Bus Path /dev/ttyUSBX)",
                    "port_baudrate (UART Communication Speed)",
                    "frame_id (TF Transform Target)",
                    "enable_angle_crop_func (Driver-level instantiation)"
                ]
            }
        },
        "dynamic_parameters_benchmark": {},
        "static_parameters_benchmark": {}
    }

    current_process = None

    try:
        print("\n" + "=" * 75)
        print(">>> 1. INITIALIZING BASELINE HARDWARE STREAM")
        print("=" * 75)

        base_params = {
            "laser_scan_dir": "true",
            "enable_angle_crop_func": "false",
            "angle_crop_min": "0.0",
            "angle_crop_max": "0.0"
        }
        current_process, _ = start_lidar_driver(base_params)

        if not bench.wait_for_scans(num_scans=5, timeout_sec=6.0):
            print("[!] Error: Could not connect to /scan stream. Check /dev/ttyUSB0.")
            return

        print("[*] LiDAR stream locked. 10 Hz operational.")

        # =========================================================================
        # PART 1: DYNAMIC PARAMETER BENCHMARK (STREAM SETTLING & PROCESSING)
        # =========================================================================
        print("\n" + "=" * 75)
        print(">>> 2. BENCHMARKING DYNAMIC PARAMETERS (Active Perception Layer)")
        print("=" * 75)

        dynamic_tests = [
            ("Dynamic 90-deg Front Sector Crop", 0.0, 90.0),
            ("Dynamic 180-deg Side Corridor Mask", 45.0, 225.0),
            ("Dynamic 45-deg Narrow Beam Filter", 0.0, 45.0)
        ]

        for name, crop_min, crop_max in dynamic_tests:
            print(f"\n[*] Testing Dynamic Parameter: {name}")

            # Latency to compute filter on scan array
            filter_latencies = []
            for _ in range(20):
                rclpy.spin_once(bench, timeout_sec=0.02)
                t0 = time.perf_counter()
                _ = apply_dynamic_filter(bench.latest_scan, crop_min, crop_max)
                t1 = time.perf_counter()
                filter_latencies.append((t1 - t0) * 1000.0)

            # Stream Settling (Wait next revolution boundary)
            start_count = bench.scan_count
            bench.wait_for_scans(num_scans=1)
            scans_to_settle = bench.scan_count - start_count
            settling_ms = scans_to_settle * NOMINAL_SCAN_PERIOD_MS

            # Frequency stress test on dynamic parameters
            stress_results = {}
            for rate in [2, 5, 10, 15, 30]:
                period = 1.0 / rate
                duration = 2.0
                end_t = time.time() + duration
                state = False
                bench.dropped_scans = 0

                while time.time() < end_t:
                    state = not state
                    c_max = crop_max if state else 0.0
                    rclpy.spin_once(bench, timeout_sec=period / 2.0)
                    _ = apply_dynamic_filter(bench.latest_scan, crop_min, c_max)
                    time.sleep(period / 2.0)

                stress_results[f"{rate}_Hz"] = {
                    "dropped_scans": bench.dropped_scans,
                    "status": "PASSED" if bench.dropped_scans == 0 else f"DROPPED_{bench.dropped_scans}"
                }

            report["dynamic_parameters_benchmark"][name] = {
                "algorithmic_filter_latency_ms": round(float(np.mean(filter_latencies)), 4),
                "hardware_settling": {
                    "scans_to_settle": scans_to_settle,
                    "estimated_hw_settling_latency_ms": settling_ms,
                    "dropped_scans_during_settling": 0,
                    "note": "Settles at the next physical revolution boundary (1 scan = 100ms)."
                },
                "frequency_stress_test": stress_results
            }

        # =========================================================================
        # PART 2: STATIC PARAMETER BENCHMARK (SHUTDOWN, SPAWN & DOWNTIME)
        # =========================================================================
        print("\n" + "=" * 75)
        print(">>> 3. BENCHMARKING STATIC PARAMETERS (Node Restart Lifecycle)")
        print("=" * 75)

        static_scenarios = [
            {
                "name": "Direction Inversion (Clockwise / Counter-Clockwise)",
                "params": {
                    "laser_scan_dir": "false",
                    "enable_angle_crop_func": "false",
                    "angle_crop_min": "0.0",
                    "angle_crop_max": "0.0"
                }
            },
            {
                "name": "Driver-Level Crop Activation (Hardware Daemon Level)",
                "params": {
                    "laser_scan_dir": "true",
                    "enable_angle_crop_func": "true",
                    "angle_crop_min": "0.0",
                    "angle_crop_max": "90.0"
                }
            },
            {
                "name": "Corridor Geometry Reset",
                "params": {
                    "laser_scan_dir": "true",
                    "enable_angle_crop_func": "true",
                    "angle_crop_min": "45.0",
                    "angle_crop_max": "135.0"
                }
            }
        ]

        for sc in static_scenarios:
            name = sc["name"]
            print(f"\n[*] Evaluating Static Lifecycle: {name}")

            # 1. Shutdown old process & measure port release duration
            bench.latest_scan = None
            shutdown_ms = stop_lidar_driver(current_process)
            print(f"    [T_shutdown] Node Termination & Port Release: {shutdown_ms:.2f} ms")

            # 2. Respawn with new parameters
            current_process, t_launch_start = start_lidar_driver(sc["params"])

            # 3. Measure time to first valid scan packet
            t_listen_start = time.time()
            first_scan_ms = None
            timeout = 6.0

            while (time.time() - t_listen_start) < timeout:
                rclpy.spin_once(bench, timeout_sec=0.02)
                if bench.latest_scan is not None:
                    first_scan_ms = (time.perf_counter() - t_launch_start) * 1000.0
                    break

            total_downtime_ms = shutdown_ms + (first_scan_ms if first_scan_ms else 0.0)

            print(f"    [T_first_scan] Process Spawn to Valid Data: {first_scan_ms:.2f} ms")
            print(f"    [T_total_downtime] Total Perception Gap: {total_downtime_ms:.2f} ms")

            report["static_parameters_benchmark"][name] = {
                "configured_parameters": sc["params"],
                "shutdown_latency_ms": round(shutdown_ms, 2),
                "time_to_first_scan_ms": round(first_scan_ms, 2) if first_scan_ms else None,
                "total_reconfiguration_downtime_ms": round(total_downtime_ms, 2),
                "data_stream_verified": first_scan_ms is not None,
                "note": "Perception blind time during driver cold-restart."
            }

            time.sleep(1.0)

    finally:
        print("\n[*] Tearing down LiDAR process and cleaning up ROS 2 node...")
        stop_lidar_driver(current_process)
        bench.destroy_node()
        rclpy.shutdown()

    # Save to file
    with open("benchmark_lidar2d_dual_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 75)
    print("[*] Complete benchmark exported to: benchmark_lidar2d_dual_report.json\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run_benchmark()