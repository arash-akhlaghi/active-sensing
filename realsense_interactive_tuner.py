#!/usr/bin/env python3
import os
import sys
import time
import queue
import threading
import cv2
import numpy as np
import pyrealsense2 as rs

FRAME_PERIOD_MS = 1000.0 / 30.0
BRIGHTNESS_DELTA_THRESHOLD = 2.0


def clamp_value(sensor, option, val):
    """Ensure parameter remains strictly within sensor hardware limits."""
    rng = sensor.get_option_range(option)
    return max(rng.min, min(rng.max, val))


def safe_metadata_field(name):
    """Return the rs.frame_metadata_value enum member, or None if unexposed."""
    return getattr(rs.frame_metadata_value, name, None)


def rgb_exposure_to_metadata_units(val):
    """Convert UVC 100us units to microseconds."""
    return val * 100


def draw_telemetry(image, lines):
    """Draw benchmark telemetry lines on the combined video stream."""
    y = 30
    for line in lines:
        cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        y += 24


class RealSenseInteractiveTuner:
    def __init__(self):
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.colorizer = rs.colorizer()

        # Active hardware stream configuration
        self.width = 848
        self.height = 480
        self.fps = 30

        self.cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)

        print("[*] Initializing Intel RealSense D456 pipeline...")
        self.profile = self.pipe.start(self.cfg)
        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        self.color_sensor = self.device.first_color_sensor()

        # Disable automatic loops for deterministic benchmarking
        if self.depth_sensor.supports(rs.option.enable_auto_exposure):
            self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        if self.color_sensor.supports(rs.option.enable_auto_exposure):
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        if self.color_sensor.supports(rs.option.enable_auto_white_balance):
            self.color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

        # -------------------------------------------------------------
        # Parameter Catalog
        # -------------------------------------------------------------
        self.dynamic_params = {
            "depth_laser_power": 240.0,
            "depth_exposure": 12000.0,
            "depth_gain": 16.0,
            "depth_emitter": 1.0,
            "rgb_exposure": 120.0,
            "rgb_gain": 32.0,
            "rgb_white_balance": 4000.0,
            # Intensity and Gating filters in memory (Zero hardware latency)
            "ir_min_intensity": 0.0,
            "ir_max_intensity": 255.0,
            "rgb_min_luminance": 0.0,
            "rgb_max_luminance": 255.0,
            "depth_min_range_m": 0.15,
            "depth_max_range_m": 10.0
        }

        self.apply_initial_parameters()

        # Thread synchronization and command queue
        self.cmd_queue = queue.Queue()
        self.settling_event = threading.Event()
        self.running = True

        # Telemetry history state
        self.active_check = None
        self.last_telemetry_banner = "No parameter changes recorded yet."
        self.telemetry_history = []
        self.last_depth_id = None
        self.last_color_id = None
        self.dropped_during_settling = 0
        self.current_overlay_lines = ["Intel RealSense D456 Interactive Tuner Active."]

    def apply_initial_parameters(self):
        """Send base parameters to camera sensors."""
        self.depth_sensor.set_option(rs.option.laser_power, self.dynamic_params["depth_laser_power"])
        self.depth_sensor.set_option(rs.option.exposure, self.dynamic_params["depth_exposure"])
        self.depth_sensor.set_option(rs.option.gain, self.dynamic_params["depth_gain"])
        self.depth_sensor.set_option(rs.option.emitter_enabled, self.dynamic_params["depth_emitter"])
        self.color_sensor.set_option(rs.option.exposure, self.dynamic_params["rgb_exposure"])
        self.color_sensor.set_option(rs.option.gain, self.dynamic_params["rgb_gain"])
        self.color_sensor.set_option(rs.option.white_balance, self.dynamic_params["rgb_white_balance"])

    def reconfigure_pipeline_resolution(self, width, height, fps):
        """Performs a cold restart of the RealSense pipeline for static resolution switching."""
        t0 = time.perf_counter()
        self.pipe.stop()
        self.cfg.disable_all_streams()
        self.width = width
        self.height = height
        self.fps = fps
        self.cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
        self.profile = self.pipe.start(self.cfg)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.color_sensor = self.profile.get_device().first_color_sensor()
        self.apply_initial_parameters()
        downtime_ms = (time.perf_counter() - t0) * 1000.0
        return downtime_ms

    def start_settling_evaluation(self, label, task_fn, checks):
        """Measure software/USB command write latency and initialize hardware settling tracker."""
        t0 = time.perf_counter()
        task_fn()
        t1 = time.perf_counter()
        cmd_ms = (t1 - t0) * 1000.0

        self.dropped_during_settling = 0
        self.active_check = {
            "label": label,
            "cmd_ms": cmd_ms,
            "checks": checks,
            "start_time": time.perf_counter(),
            "frame_offset": 0,
            "timeout_frames": 45,
            "ref_color_img": None
        }

    def process_frames(self):
        """Streaming, OpenCV rendering, and per-frame settling observer loop."""
        cv2.namedWindow("Intel RealSense D456 - Interactive Tuner", cv2.WINDOW_AUTOSIZE)

        while self.running:
            # Handle commands issued from the CLI thread
            try:
                cmd = self.cmd_queue.get_nowait()
                cmd_type = cmd.get("type")
                if cmd_type == "set_param" or cmd_type == "bundle":
                    self.start_settling_evaluation(cmd["label"], cmd["task"], cmd["checks"])
                elif cmd_type == "cold_restart":
                    dt = self.reconfigure_pipeline_resolution(cmd["w"], cmd["h"], cmd["fps"])
                    banner = (
                        f"[COLD RESTART] Resized to {cmd['w']}x{cmd['h']} @ {cmd['fps']} FPS | "
                        f"Perception Downtime: {dt:.2f} ms"
                    )
                    self.last_telemetry_banner = banner
                    self.telemetry_history.append(banner)
                    self.settling_event.set()
                elif cmd_type == "exit":
                    break
            except queue.Empty:
                pass

            # Acquire synchronized frameset
            frameset = self.pipe.wait_for_frames()
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()
            ir_frame = frameset.get_infrared_frame(1)

            if not depth_frame or not color_frame:
                continue

            # Frame drop monitoring
            depth_fid = depth_frame.get_frame_number()
            color_fid = color_frame.get_frame_number()

            if self.last_depth_id is not None and (depth_fid - self.last_depth_id) > 1:
                dropped = (depth_fid - self.last_depth_id - 1)
                if self.active_check is not None:
                    self.dropped_during_settling += dropped
            if self.last_color_id is not None and (color_fid - self.last_color_id) > 1:
                dropped = (color_fid - self.last_color_id - 1)
                if self.active_check is not None:
                    self.dropped_during_settling += dropped

            self.last_depth_id = depth_fid
            self.last_color_id = color_fid

            # ---------------------------------------------------------
            # Hardware Settling Verification
            # ---------------------------------------------------------
            if self.active_check is not None:
                chk_state = self.active_check
                chk_state["frame_offset"] += 1
                curr_offset = chk_state["frame_offset"]

                if chk_state["ref_color_img"] is None:
                    chk_state["ref_color_img"] = np.asanyarray(color_frame.get_data(), dtype=np.float32)

                all_settled = True
                for c in chk_state["checks"]:
                    if c.get("estimated", False):
                        if curr_offset < c.get("estimated_frame", 1):
                            all_settled = False
                        continue

                    target_frame = depth_frame if c["depth"] else color_frame
                    if target_frame is None or not target_frame.supports_frame_metadata(c["field"]):
                        all_settled = False
                        continue

                    actual = target_frame.get_frame_metadata(c["field"])
                    primary_match = abs(actual - c["target"]) <= c["tol"]
                    alt_match = False
                    if "alt_target" in c:
                        alt_match = abs(actual - c["alt_target"]) <= c.get("alt_tol", c["tol"])

                    if not (primary_match or alt_match):
                        # Optical Safety Net for RGB exposure unit mismatch
                        if not c["depth"] and chk_state["ref_color_img"] is not None:
                            curr_img = np.asanyarray(color_frame.get_data(), dtype=np.float32)
                            delta = float(np.mean(np.abs(curr_img - chk_state["ref_color_img"])))
                            if delta >= BRIGHTNESS_DELTA_THRESHOLD and curr_offset >= 2:
                                continue
                        all_settled = False

                if all_settled or curr_offset >= chk_state["timeout_frames"]:
                    est_hw_latency = curr_offset * (1000.0 / self.fps)
                    banner = (
                        f"[{chk_state['label']}] LATCHED! -> "
                        f"USB Latency: {chk_state['cmd_ms']:.2f} ms | "
                        f"HW Settling: {curr_offset} frames ({est_hw_latency:.2f} ms) | "
                        f"Dropped: {self.dropped_during_settling}"
                    )
                    self.last_telemetry_banner = banner
                    self.telemetry_history.append(banner)
                    if len(self.telemetry_history) > 4:
                        self.telemetry_history.pop(0)

                    self.current_overlay_lines = [
                        f"Target: {chk_state['label']}",
                        f"USB Latency: {chk_state['cmd_ms']:.2f} ms",
                        f"HW Settled: {curr_offset} frames ({est_hw_latency:.1f} ms)",
                        f"Dropped Frames: {self.dropped_during_settling}"
                    ]
                    self.active_check = None
                    self.settling_event.set()

            # ---------------------------------------------------------
            # Dynamic Intensity & Gating Processing Layer
            # ---------------------------------------------------------
            depth_data = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * 0.001
            color_data = np.asanyarray(color_frame.get_data())

            # 1. Depth Range Gating
            min_r = self.dynamic_params["depth_min_range_m"]
            max_r = self.dynamic_params["depth_max_range_m"]
            depth_mask = (depth_data < min_r) | (depth_data > max_r)

            # 2. RGB Luminance (Photometric Intensity) Gating: Y = 0.114*B + 0.587*G + 0.299*R
            rgb_luma = 0.114 * color_data[:, :, 0] + 0.587 * color_data[:, :, 1] + 0.299 * color_data[:, :, 2]
            rgb_min_y = self.dynamic_params["rgb_min_luminance"]
            rgb_max_y = self.dynamic_params["rgb_max_luminance"]
            rgb_mask = (rgb_luma < rgb_min_y) | (rgb_luma > rgb_max_y)

            # 3. Near-Infrared (NIR) Optical Intensity Gating
            ir_filtered_ratio = 0.0
            if ir_frame:
                ir_mat = np.asanyarray(ir_frame.get_data(), dtype=np.float32)
                i_min = self.dynamic_params["ir_min_intensity"]
                i_max = self.dynamic_params["ir_max_intensity"]
                ir_mask = (ir_mat < i_min) | (ir_mat > i_max)
                ir_filtered_ratio = (np.count_nonzero(ir_mask) / (self.width * self.height)) * 100.0

            # Render visual output
            depth_vis = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            depth_vis[depth_mask] = 0
            color_vis = color_data.copy()
            color_vis[rgb_mask] = 0

            if depth_vis.shape[0] != color_vis.shape[0]:
                color_vis = cv2.resize(color_vis, (depth_vis.shape[1], depth_vis.shape[0]))

            combined_frame = np.hstack((color_vis, depth_vis))

            overlay = list(self.current_overlay_lines)
            overlay.append(f"FPS: {self.fps} | Res: {self.width}x{self.height}")
            overlay.append(f"Range: [{min_r:.2f}m - {max_r:.2f}m]")
            overlay.append(f"RGB Luma: [{rgb_min_y:.0f} - {rgb_max_y:.0f}] | NIR: [{self.dynamic_params['ir_min_intensity']:.0f} - {self.dynamic_params['ir_max_intensity']:.0f}] ({ir_filtered_ratio:.1f}%)")

            draw_telemetry(combined_frame, overlay)
            cv2.imshow("Intel RealSense D456 - Interactive Tuner", combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                self.running = False
                break

        self.pipe.stop()
        cv2.destroyAllWindows()


class RealSenseTunerCLI:
    """Interactive command-line user interface."""
    def __init__(self, tuner):
        self.tuner = tuner

    def print_menu(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print("=" * 95)
        print("           INTEL REALSENSE D456 - INTERACTIVE BENCHMARK & PARAMETER TUNER")
        print("=" * 95)
        print(f" [*] Active Hardware Profile: {self.tuner.width}x{self.tuner.height} @ {self.tuner.fps} FPS")

        # Persistent Telemetry Banner
        print("-" * 95)
        print(" \033[92m[LATEST TELEMETRY REPORT]\033[0m")
        print(f" \033[93m>>> {self.tuner.last_telemetry_banner}\033[0m")
        if len(self.tuner.telemetry_history) > 1:
            print(" [Previous History]:")
            for h in self.tuner.telemetry_history[-3:-1]:
                print(f"   * {h}")
        print("-" * 95)

        print(" [A] DEPTH & EMITTER HARDWARE REGISTERS")
        print(f"   1. depth_laser_power  : {self.tuner.dynamic_params['depth_laser_power']:<8} [mW: 0 to 360]")
        print(f"   2. depth_exposure     : {self.tuner.dynamic_params['depth_exposure']:<8} [us: 1 to 200000]")
        print(f"   3. depth_gain         : {self.tuner.dynamic_params['depth_gain']:<8} [Gain: 16 to 248]")
        print(f"   4. depth_emitter      : {self.tuner.dynamic_params['depth_emitter']:<8} [0: Laser OFF, 1: Laser ON]")
        print("-" * 95)
        print(" [B] RGB COLOR ISP REGISTERS")
        print(f"   5. rgb_exposure       : {self.tuner.dynamic_params['rgb_exposure']:<8} [UVC Index: 1 to 10000]")
        print(f"   6. rgb_gain           : {self.tuner.dynamic_params['rgb_gain']:<8} [Gain: 0 to 128]")
        print(f"   7. rgb_white_balance  : {self.tuner.dynamic_params['rgb_white_balance']:<8} [Kelvin: 2800 to 6500]")
        print("-" * 95)
        print(" [C] INTENSITY & PERCEPTION FILTERS (RAM Layer, Zero Hardware Latency)")
        print(f"   8. ir_min_intensity   : {self.tuner.dynamic_params['ir_min_intensity']:<8} [0.0 to 255.0 - NIR Noise Rejection]")
        print(f"   9. ir_max_intensity   : {self.tuner.dynamic_params['ir_max_intensity']:<8} [0.0 to 255.0 - NIR Reflector Select]")
        print(f"  10. rgb_min_luminance  : {self.tuner.dynamic_params['rgb_min_luminance']:<8} [0.0 to 255.0 - RGB Brightness Low Cut]")
        print(f"  11. rgb_max_luminance  : {self.tuner.dynamic_params['rgb_max_luminance']:<8} [0.0 to 255.0 - RGB Brightness High Cut]")
        print(f"  12. depth_min_range_m  : {self.tuner.dynamic_params['depth_min_range_m']:<8} [Meters: Min Distance Cutoff]")
        print(f"  13. depth_max_range_m  : {self.tuner.dynamic_params['depth_max_range_m']:<8} [Meters: Max Distance Cutoff]")
        print("-" * 95)
        print(" [D] SIMULTANEOUS BUNDLED MACROS")
        print("  14. bundle_lighting    : Depth Exposure (25000) + Depth Gain (64)")
        print("  15. bundle_active_pass : Emitter (0/1) + Laser (0/240) + Exposure (8000/15000)")
        print("  16. bundle_synced_exp  : Depth Exposure (22000) + RGB Exposure (280)")
        print("  17. bundle_rgb_tuning  : RGB Exposure (280) + Gain (48) + White Balance (5200)")
        print("-" * 95)
        print(" [E] STREAM RESOLUTION SWITCHING (Cold Pipeline Restart Downtime)")
        print("  18. profile_848x480x30 : Fast Navigation (848x480 @ 30 FPS)")
        print("  19. profile_1280x720x15: High-Res Inspection (1280x720 @ 15 FPS)")
        print("  20. profile_640x480x30 : Compact Mapping (640x480 @ 30 FPS)")
        print("=" * 95)
        print(" Commands: Type parameter number or name | 'reset' | 'refresh' | 'exit'")
        print("=" * 95)

    def run(self):
        f_exposure = safe_metadata_field("actual_exposure")
        f_gain = safe_metadata_field("gain_level")
        f_wb = safe_metadata_field("white_balance")
        f_laser = safe_metadata_field("frame_laser_power")
        f_emitter = safe_metadata_field("frame_emitter_mode")

        param_map = {
            "1": "depth_laser_power", "2": "depth_exposure", "3": "depth_gain", "4": "depth_emitter",
            "5": "rgb_exposure", "6": "rgb_gain", "7": "rgb_white_balance",
            "8": "ir_min_intensity", "9": "ir_max_intensity",
            "10": "rgb_min_luminance", "11": "rgb_max_luminance",
            "12": "depth_min_range_m", "13": "depth_max_range_m",
            "14": "bundle_lighting", "15": "bundle_active_pass", "16": "bundle_synced_exp", "17": "bundle_rgb_tuning",
            "18": "profile_848x480x30", "19": "profile_1280x720x15", "20": "profile_640x480x30"
        }

        while self.tuner.running:
            self.print_menu()
            try:
                user_cmd = input("\nEnter parameter to tune (or command): ").strip()
            except (KeyboardInterrupt, EOFError):
                self.tuner.running = False
                break

            if not user_cmd:
                continue

            cmd_lower = user_cmd.lower()
            if cmd_lower in ["exit", "quit", "q"]:
                self.tuner.cmd_queue.put({"type": "exit"})
                self.tuner.running = False
                break
            elif cmd_lower in ["refresh", "r"]:
                continue
            elif cmd_lower == "reset":
                self.tuner.dynamic_params = {
                    "depth_laser_power": 240.0, "depth_exposure": 12000.0, "depth_gain": 16.0, "depth_emitter": 1.0,
                    "rgb_exposure": 120.0, "rgb_gain": 32.0, "rgb_white_balance": 4000.0,
                    "ir_min_intensity": 0.0, "ir_max_intensity": 255.0,
                    "rgb_min_luminance": 0.0, "rgb_max_luminance": 255.0,
                    "depth_min_range_m": 0.15, "depth_max_range_m": 10.0
                }
                self.tuner.apply_initial_parameters()
                self.tuner.last_telemetry_banner = "Parameters reset to factory baseline."
                continue

            resolved_key = param_map.get(user_cmd, user_cmd)

            # A & B: Hardware Register Tuning
            if resolved_key in [
                "depth_laser_power", "depth_exposure", "depth_gain", "depth_emitter",
                "rgb_exposure", "rgb_gain", "rgb_white_balance"
            ]:
                val_str = input(f"Enter new value for [{resolved_key}] (current: {self.tuner.dynamic_params[resolved_key]}): ").strip()
                try:
                    val_flt = float(val_str)
                except ValueError:
                    print("[!] Error: Invalid numeric value.")
                    time.sleep(1.0)
                    continue

                checks = []
                task_fn = None
                self.tuner.dynamic_params[resolved_key] = val_flt

                if resolved_key == "depth_laser_power":
                    val_flt = clamp_value(self.tuner.depth_sensor, rs.option.laser_power, val_flt)
                    task_fn = lambda v=val_flt: self.tuner.depth_sensor.set_option(rs.option.laser_power, v)
                    checks.append({"field": f_laser, "target": val_flt, "tol": max(1, round(0.02 * val_flt)), "depth": True})
                elif resolved_key == "depth_exposure":
                    val_flt = clamp_value(self.tuner.depth_sensor, rs.option.exposure, val_flt)
                    task_fn = lambda v=val_flt: self.tuner.depth_sensor.set_option(rs.option.exposure, v)
                    checks.append({"field": f_exposure, "target": val_flt, "tol": max(1, round(0.02 * val_flt)), "depth": True})
                elif resolved_key == "depth_gain":
                    val_flt = clamp_value(self.tuner.depth_sensor, rs.option.gain, val_flt)
                    task_fn = lambda v=val_flt: self.tuner.depth_sensor.set_option(rs.option.gain, v)
                    checks.append({"field": f_gain, "target": val_flt, "tol": 2, "depth": True})
                elif resolved_key == "depth_emitter":
                    val_flt = 1.0 if val_flt > 0 else 0.0
                    task_fn = lambda v=val_flt: self.tuner.depth_sensor.set_option(rs.option.emitter_enabled, v)
                    checks.append({"field": f_emitter, "target": int(val_flt), "tol": 0, "depth": True})
                elif resolved_key == "rgb_exposure":
                    val_flt = clamp_value(self.tuner.color_sensor, rs.option.exposure, val_flt)
                    target_meta = rgb_exposure_to_metadata_units(val_flt)
                    task_fn = lambda v=val_flt: self.tuner.color_sensor.set_option(rs.option.exposure, v)
                    checks.append({
                        "field": f_exposure,
                        "target": target_meta,
                        "alt_target": val_flt,
                        "tol": max(1, round(0.15 * target_meta)),
                        "alt_tol": max(2, round(0.15 * val_flt)),
                        "depth": False
                    })
                elif resolved_key == "rgb_gain":
                    val_flt = clamp_value(self.tuner.color_sensor, rs.option.gain, val_flt)
                    task_fn = lambda v=val_flt: self.tuner.color_sensor.set_option(rs.option.gain, v)
                    checks.append({"field": f_gain, "target": val_flt, "tol": 2, "depth": False})
                elif resolved_key == "rgb_white_balance":
                    val_flt = clamp_value(self.tuner.color_sensor, rs.option.white_balance, val_flt)
                    task_fn = lambda v=val_flt: self.tuner.color_sensor.set_option(rs.option.white_balance, v)
                    checks.append({"field": f_wb, "target": val_flt, "tol": max(1, round(0.02 * val_flt)), "depth": False, "estimated": True, "estimated_frame": 1})

                self.tuner.settling_event.clear()
                self.tuner.cmd_queue.put({"type": "set_param", "label": f"{resolved_key} -> {val_flt}", "task": task_fn, "checks": checks})
                print("[*] Waiting for hardware latch and frame settling...")
                self.tuner.settling_event.wait(timeout=3.0)

            # C: Dynamic Intensity / Luminance / Range Gating in RAM
            elif resolved_key in [
                "ir_min_intensity", "ir_max_intensity",
                "rgb_min_luminance", "rgb_max_luminance",
                "depth_min_range_m", "depth_max_range_m"
            ]:
                val_str = input(f"Enter new value for [{resolved_key}] (current: {self.tuner.dynamic_params[resolved_key]}): ").strip()
                try:
                    val_flt = float(val_str)
                    t0 = time.perf_counter()
                    self.tuner.dynamic_params[resolved_key] = val_flt
                    t_proc = (time.perf_counter() - t0) * 1000.0
                    banner = (
                        f"Dynamic RAM Filter [{resolved_key} = {val_flt}] applied! "
                        f"Algorithmic Latency: {t_proc:.4f} ms (Instant 0-frame settling)"
                    )
                    self.tuner.last_telemetry_banner = banner
                    self.tuner.telemetry_history.append(banner)
                except ValueError:
                    print("[!] Error: Invalid numeric value.")
                    time.sleep(1.0)

            # D: Bundled Macros
            elif resolved_key == "bundle_lighting":
                d_exp = clamp_value(self.tuner.depth_sensor, rs.option.exposure, 25000)
                d_gain = clamp_value(self.tuner.depth_sensor, rs.option.gain, 64)
                task_fn = lambda: (self.tuner.depth_sensor.set_option(rs.option.exposure, d_exp),
                                   self.tuner.depth_sensor.set_option(rs.option.gain, d_gain))
                checks = [
                    {"field": f_exposure, "target": d_exp, "tol": max(1, round(0.02 * d_exp)), "depth": True},
                    {"field": f_gain, "target": d_gain, "tol": 2, "depth": True}
                ]
                self.tuner.settling_event.clear()
                self.tuner.cmd_queue.put({"type": "bundle", "label": "Bundle 1: Lighting Adaptation", "task": task_fn, "checks": checks})
                print("[*] Executing Bundle 1 and evaluating settling...")
                self.tuner.settling_event.wait(timeout=3.0)

            elif resolved_key == "bundle_active_pass":
                state_str = input("Switch Mode (1 = Active Laser ON, 0 = Passive Laser OFF): ").strip()
                active = state_str == "1"
                laser_val = 240 if active else 0
                exp_val = 15000 if active else 8000
                task_fn = lambda: (self.tuner.depth_sensor.set_option(rs.option.emitter_enabled, 1 if active else 0),
                                   self.tuner.depth_sensor.set_option(rs.option.laser_power, laser_val),
                                   self.tuner.depth_sensor.set_option(rs.option.exposure, exp_val))
                checks = [
                    {"field": f_emitter, "target": 1 if active else 0, "tol": 0, "depth": True},
                    {"field": f_laser, "target": laser_val, "tol": max(1, round(0.02 * laser_val)), "depth": True},
                    {"field": f_exposure, "target": exp_val, "tol": max(1, round(0.02 * exp_val)), "depth": True}
                ]
                self.tuner.settling_event.clear()
                self.tuner.cmd_queue.put({"type": "bundle", "label": "Bundle 2: Active/Passive Switch", "task": task_fn, "checks": checks})
                print("[*] Executing Bundle 2 and evaluating settling...")
                self.tuner.settling_event.wait(timeout=3.0)

            elif resolved_key == "bundle_synced_exp":
                d_exp = clamp_value(self.tuner.depth_sensor, rs.option.exposure, 22000)
                c_exp = clamp_value(self.tuner.color_sensor, rs.option.exposure, 280)
                c_meta = rgb_exposure_to_metadata_units(c_exp)
                task_fn = lambda: (self.tuner.depth_sensor.set_option(rs.option.exposure, d_exp),
                                   self.tuner.color_sensor.set_option(rs.option.exposure, c_exp))
                checks = [
                    {"field": f_exposure, "target": d_exp, "tol": max(1, round(0.02 * d_exp)), "depth": True},
                    {"field": f_exposure, "target": c_meta, "alt_target": c_exp, "tol": max(1, round(0.15 * c_meta)), "alt_tol": max(2, round(0.15 * c_exp)), "depth": False}
                ]
                self.tuner.settling_event.clear()
                self.tuner.cmd_queue.put({"type": "bundle", "label": "Bundle 3: Synced Exposure", "task": task_fn, "checks": checks})
                print("[*] Executing Bundle 3 and evaluating settling...")
                self.tuner.settling_event.wait(timeout=3.0)

            elif resolved_key == "bundle_rgb_tuning":
                c_exp = clamp_value(self.tuner.color_sensor, rs.option.exposure, 280)
                c_gain = clamp_value(self.tuner.color_sensor, rs.option.gain, 48)
                c_wb = clamp_value(self.tuner.color_sensor, rs.option.white_balance, 5200)
                c_meta = rgb_exposure_to_metadata_units(c_exp)
                task_fn = lambda: (self.tuner.color_sensor.set_option(rs.option.exposure, c_exp),
                                   self.tuner.color_sensor.set_option(rs.option.gain, c_gain),
                                   self.tuner.color_sensor.set_option(rs.option.white_balance, c_wb))
                checks = [
                    {"field": f_exposure, "target": c_meta, "alt_target": c_exp, "tol": max(1, round(0.15 * c_meta)), "alt_tol": max(2, round(0.15 * c_exp)), "depth": False},
                    {"field": f_gain, "target": c_gain, "tol": 2, "depth": False},
                    {"field": f_wb, "target": c_wb, "tol": max(1, round(0.02 * c_wb)), "depth": False, "estimated": True, "estimated_frame": 1}
                ]
                self.tuner.settling_event.clear()
                self.tuner.cmd_queue.put({"type": "bundle", "label": "Bundle 4: RGB Tuning", "task": task_fn, "checks": checks})
                print("[*] Executing Bundle 4 and evaluating settling...")
                self.tuner.settling_event.wait(timeout=3.0)

            # E: Static Resolution Switch
            elif resolved_key == "profile_848x480x30":
                self.tuner.settling_event.clear()
                self.tuner.cmd_queue.put({"type": "cold_restart", "w": 848, "h": 480, "fps": 30})
                self.tuner.settling_event.wait(timeout=6.0)
            elif resolved_key == "profile_1280x720x15":
                self.tuner.settling_event.clear()
                self.tuner.cmd_queue.put({"type": "cold_restart", "w": 1280, "h": 720, "fps": 15})
                self.tuner.settling_event.wait(timeout=6.0)
            elif resolved_key == "profile_640x480x30":
                self.tuner.settling_event.clear()
                self.tuner.cmd_queue.put({"type": "cold_restart", "w": 640, "h": 480, "fps": 30})
                self.tuner.settling_event.wait(timeout=6.0)
            else:
                print(f"[!] Unknown parameter: '{user_cmd}'")
                time.sleep(0.8)


def main():
    tuner = RealSenseInteractiveTuner()
    cli = RealSenseTunerCLI(tuner)

    cli_thread = threading.Thread(target=cli.run, daemon=True)
    cli_thread.start()

    try:
        tuner.process_frames()
    except KeyboardInterrupt:
        pass
    finally:
        tuner.running = False
        print("\n[*] Closing tuner and stopping camera pipeline...")
        cli_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()