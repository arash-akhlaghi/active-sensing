import time
import json
import cv2
import numpy as np
import pyrealsense2 as rs

FRAME_PERIOD_MS = 1000.0 / 30.0

# Proxy threshold for "visually perceptible" change (mean abs pixel delta,
# 0-255 scale). ~2.0 is close to the classic ~1% Weber contrast JND for
# luminance, but this is an engineering approximation, not a validated
# human-perception result. Calibrate against your own eyes before trusting it.
BRIGHTNESS_DELTA_THRESHOLD = 2.0


def draw_telemetry(image, lines):
    """Draw benchmark telemetry lines on the combined video stream."""
    y = 30
    for line in lines:
        cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
        y += 24


def display_frame(frameset, colorizer, telemetry_lines):
    """Concatenate RGB and Depth frames horizontally and display in OpenCV."""
    depth_frame = frameset.get_depth_frame()
    color_frame = frameset.get_color_frame()

    if not depth_frame or not color_frame:
        return True

    depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    color_vis = np.asanyarray(color_frame.get_data())

    if depth_vis.shape[0] != color_vis.shape[0]:
        color_vis = cv2.resize(color_vis, (depth_vis.shape[1], depth_vis.shape[0]))

    combined_frame = np.hstack((color_vis, depth_vis))
    draw_telemetry(combined_frame, telemetry_lines)

    cv2.imshow("Intel RealSense D456 - Comprehensive Benchmark Suite", combined_frame)
    key = cv2.waitKey(1) & 0xFF
    return not (key == 27 or key == ord('q'))


def clamp_value(sensor, option, val):
    """Ensure parameter remains strictly within sensor hardware limits."""
    rng = sensor.get_option_range(option)
    return max(rng.min, min(rng.max, val))


def safe_metadata_field(name):
    """Return the rs.frame_metadata_value enum member, or None if this
    pyrealsense2 build/firmware doesn't expose it."""
    return getattr(rs.frame_metadata_value, name, None)


def rgb_exposure_to_metadata_units(val):
    """The color sensor's option.exposure is set in ~100us UVC 'absolute
    exposure time' units (per the USB Video Class spec), but the
    actual_exposure frame-metadata field reports the true value in
    microseconds. Multiply by 100 to compare them on the same scale.
    This matches current D4xx firmware behavior -- if you're on very old
    firmware, print a raw actual_exposure value once to confirm before
    trusting this multiplier."""
    return val * 100


# ---------------------------------------------------------------------------
# 1. HOST-SIDE (USB) COMMAND LATENCY
# ---------------------------------------------------------------------------
def evaluate_usb_cmd_latency(pipe, colorizer, task_fn, reset_fn, label, param_names):
    """Time for sensor.set_option() to return control to the host. This is
    ONLY the software/USB call overhead -- it says nothing about when the
    camera's ASIC actually applies the value to a frame. See
    evaluate_hardware_settling() for that."""
    params_str = " + ".join(param_names)
    cmd_latencies = []
    for i in range(5):
        reset_fn()
        time.sleep(0.04)
        t0 = time.perf_counter()
        task_fn()
        t1 = time.perf_counter()
        cmd_latencies.append((t1 - t0) * 1000.0)

        frames = pipe.wait_for_frames()
        overlay = [
            f"Testing USB Cmd Latency: {label}",
            f"Params: {params_str}",
            f"Step: USB Register Write ({i + 1}/5)",
            f"Write Latency: {cmd_latencies[-1]:.2f} ms"
        ]
        if not display_frame(frames, colorizer, overlay):
            return None

    return float(np.mean(cmd_latencies))


# ---------------------------------------------------------------------------
# 2. HARDWARE SETTLING LATENCY + FRAMES LOST WHILE SETTLING
#    Ground truth from each frame's own metadata, not host clock guesses.
# ---------------------------------------------------------------------------
def evaluate_hardware_settling(pipe, colorizer, task_fn, reset_fn, label, checks, timeout_frames=60):
    """
    checks: list of dicts {"field": rs.frame_metadata_value.X, "target": val,
            "tol": tolerance, "depth": True/False}
    Values in "target"/"tol" must already be in the SAME UNITS the metadata
    field itself reports (e.g. RGB exposure targets must be pre-converted
    with rgb_exposure_to_metadata_units()).

    Polls frames after issuing the command and reads each frame's OWN
    metadata (actual_exposure / gain_level / white_balance / frame_laser_power
    / frame_emitter_mode) to find the first frame where the camera reports
    the new value is actually in effect. That frame offset * frame period is
    the real hardware settling latency. Any frame_number gap observed while
    waiting counts as frames dropped during the transition.
    """
    params_str = ", ".join(str(c["field"]).split(".")[-1] for c in checks)

    unsupported_fields = [c for c in checks if c["field"] is None]
    if unsupported_fields:
        return {
            "metadata_supported": False,
            "frames_to_settle": None,
            "estimated_hw_settling_latency_ms": None,
            "dropped_frames_during_settling": None,
            "note": "One or more metadata fields are not exposed by this "
                    "pyrealsense2/firmware build; cannot measure directly."
        }

    reset_fn()
    for _ in range(6):
        frames = pipe.wait_for_frames()
        display_frame(frames, colorizer, [f"Stabilizing hardware for {label}..."])

    ref = pipe.wait_for_frames()
    last_depth_id = ref.get_depth_frame().get_frame_number() if ref.get_depth_frame() else None
    last_color_id = ref.get_color_frame().get_frame_number() if ref.get_color_frame() else None

    task_fn()

    dropped_during_settling = 0
    settled_at_offset = None
    metadata_readable = False

    for i in range(1, timeout_frames + 1):
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if depth_frame:
            fid = depth_frame.get_frame_number()
            if last_depth_id is not None and fid - last_depth_id > 1:
                dropped_during_settling += fid - last_depth_id - 1
            last_depth_id = fid
        if color_frame:
            fid = color_frame.get_frame_number()
            if last_color_id is not None and fid - last_color_id > 1:
                dropped_during_settling += fid - last_color_id - 1
            last_color_id = fid

        overlay = [
            f"HW Settling Test: {label}",
            f"Fields: {params_str}",
            f"Frame offset: {i}"
        ]
        if not display_frame(frames, colorizer, overlay):
            return None

        if settled_at_offset is None:
            all_match = True
            for chk in checks:
                target_frame = depth_frame if chk["depth"] else color_frame
                if target_frame is None or not target_frame.supports_frame_metadata(chk["field"]):
                    all_match = False
                    continue
                metadata_readable = True
                actual = target_frame.get_frame_metadata(chk["field"])
                if abs(actual - chk["target"]) > chk["tol"]:
                    all_match = False
            if all_match and metadata_readable:
                settled_at_offset = i

        if settled_at_offset is not None and i >= settled_at_offset + 3:
            break

    return {
        "metadata_supported": metadata_readable,
        "frames_to_settle": settled_at_offset,
        "estimated_hw_settling_latency_ms": (
            round(settled_at_offset * FRAME_PERIOD_MS, 2) if settled_at_offset else None
        ),
        "dropped_frames_during_settling": dropped_during_settling if settled_at_offset else None,
        "note": None if metadata_readable else "Metadata field present but never returned by frames"
    }


# ---------------------------------------------------------------------------
# 3. UPDATE-FREQUENCY STRESS TEST (frame drops under repeated toggling)
# ---------------------------------------------------------------------------
def evaluate_update_frequency(pipe, colorizer, toggle_state_fn, label, param_names, is_rgb_only=False):
    """Stress test update frequency while continuously pulling frames to count drops.

    is_rgb_only: track drop count against the color sensor's frame numbers
    instead of depth, for bundles/params that only touch the RGB sensor.
    """
    params_str = " + ".join(param_names)
    rates_hz = [2, 5, 10, 15, 30]
    rate_results = {}

    for rate in rates_hz:
        period = 1.0 / rate
        duration = 2.0
        end_time = time.time() + duration

        next_toggle = time.time()
        state = False
        dropped_frames = 0
        last_frame_id = None

        while time.time() < end_time:
            now = time.time()
            if now >= next_toggle:
                state = not state
                toggle_state_fn(state)
                next_toggle = now + period

            frames = pipe.poll_for_frames()
            if frames:
                target_frame = frames.get_color_frame() if is_rgb_only else frames.get_depth_frame()
                if target_frame:
                    fid = target_frame.get_frame_number()
                    if last_frame_id is not None and (fid - last_frame_id) > 1:
                        dropped_frames += (fid - last_frame_id - 1)
                    last_frame_id = fid

                    overlay = [
                        f"Stress Testing: {label}",
                        f"Params: {params_str}",
                        f"Frequency Target: {rate} Hz",
                        f"Dropped Frames: {dropped_frames}"
                    ]
                    if not display_frame(frames, colorizer, overlay):
                        return None

            time.sleep(0.001)

        rate_results[f"{rate}_Hz"] = {
            "dropped_frames": dropped_frames,
            "monitored_stream": "color" if is_rgb_only else "depth",
            "status": "PASSED" if dropped_frames == 0 else f"DROPPED_{dropped_frames}"
        }

    return rate_results


# ---------------------------------------------------------------------------
# 4. MINIMUM PERCEPTIBLE STEP (JND proxy via image statistics)
# ---------------------------------------------------------------------------
def find_perceptible_step(pipe, colorizer, sensor, option, base_value, direction, extreme_value,
                           stream="color", diff_metric="brightness",
                           threshold=BRIGHTNESS_DELTA_THRESHOLD, step_count=8):
    """
    Heuristic JND probe. Steps the parameter from base_value toward
    extreme_value in `step_count` increments, measuring image difference vs
    the baseline frame at each step, and returns the smallest step size at
    which the difference crosses `threshold`.

    stream: "color" or "infrared" -- which stream to read pixels from.
    diff_metric: "brightness" (mean abs pixel delta, all channels) or
                 "color_shift" (mean abs delta of the B-R channel difference,
                 more appropriate for white balance than raw brightness).

    NOT a validated measurement of human perception -- a statistical proxy.
    Treat the returned step as a starting point to verify by eye.
    """
    sensor.set_option(option, base_value)
    time.sleep(0.15)
    frames = None
    for _ in range(5):
        frames = pipe.wait_for_frames()
    baseline_frame = frames.get_color_frame() if stream == "color" else frames.get_infrared_frame(1)
    if not baseline_frame:
        return {"supported": False, "reason": f"{stream} stream not available"}
    baseline_img = np.asanyarray(baseline_frame.get_data()).astype(np.float32)

    step_size = abs(extreme_value - base_value) / step_count
    result = {"supported": True, "min_perceptible_step": None, "threshold_used": threshold,
              "diff_metric": diff_metric}

    for n in range(1, step_count + 1):
        trial_value = clamp_value(sensor, option, base_value + direction * step_size * n)
        sensor.set_option(option, trial_value)
        time.sleep(0.15)
        for _ in range(5):
            frames = pipe.wait_for_frames()
            display_frame(frames, colorizer, [f"Perceptibility probe: step {n}/{step_count}"])
        test_frame = frames.get_color_frame() if stream == "color" else frames.get_infrared_frame(1)
        if not test_frame:
            continue
        test_img = np.asanyarray(test_frame.get_data()).astype(np.float32)

        if diff_metric == "color_shift" and test_img.ndim == 3:
            base_bR = baseline_img[:, :, 2] - baseline_img[:, :, 0]
            test_bR = test_img[:, :, 2] - test_img[:, :, 0]
            delta = float(np.mean(np.abs(test_bR - base_bR)))
        else:
            delta = float(np.mean(np.abs(test_img - baseline_img)))

        if delta >= threshold:
            result["min_perceptible_step"] = round(abs(trial_value - base_value), 2)
            result["measured_delta"] = round(delta, 3)
            break

    sensor.set_option(option, base_value)
    if result["min_perceptible_step"] is None:
        result["note"] = f"No step up to {extreme_value} crossed the threshold"
    return result


def main():
    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)  # needed for JND probe on depth params

    print("[*] Starting Intel RealSense D456 (848x480 @ 30 FPS)...")
    profile = pipe.start(cfg)
    device = profile.get_device()

    depth_sensor = device.first_depth_sensor()
    color_sensor = device.first_color_sensor()
    colorizer = rs.colorizer()

    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
    if color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

    benchmark_data = {
        "device": device.get_info(rs.camera_info.name),
        "firmware": device.get_info(rs.camera_info.firmware_version),
        "resolution": "848x480",
        "fps": 30,
        "individual_tests": {},
        "bundled_tests": {},
        "perceptibility_tests": {}
    }

    try:
        cv2.namedWindow("Intel RealSense D456 - Comprehensive Benchmark Suite", cv2.WINDOW_AUTOSIZE)

        # -------------------------------------------------------------
        # 1. INDIVIDUAL DYNAMIC PARAMETERS
        # -------------------------------------------------------------
        single_params = [
            ("Depth Laser Power", depth_sensor, rs.option.laser_power, 0, 240,
             "frame_laser_power", True),
            ("Depth Exposure", depth_sensor, rs.option.exposure, 6000, 25000,
             "actual_exposure", True),
            ("Depth Gain", depth_sensor, rs.option.gain, 16, 64,
             "gain_level", True),
            ("RGB Exposure", color_sensor, rs.option.exposure, 80, 300,
             "actual_exposure", False),
            ("RGB Gain", color_sensor, rs.option.gain, 16, 64,
             "gain_level", False),
            ("RGB White Balance", color_sensor, rs.option.white_balance, 3000, 5500,
             "white_balance", False)
        ]

        print("\n" + "=" * 65)
        print(">>> PHASE 1: INDIVIDUAL PARAMETER BENCHMARK")
        print("=" * 65)

        for name, sensor, opt, val_low, val_high, meta_field_name, is_depth in single_params:
            if not sensor.supports(opt):
                continue
            val_low = clamp_value(sensor, opt, val_low)
            val_high = clamp_value(sensor, opt, val_high)
            is_rgb_only = sensor is color_sensor

            def task(s=sensor, o=opt, v=val_high): s.set_option(o, v)
            def reset(s=sensor, o=opt, v=val_low): s.set_option(o, v)
            def toggle(state, s=sensor, o=opt, v_h=val_high, v_l=val_low):
                s.set_option(o, v_h if state else v_l)

            print(f"\n[*] Running: {name}  |  Parameters changed: {name}")

            cmd_ms = evaluate_usb_cmd_latency(pipe, colorizer, task, reset, name, [name])
            if cmd_ms is None: return

            meta_field = safe_metadata_field(meta_field_name)
            # RGB exposure needs unit conversion (option units are ~100us
            # UVC steps, metadata reports true microseconds); everything
            # else compares directly.
            if meta_field_name == "actual_exposure" and not is_depth:
                meta_target = rgb_exposure_to_metadata_units(val_high)
            else:
                meta_target = val_high
            tol = max(1, round(0.02 * meta_target))  # 2% tolerance band
            settling = evaluate_hardware_settling(
                pipe, colorizer, task, reset, name,
                checks=[{"field": meta_field, "target": meta_target, "tol": tol, "depth": is_depth}]
            )
            if settling is None: return

            rate_data = evaluate_update_frequency(pipe, colorizer, toggle, name, [name], is_rgb_only=is_rgb_only)
            if rate_data is None: return

            benchmark_data["individual_tests"][name] = {
                "parameters_changed": [name],
                "usb_cmd_latency_ms": round(cmd_ms, 2),
                "hardware_settling": settling,
                "frequency_stress_test": rate_data
            }

        # -------------------------------------------------------------
        # 2. COMMONLY BUNDLED DYNAMIC PARAMETERS
        # -------------------------------------------------------------
        print("\n" + "=" * 65)
        print(">>> PHASE 2: BUNDLED PARAMETERS BENCHMARK")
        print("=" * 65)

        d_exp_hi = clamp_value(depth_sensor, rs.option.exposure, 25000)
        d_exp_lo = clamp_value(depth_sensor, rs.option.exposure, 6000)
        d_gain_hi = clamp_value(depth_sensor, rs.option.gain, 64)
        d_gain_lo = clamp_value(depth_sensor, rs.option.gain, 16)

        d_exp2_hi = clamp_value(depth_sensor, rs.option.exposure, 15000)
        d_exp2_lo = clamp_value(depth_sensor, rs.option.exposure, 8000)
        laser_hi = clamp_value(depth_sensor, rs.option.laser_power, 240)
        laser_lo = clamp_value(depth_sensor, rs.option.laser_power, 0)

        d_exp3_hi = clamp_value(depth_sensor, rs.option.exposure, 22000)
        d_exp3_lo = clamp_value(depth_sensor, rs.option.exposure, 7000)
        c_exp3_hi = clamp_value(color_sensor, rs.option.exposure, 280)
        c_exp3_lo = clamp_value(color_sensor, rs.option.exposure, 90)

        c_exp4_hi = clamp_value(color_sensor, rs.option.exposure, 280)
        c_exp4_lo = clamp_value(color_sensor, rs.option.exposure, 90)
        c_gain4_hi = clamp_value(color_sensor, rs.option.gain, 48)
        c_gain4_lo = clamp_value(color_sensor, rs.option.gain, 16)
        c_wb4_hi = clamp_value(color_sensor, rs.option.white_balance, 5200)
        c_wb4_lo = clamp_value(color_sensor, rs.option.white_balance, 3200)

        f_exposure = safe_metadata_field("actual_exposure")
        f_gain = safe_metadata_field("gain_level")
        f_wb = safe_metadata_field("white_balance")
        f_laser = safe_metadata_field("frame_laser_power")
        f_emitter = safe_metadata_field("frame_emitter_mode")

        # Pre-convert RGB exposure targets to metadata units (see
        # rgb_exposure_to_metadata_units) once, reused in the checks below.
        c_exp3_hi_meta = rgb_exposure_to_metadata_units(c_exp3_hi)
        c_exp4_hi_meta = rgb_exposure_to_metadata_units(c_exp4_hi)

        bundles = [
            {
                "name": "Bundle 1: Lighting Adaptation (Depth Exposure + Gain)",
                "params": ["Depth Exposure", "Depth Gain"],
                "is_rgb_only": False,
                "checks": [
                    {"field": f_exposure, "target": d_exp_hi, "tol": max(1, round(0.02 * d_exp_hi)), "depth": True},
                    {"field": f_gain, "target": d_gain_hi, "tol": 2, "depth": True},
                ],
                "task": lambda: (
                    depth_sensor.set_option(rs.option.exposure, d_exp_hi),
                    depth_sensor.set_option(rs.option.gain, d_gain_hi)
                ),
                "reset": lambda: (
                    depth_sensor.set_option(rs.option.exposure, d_exp_lo),
                    depth_sensor.set_option(rs.option.gain, d_gain_lo)
                ),
                "toggle": lambda state: (
                    depth_sensor.set_option(rs.option.exposure, d_exp_hi if state else d_exp_lo),
                    depth_sensor.set_option(rs.option.gain, d_gain_hi if state else d_gain_lo)
                )
            },
            {
                "name": "Bundle 2: Active/Passive Switching (Emitter + Laser + Exposure)",
                "params": ["Depth Emitter Enabled", "Depth Laser Power", "Depth Exposure"],
                "is_rgb_only": False,
                "checks": [
                    {"field": f_emitter, "target": 1, "tol": 0, "depth": True},
                    {"field": f_laser, "target": laser_hi, "tol": max(1, round(0.02 * laser_hi)), "depth": True},
                    {"field": f_exposure, "target": d_exp2_hi, "tol": max(1, round(0.02 * d_exp2_hi)), "depth": True},
                ],
                "task": lambda: (
                    depth_sensor.set_option(rs.option.emitter_enabled, 1),
                    depth_sensor.set_option(rs.option.laser_power, laser_hi),
                    depth_sensor.set_option(rs.option.exposure, d_exp2_hi)
                ),
                "reset": lambda: (
                    depth_sensor.set_option(rs.option.emitter_enabled, 0),
                    depth_sensor.set_option(rs.option.laser_power, laser_lo),
                    depth_sensor.set_option(rs.option.exposure, d_exp2_lo)
                ),
                "toggle": lambda state: (
                    depth_sensor.set_option(rs.option.emitter_enabled, 1 if state else 0),
                    depth_sensor.set_option(rs.option.laser_power, laser_hi if state else laser_lo),
                    depth_sensor.set_option(rs.option.exposure, d_exp2_hi if state else d_exp2_lo)
                )
            },
            {
                "name": "Bundle 3: Synced Exposure (Depth Exposure + RGB Exposure)",
                "params": ["Depth Exposure", "RGB Exposure"],
                "is_rgb_only": False,
                "checks": [
                    {"field": f_exposure, "target": d_exp3_hi, "tol": max(1, round(0.02 * d_exp3_hi)), "depth": True},
                    {"field": f_exposure, "target": c_exp3_hi_meta,
                     "tol": max(1, round(0.02 * c_exp3_hi_meta)), "depth": False},
                ],
                "task": lambda: (
                    depth_sensor.set_option(rs.option.exposure, d_exp3_hi),
                    color_sensor.set_option(rs.option.exposure, c_exp3_hi)
                ),
                "reset": lambda: (
                    depth_sensor.set_option(rs.option.exposure, d_exp3_lo),
                    color_sensor.set_option(rs.option.exposure, c_exp3_lo)
                ),
                "toggle": lambda state: (
                    depth_sensor.set_option(rs.option.exposure, d_exp3_hi if state else d_exp3_lo),
                    color_sensor.set_option(rs.option.exposure, c_exp3_hi if state else c_exp3_lo)
                )
            },
            {
                "name": "Bundle 4: RGB Tuning (Exposure + Gain + White Balance)",
                "params": ["RGB Exposure", "RGB Gain", "RGB White Balance"],
                "is_rgb_only": True,
                "checks": [
                    {"field": f_exposure, "target": c_exp4_hi_meta,
                     "tol": max(1, round(0.02 * c_exp4_hi_meta)), "depth": False},
                    {"field": f_gain, "target": c_gain4_hi, "tol": 2, "depth": False},
                    {"field": f_wb, "target": c_wb4_hi, "tol": max(1, round(0.02 * c_wb4_hi)), "depth": False},
                ],
                "task": lambda: (
                    color_sensor.set_option(rs.option.exposure, c_exp4_hi),
                    color_sensor.set_option(rs.option.gain, c_gain4_hi),
                    color_sensor.set_option(rs.option.white_balance, c_wb4_hi)
                ),
                "reset": lambda: (
                    color_sensor.set_option(rs.option.exposure, c_exp4_lo),
                    color_sensor.set_option(rs.option.gain, c_gain4_lo),
                    color_sensor.set_option(rs.option.white_balance, c_wb4_lo)
                ),
                "toggle": lambda state: (
                    color_sensor.set_option(rs.option.exposure, c_exp4_hi if state else c_exp4_lo),
                    color_sensor.set_option(rs.option.gain, c_gain4_hi if state else c_gain4_lo),
                    color_sensor.set_option(rs.option.white_balance, c_wb4_hi if state else c_wb4_lo)
                )
            }
        ]

        for b in bundles:
            print(f"\n[*] Running: {b['name']}  |  Parameters changed: {', '.join(b['params'])}")

            cmd_ms = evaluate_usb_cmd_latency(pipe, colorizer, b["task"], b["reset"], b["name"], b["params"])
            if cmd_ms is None: return

            settling = evaluate_hardware_settling(pipe, colorizer, b["task"], b["reset"], b["name"], b["checks"])
            if settling is None: return

            rate_data = evaluate_update_frequency(
                pipe, colorizer, b["toggle"], b["name"], b["params"], is_rgb_only=b["is_rgb_only"]
            )
            if rate_data is None: return

            benchmark_data["bundled_tests"][b["name"]] = {
                "parameters_changed": b["params"],
                "usb_total_cmd_latency_ms": round(cmd_ms, 2),
                "hardware_settling": settling,
                "frequency_stress_test": rate_data
            }

        # -------------------------------------------------------------
        # 3. MINIMUM PERCEPTIBLE STEP (JND proxy) PER PARAMETER
        # -------------------------------------------------------------
        print("\n" + "=" * 65)
        print(">>> PHASE 3: PERCEPTIBILITY THRESHOLD BENCHMARK (heuristic)")
        print("=" * 65)

        perceptibility_plan = [
            ("Depth Laser Power", depth_sensor, rs.option.laser_power, 0, 240, "infrared", "brightness"),
            ("Depth Exposure", depth_sensor, rs.option.exposure, 6000, 25000, "infrared", "brightness"),
            ("Depth Gain", depth_sensor, rs.option.gain, 16, 64, "infrared", "brightness"),
            ("RGB Exposure", color_sensor, rs.option.exposure, 80, 300, "color", "brightness"),
            ("RGB Gain", color_sensor, rs.option.gain, 16, 64, "color", "brightness"),
            ("RGB White Balance", color_sensor, rs.option.white_balance, 3000, 5500, "color", "color_shift"),
        ]

        for name, sensor, opt, val_low, val_high, stream, metric in perceptibility_plan:
            if not sensor.supports(opt):
                continue
            val_low = clamp_value(sensor, opt, val_low)
            val_high = clamp_value(sensor, opt, val_high)
            mid = (val_low + val_high) / 2.0

            print(f"\n[*] Probing minimum perceptible step for: {name}")
            result = find_perceptible_step(
                pipe, colorizer, sensor, opt,
                base_value=mid, direction=1, extreme_value=val_high,
                stream=stream, diff_metric=metric
            )
            if result is None: return
            benchmark_data["perceptibility_tests"][name] = result

        # Save structured log to file for later analysis
        with open("benchmark_report.json", "w") as f:
            json.dump(benchmark_data, f, indent=4)
        print("\n[*] Full benchmark data exported to: benchmark_report.json")

    except Exception as exc:
        print(f"\n[!] Benchmark error: {exc}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("[*] Pipeline stopped. Printing summary:\n")
        print(json.dumps(benchmark_data, indent=2))


if __name__ == "__main__":
    main()