import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("[!] هیچ دوربینی شناسایی نشد. اتصالات یا مجوزهای udev را بررسی کنید.")
else:
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"[*] دوربین شناسایی شد: {name} | نوع پورت ارتباطی: USB {usb}")

        print("\n[*] رزولوشن‌ها و فریم‌ریت‌های مجاز سنسور عمق:")
        depth_sensor = dev.first_depth_sensor()
        for p in depth_sensor.get_stream_profiles():
            if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
                vp = p.as_video_stream_profile()
                print(f"    - {vp.width()}x{vp.height()} @ {vp.fps()} FPS")