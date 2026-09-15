import pyrealsense2 as rs
import time

pipe = rs.pipeline()
cfg = rs.config()

# ۱. پارامترهای استاتیک (تعیین رزولوشن و نرخ فریم پیش از آغاز)
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

profile = pipe.start(cfg)

# دریافت ارجاع به سنسورها
depth_sensor = profile.get_device().first_depth_sensor()
color_sensor = profile.get_device().first_color_sensor()

try:
    frame_count = 0
    while frame_count < 200:
        frames = pipe.wait_for_frames()
        frame_count += 1

        # ۲. تغییر دینامیک پارامترها در حین استریم (مثلاً در فریم ۵۰)
        if frame_count == 50:
            # خاموش کردن نوردهی خودکار و تنظیم دستی Exposure و Laser Power
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.exposure, 12000)  # مقدار بر حسب میکروثانیه
            depth_sensor.set_option(rs.option.laser_power, 240) # بازه 0 تا 360 میلی‌وات

        # ۳. تغییر رزولوشن نیازمند توقف و راه‌اندازی مجدد است:
        if frame_count == 100:
            pipe.stop()
            cfg.disable_all_streams()
            cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
            pipe.start(cfg)

finally:
    pipe.stop()