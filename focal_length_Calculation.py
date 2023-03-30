pixel_size_mm = 1.85 / 1000

sensor_width_mm = 4096 * pixel_size_mm
sensor_height_mm = 2160 * pixel_size_mm

print('sensor height mm:', sensor_height_mm)
print('sensor width mm:', sensor_width_mm)

sensor_width_px = 4096
sensor_height_px = 2160
image_width_px = 1920
image_height_px = 1080


width_ratio = sensor_width_px/image_width_px
height_ratio = sensor_height_px/image_height_px

effective_pixel_width = pixel_size_mm * width_ratio
effective_pixel_height = pixel_size_mm * height_ratio


dx_pixels = 407
dy_pixels = 273
dx_CB = 9
dy_CB = 6
dZ = 431.8

fx = (dx_pixels/dx_CB)*dZ
fy = (dy_pixels/dy_CB)*dZ

fx_mm_L = fx * effective_pixel_width
fy_mm_L = fy * effective_pixel_height
print(fy)
print(fx)

print("Left camera focal lengths (fx_mm, fy_mm):", fx_mm_L, fy_mm_L)

fx_calc = 120
fy_calc = 107

new_x = fx_calc/ effective_pixel_width
new_y = fy_calc/ effective_pixel_height

print('new x:', new_x)
print('new y:', new_y)