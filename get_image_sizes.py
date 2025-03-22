import os
from PIL import Image

# 替换为你自己的文件夹路径
folder_path = "/home/UNT/yz0370/projects/GiT/data/nuscenes/samples/CAM_BACK"

# 定义支持的图片扩展名
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

for filename in os.listdir(folder_path):
    # 根据文件后缀名粗略判断是否是图片
    extension = os.path.splitext(filename)[1].lower()
    if extension in valid_extensions:
        file_path = os.path.join(folder_path, filename)
        with Image.open(file_path) as img:
            width, height = img.size
            print(f"文件名：{filename}, 分辨率：{width}x{height}")
