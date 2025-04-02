import os
import shutil

def clear_directory(dir_path):
    """
    清理指定目录下的所有文件和文件夹。
    
    参数:
        dir_path: 目标目录的路径（字符串）
    """
    if not os.path.exists(dir_path):
        print(f"目录 {dir_path} 不存在。")
        return

    # 遍历目录下的所有文件和子目录
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或链接
                print(f"已删除文件: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录及其所有内容
                print(f"已删除目录: {file_path}")
        except Exception as e:
            print(f"删除 {file_path} 时出现错误: {e}")

if __name__ == "__main__":
    
    target_dir = "/home/UNT/yz0370/projects/GiT/visualization/all_6_img"
    clear_directory(target_dir)
    target_dir = "/home/UNT/yz0370/projects/GiT/visualization/generateoccflow"
    clear_directory(target_dir)
    target_dir = "/home/UNT/yz0370/projects/GiT/visualization/occ_2d_box"
    clear_directory(target_dir)
    target_dir = "/home/UNT/yz0370/projects/GiT/visualization/resize_img"
    clear_directory(target_dir)
    target_dir = "/home/UNT/yz0370/projects/GiT/visualization/results_occ"
    clear_directory(target_dir)