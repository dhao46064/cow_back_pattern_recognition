import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm  # 可选，用于显示进度条


def rotate_vertical_to_horizontal(image_path, output_path, backup_original=True, backup_dir="original_backup"):
    """
    将竖版图片（高>宽）旋转为横版，横版图片保持不变
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        backup_original: 是否备份原始图片
        backup_dir: 原始图片备份目录
    Returns:
        bool: 处理是否成功
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图片 {image_path}，跳过")
            return False

        # 获取图片尺寸（高, 宽, 通道）
        h, w = img.shape[:2]

        # 备份原始图片（可选）
        if backup_original:
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, os.path.basename(image_path))
            if not os.path.exists(backup_path):
                shutil.copy2(image_path, backup_path)

        # 判断是否为竖版（高>宽）
        if h > w:
            # 旋转90度（顺时针），变为横版（宽>高）
            # 旋转参数：旋转矩阵、旋转角度、缩放比例
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # 保存旋转后的图片
            cv2.imwrite(output_path, img_rotated)
            print(f"已旋转竖版图片：{image_path} → {output_path}")
        else:
            # 横版图片直接复制
            shutil.copy2(image_path, output_path)
            print(f"横版图片无需旋转：{image_path} → {output_path}")

        return True

    except Exception as e:
        print(f"处理图片 {image_path} 时出错：{str(e)}")
        return False


def batch_process_images(input_dir, output_dir, backup_original=True):
    """
    批量处理目录下的所有图片，将竖版转为横版
    Args:
        input_dir: 输入目录（包含横竖混合图片）
        output_dir: 输出目录（所有图片为横版）
        backup_original: 是否备份原始图片
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # 获取所有图片路径
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).glob(f"**/*{ext}"))
        image_paths.extend(Path(input_dir).glob(f"**/*{ext.upper()}"))

    if not image_paths:
        print(f"在 {input_dir} 中未找到任何图片")
        return

    # 批量处理
    success_count = 0
    fail_count = 0

    print(f"\n开始处理 {len(image_paths)} 张图片...")
    for img_path in tqdm(image_paths, desc="处理进度"):
        # 构建输出路径（保持原目录结构）
        relative_path = os.path.relpath(img_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        # 创建输出子目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 处理单张图片
        if rotate_vertical_to_horizontal(str(img_path), output_path, backup_original):
            success_count += 1
        else:
            fail_count += 1

    # 输出统计结果
    print(f"\n处理完成！")
    print(f"成功处理：{success_count} 张")
    print(f"处理失败：{fail_count} 张")
    print(f"输出目录：{output_dir}")
    if backup_original:
        print(f"原始图片备份目录：original_backup")


if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = r"D:\code_project\code_cp\dataset\obb_supervise"  # 替换为你的数据集目录
    OUTPUT_DIR = "D:\code_project\code_cp\dataset_obb_ori "  # 替换为输出目录
    BACKUP_ORIGINAL = True  # 是否备份原始图片

    # 批量处理
    batch_process_images(INPUT_DIR, OUTPUT_DIR, BACKUP_ORIGINAL)