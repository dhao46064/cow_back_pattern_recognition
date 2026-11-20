import cv2
import numpy as np
import torch
import shutil
import csv
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ===================== 配置 =====================
MODEL_PATH = r"D:\code_project\cow_back_vision_y-s\runs\segment\runs\segment\cow_back_seg_7.30\weights\best.pt"

INPUT_ROOT_DIR = r"D:\code_project\triplet_code\新建文件夹"
OUTPUT_ROOT_DIR = r"D:\code_project\triplet_code\新建文件夹\result"
FAIL_CSV_PATH = r"D:\code_project\triplet_code\新建文件夹\dataset_OBB3_failed_images.csv"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.6
IMG_SIZE = 640
DEVICE = "0" if torch.cuda.is_available() else "cpu"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_NORMAL_FAIL = 10


def bbox_crop_by_mask(img, mask_resized):
    """
    不旋转：
    只根据 mask 的外接正矩形 bbox 裁剪牛背区域
    """
    contours, _ = cv2.findContours(
        mask_resized,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    all_contours = np.vstack(contours)

    x, y, w, h = cv2.boundingRect(all_contours)

    if w <= 0 or h <= 0:
        return None

    cropped = img[y:y + h, x:x + w]

    if cropped is None or cropped.size == 0:
        return None

    return cropped


def read_image_chinese_path(img_path):
    return cv2.imdecode(
        np.fromfile(str(img_path), dtype=np.uint8),
        cv2.IMREAD_COLOR
    )


def save_image_chinese_path(save_path, img):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ok, encoded = cv2.imencode(".jpg", img)

    if not ok:
        return False

    encoded.tofile(str(save_path))
    return True


def process_single_image(model, img_path, output_path):
    img = read_image_chinese_path(img_path)

    if img is None:
        return False

    results = model(
        img,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False,
        retina_masks=True,
        half=True if torch.cuda.is_available() else False
    )

    result = results[0]

    if result.masks is None:
        return False

    masks = result.masks.data

    # 取面积最大的 mask
    areas = [m.cpu().numpy().sum() for m in masks]
    best_idx = int(np.argmax(areas))

    mask = masks[best_idx]
    mask_np = mask.cpu().numpy().astype(np.uint8)

    mask_resized = cv2.resize(
        mask_np,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    cropped_img = bbox_crop_by_mask(img, mask_resized)

    if cropped_img is None or cropped_img.size == 0:
        return False

    return save_image_chinese_path(output_path, cropped_img)


def write_fail(csv_writer, cow_id, subset, img_path):
    csv_writer.writerow([
        cow_id,
        subset,
        img_path.name,
        str(img_path)
    ])


def main():
    input_root = Path(INPUT_ROOT_DIR)
    output_root = Path(OUTPUT_ROOT_DIR)
    fail_csv_path = Path(FAIL_CSV_PATH)

    fail_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("YOLO-seg + bbox裁剪处理：不旋转版本")
    print(f"输入数据集: {input_root}")
    print(f"输出数据集: {output_root}")
    print(f"失败记录CSV: {fail_csv_path}")
    print(f"normal失败阈值: > {MAX_NORMAL_FAIL} 张则删除该编号")
    print("=" * 70)

    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available():
        model = model.cuda()

    cow_dirs = [d for d in input_root.iterdir() if d.is_dir()]

    print(f"检测到牛编号数量: {len(cow_dirs)}")

    total_all = 0
    success_all = 0
    fail_all = 0
    deleted_count = 0

    with open(fail_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["cow_id", "subset", "image_name", "image_path"])

        for cow_dir in cow_dirs:
            cow_id = cow_dir.name
            cow_output_dir = output_root / cow_id

            print(f"\n{'=' * 60}")
            print(f"开始处理牛只: {cow_id}")
            print(f"{'=' * 60}")

            cow_failed = False
            normal_fail_count = 0

            for subset in ["normal", "abnormal"]:

                if cow_failed:
                    break

                input_subset_dir = cow_dir / subset

                if not input_subset_dir.exists():
                    continue

                output_subset_dir = cow_output_dir / subset
                output_subset_dir.mkdir(parents=True, exist_ok=True)

                img_paths = [
                    p for p in input_subset_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXTS
                ]

                if len(img_paths) == 0:
                    continue

                print(f"\n处理牛只: {cow_id} | 类别: {subset} | 图片数: {len(img_paths)}")

                success_count = 0
                fail_count = 0

                for img_path in tqdm(img_paths, desc=f"{cow_id}-{subset}"):

                    output_path = output_subset_dir / f"{img_path.stem}.jpg"

                    ok = process_single_image(
                        model=model,
                        img_path=img_path,
                        output_path=output_path
                    )

                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                        fail_all += 1

                        write_fail(
                            csv_writer=writer,
                            cow_id=cow_id,
                            subset=subset,
                            img_path=img_path
                        )

                        if subset == "normal":
                            normal_fail_count += 1

                            print(
                                f"[normal失败] {cow_id} | "
                                f"失败数: {normal_fail_count} | "
                                f"{img_path.name}"
                            )

                            if normal_fail_count > MAX_NORMAL_FAIL:
                                cow_failed = True
                                deleted_count += 1

                                print(
                                    f"[删除编号] {cow_id}: "
                                    f"normal失败超过 {MAX_NORMAL_FAIL} 张"
                                )

                                if cow_output_dir.exists():
                                    shutil.rmtree(cow_output_dir)

                                break

                if cow_failed:
                    break

                total_all += len(img_paths)
                success_all += success_count

                print(
                    f"完成 {cow_id}/{subset}: "
                    f"成功 {success_count}, "
                    f"失败 {fail_count}, "
                    f"总数 {len(img_paths)}"
                )

            if cow_failed:
                print(f"[跳过] {cow_id}: 已删除，不处理 abnormal")
                continue

    print("\n" + "=" * 70)
    print("全部处理完成")
    print(f"总图片数: {total_all}")
    print(f"成功数: {success_all}")
    print(f"失败数: {fail_all}")
    print(f"删除编号数量: {deleted_count}")
    print(f"失败记录CSV: {fail_csv_path}")
    print(f"输出目录: {output_root}")
    print("=" * 70)


if __name__ == "__main__":
    if not Path(INPUT_ROOT_DIR).exists():
        print(f"输入目录不存在: {INPUT_ROOT_DIR}")
    else:
        main()