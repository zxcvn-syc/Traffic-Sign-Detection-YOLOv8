from ultralytics import YOLO
import time
from pathlib import Path

# 加载训练好的最佳模型
model = YOLO("runs/detect/runs/baseline-2/weights/best.pt")

# 指定验证集路径
val_images_path = Path("output/images/val")  # 使用你的原始验证集
val_images = list(val_images_path.glob("*.*"))  # 匹配所有图片

# 如果没有找到图片，提醒
if len(val_images) == 0:
    raise FileNotFoundError(f"No images found in {val_images_path}")

# 测试FPS
start = time.time()

# 循环预测所有验证集图片
for img_path in val_images:
    model.predict(
        source=str(img_path),
        imgsz=640,
        verbose=False
    )

end = time.time()

fps = len(val_images) / (end - start)

print(f"FPS on val set ({len(val_images)} images) = {fps:.2f}")