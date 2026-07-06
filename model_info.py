from ultralytics import YOLO

model = YOLO("runs/detect/runs/baseline-2/weights/best.pt")

model.info(detailed=True)