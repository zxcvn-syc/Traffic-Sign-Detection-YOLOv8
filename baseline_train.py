from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset.yaml",
    epochs=150,
    imgsz=640,
    batch=8,
    workers=2,
    optimizer="AdamW",
    device="cpu",
    project="runs",
    name="baseline"
)