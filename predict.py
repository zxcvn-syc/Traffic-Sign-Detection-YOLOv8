from ultralytics import YOLO

model = YOLO("runs/detect/runs/baseline/weights/best.pt")

model.predict(
    source="output/images/test",
    save=True,
    conf=0.25
)