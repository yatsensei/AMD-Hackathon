from ultralytics import YOLO
model = YOLO("runs/classify/runs/classify/wafer/weights/best.pt")
model.predict("/Users/shefalisingh/Desktop/wafer-agent/yolo_wafer/images/test/w002246.png", save=True, project="newTest_result", name="run", exist_ok=True)