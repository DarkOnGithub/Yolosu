from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolov8n.pt")
    model.train(data="dataset_yolo_test_export/dataset.yaml", epochs=70, batch=48, imgsz=416, device=0)
    model.export(format="engine")    