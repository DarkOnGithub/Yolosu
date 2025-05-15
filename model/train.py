from ultralytics import YOLO




if __name__ == "__main__":
    model = YOLO(r"runs\detect\train6\weights\best.pt")
    # model.train(data="dataset_yolo_export/dataset.yaml", epochs=100, batch=64, imgsz=416, device=0, workers=4)
    model.export(format="engine", opset=12, dynamic=False)    