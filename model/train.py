from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"runs\detect\train2\weights\best.pt")
    
    model.export(format="engine")    