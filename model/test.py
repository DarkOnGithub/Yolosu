from ultralytics import YOLO


model = YOLO(r"runs\detect\train2\weights\best.onnx")  


model.predict(
    source=r"C:\Users\darke\Documents\python\Yolosu\videos\Calamity Fortune_SCV's Lunatic.mp4",    
    conf=0.5,             
    save=True,             
    show=False,
    imgsz=416
)
