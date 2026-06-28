from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26l-seg.pt")

    model.train(data="dataset/data.yaml", epochs=50)