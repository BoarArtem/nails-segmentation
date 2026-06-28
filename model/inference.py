import cv2
from ultralytics import YOLO

def realtime_inference():
    model = YOLO("../runs/segment/train-4/weights/best.pt")

    cap = cv2.VideoCapture(0)

    while True:
        res, frame = cap.read()

        if not res:
            break

        results = model(frame)

        annotated_frame = results[0].plot(boxes=False, labels=False)

        cv2.imshow("Nail Segmentation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def photo_inference():
    model = YOLO("../runs/segment/train-4/weights/best.pt")

    result = model("../img/a0efd1eb8110f2bdd88ebb34f0758a53.png")
    annotated_frame = result[0].plot(boxes=False, labels=False)

    cv2.imwrite("../result.png", annotated_frame)

if __name__ == "__main__":
    # realtime_inference()
    photo_inference()