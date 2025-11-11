from ultralytics import YOLO
import cv2

model = YOLO("yolo12x.pt")

video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Nem sikerült megnyitni a videót!")
    exit()

vehicle_classes = {"car", "truck", "bus", "motorbike", "bicycle"}

cv2.namedWindow("Jarmu-szamlalo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Jarmu-szamlalo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.resizeWindow("Jarmu-szamlalo", 1920, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    counts = {name: 0 for name in vehicle_classes}

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, cls_id in zip(boxes, cls_ids):
            x1, y1, x2, y2 = box.astype(int)
            class_name = model.names[cls_id]

            if class_name not in vehicle_classes:
                continue

            counts[class_name] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    total_vehicles = sum(counts.values())

    cv2.putText(frame, f"Jarmuvek szama (aktualis kepen): {total_vehicles}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    y_text = 80
    for name in sorted(vehicle_classes):
        if counts[name] > 0:
            cv2.putText(frame, f"{name}: {counts[name]}",
                        (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_text += 30

    cv2.imshow("Jarmu-szamlalo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()