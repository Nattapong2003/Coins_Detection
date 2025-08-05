import cv2
from ultralytics import YOLO

model = YOLO("../Coins_detection/runs/detect/train4/weights/best.pt")

coin_value = {

    "1baht": 1,
    "5baht": 5,
    "10baht": 10,
    "20baht": 20
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))  # ลดขนาดก่อนวิเคราะห์
    results = model(frame)[0]

    total_value = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label in coin_value and conf > 0.5:
            total_value += coin_value[label]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} ({coin_value.get(label, 'Unknown')} B)"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        text = f"{label} ({coin_value.get(label, 'Unknown')} B)"

        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        cv2.putText(frame, f"Total: {total_value} bath",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        
        cv2.imshow("Coin Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
