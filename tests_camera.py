import cv2, time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Cannot open camera 0")

for i in range(60):
    ok, frame = cap.read()
    if not ok: 
        time.sleep(0.03); continue
    cv2.imshow("cam", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
