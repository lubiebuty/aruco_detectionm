from ultralytics import YOLO
import cv2

# Ścieżki do pliku modelu i wideo
model_path = "runs/detect/train7/weights/best.pt"
video_path = "/Users/bartlomiejostasz/lot/n/3klatki/IMG_4699.MOV"

# Załaduj wytrenowany model
model = YOLO(model_path)


# Uruchom śledzenie z BoT‑SORT
for results in model.track(
        source=video_path,
        conf=0.25,          # niższy próg – mniej zgubionych klatek
        iou=0.5,            # odrzucaj mocno nakładające się boksy
        device="mps",       # Apple Silicon GPU; na PC → "0"
        tracker="botsort.yaml",
        persist=True,       # zachowuj ID gdy klatka wypadnie
        stream=True):       # zwracaj klatka‑po‑klatce
    cv2.imshow("Wykrywanie markerów", results.plot())
    if cv2.waitKey(1) == 27:  # ESC → wyjście
        break

cv2.destroyAllWindows()