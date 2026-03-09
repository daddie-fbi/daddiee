from ultralytics import YOLO

# Load YOLOv8 model (downloads on first run)
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("✓ Model loaded successfully")

# Change this to your image, video, or 0 for webcam
source = r"Cars Moving On Road Footage.mp4"

print(f"\nProcessing: {source}")

# Detect images, track videos automatically
if source == 0 or source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
    print("Mode: Detection (Image)")
    model.predict(source=source, conf=0.35, show=True, save=True)
else:
    print("Mode: Tracking (Video)")
    model.track(source=source, tracker="bytetrack.yaml", conf=0.35, show=True, save=True)

print("\n✓ Done! Check runs/ folder for outputs.")
print("Output saved to: runs/detect/predict/ or runs/track/predict/")
