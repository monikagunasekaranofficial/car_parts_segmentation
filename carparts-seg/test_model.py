
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

print("Model task:", model.task)
print("Classes:", model.names)

# Run prediction with a lower confidence threshold
results = model.predict(
    source=r"C:\Users\MONIKA\OneDrive\Desktop\carparts-seg\test_image\car.avif",
    imgsz=640,
    conf=0.05,
    save=True,
    show=True
)

# Check detections and segmentation masks
for result in results:

    if result.boxes is not None:
        print("Number of detections:", len(result.boxes))

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(
                "Part:", model.names[class_id],
                "| Confidence:", round(confidence, 3)
            )

    if result.masks is not None:
        print("Segmentation masks found:", len(result.masks))
    else:
        print("No segmentation masks found.")

print("Prediction completed!")
