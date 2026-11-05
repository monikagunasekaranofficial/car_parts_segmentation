
""" 
from ultralytics import YOLO

# Load your trained model
model = YOLO("best (1).pt")

# Validate using your dataset
metrics = model.val(
    data="C:\\Users\\MONIKA\\OneDrive\\Desktop\\carparts-seg\\carparts-seg.yaml",
    split="tes",
    imgsz=640,
    batch=1,
    device="cpu"
)

# Detection metrics
print("\n========== BOX METRICS ==========")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("Box mAP50:", metrics.box.map50)
print("Box mAP50-95:", metrics.box.map)

# Segmentation metrics
print("\n======= MASK METRICS ========")
print("Mask mAP50:", metrics.seg.map50)
print("Mask mAP50-95:", metrics.seg.map)

print("\nEvaluation completed!")

"""


from ultralytics import YOLO
from pathlib import Path

# ----------------------------------
# 1. Project paths
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best (1).pt"

DATA_YAML = (
    BASE_DIR
    / "datasets"
    / "carparts-seg"
    / "carparts-seg.yaml"
)


# ----------------------------------
# 2. Evaluate the model
# ----------------------------------

def evaluate_model():

    print("\nLoading YOLO26 segmentation model...")

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    if not DATA_YAML.exists():
        print(f"Dataset YAML not found: {DATA_YAML}")
        return

    model = YOLO(str(MODEL_PATH))

    print("\nModel loaded successfully!")
    print("Model classes:", model.names)

    print("\nStarting validation...")
    print("EVALUATION SPLIT:", "test")
    print("DATASET YAML:", DATA_YAML)
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=1,
        device="cpu",
        workers=0,
        plots=True
    )

    # ----------------------------------
    # 3. Display box metrics
    # ----------------------------------

    print("\n========== BOX METRICS ==========")

    print(f"Precision:       {metrics.box.mp:.4f}")
    print(f"Recall:          {metrics.box.mr:.4f}")
    print(f"Box mAP@50:      {metrics.box.map50:.4f}")
    print(f"Box mAP@50-95:   {metrics.box.map:.4f}")

    # ----------------------------------
    # 4. Display segmentation metrics
    # ----------------------------------

    print("\n========== MASK METRICS ==========")

    print(f"Precision:       {metrics.seg.mp:.4f}")
    print(f"Recall:          {metrics.seg.mr:.4f}")
    print(f"Mask mAP@50:     {metrics.seg.map50:.4f}")
    print(f"Mask mAP@50-95:  {metrics.seg.map:.4f}")

    # ----------------------------------
    # 5. Evaluation completed
    # ----------------------------------

    print("\nEvaluation completed successfully!")

    print("\nResults saved in the runs/segment folder.")


# ----------------------------------
# 6. Windows entry point
# ----------------------------------

if __name__ == "__main__":
    evaluate_model()
