
import os
import uuid
from pathlib import Path

import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO


# ==========================================
# 1. INITIALIZE FLASK APPLICATION
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

MODEL_PATH = BASE_DIR / "best (1).pt"

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Allowed image formats
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Create folders automatically
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD TRAINED YOLO26 MODEL
# ==========================================

print("Loading YOLO26 segmentation model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Place best.pt inside the project folder."
    )

model = YOLO(str(MODEL_PATH))

print("YOLO26 model loaded successfully!")
print("Model task:", model.task)
print("Available classes:", model.names)


# ==========================================
# 3. CHECK ALLOWED IMAGE FORMATS
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ==========================================
# 4. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 5. IMAGE UPLOAD AND PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check whether an image was uploaded
        if "image" not in request.files:

            return render_template(
                "index.html",
                error="No image was uploaded."
            )

        file = request.files["image"]

        # Check whether a file was selected
        if file.filename == "":

            return render_template(
                "index.html",
                error="Please select a car image."
            )

        # Validate image extension
        if not allowed_file(file.filename):

            return render_template(
                "index.html",
                error="Please upload JPG, JPEG, PNG or WEBP images."
            )

        # Generate a unique filename
        original_name = secure_filename(file.filename)

        extension = original_name.rsplit(".", 1)[1].lower()

        unique_name = f"{uuid.uuid4().hex}.{extension}"

        upload_path = UPLOAD_FOLDER / unique_name

        # Save uploaded image
        file.save(str(upload_path))

        print("Image uploaded:", unique_name)

        # ==================================
        # 6. RUN YOLO26 SEGMENTATION
        # ==================================

        print("Running YOLO26 prediction...")

        results = model.predict(
            source=str(upload_path),
            imgsz=640,
            conf=0.05,
            save=False,
            verbose=False
        )

        result = results[0]

        # ==================================
        # 7. GENERATE SEGMENTATION IMAGE
        # ==================================

        # plot() draws bounding boxes, labels and
        # segmentation masks when available.
        annotated_image = result.plot()

        result_filename = f"result_{uuid.uuid4().hex}.jpg"

        result_path = RESULT_FOLDER / result_filename

        # Save annotated image
        saved = cv2.imwrite(
            str(result_path),
            annotated_image
        )

        if not saved:
            raise RuntimeError("Could not save the prediction image.")

        # ==================================
        # 8. EXTRACT DETECTED CAR PARTS
        # ==================================

        detected_parts = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0].item())

                confidence = float(box.conf[0].item())

                class_name = result.names[class_id]

                detected_parts.append({
                    "name": class_name,
                    "confidence": confidence
                })

        # Count segmentation masks
        mask_count = (
            len(result.masks)
            if result.masks is not None
            else 0
        )

        print("Prediction completed!")
        print("Detected parts:", len(detected_parts))
        print("Segmentation masks:", mask_count)

        # ==================================
        # 9. SEND RESULTS TO HTML PAGE
        # ==================================

        return render_template(
            "index.html",
            result_image=f"results/{result_filename}",
            detected_parts=detected_parts,
            mask_count=mask_count,
            message="Image segmented successfully!"
        )

    except Exception as e:

        print("Prediction error:", str(e))

        return render_template(
            "index.html",
            error=f"An error occurred during prediction: {str(e)}"
        )


# ==========================================
# 10. HANDLE LARGE FILE UPLOADS
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        error="Image is too large. Maximum allowed size is 10 MB."
    ), 413


# ==========================================
# 11. START FLASK SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
