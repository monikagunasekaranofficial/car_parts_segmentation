# YOLO26 Car Parts Instance Segmentation Web Application

## 1. Project Overview

This project is a computer vision-based web application that uses a custom-trained YOLO26 model to detect and segment automobile parts.

Users can upload a car image through a web interface and receive an annotated image containing colored segmentation masks, detected part names, and confidence scores.

The application is developed using Python and Flask, with HTML, CSS, and JavaScript for the frontend.

## 2. Project Objectives

- Detect different automobile components.
- Generate pixel-level segmentation masks.
- Display detected car-part names and confidence scores.
- Develop a user-friendly image upload interface.
- Evaluate model performance using precision, recall, and mAP metrics.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| YOLO26 | Instance segmentation |
| PyTorch | Deep learning framework |
| Ultralytics | Model training and inference |
| OpenCV | Image processing |
| Flask | Backend development |
| HTML | Webpage structure |
| CSS | Webpage styling |
| JavaScript | Image preview |
| Google Colab | Model training |
| VS Code | Development environment |

## 4. Dataset Information

The project uses the Carparts-seg dataset.

- Training images: 3,156
- Validation images: 401
- Training epochs: 100
- Image size: 640 × 640
- Number of classes: 23
- Model: YOLO26n-seg

### Car Parts Classes

The model recognizes 23 classes, including:

- Front bumper
- Back bumper
- Hood
- Front door
- Back door
- Front glass
- Back glass
- Wheels
- Left mirror
- Right mirror
- Headlights
- Tail lights
- Tailgate
- Trunk

## 5. Model Evaluation

The selected trained checkpoint is `best (1).pt`.

Reported evaluation metrics:

| Metric | Box Detection | Mask Segmentation |
|---|---:|---:|
| Precision | 58.97% | 60.33% |
| Recall | 75.09% | 75.78% |
| mAP@50 | 67.51% | 68.48% |
| mAP@50–95 | 56.56% | 55.64% |

These results should be identified as test metrics only after confirming that the test dataset split was used.

## 6. Project Structure

```text
carparts-seg/
│
├── datasets/
│   └── carparts-seg/
│
├── runs/
│   └── segment/
│
├── static/
│   ├── uploads/
│   ├── results/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── test_image/
├── app.py
├── best (1).pt
├── best.pt
├── evaluate_model.py
├── test_model.py
├── requirements.txt
└── README.md
```

## 7. Application Workflow

1. User opens the Flask web application.
2. User uploads a car image.
3. Flask receives and saves the uploaded image.
4. YOLO26 processes the image.
5. The model detects car parts and generates segmentation masks.
6. OpenCV saves the annotated image.
7. Flask extracts class names and confidence scores.
8. Results are sent to the HTML frontend.
9. The user views and downloads the segmented image.

## 8. Backend and Frontend

**Flask Backend (`app.py`)**

- Initializes the Flask application.
- Loads the trained YOLO26 model.
- Handles uploaded images.
- Performs model inference.
- Generates annotated images.
- Extracts detected classes and confidence scores.
- Returns results to the frontend.

**HTML Frontend (`index.html`)**

- Displays the application interface.
- Provides an image upload option.
- Displays image previews.
- Shows segmentation results.
- Displays detected car parts and confidence scores.
- Provides a download option.

**CSS (`style.css`)**

Controls the webpage layout, colors, buttons, typography, and responsive design.

## 9. Installation and Execution

### Step 1: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 2: Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
python -m pip install flask ultralytics opencv-python
```

Install a compatible PyTorch version if it is not already installed.

### Step 4: Run the application

```powershell
python app.py
```

### Step 5: Open in browser

```text
http://127.0.0.1:5000/
```

Upload a car image to view its segmentation results.

## 10. Future Improvements

- Improve model performance through dataset cleaning and additional training.
- Improve segmentation accuracy.
- Optimize inference speed.
- Improve the web interface.
- Add more visualization features.
- Deploy the application to a hosting platform.

## 11. Conclusion

This project demonstrates the integration of deep learning and web development for automobile part instance segmentation.

The custom-trained YOLO26 model identifies 23 car-part classes and generates segmentation masks. Flask connects the model to a user-friendly web interface, allowing users to upload images and visualize the prediction results.

The project provides practical experience in computer vision, instance segmentation, Python, and web application development.

