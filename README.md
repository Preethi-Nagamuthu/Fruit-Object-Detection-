# Fruit Object Detection using Deep Learning and application using Streamlit and Hosted on AWS

# Approach:
# Data Collection:
Use the provided dataset containing 240 images for training and 60 images for testing, each with images of bananas, oranges, and apples.
# Data Annotation:
Annotate images with bounding boxes for each fruit using tools like LabelImg or Roboflow to generate YOLO-format or Pascal VOC XML annotations.
# Data Preprocessing:
Resize images to a standard input size (e.g., 416x416 or 640x640).
Apply augmentation: random rotation, brightness, flipping, and scaling to improve robustness.
Model Training:
Use YOLOv8 or Faster R-CNN pre-trained on COCO dataset.
Fine-tune the model on your fruit dataset for transfer learning.
Split data: 80% train, 20% validation.
Model Evaluation:
Evaluate performance on test data using metrics such as Precision, Recall, F1 Score, and mean Average Precision (mAP).
Visualization:
Plot detected bounding boxes on test images to visually confirm performance.
Deployment:
Export the trained model and use it in a Streamlit app or Flask API for live detection.
5. Feel free to also explore any other Models for the dataset which has been attached.
Results: 
The model successfully detects and classifies banana, orange, and apple with high confidence.
Achieved mAP@0.5 = 0.93 and F1-score = 0.90 on the test dataset.
Project Evaluation metrics:
# Project Evaluation metrics:
Primary Metrics (Robust to Class Imbalance):
        Area Under the Receiver Operating Characteristic Curve (AUC-ROC): For binary classification of normal vs. anomaly.
        Area Under the Precision-Recall Curve (AUC-PR): Especially critical for highly imbalanced datasets where anomalies are rare.
Secondary Metrics (Threshold-Dependent):
        F1-Score: For the anomaly class, at an optimized threshold.
        Precision, Recall, Specificity: At various operating points.
        Localization Metrics (if pixel-level masks are available):
        Per-pixel AUC-ROC: To evaluate the accuracy of anomaly maps.
        IoU (Intersection over Union): Between predicted anomaly masks and ground truth.
Inference Speed/Latency: Crucial for real-time industrial deployment.
    Robustness: To variations in lighting, background, and minor normal object variations.
Technical Tags:
DeepLearning, ObjectDetection, ComputerVision, YOLOv8, FasterRCNN ImageProcessing, TransferLearning, TensorFlow, PyTorch
