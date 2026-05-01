from ultralytics import YOLO
import os


def train_model():
    print("Starting YOLO training...")

    # Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="outputs",
        name="object_detection"
    )

    print("Training completed successfully!")


if __name__ == "__main__":
    train_model()