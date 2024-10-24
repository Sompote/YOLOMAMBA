import sys
sys.path.append('/workspace/YOLOMAMBA/ultralytics2')
from ultralytics import YOLO
# Load a model
model = YOLO("yolo11s.yaml")  # build a new model from YAML
#model = YOLO("yolo11s.yaml").load("yolo11s.pt")
#model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/workspace/6400 images/data.yaml", epochs=400, batch=60,  imgsz=640)