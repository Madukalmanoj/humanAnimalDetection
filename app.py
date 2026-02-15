import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import efficientnet_b0
import torch.nn as nn

from huggingface_hub import hf_hub_download

# ==========================
# CONFIG
# ==========================

st.set_page_config(page_title="Human / Animal Detection", layout="wide")
st.title("🧠 Human / Animal Detection System")

# Streamlit Cloud runs CPU
device = torch.device("cpu")
st.sidebar.write("Device:", device)

# ==========================
# LOAD MODELS FROM HF
# ==========================

@st.cache_resource
def load_models():

    st.write("⬇ Downloading models from HuggingFace...")

    # Download model files
    detector_path = hf_hub_download(
        repo_id="Manu542168/humananimal",
        filename="detector_finetuned.pth"
    )

    classifier_path = hf_hub_download(
        repo_id="Manu542168/humananimal",
        filename="classifier_finetuned.pth"
    )

    # Detector
    detector = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

    detector.load_state_dict(torch.load(detector_path, map_location=device))
    detector.to(device).eval()

    # Classifier
    classifier = efficientnet_b0(weights=None)
    classifier.classifier[1] = nn.Linear(1280, 2)

    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device).eval()

    return detector, classifier


detector, classifier = load_models()

cls_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================
# DETECTION FUNCTION
# ==========================

def run_detection(image):

    start = time.time()

    image_tensor = transforms.ToTensor()(image).to(device)

    with torch.no_grad():
        outputs = detector([image_tensor])[0]

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    image_np = np.array(image)

    human_count = 0
    animal_count = 0

    for box, score in zip(boxes, scores):

        if score < 0.5:
            continue

        xmin, ymin, xmax, ymax = map(int, box)

        crop = image.crop((xmin, ymin, xmax, ymax))
        crop_tensor = cls_transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            cls_output = classifier(crop_tensor)
            _, pred = torch.max(cls_output, 1)

        label = "Human" if pred.item() == 1 else "Animal"
        color = (0,255,0) if label=="Human" else (255,100,100)

        if label == "Human":
            human_count += 1
        else:
            animal_count += 1

        cv2.rectangle(image_np,(xmin,ymin),(xmax,ymax),color,2)
        cv2.putText(image_np,f"{label} {score:.2f}",
                    (xmin,ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,color,2)

    fps = 1/(time.time()-start)

    return image_np, human_count, animal_count, fps


# ==========================
# UI
# ==========================

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg","png","jpeg","mp4"]
)

if uploaded_file:

    if "image" in uploaded_file.type:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        if st.button("🔍 Run Detection"):

            output_img, h, a, fps = run_detection(image)

            st.image(output_img, caption="Detection Result")
            st.write("Humans:", h)
            st.write("Animals:", a)
            st.write("FPS:", f"{fps:.2f}")

    elif "video" in uploaded_file.type:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            output, _, _, _ = run_detection(image)

            stframe.image(output, channels="RGB")

        cap.release()
