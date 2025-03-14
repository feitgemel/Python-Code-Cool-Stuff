
import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


from huggingface_hub import hf_hub_download

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grounding DINO model
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"


groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

#SAM (Segment Anything)

# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam_checkpoint = 'Grounded-Segment-Anything/sam_vit_h_4b8939.pth'

sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))


# Stable Diffusion (Inpainting)
sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to(device)


# Inference
# ===========

local_image_path = "Grounded-Segment-Anything/a.jpg"
#local_image_path= "Grounded-Segment-Anything/assets/inpaint_demo.jpg"
image_source, image = load_image(local_image_path)
Image.fromarray(image_source)


# Inference
# ===========

# detect object using grounding DINO
def detect(image_source, image, text_prompt, model, box_threshold=0.3, text_threshold=0.25, box_thickness=30):
    # Ensure the image_source is writable
    image_source = np.array(image_source).copy()  # Make the array writable

    # Detect boxes using GroundingDINO
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    print("****** Detected boxes (raw):", boxes)  # Debugging: Print the raw box coordinates

    if len(boxes) == 0:
        print("No boxes detected.")
        return image_source, boxes

    # Convert from (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)
    boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

    print("****** Converted boxes (xyxy):", boxes)  # Debugging: Print the converted box coordinates

    # Get the dimensions of the image
    height, width, _ = image_source.shape  # Get the height and width of the image

    # Loop through each box and draw on the image
    for box in boxes:
        # Scale the box coordinates from relative to absolute pixel values
        x_min, y_min, x_max, y_max = (box * torch.tensor([width, height, width, height])).tolist()

        # Check if the box is within the image range
        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
            print(f"Box out of bounds: {box}")  # Debugging: Box is out of image bounds
            continue
        
        print(f"Drawing box: {x_min, y_min, x_max, y_max}")  # Debugging: Box coordinates to be drawn

        # Draw a thicker rectangle using OpenCV
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))
        color = (0, 0, 255)  # Red in BGR format
        thickness = box_thickness  # Adjust the thickness of the rectangle

        # Draw the rectangle with OpenCV
        image_source = cv2.rectangle(image_source, start_point, end_point, color, thickness)

    return image_source, boxes


annotated_frame, detected_boxes = detect(image_source, image, text_prompt="a teddy bear", model=groundingdino_model)
Image.fromarray(annotated_frame)


import cv2
img = cv2.imread(local_image_path)
image1_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create a figure to hold the images
plt.figure(figsize=(10, 5))

# Display the first image
plt.subplot(1, 2, 1)
plt.imshow(image1_rgb)
plt.title('Image 1')
plt.axis('off')  # Hide axis

# Display the second image
plt.subplot(1, 2, 2)
plt.imshow(annotated_frame)
plt.title('Image 2')
plt.axis('off')  # Hide axis

# Show the images
plt.show()

#cv2.imshow("annotated_frame", annotated_frame)
#cv2.waitKey(0)












cv2.destroyAllWindows()