
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
def detect(image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model, 
      image=image, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )

  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
  return annotated_frame, boxes 


annotated_frame, detected_boxes = detect(image, text_prompt="a bench", model=groundingdino_model)
Image.fromarray(annotated_frame)


# ====================
# SAM for segmentation
# ====================



def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()
  

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))



segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)



# ================================
# Stable Diffusion for inpainting
# ================================


# create mask images 
mask = segmented_frame_masks[0][0].cpu().numpy()
inverted_mask = ((1 - mask) * 255).astype(np.uint8)



# ******************************
# Generate a new image 
# *******************************


def generate_image(image, mask, prompt, negative_prompt, pipe, seed):
  # resize for inpainting 
  w, h = image.size
  in_image = image.resize((512, 512))
  in_mask = mask.resize((512, 512))

  generator = torch.Generator(device).manual_seed(seed) 

  result = pipe(image=in_image, mask_image=in_mask, prompt=prompt, negative_prompt=negative_prompt, generator=generator)
  
  result = result.images[0]

  return result.resize((w, h))


prompt="A sofa, high quality, detailed, cyberpunk, futuristic, with a lot of details, and a lot of colors"
negative_prompt="" # "low resolution, ugly"
seed = 33 # for reproducibility 
image_source_pil = Image.fromarray(image_source)
image_mask_pil = Image.fromarray(mask)
inverted_image_mask_pil = Image.fromarray(inverted_mask)

generated_image = generate_image(image=image_source_pil, mask=image_mask_pil, prompt=prompt, negative_prompt=negative_prompt, pipe=sd_pipe, seed=seed)




# Display images

import cv2
img = cv2.imread(local_image_path)
image1_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



# Create a figure to hold the images
plt.figure(figsize=(15, 10))

# Display the first image
plt.subplot(2, 3, 1)
plt.imshow(image1_rgb)
plt.title('Image 1')
plt.axis('off')  # Hide axis

# Display the second image
plt.subplot(2, 3, 2)
plt.imshow(image1_rgb)
plt.title('Detect the dog using Grounding DINO')
plt.axis('off')  # Hide axis

# Display the third image
plt.subplot(2, 3, 3)
plt.imshow(annotated_frame_with_mask)
plt.title('segmentation of the dog usign SAM')
plt.axis('off')  # Hide axis

plt.subplot(2, 3, 4)
plt.imshow(mask)
plt.title('mask')
plt.axis('off')  # Hide axis

plt.subplot(2, 3, 5)
plt.imshow(inverted_mask)
plt.title('inverted_mask')
plt.axis('off')  # Hide axis


# Show the images
plt.tight_layout()
plt.show()


# Display second the new generated image
plt.figure(figsize=(8, 6))  # Adjust width and height as needed
plt.imshow(generated_image)
plt.title('generated_image')
plt.axis('off')  # Hide axis
plt.show()







cv2.destroyAllWindows()