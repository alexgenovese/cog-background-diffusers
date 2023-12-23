# Run this before you deploy it on replicate
import os
import sys
import torch
from diffusers import AutoPipelineForInpainting

# append project directory to path so predict.py can be imported
MODEL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SDXL_CACHE = "./cache/sdxl"
DINO_CACHE = "./cache/dino"
SAM_CACHE = "./cache/sam"

# clone and install Grounded SAM
print("Installing Grounding SAM")
# os.system("git clone https://github.com/IDEA-Research/Grounded-Segment-Anything") 
os.system("cd ./Grounded-Segment-Anything && pip install -q -r requirements.txt")
os.system("cd ./Grounded-Segment-Anything/GroundingDINO && python3 -m pip install -q .")
os.system("cd ./Grounded-Segment-Anything/segment_anything && python3 -m pip install -q .")


print("Chcking folders")
# Make cache folders
if not os.path.exists(DINO_CACHE):
    os.makedirs(DINO_CACHE)

if not os.path.exists(SAM_CACHE):
    os.makedirs(SAM_CACHE)

if not os.path.exists(SDXL_CACHE):
    os.makedirs(SDXL_CACHE)


# Cache locally pretrained model
if not os.path.exists(SDXL_CACHE): 
    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.save_pretrained(SDXL_CACHE, safe_serialization=True)
