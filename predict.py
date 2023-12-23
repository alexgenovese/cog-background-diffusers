# Prediction interface for Cog
import os, sys, math, torch, uuid

# Grunding DINO
sys.path.append( "./Grounded-Segment-Anything/GroundingDINO" )
from groundingdino.util.inference import load_image

from cog import BasePredictor, Input, Path

from PIL import Image
from diffusers import (AutoPipelineForInpainting, AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler,
    HeunDiscreteScheduler, 
    PNDMScheduler
)
import numpy as np

from lib.utils import remove_bg, generate_image, download_image
from lib.DINO import DINO, CACHE_DINO
from lib.SAM import SAM


MODEL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SDXL_CACHE = "./cache/sdxl"
DINO_CACHE = "./cache/dino"
SAM_CACHE = "./cache/sam"
OUTPUT_FILENAME = uuid.uuid4().hex

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


class Predictor(BasePredictor):

    def get_device_type(self):
        if torch.backends.mps.is_available():
            return "mps"
        
        if torch.cuda.is_available():
            return "cuda"
        
        print("------ CPU Device type -------")
        return "cpu"


    def setup(self) -> None:
        self.device = self.get_device_type()

        # Download weights
        self.dino = DINO(self.device)
        self.sam = SAM(self.device)

        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        if not os.path.exists(SDXL_CACHE):
            os.makedirs(SDXL_CACHE)
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                MODEL_NAME,
                vae=vae,
                torch_dtype=torch.float16
            )
            self.pipe.save_pretrained(SDXL_CACHE)
        else:
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                MODEL_NAME,
                vae=vae,
                torch_dtype=torch.float16,
                cache_dir=SDXL_CACHE
            )
        
        # to GPU
        self.pipe.to(self.device)


    # Output PIL
    def scale_down_image(self, image: Image, max_size) -> Image:
        width, height = image.size
        scaling_factor = min(max_size/width, max_size/height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = image.resize((new_width, new_height))
        cropped_image = self.crop_center(resized_image)
        return cropped_image

    def crop_center(self, pil_img):
        img_width, img_height = pil_img.size
        crop_width = self.base(img_width)
        crop_height = self.base(img_height)
        return pil_img.crop(
            (
                (img_width - crop_width) // 2,
                (img_height - crop_height) // 2,
                (img_width + crop_width) // 2,
                (img_height + crop_height) // 2)
            )

    def base(self, x):
        return int(8 * math.floor(int(x)/8))
    
    def predict(
        self,
        image: Path = Input(description="Input image"),
        mask: Path = Input(description="Mask image"),
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="monochrome, lowres, bad anatomy, worst quality, low quality",
        ),
        caption: str = Input(
            description="Text for object identification with Grounding DINO",
            default=None,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        guidance_scale: float = Input(
            description="Guidance scale", ge=0, le=10, default=8.0
        ),
        steps: int = Input(
            description="Number of denoising steps", ge=1, le=80, default=20),
        strength: float = Input(
            description="1.0 corresponds to full destruction of information in image", ge=0.01, le=1.0, default=0.7),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if image is None:
            raise Exception("No Image found in the request")

        if prompt == "" or negative_prompt == "":
            raise Exception("No Prompt or Negative prompt found in the request")

        if caption is None or caption == "":
            raise Exception("No caption found in the request")
        
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)

        # save local temp file
        tmp_pil_img = download_image(image)
        input_image = self.scale_down_image(tmp_pil_img, 1024)

        # Create PIL Image and Tensor
        image_source, image_tensor = load_image(input_image)
        
        annotated_frame, detected_boxes = self.dino.detect(image_tensor, image_source, text_prompt=caption)

        if detected_boxes.numel() == 0:
            print(f"Detected Boxes {detected_boxes.numel()}")
            return "No object found in image. Try to change the caption text."
        
        # Use SAM to create Mask
        segmented_frame_masks = self.sam.segment(image_source, boxes=detected_boxes)

        # create mask images 
        mask = segmented_frame_masks[0][0].cpu().numpy()
        inverted_mask = ((1 - mask) * 255).astype(np.uint8)

        image_mask_pil = Image.fromarray(mask)
        inverted_image_mask_pil = Image.fromarray(inverted_mask)

        # Inference
        result = generate_image(image=input_image, 
                mask=inverted_image_mask_pil, 
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                pipe=self.pipe, 
                seed=seed,
                guidance_scale=guidance_scale,
                steps=steps,
                strength=strength,
                device = self.device
            )

        # Paste the object for a better resolution 
        rem_data = remove_bg(input_image)
        result.paste(rem_data, (0,0), mask = rem_data)

        # Save
        result.save('./'+OUTPUT_FILENAME)

        return './'+OUTPUT_FILENAME
    


    # Only for local test
    def predict_mps(
        self,
        image: str = "./_tests/dog.png",
        prompt: str = "A dog on seaside",
        negative_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality",
        caption: str = "dog",
        scheduler: str = "DPMSolverMultistep",
        guidance_scale: float = 8.0,
        steps: int = 25,
        strength: float = 0.99,
        seed: int = None,
        debug: bool = True
    ) -> str:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(self.pipe.scheduler.config)

        # save local temp file
        tmp_pil_img = download_image(image)
        input_image = self.scale_down_image(tmp_pil_img, 1024)
        
        if debug: 
            input_image.save('0_tmp_image.png')

        # Create PIL Image and Tensor
        image_source, image_tensor = load_image(input_image)
        
        annotated_frame, detected_boxes = self.dino.detect(image_tensor, image_source, text_prompt=caption)

        if detected_boxes.numel() == 0:
            print(f"Detected Boxes {detected_boxes.numel()}")
            return "No object found in image. Try to change the caption text."
        
        if debug:
            # To show the annotation on image
            image_with_box = Image.fromarray(annotated_frame)
            image_with_box.save('1_image_with_box.png')
        
        # Use SAM to create Mask
        segmented_frame_masks = self.sam.segment(image_source, boxes=detected_boxes)

        # Show the annotation and the mask on the image
        if debug:
            annotated_frame_with_mask = self.sam.draw_mask(segmented_frame_masks[0][0], annotated_frame)
            annotated_frame_with_mask = Image.fromarray(annotated_frame_with_mask) # To show the annotation on image
            annotated_frame_with_mask.save('2_annotated_frame_with_mask.png')

        # create mask images 
        mask = segmented_frame_masks[0][0].cpu().numpy()
        inverted_mask = ((1 - mask) * 255).astype(np.uint8)

        image_mask_pil = Image.fromarray(mask)
        inverted_image_mask_pil = Image.fromarray(inverted_mask)
        
        if debug:
            image_mask_pil.save('3_image_mask_pil.png')
            inverted_image_mask_pil.save('4_inverted_image_mask_pil.png')

        # Inference
        result = generate_image(image=input_image, 
                mask=inverted_image_mask_pil, 
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                pipe=self.pipe, 
                seed=seed,
                guidance_scale=guidance_scale,
                steps=steps,
                strength=strength,
                device = self.device
            )

        # Paste the object for a better resolution 
        rem_data = remove_bg(input_image)
        result.paste(rem_data, (0,0), mask = rem_data)

        # Save
        result.save('./'+OUTPUT_FILENAME)

        return './'+OUTPUT_FILENAME




if __name__ == "__main__":    
    pred = Predictor()
    pred.setup()
    pred.predict_mps()