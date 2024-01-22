import sys
from cog import BasePredictor, BaseModel, Input, Path
from typing import Optional, List

sys.path.append('./cache/sam')
sys.path.append('./cache/dino')

# ----SAM
from segment_anything import SamPredictor, sam_model_registry
# ----Stable Diffusion
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, AutoencoderKL, DPMSolverMultistepScheduler
# ----GroundingDINO
from groundingdino.util.inference import load_model, predict as predict_dino, annotate
from groundingdino.util import box_ops
import groundingdino.datasets.transforms as T
# ----Extra Libraries
import os, cv2, torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from diffusers.utils import make_image_grid, load_image
from rembg import new_session, remove

from download_weights import download_grounding_dino_weights, download_diffusion_weights, DINO_CACHE, SDXL_CACHE


class Predictor(BasePredictor):

    def get_device_type(self):
        if torch.backends.mps.is_available():
            return "mps"
        
        if torch.cuda.is_available():
            return "cuda"
        
        print("------ CPU Device type -------")
        return "cpu"


    def show_mask(self, mask, image, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    

    def make_canny_condition(self, image, low_threshold, max_threshold):
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, max_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image


    def setup(self) -> None:
        self.device = self.get_device_type()
        download_diffusion_weights()
        download_grounding_dino_weights()

        self.model_dino = load_model(
            f"{DINO_CACHE}/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            f"{DINO_CACHE}/groundingdino/weights/groundingdino_swint_ogc.pth",
            device=self.device,
        )

        self.rmsession = new_session("u2net")
        

    # a photo of bag on a table of elegant room hotel, RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3
    def predict(
        self,
        image: str = "./_tests/parfum.webp",
        prompt: str = "parfum is on the seaside, sunny weather background",
        # prompt:str = "a photo of bag on the table of elegant room hotel, RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        negative_prompt: str = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
        caption: str = "parfum",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        controlnet_conditioning_scale: float = 0.2, # recreate completely background
        low_threshold: int = 400,
        max_threshold: int = 800,
        seed: int = 123456
    ) :
        # Image manipulation
        init_image = load_image( image )
        init_image.convert("RGB")

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        src = np.asarray(init_image)
        img, _ = transform(init_image, None)
        
        # ------SAM Parameters
        model_type = "vit_h"
        sam_weights = os.path.join(DINO_CACHE, 'groundingdino', 'weights', 'sam_vit_h_4b8939.pth')
        predictor = SamPredictor(sam_model_registry[model_type](checkpoint=sam_weights).to(device=self.device))
        
        # ------Stable Diffusion
        controlnet = ControlNetModel.from_pretrained( "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16 )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            cache_dir=SDXL_CACHE
        ).to(self.device)
        # pipe.enable_model_cpu_offload()

        boxes, logits, phrases = predict_dino(
            model=self.model_dino,
            image=img,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        # img_annnotated = annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)[...,::-1]
        
        predictor.set_image(src)
        H, W, _ = src.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        new_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, src.shape[:2]).to(self.device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = new_boxes,
            multimask_output = False,
        )

        mask = Image.fromarray(masks[0][0].cpu().numpy())
        inverted_mask = ImageOps.invert(mask)
        image_canny = self.make_canny_condition(init_image, low_threshold, max_threshold) # case reimagine the background
        rem_data = remove(init_image, session=self.rmsession )

        if seed is None: 
            generator = torch.Generator(device="cpu").manual_seed(1)
        else: 
            generator = torch.Generator(device="cpu").manual_seed(seed)

        image = pipe(
            prompt,
            negative_prompt = negative_prompt,
            image = init_image,
            mask_image = inverted_mask,
            control_image = image_canny,
            # guidance_scale = 7.0,
            # strength = 0.75,
            controlnet_conditioning_scale = controlnet_conditioning_scale,
            num_inference_steps = 25,
            generator=generator
            ).images[0]

        # image.paste(rem_data, (0,0), mask = rem_data)

        image.save('./output.png')

        return image



if __name__ == "__main__":
    pred = Predictor()
    pred.setup()
    pred.predict()
