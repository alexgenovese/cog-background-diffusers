import torch, os
import numpy as np
from groundingdino.util import box_ops
from PIL import Image
# segment anything
from segment_anything import build_sam, SamPredictor 
import numpy as np
# Download the model from HF
from huggingface_hub import hf_hub_download

CACHE_SAM = './cache/sam'


class SAM: 

    def __init__(self, device) -> None:
        self.device = device

        self.checkpoint = self.load_model_hf( 'ybelkada/segment-anything', 'checkpoints', 'sam_vit_h_4b8939.pth')
        self.model = SamPredictor(build_sam(checkpoint=self.checkpoint).to(device))
        print(f"---> Instanciated SAM {device}")


    def load_model_hf(self, repo_id, subfolder, ckpt_config_filename):
        if not os.path.exists(CACHE_SAM):
            os.makedirs(CACHE_SAM)
            cache_config_file = hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=ckpt_config_filename, local_dir=CACHE_SAM)
        else:
            cache_config_file = hf_hub_download(repo_id=repo_id, subfolder=subfolder, filename=ckpt_config_filename, cache_dir=CACHE_SAM)

        return cache_config_file

    def segment(self, image, boxes):
        self.model.set_image(image)
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.model.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image.shape[:2])
        masks, _, _ = self.model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
            )
        return masks.cpu()
        

    def draw_mask(self, mask, image, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))