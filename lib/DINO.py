import torch, os
import numpy as np
# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
# Download the model from HF
from huggingface_hub import hf_hub_download

CACHE_DINO = './cache/dino'

class DINO:

    def __init__(self, device) -> None:
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        self.device = device
        self.model = self.load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
        print(f"---> Instanciated DINO {device}")


    def load_model_hf(self, repo_id, filename, ckpt_config_filename, device='cpu'):
        if not os.path.exists(CACHE_DINO):
            os.makedirs(CACHE_DINO)
            cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename, local_dir=CACHE_DINO, force_download=True)
            cache_file = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=CACHE_DINO, force_download=True)
        else: 
            cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename, cache_dir=CACHE_DINO)
            cache_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=CACHE_DINO)

        # Set the configuration
        args = SLConfig.fromfile(cache_config_file) 
        args.device = device
        model = build_model(args)
        # load the model
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model   
    
    # detect object using grounding DINO
    def detect(self, image_tensor, image_pil, text_prompt, box_threshold = 0.3, text_threshold = 0.25):
        boxes, logits, phrases = predict(
            model=self.model, 
            image=image_tensor, 
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device = 'cpu'
        )

        annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
        return annotated_frame, boxes 
            
