import subprocess, tarfile, os, torch, time
from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetInpaintPipeline

DINO_CACHE = "./cache/dino"
SDXL_CACHE = "./cache/sdxl"
SEGMENT_CACHE = "./cache/sam"


WEIGHTS_URL_DIR_MAP = {
    #"GROUNDING_DINO_WEIGHTS_URL": "https://weights.replicate.delivery/default/grounding-dino/grounding-dino.tar",
    #"HF-CACHE": "https://weights.replicate.delivery/default/grounding-dino/bert-base-uncased.tar",
    "DINO-SWINT" : "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "SAM-MODEL" : "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def download_grounding_dino_weights():
    if not os.path.exists(DINO_CACHE):
        # download_weights( url=WEIGHTS_URL_DIR_MAP["GROUNDING_DINO_WEIGHTS_URL"], dest=grounding_dino_weights_dir )
        os.system(f"git clone https://github.com/IDEA-Research/GroundingDINO {DINO_CACHE}")
        # Install DINO as library
        os.system(f"cd {DINO_CACHE} && pip install -e .")
    
    dino_weights = os.path.join(DINO_CACHE, 'groundingdino', 'weights')

    if not os.path.exists( dino_weights ):
        os.makedirs( dino_weights )

    if not os.path.isfile( os.path.join(dino_weights, 'sam_vit_h_4b8939.pth' ) ):
        download_weights(
            url=WEIGHTS_URL_DIR_MAP["SAM-MODEL"],
            dest=dino_weights
        )
    
    if not os.path.isfile( os.path.join(dino_weights, 'groundingdino_swint_ogc.pth' ) ):
        download_weights(
            url=WEIGHTS_URL_DIR_MAP["DINO-SWINT"],
            dest=dino_weights
        )


def download_diffusion_weights():
    if not os.path.exists(SDXL_CACHE): 
        controlnet = ControlNetModel.from_pretrained( "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16 )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            cache_dir=SDXL_CACHE
        )
        pipe.save_pretrained(SDXL_CACHE)


def download_segment_anything():
    if not os.path.exists(SEGMENT_CACHE):
        os.makedirs(SEGMENT_CACHE)
        os.system(f'git clone https://github.com/facebookresearch/segment-anything {SEGMENT_CACHE}')
        # Install DINO as library
        #subprocess.check_output(["pip", "install", "-q", "-e", "."])
        os.system(f"cd {SEGMENT_CACHE} && pip install -q -e .")


if __name__ == "__main__":
    download_diffusion_weights()
    download_grounding_dino_weights()
    download_segment_anything()