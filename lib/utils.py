# from diffusers.utils import load_image
import torch, requests, os
from PIL import Image
from io import BytesIO
from rembg import new_session, remove
from tempfile import NamedTemporaryFile


def download_image(image_url: str) -> Image:
    if not os.path.isfile(image_url): 
        r = requests.get(image_url, timeout=4.0)
        if r.status_code != requests.codes.ok:
            assert False, 'Status code error: {}.'.format(r.status_code)

        pil_img = Image.open(BytesIO(r.content)).convert("RGB")
    else:
        pil_img = Image.open(image_url).convert("RGB")

    #with pil_img as im:
    #    im.save(output_path)
    # print('Image downloaded from url: {} and saved to: {}.'.format(image_url, output_path))
    with NamedTemporaryFile(suffix=".jpg") as temp_file:
        pil_img.save(temp_file.name)
        print (f"save {temp_file.name} in ")

        return pil_img



def generate_image(image, mask, prompt, negative_prompt, pipe, seed, guidance_scale, steps, strength, device):
    # resize for inpainting 
    w, h = image.size
    in_image = image.resize((1024, 1024))
    in_mask = mask.resize((1024, 1024))

    generator = torch.Generator( device ).manual_seed(seed)

    result = pipe(
        image=in_image, 
        mask_image=in_mask, 
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        strength=strength,
        width=image.width,
        height=image.height
    ).images[0]

    return result.resize((w, h))


def remove_bg(image_source_pil):
    rmsession = new_session("u2net")
    return remove(image_source_pil, session=rmsession)