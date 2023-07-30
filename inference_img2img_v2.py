from diffusers  import StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import random

# load model 
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id =  "output_models/models-ultra-step500-stable2-benign-malignant"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

from PIL import Image
from pathlib import Path
import torch
import re

steps = 20
scale = 8.5
num_images_per_prompt = 1
# seed = torch.randint(0, 1000000, (1,)).item()
seed = 9
generator = torch.Generator(device=device).manual_seed(seed)

images = [f"./Dataset_BUSI_with_GT/normal/normal ({i}).png" for i in range(1,11)]
num_imgs = len(images)
init_images = [Image.open(image).convert("RGB").resize((768,768)) for image in images] 


prompt_templates = ["a ultrasound image of breast with malignant tumor in the middle",
           "a ultrasound image of breast with a large malignant tumor of size 2cm", 
           "a ultrasound image of breast with a malignant tumor with unclear boundary"]

prompts = random.choices(prompt_templates, k=num_imgs)


DIR_NAME="saved_images/resized_img/"
dirpath_preprocessed = Path(DIR_NAME)
dirpath_preprocessed.mkdir(parents=True, exist_ok=True)
exp_name=model_id.split("/")[-1]

for idx, (image,prompt) in enumerate(zip(init_images, prompts )):
    image_name = f'{exp_name}-{idx}.png'
    image_path = dirpath_preprocessed / image_name
    image.save(image_path)

negative_prompts = ["ultrasound image of normal breast"]*num_imgs
output =  pipe(prompts, negative_prompt=negative_prompts, image=init_images, num_inference_steps=steps,
guidance_scale=scale, num_images_per_prompt=num_images_per_prompt, generator=generator)



DIR_NAME="./saved_images/img2img/"
dirpath = Path(DIR_NAME)
dirpath.mkdir(parents=True, exist_ok=True)


for idx, (image,prompt) in enumerate(zip(output.images, prompts*num_images_per_prompt)):
    image_name = f'{exp_name}-{idx}.png'
    image_path = dirpath / image_name
    image.save(image_path)