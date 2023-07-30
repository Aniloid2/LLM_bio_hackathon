# from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
# import torch

# pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.to("cuda")

# # pipe.unet.load_attn_procs("patrickvonplaten/lora_dreambooth_dog_example")
# pipe.unet.load_attn_procs("path_to_saved_model")

# image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]




from diffusers  import StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from pathlib import Path
import re
from PIL import Image

def slugify(text):
   text = re.sub(r"[^\w\s]", "", text)
   text = re.sub(r"\s+", "-", text)
   return text


device = "cuda" if torch.cuda.is_available() else "cpu"
# Step 2: Create the pipeline with your UNet model
model_id =  "stabilityai/stable-diffusion-2"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

pipe.unet.load_attn_procs("normal_to_malignant_image2image")


 
steps = 20
scale = 9
num_images_per_prompt = 1
# seed = torch.randint(0, 1000000, (1,)).item()
seed = 9
generator = torch.Generator(device=device).manual_seed(seed)

# images = ["/home/brianformento/Dataset_BUSI_with_GT/normal/normal (1).png",
# "/home/brianformento/Dataset_BUSI_with_GT/benign/benign (1).png" ,
#  "/home/brianformento/Dataset_BUSI_with_GT/malignant/malignant (1).png"]

# images = ["/home/brianformento/Dataset_BUSI_with_GT/normal/normal (5).png",
# "/home/brianformento/Dataset_BUSI_with_GT/normal/normal (6).png" ,
#  "/home/brianformento/Dataset_BUSI_with_GT/normal/normal (7).png"]


# images = ["./ultrasound/normal/normal (5).png",
# "./ultrasound/normal/normal (6).png" ,
#  "./ultrasound/normal/normal (7).png"]

from os import listdir
from os.path import isfile, join
images = ["./ultrasound/normal/"+f for f in listdir("./ultrasound/normal/") if isfile(join("./ultrasound/normal/", f))][:10]



init_images = [Image.open(image).convert("RGB").resize((768,768)) for image in images] 


img_list_len = len(images)
prompts = ["ultra sound image of breast with malignant cancer"]*img_list_len


DIR_NAME="./full_images_preprocessed/"
dirpath_preprocessed = Path(DIR_NAME)
# create parent dir if doesn't exist
dirpath_preprocessed.mkdir(parents=True, exist_ok=True)

edited_image_save_name = ["normal_image_768"]*img_list_len

for idx, (image,prompt) in enumerate(zip(init_images, edited_image_save_name )):
    image_name = f'{slugify(prompt)}-{idx}.png'
    image_path = dirpath_preprocessed / image_name
    image.save(image_path)

negative_prompts = ["ultrasound scanning device" ]*img_list_len

for i in range(img_list_len):
    output =  pipe(prompts[i], negative_prompt=negative_prompts[i], image=init_images[i], num_inference_steps=steps,
    guidance_scale=scale, num_images_per_prompt=num_images_per_prompt, generator=generator)




    DIR_NAME="./full_images_generated_image_to_image/"
    dirpath = Path(DIR_NAME)
    # create parent dir if doesn't exist
    dirpath.mkdir(parents=True, exist_ok=True)



    for idx, (image,prompt) in enumerate(zip(output.images, prompts[i]*num_images_per_prompt)):
        image_name = f'{slugify(prompt)}-{i}.png'
        image_path = dirpath / image_name
        image.save(image_path)