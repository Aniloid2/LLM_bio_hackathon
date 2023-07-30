from diffusers import StableDiffusionPipeline
import torch
import random


########### load model ###########

##### stable diffusion v1 #####

diff_version = "v2"
# class_type = "normal"
# class_type = "benign"
class_type = "malignant"


normal_templates = ["a ultrasound image of normal breast"]
benign_templates = ["a ultrasound image of breast with two small benign tumor", 
                    "a ultrasound image of breast with a benign tumor in the middle"]
malignant_templates = ["a ultrasound image of breast with malignant tumor in the middle",
           "a ultrasound image of breast with a large malignant tumor of size 2cm", 
           "a ultrasound image of breast with a malignant tumor with unclear boundary"]


if class_type == "normal":
    prompt_templates = normal_templates
elif class_type == "benign":
    prompt_templates = benign_templates
elif class_type == "malignant":
    prompt_templates = malignant_templates
else:
    print("don't support  this class type", class_type)


if diff_version == "v1":
    if class_type == "normal":
        model_id = "./output_models/models-ultra-step100" 
    elif class_type == "benign":
        model_id = "./output_models/models-ultra-step100-benign" 
    elif class_type == "malignant":
        model_id = "./output_models/models-ultra-step100-benign-malignant"
    else:
        print("do not support")
elif diff_version == "v2":
    if class_type == "normal":
        model_id = "output_models/models-ultra-step100-stable2" 
    elif class_type == "benign":
        model_id = "output_models/models-ultra-step100-stable2-benign" 
    elif class_type == "malignant":
        model_id = "output_models/models-ultra-step100-stable2-benign-malignant"
    elif class_type == "malignant-step500":
        model_id = "output_models/models-ultra-step500-stable2-benign-malignant"
    else:
        print("do not support")
else:
    print("do not support")



pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.safety_checker = None
pipe.requires_safety_checker = False
          



for i in range(5):
    # img_name = "malignant"
    prompt = random.choice(prompt_templates)
    img_name=f"{class_type}_{diff_version}_{prompt}"

    image = pipe(prompt, num_inference_steps=20, guidance_scale=8.5).images[0]
    
    image.save(f"saved_images/text2img/{img_name}_{i}.png")


