from diffusers  import StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
# Step 2: Create the pipeline with your UNet model
model_id =  "stabilityai/stable-diffusion-2"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)



pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

pipe.unet.load_attn_procs("normal_to_malignant_image2image")


from PIL import Image
from pathlib import Path
import torch
import re

import gradio as gr

import random



def fake_gan(prompt_1, prompt_2, image):
    steps = 20
    scale = 9
    num_images_per_prompt = 1 
    seed = 9
    generator = torch.Generator(device=device).manual_seed(seed)

    # images = [ 
    #             "./ultrasound/normal/normal (5).png",
    #             "./ultrasound/normal/normal (6).png" ,
    #             "./ultrasound/normal/normal (7).png",
    #             "./ultrasound/normal/normal (8).png",
    #             "./ultrasound/normal/normal (9).png"
    # ]
    
    # images = [image]
 
    # images = [ 
    #             "./ultrasound/normal/normal (5).png",
    #             "./ultrasound/normal/normal (6).png" ,
    #             "./ultrasound/normal/normal (7).png",
    #             "./ultrasound/normal/normal (8).png",
    #             "./ultrasound/normal/normal (9).png"
    # ]

    # init_images = [Image.open(image).convert("RGB").resize((768,768)) for image in images] 

    init_images = [Image.fromarray(image).convert("RGB").resize((768,768))]

   

    # prompts = ["ultra sound image of breast with malignant cancer" ]
    prompts = [prompt_1]
    

    # negative_prompts = ["ultrasound scanning device" ]
    negative_prompts = [prompt_2]

    output =  pipe(prompts, negative_prompt=negative_prompts, image=init_images, num_inference_steps=steps,
        guidance_scale=scale, num_images_per_prompt=num_images_per_prompt, generator=generator)
    images = output.images

    return images
    
    # seeds = [1,2,3,9]
    # outputted_images = []
    # for i in seeds:
    #     seed = i
    #     generator = torch.Generator(device=device).manual_seed(seed)
    #     output =  pipe(prompts, negative_prompt=negative_prompts, image=init_images, num_inference_steps=steps,
    #     guidance_scale=scale, num_images_per_prompt=num_images_per_prompt, generator=generator)
    #     # outputted_images.append((output.images,f'label {i}'))
    #     outputted_images.append(output.images)

    # images = outputted_images


            
    # new_images = [
    #     (random.choice(
    #                  outputted_images[0],
    #         # [            outputted_images[0],
    #                         # outputted_images[1],
    #                         # outputted_images[2],
    #                         # outputted_images[3]
    #                         # ]
    #     ), f"label {i}" if i != 0 else "label" * 50)
    #     for i in range(3)
    # ]




    
    return new_images



with gr.Blocks() as demo:
    with gr.Row(variant="compact"):
        with gr.Column(variant="panel"):
            upload = gr.Image(shape=(200, 200))
        with gr.Column(variant="panel"):
            # with gr.Row(variant="compact"):
            prompt_1 = gr.Textbox(
                label="Enter Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter Prompt",
            ).style(
                container=False,
            )
            # with gr.Row(variant="compact"):
            prompt_2 = gr.Textbox(
                label="Enter negative prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter negative prompt",
            ).style(
                container=False,
            )
            # with gr.Row(variant="compact"):
            btn = gr.Button("Generate image").style(full_width=False)
    # with gr.Column(variant="panel"):

    gallery = gr.Gallery(
        label="Generated images", show_label=False, elem_id="gallery"
    ).style(columns=[2], rows=[2], object_fit="contain", height="auto")

    btn.click(fake_gan, inputs=[prompt_1, prompt_2, upload], outputs=gallery)
    # btn.click(fake_gan, inputs=prompt_1,prompt_2,upload, gallery)

if __name__ == "__main__":
    demo.launch(share=True)