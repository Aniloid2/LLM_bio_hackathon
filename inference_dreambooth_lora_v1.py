from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import matplotlib.pyplot as plt

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

pipe.unet.load_attn_procs("dreambooth_lora_v1/checkpoint-3000")
prompt = 'a healthy breast ultrasound image'
image = pipe(prompt, num_inference_steps=25).images[0]
plt.title(prompt)
plt.axis('off')
plt.imshow(image)
