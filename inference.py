from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# pipe.unet.load_attn_procs("patrickvonplaten/lora_dreambooth_dog_example")
pipe.unet.load_attn_procs("path_to_saved_model")

image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]