# LLM_bio_hackathon

hackathon repository for llm bio hackathon 2023

install conda on linux with

wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh

or the latest version from https://www.anaconda.com/download#downloads for your particular machine

Then do:

bash Anaconda3-2023.07-1-Linux-x86_64.sh

and follow the installation guide

create a conda enviroment with

```
conda env create -f environment.yml
```

go into diffusers folder

cd diffusers

then do

`accelerate config`

the file (default_config.yaml) in the following folder (~/.cache/huggingface/accelerate) has to look like the following

compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '1'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

train model with
bash train_start.sh

This will generate 100 class specific images and insert them in the ultrasoung_classbase folder

Then, it will run 1000 steps of stable diffusion on the DiffusionPipeline

Next, use the inference_img_to_img.ipynb

The first cell is to do text-to-image inference

for example:

using the prompt: ultrasound image of breast with cancer malignant

will output a random image of what soft tissue would look like if it had cancer.

The next cell, loads the finetuned UNet from the DiffusionPipeline and inserts this into the StableDiffusionImg2ImgPipeline pipeline

The third cell does generation on a few images from the ultrasound/normal directory. It loads one normal image and a user prompt. The prompt and image are later parsed through the StableDiffusionImg2ImgPipeline and a new image, in the style of the input image is generated with changes directed by the input prompt.

For example,

If we input ultrasound/normal (5).png and the prompt "ultra sound image of breast with malignant cancer". we would generate the following image.

![alt img](https://github.com/Aniloid2/LLM_bio_hackathon/ultrasound/normal/normal (5).png)
