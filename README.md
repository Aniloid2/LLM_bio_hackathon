# LLM_bio_hackathon
hackathon repository for llm bio hackathon 2023

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

![Local Image](./ultrasound/normal/normal (5).png)
