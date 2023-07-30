MODEL_NAME="CompVis/stable-diffusion-v1-4"
INSTANCE_DIR="./dream_ultra/normal"
OUTPUT_DIR="./output_models/models-ultra-step100"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port=5678 diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a ultrasound image of normal breast" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100


MODEL_NAME="./output_models/models-ultra-step100"
INSTANCE_DIR="./dream_ultra/benign"
OUTPUT_DIR="./output_models/models-ultra-step100-benign"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port=5678 diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a ultrasound image of breast with benign tumor" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100



MODEL_NAME="./output_models/models-ultra-step100-benign"
INSTANCE_DIR="./dream_ultra/malignant"
OUTPUT_DIR="./output_models/models-ultra-step100-benign-malignant"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port=5678 diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a ultrasound image of breast with malignant tumor" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100
