
MODEL_NAME="stabilityai/stable-diffusion-2"
OUTPUT_DIR="./output_models/models-ultra-step100-stable2"
INSTANCE_DIR="./dream_ultra/normal"
gpus=0,1
CUDA_VISIBLE_DEVICES=$gpus accelerate launch --main_process_port=5678 --config_file=acc_config.yaml diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a ultrasound image of normal breast" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100



MODEL_NAME="./output_models/models-ultra-step100-stable2"
INSTANCE_DIR="./dream_ultra/benign"
OUTPUT_DIR="./output_models/models-ultra-step100-stable2-benign"

gpus=0,1
CUDA_VISIBLE_DEVICES=$gpus accelerate launch --main_process_port=5678 --config_file=acc_config.yaml diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a ultrasound image of breast with benign tumor" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100



MODEL_NAME="./output_models/models-ultra-step100-stable2-benign"
INSTANCE_DIR="./dream_ultra/malignant"
OUTPUT_DIR="./output_models/models-ultra-step500-stable2-benign-malignant"
gpus=0,1
CUDA_VISIBLE_DEVICES=$gpus accelerate launch --main_process_port=5678 --config_file=acc_config.yaml diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a ultrasound image of breast with malignant tumor" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500
