# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2"
export INSTANCE_DIR="ultrasound/malignant"
export OUTPUT_DIR="path_to_saved_model"
export CLASS_DIR="ultrasound_classbase/ultrasound-class"

# accelerate launch diffusers/examples/dreambooth/train_dreambooth.py   --pretrained_model_name_or_path=$MODEL_NAME    --instance_data_dir=$INSTANCE_DIR   --output_dir=$OUTPUT_DIR   --instance_prompt="a photo of sks dog"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --max_train_steps=400   --push_to_hub;


# accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks dog" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=1 

# accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="an ultrasound image of malignant breast" \
#   --class_prompt="an ultrasound image of breast" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="tensorboard" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=1 \
#   --validation_prompt="A photo of sks dog in a bucket" \
#   --validation_epochs=50 \
#   --seed="0" 


# accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="an ultrasound image of malignant breast" \
#   --class_prompt="an ultrasound image of breast" \
#   --with_prior_preservation --prior_loss_weight=0.5 \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --checkpointing_steps=100 \
#   --learning_rate=1e-4 \
#   --report_to="tensorboard" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=100 \
#   --validation_prompt="an ultrasound image of malignant breast" \
#   --num_class_images=100 \
#   --validation_epochs=1 \
#   --seed=42



# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export INSTANCE_DIR="ultrasound/normal"
# export CLASS_DIR='ultrasound'
# export OUTPUT_DIR="model"

accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py  \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="an ultrasound image of malignant breast" \
  --class_prompt="an ultrasound image of breast" \
  --with_prior_preservation --prior_loss_weight=0.5 \
  --resolution=768 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to='tensorboard' \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --num_class_images=100 \
  --seed=42
# #   --validation_prompt="an ultrasound image of benign breast" \
# #   --validation_epochs=50 \
# #   --seed="0"