export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="ultrasound/normal"
export CLASS_DIR='ultrasound'
export OUTPUT_DIR="dreambooth_lora_v1"

accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="ultrasound image without cancer" \
  --class_prompt="ultrasound images of breast" \
  --with_prior_preservation --prior_loss_weight=0.5 \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to='tensorboard' \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --seed="42" 




export resume_ckt='dreambooth_lora_v1/checkpoint-1000'
export INSTANCE_DIR="ultrasound/benign"

accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="ultrasound image with benign tumor" \
  --class_prompt="ultrasound images of breast" \
  --with_prior_preservation --prior_loss_weight=0.5 \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to='tensorboard' \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --resume_from_checkpoint=$resume_ckt \
  --seed="42" 





export resume_ckt='dreambooth_lora_v1/checkpoint-2000'
export INSTANCE_DIR="ultrasound/malignant"

accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="ultrasound image with malignant tumor" \
  --class_prompt="ultrasound images of breast" \
  --with_prior_preservation --prior_loss_weight=0.5 \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to='tensorboard' \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --resume_from_checkpoint=$resume_ckt \
  --seed="42" 


