accelerate launch --multi_gpu --mixed_precision=fp16 /home/aihao/workspace/StableDiffusionReferenceOnly/train/train_sd_reference_only.py \
    --pretrained_model_name_or_path='/home/aihao/workspace/DeepLearningContent/models/sd_reference_only/sd-2-1-init' \
    --dataset_name='/home/aihao/workspace/DeepLearningContent/datasets/characters' \
    --dataset_config_name='reference_only_for_automatic_coloring' \
    --train_data_dir='/home/aihao/workspace/DeepLearningContent/datasets/characters' \
    --output_dir="/home/aihao/workspace/DeepLearningContent/models/sd_reference_only/automatic_coloring_2-1_train_image_encoder_0.1" \
    --train_batch_size=6 \
    --max_train_steps=200000 \
    --learning_rate=5e-5 \
    --checkpointing_steps=1000 \
    --validation_steps=1000 \
    --validation_prompt "validation_images/1/1.png" "validation_images/2/1.png" "validation_images/3/1.png" "validation_images/4/1.png" \
    --validation_blueprint "validation_images/1/2.png" "validation_images/2/2.png" "validation_images/3/2.png" "validation_images/4/2.png" \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --seed 2221101 \
    --mixed_precision=fp16 \
    --train_image_encoder \
    --resolution=256 \
    --resume_from_checkpoint="latest"
# --load_dataset_streaming
