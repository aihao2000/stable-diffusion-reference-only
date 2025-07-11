accelerate launch --multi_gpu --mixed_precision=fp16 train/train_sd_reference_only_without_blueprint.py \
    --ddp_find_unused_parameters \
    --pretrained_model_name_or_path='/home/aihao/workspace/DeepLearningContent/models/sd_reference_only_without_blueprint/sd-2-1-init' \
    --dataset_name='/home/aihao/workspace/DeepLearningContent/datasets/characters' \
    --dataset_config_name='similar_pairs' \
    --train_data_dir='/home/aihao/workspace/DeepLearningContent/datasets/characters' \
    --load_dataset_num_proc=8 \
    --cache_dir='/home/aihao/workspace/DeepLearningContent/datasets/characters/.cache' \
    --output_dir="/home/aihao/workspace/DeepLearningContent/models/sd_reference_only_without_blueprint/associative_paint_0.1" \
    --tracker_project_name='associative_paint' \
    --prompt_column='image1' \
    --image_column='image2' \
    --train_batch_size=2 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --checkpointing_steps=1000 \
    --validation_steps=1000 \
    --validation_prompt "validation_images/1/1.jpg" "validation_images/2/1.jpg" "validation_images/3/1.jpg" "validation_images/4/1.jpg" \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --seed 2221101 \
    --mixed_precision=fp16 \
    --resolution=512
# --resume_from_checkpoint="latest"
# --load_dataset_streaming
