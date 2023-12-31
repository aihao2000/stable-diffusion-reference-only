accelerate launch  --mixed_precision=fp16 train/train_sd_reference_only_automatic_coloring.py \
    --ddp_find_unused_parameters \
    --pretrained_model_name_or_path='/home/aihao/workspace/DeepLearningContent/models/sd_reference_only/init_0.1' \
    --report_to="tensorboard" \
    --dataset_name='/home/aihao/workspace/DeepLearningContent/datasets/characters' \
    --dataset_config_name='similar_pairs' \
    --train_data_dir='/home/aihao/workspace/DeepLearningContent/datasets/characters' \
    --load_dataset_num_proc=8 \
    --output_dir="/home/aihao/workspace/DeepLearningContent/models/sd_reference_only/automatic_coloring_0.1" \
    --tracker_project_name='automatic_coloring' \
    --train_batch_size=1 \
    --max_train_steps=500000 \
    --learning_rate=1e-5 \
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
