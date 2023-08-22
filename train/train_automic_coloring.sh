accelerate launch /home/aihao/workspace/StableDiffusionReferenceOnly/train/train_sd_reference_only.py \
    --pretrained_model_name_or_path='/home/aihao/workspace/DeepLearningContent/models/sd_reference_only/init' \
    --dataset_name='/home/aihao/workspace/DeepLearningContent/datasets/images' \
    --dataset_config_name='reference_only_for_automatic_coloring' \
    --train_data_dir='/home/aihao/workspace/DeepLearningContent/datasets/images' \
    --output_dir="/home/aihao/workspace/DeepLearningContent/models/sd_reference_only/automatic_coloring" \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=100000 \
    --learning_rate=1e-5 \
    --checkpointing_steps=1000 \
    --validation_steps=1000 \
    --validation_prompt "validation_images/1/1.jpg" "validation_images/2/1.jpg" "validation_images/3/1.jpg" "validation_images/4/1.jpg" \
    --validation_blueprint "validation_images/1/2.jpg" "validation_images/2/2.png" "validation_images/3/2.jpg" "validation_images/4/2.jpg" \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --seed 2221101 \
    --resume_from_checkpoint="latest"
# --mixed_precision=fp16 \
# --gradient_checkpointing
# "/home/aihao/workspace/DeepLearningContent/datasets/images/data/pixiv/狗脸脸dogface/Illustration/_狗脸脸dogface - pixiv__Illustration_Genshin Impact,girl,Klee,Klee (Genshin Impact),星燭に揺れる爛花_109735258_p000.jpg"
