accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 /home/aihao/workspace/StableDiffusionReferenceOnly/train/train_sd_reference_only.py \
    --pretrained_model_name_or_path='/home/aihao/workspace/DeepLearningContent/models/sd_ro/init' \
    --dataset_name='/home/aihao/workspace/DeepLearningContent/datasets/images' \
    --dataset_config_name='reference_only_for_automatic_coloring' \
    --train_data_dir='/home/aihao/workspace/DeepLearningContent/datasets/images' \
    --output_dir="/home/aihao/workspace/DeepLearningContent/models/sd_ro/automatic_coloring" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --max_train_steps=50000 \
    --learning_rate=1e-5 \
    --checkpointing_steps=1000 \
    --validation_steps=1000 \
    --validation_prompt "validation_images/1/1.jpg" "validation_images/2/1.jpg" "validation_images/3/1.jpg" "validation_images/4/1.jpg" \
    --validation_blueprint "validation_images/1/2.jpg" "validation_images/2/2.png" "validation_images/3/2.jpg" "validation_images/4/2.jpg" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    --seed 2221101
# "/home/aihao/workspace/DeepLearningContent/datasets/images/data/pixiv/狗脸脸dogface/Illustration/_狗脸脸dogface - pixiv__Illustration_Genshin Impact,girl,Klee,Klee (Genshin Impact),星燭に揺れる爛花_109735258_p000.jpg"
