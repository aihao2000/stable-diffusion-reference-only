accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 '/home/aihao/workspace/StableDiffusionJointControl/train/train_sdxl_jc_lora_precomputed_embeds.py' \
    --pretrained_model_name_or_path='/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc-base-1.0' \
    --dataset_name='/home/aihao/workspace/DeepLearningContent/datasets/images' \
    --dataset_config_name='precomputed_embeds_for_sdxl_jc_train' \
    --train_data_dir='/home/aihao/workspace/DeepLearningContent/datasets/images' \
    --output_dir="/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc_lora-0.1" \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=2000 \
    --learning_rate=1e-5 \
    --checkpointing_steps=1000 \
    --validation_steps=1000 \
    --validation_prompt="a girl in a witch hat and dress with flowers" \
    --validation_image_prompt_path="/home/aihao/workspace/DeepLearningContent/datasets/images/data/pixiv/喵咕君QAQ(KH3)/Illustration/_喵咕君QAQ(KH3)'s illustrations／manga - pixiv__Illustration_Genshin Impact,GenshinImpact,girl,Klee,klee,Klee (Genshin Impact),星燭に揺れる爛花,drawers,ドロチラ_109887764_p000.jpg" \
    --use_8bit_adam \
    --gradient_checkpointing \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention
# "/home/aihao/workspace/DeepLearningContent/datasets/images/data/pixiv/狗脸脸dogface/Illustration/_狗脸脸dogface - pixiv__Illustration_Genshin Impact,girl,Klee,Klee (Genshin Impact),星燭に揺れる爛花_109735258_p000.jpg"
