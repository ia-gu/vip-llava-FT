paths=('rename_full_hand_made_v3_for_conversation')
lrs=(2e-7 4e-7 6e-7 8e-7 2e-6 4e-6 6e-6 8e-6 2e-5 4e-5 6e-5 8e-5)
task='rename_full_hand_made_v3_ft_from_v2'

for path in "${paths[@]}" ; do
    for lr in "${lrs[@]}" ; do
        echo $path/$lr
        #!/bin/bash
        PROMPT_VERSION=llava_v1
        # DATA_ROOT=./playground/data
        model_size=7b

        deepspeed --master_port 12347 llava/train/train_mem.py \
            --deepspeed ./scripts/zero2.json \
            --model_name_or_path /home/ueno/vip-llava/checkpoints/rename_full_no_bias/2e-5/40epoch/vip-llava-7b/20240508-180246 \
            --version $PROMPT_VERSION \
            --data_path jsons/$path.json \
            --image_folder datasets/ \
            --vision_tower clip_4layers_336 \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --bf16 True \
            --output_dir ./checkpoints/$task/$lr \
            --num_train_epochs 100 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --save_epochs 10 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 500000 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to wandb \
            --wandb_project vip-llava \
            --wandb_run_name $task$lr
    done
done