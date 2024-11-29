# DDP
export OMP_NUM_THREADS=24
num_epochs=5
num_gpu=6
run_name=BLIP-V2_256_Vector


CUDA_VISIBLE_DEVICES=0,1,2,3,5,7 torchrun --nproc_per_node $num_gpu main.py \
    --num_device $num_gpu \
    --run_name $run_name \
    --output_dir /data/match_element_retrieval/element_to_element/trained_model_weights/$run_name \
    --backbone_name BLIP-V2 \
    --element_vector_cache_path /data/match_element_retrieval/element_to_element/cache/search_dataset/element \
    --model_name_or_path Salesforce/blip2-opt-2.7b \
    --train_dataset_path /data/match_element_retrieval/element_to_element/data/csv/train_dataset.csv \
    --eval_dataset_path /data/match_element_retrieval/element_to_element/data/csv/test_dataset.csv \
    --test_dataset_path /data/match_element_retrieval/element_to_element/data/csv/total_dataset.csv \
    --search_test_dataset_path /data/match_element_retrieval/element_to_element/data/csv/search_test_dataset.csv \
    --element_image_path /data/s3_dataset/collection_element_dataset \
    --remove_unused_columns False \
    --cache_dir /data/match_element_retrieval/element_to_element/pretrained_weight \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 4 \
    --lr_scheduler_type constant_with_warmup \
    --num_train_epochs $num_epochs \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --eval_ratio 0.1 \
    --weight_decay 0.1 \
    --eval_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 10 \
    --load_best_model_at_end \
    --log_level info \
    --gradient_accumulation_steps 4 # > logs/$run_name.out &


# test_ddp
# export OMP_NUM_THREADS=32
# num_epochs=5
# num_gpu=8
# run_name=eval_test

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup torchrun --nproc_per_node $num_gpu run_train.py \
#     --num_device $num_gpu \
#     --run_name $run_name \ Salesforce/blip2-opt-2.7b
#     --output_dir ./ckpt/$run_name \
#     --model_name_or_path google/efficientnet-b4 \
#     --dataset_name /mnt/raid6/dltmddbs100/miricanbus/train/train_v01_discard_text_1 \
#     --image_column image_path \
#     --remove_unused_columns False \
#     --do_eval \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 16 \
#     --dataloader_num_workers 32 \
#     --lr_scheduler_type constant_with_warmup \
#     --num_train_epochs $num_epochs \
#     --learning_rate 5e-5 \
#     --warmup_ratio 0.1 \
#     --weight_decay 0.1 \
#     --eval_strategy steps \
#     --eval_steps 470 \
#     --save_steps 470 \
#     --logging_steps 1 \
#     --save_strategy steps \
#     --save_total_limit 2 \
#     --load_best_model_at_end \
#     --gradient_accumulation_steps 2 > logs/eval_test.out &