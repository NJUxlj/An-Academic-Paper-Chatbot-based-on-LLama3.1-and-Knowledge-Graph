
#!/bin/bash
# MiniQwen Comprehensive Pretraining Script

# Set base directory for outputs
OUTPUT_DIR="./miniqwen_pretrained"
mkdir -p $OUTPUT_DIR

# Run pretraining
python comprehensive_pretraining.py \
  --model_path ./miniqwen_checkpoints/final_model \
  --dataset_config config.json \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --max_steps 100000 \
  --warmup_steps 10000 \
  --logging_steps 100 \
  --save_steps 5000 \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --metric_for_best_model perplexity \
  --greater_is_better false \
  --streaming True \
  --max_seq_length 1024 \
  --block_size 1024 \
  --preprocessing_num_workers 4 \
  --dataset_sampling_rates "0.3,0.2,0.2,0.15,0.15" \
  --pre_2020_sampling_rate 0.2 \
  --fp16 True \
  --fp16_full_eval True \
  --report_to wandb \
  --dataloader_num_workers 4 \
  --ddp_find_unused_parameters False

# The sampling rates above are distributed as follows:
# - Books: 30%
# - Code: 20%
# - Reddit: 20%
# - ArXiv (post-2020 prioritized): 15%
# - PubMed (post-2020 prioritized): 15%
