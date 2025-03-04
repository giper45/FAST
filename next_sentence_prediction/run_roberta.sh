#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="0" python run_classifier.py \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name mrpc \
--do_train \
--do_eval \
--eval_all_checkpoints \
--data_dir /home/fast/FAST/data/next_sentence_prediction \
--train_file realnews_human_train.tsv \
--dev_file realnews_human_val.tsv \
--output_dir  /home/fast/FAST/data/models/realnews_human_next_sentence_prediction_roberta_large \
--max_seq_length 128 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 8 \
--learning_rate 1e-5 \
--num_train_epochs 5
