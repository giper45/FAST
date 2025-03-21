#!/usr/bin/env bash

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="0" python FAST/run_classifier_add_wiki.py \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name deepfake \
--train_file grover_kws_graph_info_addsenidx.jsonl \
--dev_file grover_kws_graph_info_addsenidx.jsonl \
--test_file grover_kws_graph_info_addsenidx.jsonl \
--eval_all_checkpoints \
--data_dir data \
--output_dir data/models/roberta_base_grover_sens_lstm_nsp_score_weighted_wiki \
--max_seq_length 512 \
--do_eval \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--learning_rate 1e-5 \
--num_train_epochs 6 \


