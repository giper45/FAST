#!/bin/bash

# if [ ! -f /home/fast/FAST/data/p0.94.jsonl ]; then
#     mkdir -p /home/fast/FAST/data
#     wget https://storage.googleapis.com/grover-models/generation_examples/generator=mega~dataset=p0.94.jsonl -O /home/fast/FAST/data/p0.94.jsonl
# fi

#       inf = 'data/grover_kws_graph_info_addsenidx.jsonl'
#       outp = 'data/grover_kws_graph_info_nsp_hm.jsonl'
echo "[+] Calculate sentence pair score (input: data/grover_kws_graph_info_addsenidx.jsonl, output: data/grover_kws_graph_info_nsp_hm.jsonl)"
python next_sentence_prediction/calculate_sentence_pair_score.py

echo "[+] Run FAST model for testing"
bash FAST/run_roberta_add_wiki_do_eval.sh

# exec "$@"
