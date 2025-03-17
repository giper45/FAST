#!/bin/bash
set -e
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <JSONL article file> <JSONL with keywords> <JSONl with graph info and kws>"
    exit
fi

article_file=${1}
kws_file=${2} 
graph_file=${3} 

echo "[+] Extracting keywords for ${article_file} => ${kws_file}"
python  data_process/extract_keywords.py --input_file ${article_file} --output_file ${kws_file}

echo "[+] Constructing graph for ${kws_file} => ${graph_file}"
python graph_construction/construct_graph_deepfake.py --input_file ${kws_file} --output_file ${graph_file}