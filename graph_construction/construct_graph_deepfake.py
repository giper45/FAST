from tqdm import tqdm
import sys
sys.path.append('.')
import argparse
from utils.common import *
from graph_construction.build_graph import *
from transformers import (
    RobertaTokenizer
)
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and analyze JSONL data to extract keywords.')

    parser.add_argument('--input_file', type=str, default='data/p_0.94_kws.jsonl', help='Path to the input JSONL file (default: data/p0.94.jsonl).')
    parser.add_argument('--output_file', type=str, default='data/grover_kws_graph_info_addsenidx.jsonl', help='Path to the output JSONL file (default: data/p_0.94_kws.jsonl).')

    args = parser.parse_args()

    inf = args.input_file
    outp = args.output_file
    
    data = read_data(inf)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=False)
    max_seq_length = 512
    no_node = 0
    with open(outp,'w',encoding='utf8') as outf:
        for id,line in tqdm(enumerate(data)):
            kws = line['information']['keywords']
            nodes, edges, entity_occur, sens,sen2node= build_graph(kws)
            if not nodes:
                no_node+=1
            nodes, all_tokens, drop_nodes ,sen_idx_pair = generate_rep_mask_based_on_graph(nodes,sens,tokenizer,max_seq_length)

            line['information']['graph'] = {'nodes':nodes,'edges':edges,'all_tokens':all_tokens,'drop_nodes':drop_nodes,'sentence_to_node_id':sen2node,
                                            'sentence_start_end_idx_pair':sen_idx_pair}
            outf.write(json.dumps(line)+'\n')
    print('{} instances have no graph'.format(no_node))