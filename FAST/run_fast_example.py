import os
import sys
from typing import List
import numpy as np
import torch
from tqdm import tqdm
import nltk
from datetime import datetime


import logging
from run_classifier_add_wiki import generate_shaped_edge_mask, generate_shaped_nodes_mask
from wikipedia2vec import Wikipedia2Vec

def get_device():                                                                                                    
    if torch.cuda.is_available():                                                                                               
        device = torch.device("cuda")                                                                                  
    elif torch.backends.mps.is_available():                                                                                     
        device = torch.device("mps")                                                                                   
    else:                                                                                                                       
        device = torch.device("cpu")                                                                                   
                                                                                                                                
    # return torch.device("cpu")                                                                                                               
    return device




logger = logging.getLogger(__name__)
sys.path.append('.')
from modeling_roberta import RobertaForGraphBasedSequenceClassification
from utils_graph_add_wiki import DeepFakeProcessor, InputExample, InputFeatures, acc_and_f1, glue_convert_examples_to_features
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    get_linear_schedule_with_warmup,
)
MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaForGraphBasedSequenceClassification, RobertaTokenizer),
}
def get_labels(self):
    """See base class."""
    return ["human", "machine"]

MODEL_FILE = 'data/enwiki_20180420_100d.pkl'
from calculate_test_score_grover import score
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)

PRETRAINED_MODEL_PATH = "data/models/roberta_base_grover_sens_lstm_nsp_score_weighted_wiki"
MAX_LENGTH = 512
PAD_ON_LEFT = False
config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
tokenizer = tokenizer_class.from_pretrained(PRETRAINED_MODEL_PATH)
PAD_TOKEN = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
PAD_TOKEN_SEGMENT_ID = 0
OUTPUT_MODE = "classification"
#parser.add_argument("--max_nodes_num", type=int, default=60, help="maximum number of nodes")
#parser.add_argument("--max_sentences", type=int, default=30, help="maximum number of sentences")
MAX_NODES_NUM = 60
MAX_SENTENCES = 30
EVAL_BATCH_SIZE= 4
DEVICE = get_device()

def get_dataset(test_data) -> TensorDataset:
    """Loads the dataset."""
    processor = DeepFakeProcessor()
    label_list = processor.get_labels()
    test_data = processor.get_dev_examples('data', 'grover_kws_graph_info_addsenidx.jsonl')
    print("Convert examples to features")
    features = glue_convert_examples_to_features(
        test_data,
        tokenizer,
        label_list=label_list,
        max_length=MAX_LENGTH,
        output_mode=OUTPUT_MODE,
        pad_on_left=PAD_ON_LEFT,
        pad_token=PAD_TOKEN,
        pad_token_segment_id=PAD_TOKEN_SEGMENT_ID
    )
    # Convert to Tensors and build dataset
    filtered_features = [
        f for f in features 
        if all(not isinstance(item, tuple) for item in f.input_ids)  # Check if no element is a tuple
    ]


    all_input_ids = torch.tensor([f.input_ids for f in filtered_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in filtered_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in filtered_features], dtype=torch.long)
    # Classification task
    all_labels = torch.tensor([f.label for f in filtered_features], dtype=torch.long)
    # print(all_labels)
    all_nodes_index_mask = []  # seq_length, each position indicate whether it store the token from the the node
    all_adj_metric = []  # node adjacent metric
    all_node_mask = []  # whether the nodes are appended
    all_sen2node = []
    all_sen_mask = []
    all_sen_length = []
    all_nsp_score = []
    all_nodes_ent_emb = []
    no_ent_emb,all_ent = 0,0
    for f in filtered_features:
        nodes_mask, node_num = generate_shaped_nodes_mask(f.nodes_index, MAX_LENGTH, MAX_NODES_NUM)
        # nmask = np.zeros(args.max_nodes_num)
        nmask = np.zeros(MAX_NODES_NUM)
        nmask[:node_num] = 1
        all_node_mask.append(nmask)

        adj_metric = generate_shaped_edge_mask(f.adj_metric, node_num, MAX_NODES_NUM)
        all_nodes_index_mask.append(nodes_mask)
        all_adj_metric.append(adj_metric)

        sen2node_mask = np.zeros(shape=(MAX_SENTENCES, MAX_NODES_NUM))
        # try1: sentence mask
        sen_mask = np.zeros(MAX_SENTENCES - 1)
        sen_mask[:len(f.sen2node) - 1] = 1
        all_sen_mask.append(sen_mask)
        # try2: sentence length
        all_sen_length.append(len(f.sen2node) if len(f.sen2node) <= MAX_SENTENCES else MAX_SENTENCES)

        for idx in range(len(f.sen2node)):
            if idx >= MAX_SENTENCES:
                break
            all_sennodes = f.sen2node[idx]
            for sennode in all_sennodes:
                if (sennode < MAX_NODES_NUM):
                    sen2node_mask[idx, sennode] = 1
        all_sen2node.append(sen2node_mask)

        # try3: add next sentence prediction score
        nsp_score = np.zeros(MAX_SENTENCES - 1)
        length = min(len(f.sen2node) - 1, MAX_SENTENCES - 1)
        # logger.info('length {}'.format(length))
        if length != 0:
            nsp_score[:length] = f.nsp_score[:length, 1]
        all_nsp_score.append(nsp_score)

        ins_ent_embs = np.zeros(shape=(MAX_NODES_NUM,100))

        for idx in range(node_num):
            all_ent+=1
            ent = f.nodes_ent[idx]
            ent_exist = True
            word_exist = True
            try:
                ent_emb = wiki2vec.get_entity_vector(f.nodes_ent[idx])
            except Exception:
                ent_exist = False

            if ent_exist==False:
                    words = [w.lower() for w in nltk.word_tokenize(ent)]
                    word_embs = []
                    for word in words:
                        try:
                            word_emb = wiki2vec.get_word_vector(word)
                            word_embs.append(word_emb)
                        except Exception:
                            continue
                    if word_embs:
                        ent_emb = np.average(np.array(word_embs))
                    else:
                        ent_emb = np.random.randn(100)
                        word_exist = False
            if word_exist==False and ent_exist==False:
                no_ent_emb+=1
            ins_ent_embs[idx,:] = ent_emb

        all_nodes_ent_emb.append(ins_ent_embs)

    all_nodes_index_mask = torch.tensor(all_nodes_index_mask, dtype=torch.float)
    all_node_mask = torch.tensor(all_node_mask, dtype=torch.int)
    all_adj_metric = torch.tensor(all_adj_metric, dtype=torch.float)
    all_sen2node_mask = torch.tensor(all_sen2node, dtype=torch.float)
    all_sen_mask = torch.tensor(all_sen_mask, dtype=torch.float)
    all_sen_length = torch.tensor(all_sen_length, dtype=torch.long)
    all_nsp_score = torch.tensor(all_nsp_score, dtype=torch.float)
    all_nodes_ent_emb = torch.tensor(all_nodes_ent_emb, dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_nodes_index_mask,
                            all_adj_metric, all_node_mask, all_sen2node_mask, all_sen_mask, all_sen_length,
                            all_nsp_score, all_nodes_ent_emb)
    return dataset




# model.eval()

processor = DeepFakeProcessor()
label_list = processor.get_labels()
test_data : List[InputExample]= processor.get_dev_examples('data', 'grover_kws_graph_info_addsenidx.jsonl')

dataset : TensorDataset = get_dataset(test_data)
eval_sampler = SequentialSampler(dataset)
eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)
logger.info("***** Running evaluation ****")
logger.info("  Num examples = %d", len(dataset))
logger.info("  Batch size = %d", EVAL_BATCH_SIZE)

eval_loss = 0.0
nb_eval_steps = 0
preds = None
out_label_ids = None
results = {}
model = model_class.from_pretrained(PRETRAINED_MODEL_PATH)
model.to(DEVICE)
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    model.eval()
    batch = tuple(t.to(DEVICE) for t in batch)

    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],
                    'nodes_index_mask': batch[4], 'adj_metric': batch[5], 'node_mask': batch[6],
                    'sen2node': batch[7], 'sentence_mask': batch[8], 'sentence_length': batch[9],
                    'nsp_score': batch[10],
                    'nodes_ent_emb': batch[11]}
        # if args.model_type != "distilbert":
        #     inputs["token_type_ids"] = (
        #         batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
        #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()
    
    nb_eval_steps += 1
    if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

logger.info("Evaluation completed")
probs = preds
eval_loss = eval_loss / nb_eval_steps
# Classification task
preds = np.argmax(preds, axis=1)
result = acc_and_f1(preds, out_label_ids)
results.update(result)

prefix = datetime.now().strftime("%Y-%d-%m-%H-%M-%S")
output_folder = os.path.join("data", prefix)
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

output_eval_file = os.path.join(output_folder, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))



# list_labels = processor.get_labels()
# print(list_labels)


# def predict(text):
#     """Predicts the label for the given text."""
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     inputs = {k: v.to('cuda') for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     predicted_class_idx = torch.argmax(probs, dim=-1).item()
#     labels = get_labels()
#     return labels[predicted_class_idx], probs.tolist()[0]

# # Example usage:
# example_text_human = "The quick brown fox jumps over the lazy dog. I am writing this text."
# example_text_machine = "Generate a summary of the following document. This document is about artificial intelligence."

# human_prediction, human_probs = predict(example_text_human)
# machine_prediction, machine_probs = predict(example_text_machine)

# print(f"Text: '{example_text_human}'")
# print(f"Prediction: {human_prediction}")
# print(f"Probabilities: {human_probs}")

# print(f"\nText: '{example_text_machine}'")
# print(f"Prediction: {machine_prediction}")
# print(f"Probabilities: {machine_probs}")

# # # Example of more text.
# # example_text_human2 = "I went to the store and bought some groceries. Then I came home and made dinner."
# # example_text_machine2 = "Create a python function to calculate the factorial of a given number. Include comments."

# # human_prediction2, human_probs2 = predict(example_text_human2)
# # machine_prediction2, machine_probs2 = predict(example_text_machine2)

# # print(f"\nText: '{example_text_human2}'")
# # print(f"Prediction: {human_prediction2}")
# # print(f"Probabilities: {human_probs2}")

# # print(f"\nText: '{example_text_machine2}'")
# # print(f"Prediction: {machine_prediction2}")
# # print(f"Probabilities: {machine_probs2}")