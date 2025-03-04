# Citation
Source code for the EMNLP2020 paper [Neural Deepfake Detection with Factual Structure of Text](https://aclanthology.org/2020.emnlp-main.193.pdf). If you find the code useful, please cite our paper:
```
@inproceedings{zhong-etal-2020-neural,
    title = "Neural Deepfake Detection with Factual Structure of Text",
    author = "Zhong, Wanjun  and
      Tang, Duyu  and
      Xu, Zenan  and
      Wang, Ruize  and
      Duan, Nan  and
      Zhou, Ming  and
      Wang, Jiahai  and
      Yin, Jian",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.193",
    doi = "10.18653/v1/2020.emnlp-main.193",
    pages = "2461--2470",
    abstract = "Deepfake detection, the task of automatically discriminating machine-generated text, is increasingly critical with recent advances in natural language generative models. Existing approaches to deepfake detection typically represent documents with coarse-grained representations. However, they struggle to capture factual structures of documents, which is a discriminative factor between machine-generated and human-written text according to our statistical analysis. To address this, we propose a graph-based model that utilizes the factual structure of a document for deepfake detection of text. Our approach represents the factual structure of a given document as an entity graph, which is further utilized to learn sentence representations with a graph neural network. Sentence representations are then composed to a document representation for making predictions, where consistent relations between neighboring sentences are sequentially modeled. Results of experiments on two public deepfake datasets show that our approach significantly improves strong base models built with RoBERTa. Model analysis further indicates that our model can distinguish the difference in the factual structure between machine-generated text and human-written text.",
}
```
# Code Usage

### Extract keywords
```python
python data_process/extract_keywords.py 
```

Use the NER model from allenlp
```
predictor_ner = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz", cuda_device=0)
````



Input: `p0.94.json` (Downloaded in the `docker_entrypoint.sh`) in the format:
```

  "article",
  "authors",
  "date",
  "domain",
  "ind30k",
  "label",
  "orig_split",
  "random_score",
  "split",
  "title",
  "url"
]
```
Output: `p_0.94_kws.jsonl`  in the format:
```
[
  "article",
  "authors",
  "date",
  "domain",
  "ind30k",
  "information",
  "label",
  "orig_split",
  "random_score",
  "split",
  "title",
  "url"
]
```
Add an `information` field that is filled with several fields.

##  contruct_graph_deepfake.py

```python
python graph_construction/construct_graph_deepfake.py
```

Input: kws with deps
Output: grover_kws_graph_info_addsenidx.jsonl

## process news data
Construct the training data for the next sentence prediction NSP model 
### Validation data
```python
python next_sentence_prediction/process_news_data.py
```
Input: `data/p0.94.jsonl`
Output: `data/realnews_human_val.tsv` 


### Training data
```python
python next_sentence_prediction/process_news_data.py --train
```
Input: `data/p0.94.jsonl`
Output: `data/realnews_human_train.tsv` 





## Train the classifier
```bash
./next_sentence_prediction/run_roberta.sh
```
Input: 
* data_dir (`data` folder)
* train_file `realnews_human_train.tsv` file obtained previously
* dev_file `realnews_human_val.tsv` file obtained previously
Output:
* `data/models/realnews_human_next_sentence_prediction_roberta_large`




## calculate_sentence_pair_score
After trained the next sentence prediction model, it is possible to add the nsp to the graph_info
```
python next_sentence_prediction/calculate_sentence_pair_score.py
```

Input: `data/grover_kws_graph_info_addsenidx.jsonl` (obtained by constructing the graph)

Output: `data/grover_kws_graph_nsp_hm.jsonl`
Basically, contains the score of next sentence predictions.

## Train final classifier
```python
./FAST/run_roberta_add_wiki.sh
```
 

## code file
### Folder
| code_file | function | usage |
| --- | --- | --- |
| data_process | | |
| extract_keywords.py | extract entities from gpt2 and grover dataset | python extract_keywords.py; need to change data folder and dataset type |
| graph_construction | | |
| contruct_graph_deepfake.py | Main function to construct graph, extract nodes, edges, tokens, and start and end idx for each sentence. Finally, each entity and sentence will be recorded by a mask index in the whole input sequence| python construct_graph_deepfake.py; need to change data path |
| build_graph.py | define functions for constructing graph | func build_graph(all_info); generate_rep_mask_based_on_graph(nodes, sens, tokenizer,max_seq_length); |
| next_sentence_prediction | | |
| process_news_data.py | construct training data for NSP model. positive ins is the NSP sentence pair. negative is the most similar sentence with B, suppose positive sentence pair is (A,B). | python process_news_data.py; need to change data path |
| run_classifier.py | training the NSP model | bash run_roberta.sh | 
| discriminator | | |
| transformers_graph_wiki/modeling_roberta.py | code for graph based model, add nsp score and wiki knowledge | 
| run_classifier_add_wiki.py | code for training and evaluate final model for the paper. (add nsp score, wiki knowledge) | bash run_roberta_add_wiki.sh |
| utils_graph_add_wiki | code for processing data for graph+nsp+wiki model | none |
| calculate_test_score_grover.py | calculate test score based on evaluation script for grover | python calculate_test_score_grover; need to change input data path and score file |


