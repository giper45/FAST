
import json
import nltk
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import re
import argparse

# predictor_ner = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz", cuda_device=0)
predictor_ner = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
def check_contain_upper(password):
    pattern = re.compile('[A-Z]+')
    match = pattern.findall(password)
    if match:
        return True
    else:
        return False

def extract_entity_allennlp(words, tags):
    all_ents = []
    all_ents_test = []
    flag = True
    tmp = []
 
    for i, tag in enumerate(tags):
        flag = True if (check_contain_upper(words[i])) else False
        if (tag != 'O' or flag):
            tmp.append(words[i])
        if (len(tmp) != 0 and ((tag == 'O' and flag == False) or i == (len(tags) - 1))):
            all_ents.append(' '.join(tmp))
            tmp = []
    return all_ents

def found_key_words(claims):

    # all_tokens = self.predictor_cp.predict_batch_json(inputs=[{'sentence': text} for text in claims])
    all_ent_res = predictor_ner.predict_batch_json(inputs=[{'sentence': text} for text in claims])

    all_keywords = []
    for i in range(len(claims)):
        claim = claims[i]
        # tokens = all_tokens[i]
        ent_res = all_ent_res[i]

        # tokens = self.predictior_cp.predict(sentence=claim)
        key_words = {'noun': [], 'claim': claim, 'subject': [], 'entity': []}
        # nps = []
        # tree = tokens['hierplane_tree']['root']
        # noun_phrases = self.get_NP(tree, nps)
        # key_words['noun'].extend(noun_phrases)
        # subjects = self.get_subjects(tree)
        # for subject in subjects:
        #     if len(subject) > 0:
        #         key_words['subject'].append(subject)

        all_ents = extract_entity_allennlp(ent_res['words'], ent_res['tags'])
        key_words['entity'].extend(all_ents)
        key_words = {'keywords': key_words, 'sentence': claim}

        all_keywords.append(key_words)
    return all_keywords
def analyze_document(doc):
    sens = nltk.sent_tokenize(doc)
    resplit_sens = []
    for sen in sens:
        resplit_sens+=[s.strip() for s in sen.split('\n') if s.strip()!='']
    sens = resplit_sens
    # print(sens)
    # input()
    try:
        all_keywords = found_key_words(sens)
    except Exception as e:
        print(e)
        all_keywords = []
    # for sen in sens:
    #     kws = RP.found_key_words(sen)
    #     all_keywords.append(kws)
    # srls = RP.found_openie_srl(sens)

    return all_keywords

if __name__ == '__main__':
    
    #data process for grover dataset
    parser = argparse.ArgumentParser(description='Process and analyze JSONL data to extract keywords.')

    parser.add_argument('--input_file', type=str, default='data/p0.94.jsonl', help='Path to the input JSONL file (default: data/p0.94.jsonl).')
    parser.add_argument('--output_file', type=str, default='data/p_0.94_kws.jsonl', help='Path to the output JSONL file (default: data/p_0.94_kws.jsonl).')

    
    args = parser.parse_args()

    file = open(args.input_file,'r',encoding='utf8')
    out_file = open(args.output_file,'w',encoding='utf8')
    
    # out_file_bea = open('/mnt/wanjun/data/p_0.96_kws_beautiful.jsonl', 'w', encoding='utf8')
    data = file.readlines()
    for line in tqdm(data):
        # print(line)
        line = json.loads(line)
        doc = line['article']
        h_kws = analyze_document(doc)
        line['information'] = {'keywords':h_kws}
        # out_file_bea.write(json.dumps(line,indent=4)+'\n')
        out_file.write(json.dumps(line)+'\n')
    
    
    
