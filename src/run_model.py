import json
from os import path
from argparse import ArgumentParser
import modeling.bert_framework_for_dreaddit as bert_dreaddit_fw
import modeling.bert_framework_for_stance as bert_stance_fw
from modeling.lr_framework import LRFramework
from modeling.models import *

def main(model, dataset):
    # load configurations from config.json
    with open(path.join('src', 'config.json'), 'r') as f:
        config = json.load(f)

    if dataset == 'stress_detection':
        raw_data_path = path.join('data', 'dreaddit')
    else:
        raw_data_path = path.join('data', 'stance')

    if model == 'lr':
        if dataset == 'stance_detection':
            print("Error: logistic regression is not applied on the stance detection task")
            return
        fworkf = LRFramework
        modelf = LogisticRegression
        modelframework = fworkf(config['lr'], modelf)
        modelframework.fit(raw_data_path=raw_data_path)
    elif model == 'bert':
        fworkf = bert_dreaddit_fw.BERTFramework if dataset == 'stress_detection' else bert_stance_fw.BERTFramework
        modelf = BertModelClassification
        modelframework = fworkf(config['bert'], modelf)
        modelframework.fit()
    elif model == 'roberta':
        fworkf = bert_dreaddit_fw.RoBERTaFramework if dataset == 'stress_detection' else bert_stance_fw.RoBERTaFramework
        modelf = RoBertaModelClassification
        modelframework = fworkf(config['roberta'], modelf)
        modelframework.fit()
    elif model == 'roberta_with_features':
        fworkf = bert_dreaddit_fw.RoBERTaFramework if dataset == 'stress_detection' else bert_stance_fw.RoBERTaFramework
        modelf = RoBertaWFeaturesModelClassification
        modelframework = fworkf(config['roberta'], modelf, with_features=True)
        modelframework.fit()
    elif model == 'gpt2':
        fworkf = bert_dreaddit_fw.GPT2Framework if dataset == 'stress_detection' else bert_stance_fw.GPT2Framework
        modelf = GPT2ModelClassification
        modelframework = fworkf(config['gpt2'], modelf)
        modelframework.fit()
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-m', 
        '--model', 
        help='Choose model to execute', 
        required=False, 
        choices=['lr', 'bert', 'roberta', 'roberta_with_features', 'gpt2'], 
        default='lr'
    )
    parser.add_argument(
        '-d', 
        '--dataset', 
        help='Rather running the stress_detection or stance_detection dataset', 
        required=False, 
        choices=['stress_detection', 'stance_detection'], 
        default='stress_detection'
    )

    args = parser.parse_args()
    kwargs = vars(args)
    model = kwargs['model']
    dataset = kwargs['dataset']

    main(model, dataset)