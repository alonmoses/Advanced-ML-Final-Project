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

    if dataset == 'dreaddit':
        raw_data_path = path.join('data', 'dreaddit')
    else:
        raw_data_path = path.join('data', 'stance')

    if model == 'lr':
        fworkf = LRFramework
        modelf = LogisticRegression
        modelframework = fworkf(config['lr'], modelf)
    elif model == 'bert':
        fworkf = bert_dreaddit_fw.BERTFramework if dataset == 'dreaddit' else bert_stance_fw.BERTFramework
        modelf = BertModelClassification
        modelframework = fworkf(config['bert'], modelf)
    elif model == 'roberta':
        fworkf = bert_dreaddit_fw.RoBERTaFramework if dataset == 'dreaddit' else bert_stance_fw.RoBERTaFramework
        modelf = RoBertaModelClassification
        modelframework = fworkf(config['roberta'], modelf)
    elif model == 'roberta_with_features':
        fworkf = bert_dreaddit_fw.RoBERTaFramework if dataset == 'dreaddit' else bert_stance_fw.RoBERTaFramework
        modelf = RoBertaWFeaturesModelClassification
        modelframework = fworkf(config['roberta'], modelf, with_features=True)
    elif model == 'gpt2':
        fworkf = bert_dreaddit_fw.GPT2Framework if dataset == 'dreaddit' else bert_stance_fw.GPT2Framework
        modelf = GPT2ModelClassification
        modelframework = fworkf(config['gpt2'], modelf)
    
    modelframework.fit(raw_data_path=raw_data_path)


if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        '-m', 
        '--model', 
        help='Rather running the baseline or new model', 
        required=False, 
        choices=['lr', 'bert', 'roberta', 'roberta_with_features', 'gpt2'], 
        default='lr'
    )
    parser.add_argument(
        '-d', 
        '--dataset', 
        help='Rather running the dreaddit or stance dataset', 
        required=False, 
        choices=['dreaddit', 'stance'], 
        default='dreaddit'
    )

    args = parser.parse_args()
    kwargs = vars(args)
    model = kwargs['model']
    dataset = kwargs['dataset']

    main(model, dataset)