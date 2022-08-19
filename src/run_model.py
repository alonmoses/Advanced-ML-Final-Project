import json
from os import path
from argparse import ArgumentParser
from modeling.bert_framework import BERTFramework, RoBERTaFramework, GPT2Framework
from modeling.lr_framework import LRFramework
from modeling.models import *

def main(model, raw_data_path):
    # load configurations from config.json
    with open(path.join('src', 'config.json'), 'r') as f:
        config = json.load(f)

    if model == 'lr':
        fworkf = LRFramework
        modelf = LogisticRegression
        modelframework = fworkf(config['lr'], modelf)
    elif model == 'bert':
        fworkf = BERTFramework
        modelf = BertModelForStanceClassification
        modelframework = fworkf(config['bert'], modelf)
    elif model == 'roberta':
        fworkf = RoBERTaFramework
        modelf = RoBertaModelForStanceClassification
        modelframework = fworkf(config['roberta'], modelf)
    elif model == 'roberta_with_features':
        fworkf = RoBERTaFramework
        modelf = RoBertaWFeaturesModelForStanceClassification
        modelframework = fworkf(config['roberta'], modelf, with_features=True)
    elif model == 'gpt2':
        fworkf = GPT2Framework
        modelf = GPT2ModelForStanceClassification
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

    if dataset == 'dreaddit':
        raw_data_path = path.join('data', 'dreaddit')
    else:
        raw_data_path = path.join('data', 'stance')

    main(model, raw_data_path)