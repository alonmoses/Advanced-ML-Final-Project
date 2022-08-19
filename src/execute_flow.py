import json
from argparse import ArgumentParser
from data_processing import ProcessData
from models import LogisticRegressionExecute

USE_SAVED = False

def main(kwargs, config):
    # get arhuments
    model = kwargs['model']
    word2vec_model = kwargs['word2vec']

    # process the imput data and create dataloaders
    dataset = ProcessData(model, word2vec_model, config)
    dataset.create_dataloaders(use_saved=USE_SAVED)

    # execute logistic regression model
    lr_executer = LogisticRegressionExecute(config[model], dataset.train_dl, dataset.test_dl)
    lr_executer.fit()


if __name__ == '__main__':
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help='Choose the model to train', required=False, choices=['logistic_regression'], default='logistic_regression')
    parser.add_argument('-wm', '--word2vec', help='Choose the model to train', required=False, choices=['word2vec'], default='word2vec')
    args = parser.parse_args()
    kwargs = vars(args)

    # load configurations from config.json
    with open('src/config.json', 'r') as f:
        config = json.load(f)

    main(kwargs, config)