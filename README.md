# Advanced-ML-Final-Project
Development Repository for Advanced Machine Learning Final Project-
`Dreaddit: A Reddit Dataset for Stress Analysis in Social Media`

## Environment
There are two environments that you can run this project scripts through: local Docker container or Google Colab notebook 
### Docker Container
1. Create the requested environment using docker-compose: `docker-compose up -d`
2. Attach to the running container with bash: `docker exec -it {container_id} bash`
3. Run python scripts (detailed below)
### Google Colab
1. Copy this entire repository into your Google Drive (including the data folder)
2. Open the "run_project.ipynb" notebook that you can find in the root folder of this Repo.
3. Change the hard-coded Drive path to this folder (second code cell)
4. Run cells
** Note that for this example we changed the maximum number of epochs to be 2

* The `src/config.json` file lists the hyperparameters used during training and evaluation of the model, change them to get different results.

## Data
### Dreaddit challenge
- Make the Dreadit dataset is located under `data/` directory

### Stance challenge
Download data into the "data" folder inside the root directory. 
The structure should be:
- data
    - rumoureval-2019-test-data
        - reddit-test-data
        - twitter-en-test-data
    - rumoureval-2019-training-data
        - reddit-dev-data
        - reddit-training-data
        - twitter-english
        - dev-key.json
        - train-key.json
    - final-eval-key.json

## Relevant Scripts
- Training word2vec model - Run the "train_word2vec_on_reddit_data.py" script to train the Word2Vec model on the whole dataset of the paper.
If you preffer you can download the model files from our shared Drive folder - __________

### Dreaddit challenge
There are two main scripts you have to run:
1. data processing: `python src/data_processing_dreaddit.py`. 
The preprocessed data (output) should be available in data_preprocessing/saved_data_RumEval2019.
** Notice that this script is taking a long time to run due to the large processing of the data.
2. Run a model: `python src/run_model.py -m {model_name}`. You need to provide the script the name of the model you want to run.
Valid options are:
    - baseline - our baseline - BERT
    - gpt2 - GPT2 model
    - roberta - Plain RoBERTa model
    - roberta_with_features - RoBERTa model with the best features combinations we found.

    For example: `python src/run_model.py -m gpt2`

** Notice - you need to run everything from root directory (so the working directory is root directory)

### Stance challenge
There are two main scripts you have to run:
1. data processing: `python src/data_processing_stance.py`. 
The preprocessed data (output) should be available in data_preprocessing/saved_data_RumEval2019.
** Notice that this script is taking a long time to run due to the large processing of the data.
2. Run a model: `python src/run_model.py -m {model_name}`. You need to provide the script the name of the model you want to run.
Valid options are:
    - baseline - our baseline - BERT
    - gpt2 - GPT2 model
    - roberta - Plain RoBERTa model
    - roberta_with_features - RoBERTa model with the best features combinations we found.

    For example: `python src/run_model.py -m gpt2`

** Notice - you need to run everything from root directory (so the working directory is root directory)

## Results-
- The flow prints to the terminal the loss, accuracy, and F1 score for each epoch.
- At the end of the execution `plots/` directory will include 3 plots of the loss, accuracy, and F1 progress with respect to the epoch number.