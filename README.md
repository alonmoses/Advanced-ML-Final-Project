# Advanced-ML-Final-Project
Development Repository for Advanced Machine Learning Final Project-
`Dreaddit: A Reddit Dataset for Stress Analysis in Social Media`

## Environment
- Use docker to create the requested environment
`docker-compose up -d`
- In order to execute the model run under the root directory:
`python src/execute_flow.py`

* The `src/config.json` file lists the hyperparameters used during training and evaluation of the model, change them to get different results.

## Data
- Make the Dreadit dataset is located under `data/` directory

## Results-
- The flow prints to the terminal the loss, accuracy, and F1 score for each epoch.
- At the end of the execution `plots/` directory will include 3 plots of the loss, accuracy, and F1 progress with respect to the epoch number.