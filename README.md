# Object Recognition with CERFA-10 dataset

## Getting started

1. Clone or fork this repository
2. Install [Poetry](https://python-poetry.org/docs/) (if it's not already present in your system)
3. Go to the repository location on your system and create a virtual environment: `python -m venv .venv`
4. Install the project dependencies: `poetry install`
5. Activate the virtual environment: `poetry shell`

## Available commands

Once you have installed the project dependencies and activated your virtual environment, you can use the following commands that were already implemented:

- `prepare`: Performs data preparation operations on the CERFA-10 dataset
- `train`: Performs training on an CERFA-10 dataset and saves a model at a given location
- `evaluate`: Retrieves a model from a given location, apply it on a test dataset and prints some evaluation metrics (accuracy, precision and recall)
