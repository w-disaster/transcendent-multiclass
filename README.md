# Transcendent Multiclass

![CI status](https://github.com/w-disaster/transcendent-multiclass/actions/workflows/check.yml/badge.svg) 
![Version](https://img.shields.io/github/v/release/w-disaster/transcendent-multiclass?style=plastic)

This repository allows the user to apply Transcendent-like Concept drift detection for both binary and multiclass problems.

Modifications have been made to the ICE implementation specifically, leaving the other solutions out of the scope (e.g. TCE, CCE etc.). Furthermore, the thresholding phase is temporarily disabled due to time-contraints, so the threshold should be derived manually once the calibration phase completes.

This project extends [Transcendent](https://github.com/s2labres/transcendent-release/tree/main) by implementing a Non-Conformity Measure (NCM) based on Random Forest proximities, introduced from the paper ["Prediction with Confidence Based on a Random Forest Classifier"](https://s2lab.cs.ucl.ac.uk/projects/transcend/).


## Prerequisites

- Make sure you have a running and active version of [Docker](https://docs.docker.com/engine/install/).

## Usage:

1. Clone the repository and change directory:
    ```bash
    git clone git@github.com:w-disaster/transcendent-multiclass.git && cd transcendent-multiclass
    ```

2. Setup `docker-compose.yaml` and the directory containing the Training and Testing set:

    `ice.py` looks for the training and testing dataset, which should be mounted inside the Docker container.
    As default, `docker-compose.yaml` maps the local directory `./splitted_dataset/` inside the container.
    Also, two environment variables should be set: `PE_DATASET_TYPE` and `TRAIN_TEST_SPLIT_TYPE`, which allows to find the specific train/test split for a specific dataset. 
    In other terms, `splitted_dataset/` directory should have this structure:


    ```plaintext
    splitted_dataset/
    ├── PE_DATASET_TYPE/
    |   ├── TRAIN_TEST_SPLIT_TYPE/
    │   │    ├── X_train.csv
    │   │    ├── y_train.csv
    │   │    ├── X_test.csv
    │   │    └── y_test.csv
    │   └──
    └── 
    ```

    So that you can configure the pipeline for different datasets and train/test splits. E.g.:

    ```plaintext
    splitted_dataset/
    ├── ember/
    |   ├── random_split/
    │   │    ├── X_train.csv
    │   │    ├── y_train.csv
    │   │    ├── X_test.csv
    │   │    └── y_test.csv
    │   └──
    │   ├── time_based/
    │   │    ├── X_train.csv
    │   │    ├── y_train.csv
    │   │    ├── X_test.csv
    │   │    └── y_test.csv
    │   └──
    └── 
    ├── motif/
    │   ...
    └── 
    ```

3. *Deploy* the Concept Drift Pipeline
    
    A `results/` directory will be locally created containing the credibility ($p$-values) and confidence scores for both calibration and testing sets.


