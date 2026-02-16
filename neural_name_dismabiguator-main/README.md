# LSPO Dataset and Neural Author Name Disambiguator

The Neural Author Name Disambiguator (NAND) is a baseline method trained on the [LSPO dataset](https://zenodo.org/records/11489161) for author name disambiguation.

## Repository Contents

- **AND_dataset_builder.py**: Builds the pairs for training the AND_nn_exp.py models with the chosen hyperparameters. Here, the specter + chars2vec model architecture is displayed.
- **AND_readdata_exp.py**: Contains the necessary data module for the NAND model.
- **experiment.py**: Runs the training process for various seeds to later test the standard error of the NAND model training.
- **results.py**: Evaluates the trained models.

## NAND Model Training, Validation, and Testing Process

The NAND model undergoes a structured training, validation, and testing process. This ensures robust performance and accurate author name disambiguation. The details of this process are documented within the scripts provided in this repository.

## Environment Setup
conda env create -f environment.yml

## Run an Example 

1. **Building Training Pairs**:
   python AND_dataset_builder.py

The embeddings must be output and included to build the pairs (e.g., the transformers library can be used to output embeddings). 

2. **Training the Model**:
   python experiment.py 

Which includes the AND_nn_exp.py and AND_readdata_exp.py to run an experiment. 

3. **Evaluating the Model**:
   python results.py

The checkpoint files must be updated for different experiments. 

Each of the files must be updated for the different representation method used for outputting embeddings. 

## Model Architecture

The AND_nn_exp.py file contains the architecture utilzied for the Specter and chars2vec embeddings. 

## Hyperparameters

Hyperparameters for training can be adjusted in the corresponding script files. The current setup displays the Specter and chars2vec model architecture.

## Evaluation

The evaluation process includes testing the standard error of the NAND model training. This ensures the reliability and accuracy of the model across different runs.

## Dataset

The LSPO dataset is crucial for training the NAND model. You can access the dataset [here](https://zenodo.org/records/11489161).

---

For more detailed information, please refer to the individual script files and their inline documentation.

License
-------

This project is Copyright (c) DeepThought Initiative and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.

