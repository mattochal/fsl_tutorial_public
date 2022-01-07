# Few-Shot Learning Tutorial
This repository contains code for my FSL tutotial.

Before jumping into the tutorial, make sure that all the conda environments are 

The goals of this tutorial are as follows:
 * Understanding how to train FSL algorithms
 * Understanding how baseline FSL algorithms work
 * Experience training and evaluating FSL algorithms

The tasks are marked in files with a "TODO:" note:
 1. Study this repository and understand how things fit together
    * Ask questions if you are not sure!
    * Unzip the fish data set in to the [./data/](./data/) folder
    * You can visualise the content of the dataset using the [./visualiser.ipynb](./visualiser.ipynb)
 2. Fix Task Sampler:
    * Go to: [./src/tasks/fsl.py](./src/tasks/fsl.py)
 3. Fix Prototypical Networks:
    * Original Paper: [(Snell et al., 2017)](https://arxiv.org/pdf/1703.05175.pdf)
    * Code file: [./src/models/protonet.py (line 66)](./src/models/protonet.py)
 4. Fix MAML:
    * Original Paper: [(Finn et al., 2017)](https://arxiv.org/pdf/1703.03400.pdf)
    * Code file: [./src/models/maml.py (lines 137 and 164)](./src/models/maml.py)
 5. Fix Proto-MAML
    * Original Paper: [(Triantafillou et al., 2017)](https://arxiv.org/pdf/1903.03096.pdf)
    * Code file: [./src/models/protomaml.py (lines 30 and 54)](./src/models/protomaml.py)

## Running Repository
### Dependecies

* numpy
* python 3.8+
* pytorch
* tqdm
* pillow
* scikit-learn

This code was tested on:
 * Ubuntu 16.04, cuda release 10.1
 * Ubuntu 20.04, cuda release 10.1

To set up the specific conda environment run:
```
conda env create --file fsl_tutorial.yml
```

OR set it up manually:
```
conda create -n fsl_tutorial python=3.8
conda activate fsl_tutorial
conda install -y pytorch torchvision -c pytorch
conda install -y -c conda-forge tqdm
conda install -y -c anaconda pillow scikit-learn
conda install -y -c anaconda pytest

conda install -y -c conda-forge ipykernel
conda install -y -c conda-forge matplotlib
```

### Structure

The framework is structured as follows:

```
.
├── generator.py          # Experiment generator
├── data/                 # Default data source
├── [experiments/]        # Default script, config and results destination
└── src
    ├── main.py           # Main program
    ├── datasets          # Code for loading datasets
    ├── models            # FSL methods, baselines, backbones
    ├── tasks             # Standard FSL Task Sampler and Batch Sampler
    └── utils             # Utils, experiment builder, performance tracker, dataloader
```

### Data

See ```./data/README.md```


## Generating Main Experiments

To generate the training scripts:
```
python generator.py
```

Add ```--gpu <GPU>``` to specify the GPU ID or ```cpu```

To generate the evaluation:
```
python generator.py --test
```

## Running main program

To run a specific experiment setting from a configuration file:
```
python src/main.py --args_file <CONFIGPATH> --gpu <GPU>
```
