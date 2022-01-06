# Few-Shot Learning Tutorial
This repository contains code for my FSL tutotial.

Before jumping into the tutorial, make sure that all the conda environments are 

The goals of this tutorial are as follows:
 * To implement task sampler for FSL tasks to gain an intuition of FSL problem setup. The relevant 
 * To implement a couple of FSL methods to gained an intuition of meta-learning works

The task 
 Prototypical Networks (Snell et al., 2017)
 * To implement Model-Agnostic Meta-Learning (MAML, Finn et al., 2017)
 * To implement Proto-MAML (Triantafillou et al., 2017)



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
conda install -y -c intel scikit-learn
conda install -y -c anaconda pytest
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
____

### Contributions
This repository is based the following GITHUB repository:
 * https://github.com/mattochal/imbalanced_fsl_public
