# Uncertainty in Deep Learning 2024

## Jensen-Shannon Divergence For Variational Inference In Continual Learning

This project aims to investigate how changing the Kullback-Leibler divergence 
function to the Jensen-Shannon divergence impacts the performance of Variational
Continual Learning, within an experimental setup specifically tailored for 
assessing continual learning tasks.

## Getting started:

Start by cloning the repository:

```
git clone https://github.com/anonymousoxford2024/UDL2024.git
```

Or via SSH:

```
git clone git@github.com:anonymousoxford2024/UDL2024.git
```

If you don't already have Poetry installed, by running the following command. 
More information on this can be found in the 
[Poetry Documentation](https://python-poetry.org/docs/).

```
pip install poetry
poetry install --all-extras
```

Activate the virtual environment with:

```
poetry shell
```

You're now ready to replicate the study's findings.

## Reproduce results:

Running the following commands will retrain and evaluate the models detailed 
in the report, on a dataset of your choice. 
This will use the final hyperparameters, that are stored in `hyper_params.json`.

For MNIST:
```
python src/continual_learning/run_experiments.py --dataset MNIST
```

For CIFAR10:
```
python src/continual_learning/run_experiments.py --dataset CIFAR10
```

For CIFAR100:
```
python src/continual_learning/run_experiments.py --dataset CIFAR100
```

## Model Retraining and Hyperparameter Search

To redo hyperparameter searches, choose one of the commands below. 
You can specify a dataset with --dataset DATA. 
If unspecified, it defaults to MNIST.

VCL Singlehead:
```
python src/continual_learning/vcl/run_vcl_singlehead.py 
```

VCL Multihead:
```
python src/continual_learning/vcl/run_vcl_multihead.py 
```

EWC Singlehead:
```
python src/continual_learning/ewc/run_ewc_singlehead.py 
```

EWC Multihead:
```
python src/continual_learning/ewc/run_ewc_multihead.py 
```


