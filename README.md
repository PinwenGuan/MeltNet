# MeltNet
A deep neural network predicting melting temperature of alloy with arbitrary composition

## Installation

Download the files and put them in a folder. In the folder, type:<br>
```
tar zxvf liq.tgz
```
The code is dependent on pymatgen, PyTorch and scikit-learn, which should be installed first.<br>
For Bayesian optimisation, dragonfly (https://github.com/dragonfly/dragonfly) should be installed first.<br>

## General instructions

Adjustable parameters and their explanations are listed in the beginning of each script.<br>

## Generate data

```
python generate_data.py
```
This command reads liquidus data from the files in the liq/ folder. A folder with default name "cv" containing all the necessary data for training will be generated.<br>

## Train MelNet

```
python train.py
```

## Train MelNet using the ensemble method

```
python ensemble_train.py
```

## Bayesian Optimization of hyperparameters

```
python bo.py
``` 
The configuration file should be present in the folder, and the default name is "config1.json", which is adjustable. The hyperparameters and associated loss function value in each evaluation will be printed.<br> 

## Author
This software was written by Pin-Wen Guan who was advised by Prof. Venkat Viswanathan.

## Reference

Please cite the reference below if you use MeltNet in your work:<br>

Pin-Wen Guan and Venkatasubramanian Viswanathan. “MeltNet: Predicting alloy melting temperature by machine learning”. In: arXiv preprint arXiv:2010.14048 (2020).<br>

```
@article{guan2020meltnet,
  title={MeltNet: Predicting alloy melting temperature by machine learning},
  author={Guan, Pin-Wen and Viswanathan, Venkatasubramanian},
  journal={arXiv preprint arXiv:2010.14048},
  year={2020}
}
```
