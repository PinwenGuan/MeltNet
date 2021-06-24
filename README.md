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

## Uncertainty quantification of thermodynamic properties

The script gpaw/relax-n.py demonstrate using BEEF (Bayesian error estimation functional) to calculate an ensemble of E-V data by the code GPAW. Supposing you generate 2000 e-v.dat files from this calculations, i.e., e-v1.dat, e-v2.dat, ..., e-v2000.dat, then you can using a bash script like the following to obtain an ensemble of thermodynamic properties:<br>
```
for i in $(seq 1 2000)
do
python -c "from post import eos_gpaw;eos_gpaw(ev='e-v`echo $i`.dat',struc='POSCAR')"
done
```
In this way, an estimation of uncertainty of thermodynamic properties caused by DFT functionals can be attained. There are also functions in the post module for generating probability distribution function of thermodynamic properties, calculating histograms of thermodynamic properties at certain temperature and their statistics and plotting results. 

## Author
This software was primarily written by Pin-Wen Guan who was advised by Prof. Venkat Viswanathan.

## Reference

Please cite the reference below if you use dePye in your work:<br>

Pin-Wen Guan, Gregory Houchins, Venkatasubramanian Viswanathan. Uncertainty quantification of DFT-predicted finite temperature thermodynamic properties within the Debye model. The Journal of Chemical Physics, 2019, 151(24): 244702.<br>

```
@article{guan2019uncertainty,
  title={Uncertainty quantification of DFT-predicted finite temperature thermodynamic properties within the Debye model},
  author={Guan, Pin-Wen and Houchins, Gregory and Viswanathan, Venkatasubramanian},
  journal={The Journal of Chemical Physics},
  volume={151},
  number={24},
  pages={244702},
  year={2019},
  publisher={AIP Publishing LLC}
}
```
