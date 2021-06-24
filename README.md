# MeltNet
A deep neural network predicting melting temperature of alloy with arbitrary composition

## Installation

Download the files and put them in a folder. In the folder, type:<br>
```
tar zxvf liq.tgz
```
The code is dependent on pymatgen, PyTorch and scikit-learn, which should be installed first.<br>
For Bayesian optimisation, dragonfly (https://github.com/dragonfly/dragonfly) should be installed first.<br>

## Prepare the input files

Since you have finished the energy calculations, you should already have the structure file in your working folder (if not you need to do so). Currently dePye supports the structure files of VASP and Quantum ESPRESSO (QE), i.e., POSCAR and the .in file.<br> 
Normally, the only input file you need to prepare is e-v.dat (can be a name ended with .dat) such as:<br>
```
# V(angstrom^3) E(eV) vol_0.98 vol_0.99 vol_1.00 vol_1.01 vol_1.02

69.131955   -7728.27656469   -7728.27686878   …
71.217852   -7728.28518104   -7728.29116264   …
73.345300   -7728.25348998   -7728.25450442   …
75.514684   -7728.18594466   -7728.17339096   …
77.726434   -7728.08681822   -7728.05316957   …
```
The comment lines do not matter. The first column is the volume (in angstrom^3), and the second, third, … is the energies (in eV). At least 5 V-E data points are needed. In the case of multiple sets of energies, each energy set will generate a result. For example, one can get two E-V datasets with and without relaxation, and get two sets of properties.<br>
Note: the input data should be corresponded to the number of atoms in the structure file. 

## Run dePye

First activate the virtual environment where pymatgen is installed:<br>
```
source activate (pymatgen environment)
```
In your working folder, type:<br>
```
depye -tmax=1000 –tstep=10 POSCAR
```
Here,  tmax means the maximum temperature you want to calculate, tstep is the temperature step in unit of 10 K in the output file, and POSCAR is the structure file. If not specified, the default setting will be used, i.e., tmax=1000, tstep=1 and POSCAR. The order of these flags does not matter. Another example:<br>
```
depye -tmax=1000 cu.in
```
where the default tstep=1 (10 K) and the QE input file cu.in will be used.<br>
You can also use an input file whose name is not e-v.dat (but should end with .dat):<br>
```
depye POSCAR e-v2.dat
```
You can adjust the scaling factor in the Debye temperature (default is 0.617):<br>
```
depye POSCAR s=0.8
```
You can put a 'poisson' file which only contains the value of the poisson ratio of the material, then the code will calculate the scaling factor from it and override the default or the assigned s.<br>
You can select EOS, e.g.:<br>
```
depye POSCAR eos='birch_murnaghan'
```
The available EOS options are:<br>
```
        "vinet": Vinet,
        "murnaghan": Murnaghan,
        "birch": Birch,
        "birch_murnaghan": BirchMurnaghan,
        "pourier_tarantola": PourierTarantola,
        "deltafactor": DeltaFactor,
        "numerical_eos": NumericalEOS
```
You can turn off showing figures by adding show='F':<br>
```
depye POSCAR show='F'
```
You can determine how many atoms the output quantities are corresponded to:<br>
```
depye POSCAR nat=1
```
The above outputs the quantities per atom.<br>
Note: the default setting is per formula.<br>
You can decide which properties are kept in the outputed figures:<br>
```
depye POSCAR prop=[0,3]
```
The above shows thermodynamic quantities for only Gibbs energy and entropy. The quantity codes are<br>
```
G V B S H TEC Cp Cv Bp
0 1 2 3 4  5  6  7  8
```
You can append experimental data (or data you get by other methods, e.g., phonon method):<br>
```
depye POSCAR data.expt
```
Note: the default name of the experimental file is expt. If you want to use another name, the file should end with .expt.<br> 
Note: for VASP, the structure file should be named “POSCAR”. For QE, it should be ended with “.in”. The expt file should look like:<br>
```
# T (K) G (kJ/mol) V (cm^3/mol) B (GPa) S (J/mol/K) H (kJ/mol) TEC (1e-6/K) Cp (J/mol/K) Cv (J/mol/K)
T B ref
298 100 1
298 95 2 
500  90 2

T TEC ref
298 40 3

#You can name the references for literature data shown in the figures by alias below. The names are in the order as 1,2,3
#alias: author A, author B, author C
```

## Get the results

After dePye running, a series of figures about EOS and thermodynamic properties will be prompted out.  The output file “Debye-thermo” contains the EOS parameters and the data of Gibbs energy, volume, bulk modulus, entropy, enthalpy, thermal expansion and heat capacity as functions of temperature, which can be further as inputs for thermodynamic modelling.<br> 

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
