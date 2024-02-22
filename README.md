# SenCID
Senescence Identification of a Cell

### Requirements

+ Linux or UNIX system
+ Python >= 3.6
+ tensorflow >= 2.0, <2.5

### Create environment

```
conda create -n SenCID python=3.6
conda activate SenCID
```

### Installation

The `SenCID` python package is in the folder SenCID. You can simply install it from the root of this repository using

```
pip install . 
```

Alternatively, you can also install the package directly from GitHub via

```
pip install git+https://github.com/JackieHanLab/SenCID.git
```

In case of any version inconsistency, you may use 
```
pip install --upgrade pip
```
or
```
conda env create -f SenCID.yaml
```
to clone the environment before the installation.

### Tutorial

See SenCID/demo/SenCID_tutorial.ipynb
