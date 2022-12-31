# Installation

This document contains detailed instructions for installing dependencies for DPINet. The code is tested on an Ubuntu 20.04 system with Nvidia GPU.

### Requirments
* Conda with Python 3.8
* Nvidia GPU
* PyTorch 1.10.1
* pyyaml
* yacs
* tqdm
* matplotlib
* OpenCV

## Step-by-step instructions

#### Create environment and activate
```bash
conda create --name DPINet python=3.8
source activate DPINet
```

#### Install numpy/pytorch/opencv
```bash
conda install numpy
conda install pytorch=1.10.1 torchvision cudatoolkit=11.0 -c pytorch
pip install opencv-python
```

#### Install other requirements
```bash
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboard future mpi4py optuna
```

#### Build extensions
```bash
python setup.py build_ext --inplace
```



