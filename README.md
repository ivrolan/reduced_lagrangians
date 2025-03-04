# Reduced-Order Lagrangian Dynamics


Source code for the learning of high-dimensional Lagrangian systems in structure-preserving latent spaces.
For details on the architecture, check out [A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics]( https://arxiv.org/abs/2410.18868 ).

## getting started


### download example data

To run our examples, you need to download & unzip the data-folder inside the repository.
Download at: https://drive.google.com/file/d/1ZtJsd2Ya46SgkByPNcmTzX_Z73UzN2lA/view?usp=sharing

The folder structure should now look like this:
```
reduced_lagrangians/  
│── data/  # datasets  
│── models/ # network implementations   
│── pretrained/ # trained model parameters  
│── utils/ # e.g. Biorthogonal and SPD Manifolds for network parameter optimization  
│── train_lnn.py # script to train a geometric Lagrangian Network   
└── train_ro_lnn.py # script to train a reduced-order Lagrangian Network
```

### environment
You can set up a conda environment as follows:
```
conda env create -f reduced_lagrangians.yml
conda activate reduced_lagrangians
```

## example scripts

### Reduced-Order Lagrangian Neural Network

This script trains and evaluates a Reduced-Order Lagrangian Network (`models/reduced_LNN`) on a dataset of a simulated cloth (600DoFs).

Call:
```
python train_ro_lnn.py <OPTIONS>
```

A pre-trained model is included to skip training times, if called without arguments, the script will only evaluate predictions from a pretrained model.

The following arguments are available:

| **flag**                   | **description**                                                         |
|----------------------------|-------------------------------------------------------------------------|
| `--train`                  | re-train model                                                          |
| `--save_params`            | save parameters of trained model                                     |
| `--save_model_to <path>`   | path location for saved model, default: `'./pretrained/ro_lnn_cloth_v1'` |
| `--load_model_from <path>` | path location to load model from, default: `'./pretrained/ro_lnn_cloth'` |

Other hyperparameters can be modified directly in the script.



### Geometric Lagrangian Neural Network
This example features the training and evaluation of a geometric Lagrangian Network (`models/geom_LNN`) on a dataset of a simulated double pendulum. This version does not include a latent representation and is suitable for low-dimensional systems. 

Call:
```
python train_lnn.py
```
Hyperparameters can be adjusted directly in the script.


## reference

If this work was useful for you, we would highly appreciate citations of the corresponding reference:
```
@inproceedings{friedl2025reduced,
 author = {Friedl, Katharina and Jaquier, No{\'e}mie and Lundell, Jens and Asfour, Tamim and Kragic, Danica},
 title = {A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics},
 booktitle = {Intl. Conf. on Learning Representations (ICLR)},
 year = {2025},
}
```





