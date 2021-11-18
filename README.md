# Holistic Deep Learning

To train a basic feed-forward neural network go to ```src``` and execute:

```python train.py --batch_range 64 --network size 256 128 --stab_ratio_range 0.8 --l2 1e-5 --data_set uci10 --train_size 0.8 --lr 3e-4 --val_size 0.2```

To train a stable network, add: ```--is_stable``` and tune ```--stab_ratio_range 0.8```.
E.g., ```python train.py --data_set uci10 --is_stable --stab_ratio_range 0.8```

To train a robust neural network, add: ```-r 1e-3```.

To train a sparse neural network, add: ```--l0 1e-5```.

The folder ```src``` contains:
- ```config.json```: the configuration file to determine some hyperparameters in your experiments.
- ```input_data.py```: the file to load and prepare the different datasets.
- ```pgd_attack```: implements the neural network attacks
- ```train.py```: runs the training. The most important file.
- ```Networks```: this folder contains the neural network architectures. We provide feed-forward and an old version of CNN_model.py that needs to be adapted to fit this code.

The folder ```utils``` contains:
- ```utils_print.py```: contains the printing functions.
- ```utils_model.py```: contains the loss functions and model dictionaries.
- ```utils_init.py```: contains function to load the different arguments.
- ```utils_MLP_model.py```: contains initialization function for MLP models. Needs to be adapted for CNNs.
- ```utils.py```: contains miscellaneous elements.

You can modify parameters in the config file with the desired values.

The documentation is a work in progress and will be updated soon.



