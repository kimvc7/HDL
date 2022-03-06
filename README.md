# Holistic Deep Learning

This code corresponds to the paper Holistic Deep Learning: https://arxiv.org/abs/2110.15829
by Dimitris Bertsimas, Léonard Boussioux, Kimberly Villalobos Carballo, Michael Linghzi Li, Alex Paskov, Ivan Paskov.

There is much interest in deep learning to solve challenges that arise in applying neural network models in real-world environments. In particular, three areas have received considerable attention: adversarial robustness, parameter sparsity, and output stability. Despite numerous attempts on solving these problems independently, there is very little work addressing the challenges simultaneously. In this paper, we address this problem of constructing holistic deep learning models by proposing a novel formulation that solves these issues in combination. Real-world experiments on both tabular and MNIST dataset show that our formulation is able to simultaneously improve the accuracy, robustness, stability, and sparsity over traditional deep learning models among many others.

# Introduction to the code

To train a basic feed-forward neural network go to ```src``` and execute:

```python3 train.py --batch_range 64 --network_size 256 128 --l2 1e-5 --data_set uci10 --train_size 0.8 --lr 3e-4 ```

To train a VGG3 network:
```python3 train.py --data_set mnist --network_type VGG3```

Overall, one can replace VGG3 by pre-supported architectures, such as ALEX (AlexNet) or VGG16. You can develop your own network architecture and name it.

To train a stable network, add: ```--is_stable``` and tune ```--stab_ratio_range 0.8```.

E.g., ```python train.py --data_set uci10 --is_stable --stab_ratio_range 0.8```

To train a robust neural network, add: ```-r 1e-3``` and tune the regularization value.

To train a sparse neural network, add: ```--l0 1e-5``` and tune the regularization value.

The folder ```src``` contains:
- ```config.json```: the configuration file to determine some hyperparameters in your experiments.
- ```input_data.py```: the file to load and prepare the different datasets.
- ```pgd_attack```: implements the neural network attacks (Copyright Madry et al)
- ```train.py```: runs the training. The most important file.
- ```Networks```: this folder contains the neural network architectures. We provide feed-forward and convolutional neural networks that can be combined into a desired network.

The folder ```utils``` contains:
- ```utils_print.py```: contains the printing functions.
- ```utils_model.py```: contains the loss functions and model dictionaries.
- ```utils_init.py```: contains function to load the different arguments.
- ```utils_MLP_model.py```: contains initialization function for MLP models. Needs to be adapted for CNNs.
- ```utils.py```: contains miscellaneous elements.

You can modify parameters in the config file with the desired values.

The documentation is in progress and will be updated regularly.



