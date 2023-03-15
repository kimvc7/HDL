# Holistic Deep Learning

This code corresponds to the paper Holistic Deep Learning: https://arxiv.org/abs/2110.15829
by Dimitris Bertsimas, Kimberly Villalobos Carballo, Léonard Boussioux, Michael Linghzi Li, Alex Paskov, Ivan Paskov.

This paper presents a novel holistic deep learning framework that simultaneously addresses the challenges of vulnerability to input perturbations, overparametrization, and performance instability from different train-validation splits. The proposed framework holistically improves accuracy, robustness, sparsity, and stability over standard deep learning models, as demonstrated by extensive experiments on both tabular and image data sets. The results are further validated by ablation experiments and SHAP value analysis, which reveal the interactions and trade-offs between the different evaluation metrics. To support practitioners applying our framework, we provide a prescriptive approach that offers recommendations for selecting an appropriate training loss function based on their specific objectives.

# Introduction to the code

To train a basic feed-forward neural network go to ```src``` and execute for instance:

```python train.py --data_set cifar10 --network_type ALEX```

Valid data_set values include "mnist", "fashion_mnist", "cifar10" as well as  "1", ..., "44" corresponding to the UCI data sets described in the config file.

To train a VGG3 network:
```python train.py --data_set mnist --network_type VGG3```

Overall, one can replace VGG3 by pre-supported architectures, such as ALEX (AlexNet) or VGG16. You can develop your own network architecture and name it.

To train a stable network, add: ```--is_stable``` and tune ```--stab_ratio_range 0.8```.

E.g., ```python train.py --data_set cifar10 --is_stable --stab_ratio_range 0.8```

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
- ```utils_init.py```: contains function to load the different arguments and launch a grid search.
- ```utils_model.py```: contains initialization function for the neural network models.
- ```utils.py```: contains miscellaneous elements.

You can modify parameters in the config file with the desired values.

The documentation is in progress and will be updated regularly.
