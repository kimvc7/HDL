# DHL
NOTE: Requires Tensorflow 2 and Keras.
 Modify parameters in config file with the desired values.
 To train with stability run train.py
 To train without stability you need to replace model.max_xent with model.xent inside the Adam Optimizer in file train.py
 Includes code to visualize Tensorboards.

To use the code for montecarlo Stability:
1) Go to ```Stability```.

2) Then ```python train.py --subset-ratio 0.8 --num-subsets 5 --batch-size 64```

You can choose these parameters directly from the command line and have more modularity in the config file, but these 3 parameters are the most important a priori.

To use the code for dual Stability:
1) Go to ```StabilityDual```.

2) Check the ```config.json``` file to put the desired config.

3) Then ```python train.py --ratio_range 0.7 0.8 0.9 --batch_range 16 32 64 128 --stable --data_set=cifar --dropout 0.9 --l2 0.1``` for stability or
```python train.py --ratio_range 0.7 0.8 0.9 --batch_range 16 32 64 128 --data_set=cifar``` for no stability.

If possible: get the performance on validation set every 500 updates or so.
TODO:

- Merge MC and StabilityDual in one ```train.py``` to avoid duplication of efforts.
- Code the GINI index for stability.
- Code a better architecture for Cifar: using CNN

A PRIORI DONE:
- Code the std of the weights using the 10 last logits. Be careful, one std per logit per XX seeds, then averaged the size of test set.

