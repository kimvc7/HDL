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

2) Then ```python train.py --subset-ratio 0.8 --batch-size 64 --stable=1```


TODO:

- Get the performance on validation set every 500 updates or so.

- Code the std of the weights using the 10 last logits. Be careful, one std per logit per XX seeds, then averaged the size of test set.

