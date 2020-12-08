
**Experiment 1

**Investigation**:

Find how the number of iterations influence the test loss for the reward network. 

**Motivation**: 

To compare different reward netwrok parameters, losses on each model set up must converge to eliminate the effect of the number of iterations on the model performance. 

**Result**: 

Figure 1 compares 2 runs with the same hyperparameters, but with different number of iteration steps. To ensure reprodicibility the training script included https://github.com/SergeyAnufriev/Mol_gan/blob/10403a118e38cf610be57ab2a954eb7685098413/train.py#L20

![alt text](https://github.com/SergeyAnufriev/Mol_gan/blob/master/figures/Fig.1.png)
[Fig.1]

The model, which run longer has lower test loss. Fig.2 illustrates approximate convergence for the test loss. 

![alt text](https://github.com/SergeyAnufriev/Mol_gan/blob/master/figures/Fig2.png)
[Fig.2]

**Conclusion**:

Longer  run loss: 0.0014
Shorter run loss: 0.0022

The model converges at 10^3 iteration steps [Fig.2] 

**To do next**

Slow convergence requires more time to run each experiment. Time is important,since many different model configurations must be tested. 


* Measure time to train one model configuration. 
