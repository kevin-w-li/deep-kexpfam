# deep-kexpfam
This repo has the code for the paper:

> [Learning deep kernels for exponential family densities](https://arxiv.org/abs/1811.08357)\
> by [Li Wenliang](https://kevin-w-li.github.io/)\*, 
[Danica J. Sutherland](http://djsutherland.ml)\*, 
[Heiko Strathmann](http://herrstrathmann.de/) and 
[Arthur Gretton](http://www.gatsby.ucl.ac.uk/~gretton/)

To install and run 
1. clone the repo
2. inside the repo run `pip install -r requirements.txt`
3. usage is in `training_demo.ipynb` notebook
4. `MoG.ipynb` reproduces the mixture of Gaussian example in Figure 1.

The code uses TensorFlow 1.4, with CUDA 8.0 and cuDNN 6.0
