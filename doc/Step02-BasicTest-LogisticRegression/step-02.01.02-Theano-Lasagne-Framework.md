Logistic Regression with Theano and Lasagne Framework [[back](index.md)]
==========================

At this point Theano is already installed in the system.

1. Install lateset Lasagne framework:
```
sudo pip install Lasagne
```

2. Run prepared emxaple with jupyter notebook:
```
$ cd ~/gitlab.altoros/776_DL_Libs_Benchmark.git/src/Step02/SubStep-01.02-Theano-Lasagne
$ jupyter notebook Theanoe_Lasagne_LogisticRegression.ipynb
```

Go to the URL: [http://ec2-54-172-161-206.compute-1.amazonaws.com:9999/](http://ec2-54-172-161-206.compute-1.amazonaws.com:9999/)


To work with the MNIST dataset we will use a helper method from Tensorflow package:

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
```

--------
Traing process:

![Loss plot](img/Step02/Theano_Lasagne/Theano_Lasagne_Train_Plot.png)

--------
Weights visualisation:


![Weights](img/Step02/Theano_Lasagne/Theano_Lasagne_Weight_Matrix.png)
