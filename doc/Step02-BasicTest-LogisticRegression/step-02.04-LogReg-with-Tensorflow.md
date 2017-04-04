Logistic Regression with TensorFlow [[back](index.md)]
==========================


To work with the MNIST dataset we will use a helper method from Tensorflow package:


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
```

1. You can Run TensorFlow Logistic Regression example from command-line:
```
$ cd ~/gitlab.altoros/776_DL_Libs_Benchmark.git/src/Step02/SubStep-04-Tensorflow
$ python ./TensorFlow_LogisticRegression.py
```

or with jupyter notebook:
```
$ cd ~/gitlab.altoros/776_DL_Libs_Benchmark.git/src/Step02/SubStep-04-Tensorflow
$ jupyter notebook TensorFlow_LogisticRegression_Notebook.ipynb
```

and then open in browser URL: [http://ec2-54-86-145-119.compute-1.amazonaws.com:9999](http://ec2-54-86-145-119.compute-1.amazonaws.com:9999)

--------
Accuracy plot:


![TensorFlow LogReg AccPLot](img/Step02/TensorFlow/TensorFlow_MNIST_LogReg_AccPlot.png)


Network weights visualization:

![TensorFlow LogReg AccPLot](img/Step02/TensorFlow/TensorFlow_MNIST_LogReg_Weights.png)


