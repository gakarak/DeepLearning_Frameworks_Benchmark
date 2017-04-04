Logistic Regression (quick intro) [back](README.md)
==========================

[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) - one of
the ways to build a linear model for the available
training set data (X_i,Y_i): X=>Y.

On the other hand, this method (Linear Regression) can
be regarded as a degenerate case of MultiLayer Fully Connectad Netwok.

![Logistic Regression as MLP](img/Step02/LogReg_as_MLP_crop.png)

As parameters LogisticRegression model has a matrix of weights with size NxK

and biases vector with size Nx1, where N - Size of feature vector (if feature

vector is gray-scale image with size LxL, then N=L*L), and K - is number of

classes.

Non-linearity in the model is selected as SoftMax: Y=SoftMax(W*X + B).

CrossEntropy is used as a loss function for multi categorical classification.

[MNIST](http://yann.lecun.com/exdb/mnist/) image database used as training set.

---------------
Next: [2.1 Logistic Regression with Theano](doc/Step02-BasicTest-LogisticRegression/step-02.01-LogReg-with-Theano.md)

---------------
- [2.1 Logistic Regression with Theano](doc/Step02-BasicTest-LogisticRegression/step-02.01-LogReg-with-Theano.md)
    - [2.1.1 Theano: Blocks Framework](doc/Step02-BasicTest-LogisticRegression/step-02.01.01-Theano-Blocks-Framework.md)
    - [2.1.2 Theano: Lasagne Framework](doc/Step02-BasicTest-LogisticRegression/step-02.01.02-Theano-Lasagne-Framework.md)
    - [2.1.3 Theano: Keras Framework](doc/Step02-BasicTest-LogisticRegression/step-02.01.03-Theano-Keras-Framework.md)
- [2.2 Logistic Regression with Torch](doc/Step02-BasicTest-LogisticRegression/step-02.02-LogReg-with-Torch.md)
- [2.3 Logistic Regression with Caffe](doc/Step02-BasicTest-LogisticRegression/step-02.03-LogReg-with-Caffe.md)
- [2.4 Logisitc Regression with Tensorflow](doc/Step02-BasicTest-LogisticRegression/step-02.04-LogReg-with-Tensorflow.md)
- [2.5 Logistic Regression with Deeplearning4J](doc/Step02-BasicTest-LogisticRegression/step-02.05-LogReg-with-Deeplearning4J.md)

