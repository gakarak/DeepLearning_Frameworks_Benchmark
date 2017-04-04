#!/usr/bin/python

import numpy as np
import os
import matplotlib.pyplot as plt

import fuel
from fuel.datasets import MNIST
import theano
from theano import tensor as T
import blocks
import blocks.bricks
from blocks.bricks import Linear, Rectifier, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.cost import MisclassificationRate
#
from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
#
from blocks.initialization import IsotropicGaussian, Constant
#
from blocks.model import Model
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing

if __name__=='__main__':
    x=T.matrix('features')
    input_to_output=Linear(name='input_to_output', input_dim=784, output_dim=10)
    h=Rectifier().apply(input_to_output.apply(x))
    y_hat=Softmax().apply(h)
    #
    y=T.lmatrix('targets')
    cost=CategoricalCrossEntropy().apply(y.flatten(), y_hat)
    error_rate = MisclassificationRate().apply(y.flatten(), y_hat)
    #
    cg=ComputationGraph(cost)
    W1=VariableFilter(roles=[WEIGHT])(cg.variables)[0]
    cost = cost + 0.005 * (W1 ** 2).sum()
    cost.name = 'cost_with_regularization'
    #
    input_to_output.weights_init = IsotropicGaussian(0.01)
    input_to_output.biases_init = Constant(0)
    input_to_output.initialize()
    #
    mnist=MNIST(('train',))
    data_stream = Flatten(DataStream.default_stream(
            mnist,
            iteration_scheme=SequentialScheme(mnist.num_examples, batch_size=256)))
    #
    mnist_test = MNIST(("test",))
    data_stream_test = Flatten(DataStream.default_stream(
            mnist_test,
            iteration_scheme=SequentialScheme(mnist_test.num_examples, batch_size=1024)))
    #
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                step_rule=Scale(learning_rate=0.1))
    #
    monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=data_stream_test, prefix="test-FUCK")
    #
    main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                         model=Model(error_rate),
                         extensions=[monitor, FinishAfter(after_n_epochs=30), Printing()])
    main_loop.run()
    #
    numx=4
    numW=10
    numy=numW/numx
    if numx*numy<numW:
        numy+=1
    plt.figure(figsize=(10,10))
    for xx in xrange(numx):
        for yy in xrange(numy):
            pos=yy*numx+xx
            if pos<numW:
                plt.subplot(numy,numx,pos+1)
                plt.imshow(input_to_output.W.eval()[:,pos].reshape(28,28))
    plt.show()
