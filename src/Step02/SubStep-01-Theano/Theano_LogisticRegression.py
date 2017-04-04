#!/usr/bin/python
# coding: utf-8

import sys
import os
import timeit
import time
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from tensorflow.examples.tutorials.mnist import input_data

"""
Basic code from:
http://deeplearning.net/tutorial/logreg.html
"""
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


if __name__=='__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    learning_rate=0.01
    n_epochs=100
    batch_size=256
    test_set_x,  test_set_y  = shared_dataset( (mnist.test.images,
                                                mnist.test.labels) )
    valid_set_x, valid_set_y = shared_dataset( (mnist.validation.images,
                                                mnist.validation.labels) )
    train_set_x, train_set_y = shared_dataset( (mnist.train.images,
                                                mnist.train.labels) )
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0] // batch_size
    print('... building the model')
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    cost = classifier.negative_log_likelihood(y)
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    loss_validation = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    loss_train = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('... training the model')
    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    lstScores=[]
    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
        validation_error = [validate_model_error(i) for i in range(n_valid_batches)]
        this_validation_error = np.mean(validation_error)
        #
        chk_loss_train = np.mean([loss_train(i) for i in range(n_train_batches)])
        chk_loss_validation = np.mean([loss_validation(i) for i in range(n_valid_batches)])
        #
        lstScores.append((epoch, chk_loss_train, chk_loss_validation, this_validation_error * 100.))
        print 'epoch %i, validation error %f %%' % (epoch, this_validation_error * 100.)
    #
    test_losses = [test_model(i) for i in range(n_test_batches)]
    test_score = np.mean(test_losses)
    end_time = timeit.default_timer()
    #
    print '-------------'
    print 'Optimization complete with best validation score of %f %%, with test performance %f %%' % (best_validation_loss * 100., test_score * 100.)
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print 'The code for file [%s]  ran for %.1fs' % (os.path.basename(sys.argv[0]), end_time - start_time)
    #
    tres=np.array(lstScores)
    plt.figure()
    plt.subplot(1,2,1)
    plt.hold(True)
    plt.plot(tres[:,0],tres[:,1])
    plt.plot(tres[:,0],tres[:,2])
    plt.hold(False)
    plt.legend(('Loss: train','Loss: Validation'))
    plt.title('Loss')
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(tres[:,0],100.-tres[:,3])
    plt.title('Classification Accuracy (%)')
    plt.grid(True)
    #
    plt.figure()
    dataW=classifier.W.get_value()
    numx=5
    numy=2
    cnt=0
    for xx in xrange(numx):
        for yy in xrange(numy):
            plt.subplot(numy,numx,cnt+1)
            plt.imshow(dataW[:,cnt].reshape((28,28)))
            plt.title('W_%d' % cnt)
            cnt+=1
    plt.show()