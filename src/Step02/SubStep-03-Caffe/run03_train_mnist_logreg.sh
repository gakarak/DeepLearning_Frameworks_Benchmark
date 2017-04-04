#!/bin/bash

##./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt
caffe.bin train --solver=mnist_logreg_solver.prototxt
