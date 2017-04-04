# **Caffe** installation & quick check setup [[back](index.md)]

Main article with installation instructions [caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html)

(1) Install Boost 1.55 version:
```sh
$ sudo apt-get install libboost1.55-all-dev
```

(2) Install Protobuf:
```sh
$ sudo apt-get install libprotobuf-dev libprotoc-dev libprotobuf-c0-dev python-protobuf protobuf-c-compiler protobuf-compiler
```

(3) Install Glog:
```sh
$ sudo apt-get install libgoogle-glog-dev
```

(4) Install Gflags:
```sh
$ sudo apt-get install libgflags-dev python-gflags
```

(5) Install HDF5:
```
$ sudo apt-get install libhdf5-dev hdf5-tools python-h5py
```

(6) Install lmdb, leveldb, snappy:
```sh
$ sudo apt-get install libsnappy-dev python-snappy  liblmdb-dev  libleveldb-dev
$ sudo pip install leveldb -U
$ sudo pip install lmdb -U
```

(7) Compile Caffe:
```sh
$ cd ~/deep-learning
$ cp Makefile.config.example Makefile.config
```

(7.1) Check flags in config:
 * USE_CUDNN := 1
 * BLAS := open
 * WITH_PYTHON_LAYER := 1

(7.2) Compile:
```sh
$ make -j8
$ make pycaffe
$ make distribute
```

(7.3) Create Caffe Env setuper:
```sh
$ cat ~/bin/set-caffe.sh
```
```
  #!/bin/bash
  bd="$HOME/deep-learning/caffe.git/distribute"
  LD_LIBRARY_PATH="${bd}/lib:$LD_LIBRARY_PATH"
  PYTHONPATH="${bd}/python:$PYTHONPATH"
  PATH="${bd}/bin:$PATH"
  export PATH LD_LIBRARY_PATH PYTHONPATH
```

```sh
$ echo "
source $HOME/bin/set-caffe.sh
" >> ~/.bashrc
$ echo "
source $HOME/bin/set-caffe.sh
" >> ~/.profile
```

(8) **Check Caffe installation:**

(8.1)Load MNIST Data:
```sh
$ cd ~/deep-learning/caffe.git/data/mnist/
$ ./get_mnist.sh
$ cd ~/deep-learning/caffe.git/
```

(8.2) Create dataset:
```sh
$ examples/mnist/create_mnist.sh
```

(8.3) Train on GPU #0
```sh
$ caffe.bin train -solver examples/mnist/lenet_solver.prototxt -gpu 0
```
```
I0219 17:30:42.840487  7692 caffe.cpp:185] Using GPUs 0
I0219 17:30:43.106034  7692 caffe.cpp:190] GPU 0: GRID K520
I0219 17:30:43.226626  7692 solver.cpp:48] Initializing solver from parameters: 
test_iter: 100
test_interval: 500
base_lr: 0.01
display: 100
max_iter: 10000
lr_policy: "inv"
gamma: 0.0001
power: 0.75
momentum: 0.9
weight_decay: 0.0005
snapshot: 5000
snapshot_prefix: "examples/mnist/lenet"
solver_mode: GPU
device_id: 0
net: "examples/mnist/lenet_train_test.prototxt"
.......
I0219 17:31:27.672592  7692 solver.cpp:338] Iteration 10000, Testing net (#0)
I0219 17:31:27.864341  7692 solver.cpp:406]     Test net output #0: accuracy = 0.9921
I0219 17:31:27.864397  7692 solver.cpp:406]     Test net output #1: loss = 0.0257289 (* 1 = 0.0257289 loss)
I0219 17:31:27.864413  7692 solver.cpp:323] Optimization Done.
I0219 17:31:27.864429  7692 caffe.cpp:222] Optimization Done.
```

**... [Ok]**
