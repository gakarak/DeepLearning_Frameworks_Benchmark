#!/usr/bin/env sh

DATA=./mnist-raw
BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf ./mnist_train_${BACKEND}
rm -rf ./mnist_test_${BACKEND}

convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte ./mnist_train_${BACKEND} --backend=${BACKEND}
convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte ./mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."
