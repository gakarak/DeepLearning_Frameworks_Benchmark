# Development Tools Installation [[back](index.md)]

(1) Install Git:
```sh
$ sudo apt-get install git
```

(2) Install cmake:
```sh
$ sudo apt-get install cmake
```

(3) Install OpenCV:
```sh
$ sudo apt-get install libopencv-dev libcv-dev python-opencv
```

(3.1) Check OpenCV Installation:
```sh
$ mkdir -p ~/img
$ cd ~/img
$ wget https://raw.github.com/gakarak/PTD-data-processing/master/data/doge2.jpg
$ wget https://raw.github.com/gakarak/PTD-data-processing/master/data/lena.jpg
$ wget https://raw.github.com/gakarak/PTD-data-processing/master/data/lena.png
```

![alt Wow! Doge!](../../img/doge2.jpg) 

(3.2) Try some OpenCV code (Python API):
```sh
$ python -c "import cv2 ; q=cv2.imread('/home/ubuntu/img/doge2.jpg') ; print q.shape"
```
```
(225, 225, 3)
```

***Yep! Ok, go to the next step.***
