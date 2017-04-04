# Alternative installation NVidia driver for **GRID K520** [[back](index.md)]


**!!!!!! Warning !!!!!!!**
> This guide is a bad idea: too many dependencies in Nvidia CUDA-repo,
> and this repo can contain invalid driver.
> 
> Skip this method if you are
> not sure of the technical support provided by NVidia.
>
> This method work on **Amazon g2.2xlarge**.
>

(1) Download cuda repo from nvidia site:
```sh
$ mkdir -p ~/i/nvidia.com/cuda
$ cd ~/i/nvidia.com/cuda
$ wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
```

(2) Install repo and update package list:
```sh
$ sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
$ sudo apt-get update
```

(3) Install CUDA:
```sh
$ sudo apt-get install cuda
```
