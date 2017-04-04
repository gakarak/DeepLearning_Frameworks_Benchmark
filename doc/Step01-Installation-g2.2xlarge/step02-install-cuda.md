# NVidia CUDA Installation [[back](index.md)]

(1) Download CUDA-Installer from nvidia site:
```sh
$ mkdir -p ~/i/nvidia.com/cuda
$ cd ~/i/nvidia.com/cuda
$ wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
```

(2) Install CUDA-SDK (skip driver installation):
```sh
$ chmod u+x ./cuda_7.5.18_linux.run
$ sudo ./cuda_7.5.18_linux.run
```
```
----> Installation & Configuration output
 Do you accept the previously read EULA? (accept/decline/quit): accept
 Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 352.39? ((y)es/(n)o/(q)uit): n
 Install the CUDA 7.5 Toolkit? ((y)es/(n)o/(q)uit): y
 Enter Toolkit Location [ default is /usr/local/cuda-7.5 ]: 
 Do you want to install a symbolic link at /usr/local/cuda? ((y)es/(n)o/(q)uit): y
 Install the CUDA 7.5 Samples? ((y)es/(n)o/(q)uit): y
 Enter CUDA Samples Location [ default is /home/ubuntu ]: y        
 Samples location must be an absolute path
 Enter CUDA Samples Location [ default is /home/ubuntu ]: /home/ubuntu/dev
 Installing the CUDA Toolkit in /usr/local/cuda-7.5 ...
---->
```

(3) Setup Environement:
(3.1) Create Setup-File:
```sh
$ mkdir -p ~/bin
$ echo "#!/bin/bash
CUDA_HOME="/usr/local/cuda"
PATH="\$CUDA_HOME/bin:\$PATH"
LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$CUDA_HOME/lib:\$LD_LIBRARY_PATH"
export PATH LD_LIBRARY_PATH CUDA_HOME
" > ~/bin/set-cuda.sh
```

(3.2) Append setup CUDA Env to '~/.bashrc' and '~/.profile' (for reliability)
```sh
$ chmod u+x ~/bin/set-cuda.sh
$ echo "
source $HOME/bin/set-cuda.sh
" >> ~/.bashrc
$ echo "
source $HOME/bin/set-cuda.sh
" >> ~/.profile
```

(4) Check installation:
```sh
$ cd ~/dev/NVIDIA_CUDA-7.5_Samples
$ cat /proc/cpuinfo | grep '^processor' | wc -l
8
$ make -j8
```
*... after 15 minutes ...*

(4.1) Try to run some examples:
```sh
$ ./bin/x86_64/linux/release/simpleCUBLAS 
GPU Device 0: "GRID K520" with compute capability 3.0
```
```
simpleCUBLAS test running..
simpleCUBLAS test passed.
```

(4.2) Ohh... :( to bad CUDA compute compatibility version: only 3.0,
Tensorflow (without hacks) required 3.5 and higher

(5) Ok, go to next step!

