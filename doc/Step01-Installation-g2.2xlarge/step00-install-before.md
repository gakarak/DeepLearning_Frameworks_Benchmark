# Install basic software [[back](index.md)]

(1) Install **Midnight Commander** (I like it!)
```sh
$ sudo apt-get install mc
```

(2) Install minimal build-environement:
```sh
$ sudo apt-get install build-essential
```

(3) Install linux Extra Virtual Image to suport DRM
and compile Nvidia driver:
```sh
$ sudo apt-get install linux-image-extra-virtual
```

(3.1) And install kernel-headers for this kernel:
```sh
$ sudo apt-get install linux-headers-virtual
```

(4) Disabe 'Nouveau' driver:
```sh
sudo echo "
blacklist nouveau
options nouveau modeset=0
" > /etc/modprobe.d/nvidia-installer-disable-nouveau.conf
```

(4.1) Reboot system or unload module 'nouveau':
```sh
$ sudo rmmod nouveau
```


## **** INFO **** After this step you can install NVidia driver [step1](step01-install-nvidia-driver.md)

***

(5) Install Networking utilites:
(5.1) Nmap:
```sh
$ sudo apt-get install nmap
```
