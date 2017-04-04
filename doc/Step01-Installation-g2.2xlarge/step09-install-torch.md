# **Thorch** installation & quick check setup [[back](index.md)]

(1) Install Torch ([Official Guide](http://torch.ch/docs/getting-started.html)):
```sh
$ mkdir -p ~/i/torch
$ cd ~/i/torch
$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps >install-deps.bash
$ cat install-deps.bash | bash
```
```sh
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; ./install.sh
```

```sh
$ source ~/.bashrc
$ luarocks install image
```

(2) **Check Torch GPU installation:**
```sh
$ echo "cutorch.test()" | luajit -lcutorch
```
```
seed: 	1455899655
Running 142 tests
________________________________________________  ==> Done

Completed 68186 asserts in 142 tests with 0 errors
```
**... [Ok]**