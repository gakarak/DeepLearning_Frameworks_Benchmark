# **Theano** installation & quick check setup [[back](index.md)]

(1) Install Theano (pip-installation):
```sh
$ sudo pip install Theano -U
```

(2) **Check Theano installation and GPU usage:**
```sh
$ THEANO_FLAGS=floatX=float32,device=gpu python -c "import theano as th ; print th.version.full_version"
```
```
Using gpu device 0: GRID K520
0.7.0.dev-RELEASE
```

**... [Ok]**


