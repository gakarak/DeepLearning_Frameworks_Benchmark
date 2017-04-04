# **Deeplearning4J** installation & quick check setup [[back](index.md)]

(1) At this moment Maven is installed, then:
```sh
$ cd ~/deep-learning
$ git clone https://github.com/deeplearning4j/dl4j-0.4-examples.git dl4j-0.4-examples.git
$ cd dl4j-0.4-examples.git
```

(2) Build DL4J Examples:
```sh
$ mvn clean package
```

(3) **Check DL4J installation:**
```sh
$ time java -cp target/deeplearning4j-examples-0.4-rc0-SNAPSHOT.jar org.deeplearning4j.examples.multinetwork.DBNMnistFullExample
```
```
o.d.e.m.DBNMnistFullExample - Load data....
o.d.b.MnistFetcher - Downloading mnist...

Feb 19, 2016 11:13:17 PM com.github.fommil.jni.JniLoader liberalLoad
INFO: successfully loaded /tmp/jniloader4273761144438629527netlib-native_system-linux-x86_64.so
o.d.e.m.DBNMnistFullExample - Build model....
o.d.e.m.DBNMnistFullExample - Train model....
o.d.n.m.MultiLayerNetwork - Training on layer 1 with 100 examples
...
o.d.o.s.BaseOptimizer - Objective function automatically set to minimize. Set stepFunction in neural net configuration to change default settings.
o.d.o.l.ScoreIterationListener - Score at iteration 0 is 0.7298707580566406
o.d.e.m.DBNMnistFullExample - ****************Example finished********************

real	0m11.385s
user	0m25.023s
sys	0m54.658s
```

**... [Ok]**

(4) Setup GPU version of dl4j:
```sh
$ sudo apt-get install libblas*
```

(5) Change maven-config to allow jcublas [nd4j-jcublas-7.5/0.4-rc3.8](http://mvnrepository.com/artifact/org.nd4j/nd4j-jcublas-7.5/0.4-rc3.8)
```sh
$ cd ~/deep-learning/dl4j-0.4-examples.git
$ cp pom.xml pom.xml-cpu-x86
$ cat ./pom.xml-cpu-x86 | sed 's/nd4j-x86/nd4j-jcublas-7.5/g' > ./pom.xml
```

(5.1) Rebuild project:
```sh
$ mvn clean package
```

(5.2) Check output Jar-package:
```sh
$ jar -tf ./target/deeplearning4j-examples-0.4-rc0-SNAPSHOT.jar | grep 'so$'
```
```
META-INF/native/linux32/libleveldbjni.so
META-INF/native/linux64/libleveldbjni.so
  libsigar-amd64-linux-1.6.4.so
  libsigar-amd64-solaris-1.6.4.so
  libsigar-x86-linux-1.6.4.so
* lib/libJCublas-linux-x86_64.so
* lib/libJCusparse-linux-x86_64.so
* lib/libJCurand-linux-x86_64.so
* lib/libJCudaRuntime-linux-x86_64.so
* lib/libJCusolver-linux-x86_64.so
* lib/libJCufft-linux-x86_64.so
* lib/libJCudaDriver-linux-x86_64.so
* lib/libJCublas2-linux-x86_64.so
```

**... Ok, looks good **

(5.3) Try to run the test application with GPU:
```sh
$ java -cp target/deeplearning4j-examples-0.4-rc0-SNAPSHOT.jar org.deeplearning4j.examples.multinetwork.DBNMnistFullExample
```
```
o.d.e.m.DBNMnistFullExample - Load data....

o.n.l.j.k.KernelFunctionLoader - Registering cuda functions...
o.n.l.j.k.KernelFunctionLoader - Compiling cuda kernels
java.lang.RuntimeException: java.lang.IllegalStateException: Unable to find path /tmp/nd4j-kernels/output/std_strided.cubin. Recompiling
    at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.compileAndLoad(KernelFunctionLoader.java:297)
    at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.compileAndLoad(KernelFunctionLoader.java:219)
....
    at org.deeplearning4j.datasets.iterator.BaseDatasetIterator.next(BaseDatasetIterator.java:33)
    at org.deeplearning4j.examples.multinetwork.DBNMnistFullExample.main(DBNMnistFullExample.java:39)
Caused by: java.lang.IllegalStateException: Unable to find path /tmp/nd4j-kernels/output/std_strided.cubin. Recompiling
    at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.loadModules(KernelFunctionLoader.java:331)
    at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.compileAndLoad(KernelFunctionLoader.java:282)
    ... 14 more
o.n.l.j.k.KernelFunctionLoader - Registering cuda functions...
o.n.l.j.k.KernelFunctionLoader - Compiling cuda kernels
Exception in thread "main" java.lang.RuntimeException: java.lang.RuntimeException: java.lang.IllegalStateException: Unable to find path /tmp/nd4j-kernels/output/std_strided.cubin. Recompiling
    at org.nd4j.linalg.jcublas.context.ContextHolder.configure(ContextHolder.java:250)
    at org.nd4j.linalg.jcublas.context.ContextHolder.getInstance(ContextHolder.java:116)
....
    at org.deeplearning4j.datasets.iterator.BaseDatasetIterator.next(BaseDatasetIterator.java:33)
    at org.deeplearning4j.examples.multinetwork.DBNMnistFullExample.main(DBNMnistFullExample.java:39)
Caused by: java.lang.RuntimeException: java.lang.IllegalStateException: Unable to find path /tmp/nd4j-kernels/output/std_strided.cubin. Recompiling
    at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.compileAndLoad(KernelFunctionLoader.java:297)
    at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.compileAndLoad(KernelFunctionLoader.java:219)
    at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.load(KernelFunctionLoader.java:186)
    at org.nd4j.linalg.jcublas.context.ContextHolder.configure(ContextHolder.java:248)
    ... 10 more
Caused by: java.lang.IllegalStateException: Unable to find path /tmp/nd4j-kernels/output/std_strided.cubin. Recompiling
    at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.loadModules(KernelFunctionLoader.java:331)
    at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.compileAndLoad(KernelFunctionLoader.java:282)
    ... 13 more
```

**... Ohh, damn it! [Error]. From Deeplearnin4J Gitter channel [https://gitter.im/deeplearning4j/deeplearning4j/archives/2016/01/12]:**

```
> [wiku] (Jan 12 13:53)
> 	now I have this:
> 	Caused by: java.lang.IllegalStateException: Unable to find path /tmp/nd4j-kernels/output/std_strided.cubin. Recompiling
> 	at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.loadModules(KernelFunctionLoader.java:331)
> 	at org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader.get(KernelFunctionLoader.java:141)
> [raver119] (Jan 12 13:54)
> 	@wiku iâve told you - cuda isnât supposed to be used in few latest releases
> wiku (Jan 12 13:55)
> 	okay okay
> [raver119] (Jan 12 13:55)
> 	itâs available there only due to maven build. but itâs not usable. and wonât untill weâll finish overhaul :)
> [wiku] (Jan 12 13:55)
> 	I give up :D
> 	don't shoot
```

[Mac OSX stack trace running example with 7.5](https://webcache.googleusercontent.com/search?q=cache:1lEUs0ZId5gJ:https://github.com/deeplearning4j/nd4j-kernels/issues/3+&cd=1&hl=ru&ct=clnk&gl=by)
```
For some reason there are duplicate classes (with different behavior)
in nd4j-jcublas-7.5 and nd4j-jcublas-common.
The 7.5 version of org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader
seems to be totally broken :disappointed:
```
