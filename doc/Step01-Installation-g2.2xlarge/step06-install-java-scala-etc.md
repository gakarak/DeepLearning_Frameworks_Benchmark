# **Java**, **Scala** and etc. installation [[back](index.md)]

(1) Install apt-repository:
```sh
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
```

(2) Install oracle-jdk
```sh
$ sudo apt-get install oracle-java8-installer
```

(3) Check installation:
```sh
$ java -version
```
```
java version "1.8.0_72"
Java(TM) SE Runtime Environment (build 1.8.0_72-b15)
Java HotSpot(TM) 64-Bit Server VM (build 25.72-b15, mixed mode)
```

(4) Download scala and unpack scala
```sh
$ cd ~/dev
$ wget http://downloads.typesafe.com/scala/2.11.7/scala-2.11.7.tgz
$ tar xzf scala-2.11.7.tgz
$ ln -s scala-2.11.7 scala
```

(5) Setup Scala ENV:
```sh
$ echo "
SCALA_HOME=\"\$HOME/dev/scala\"
PATH=\"\$SCALA_HOME/bin:\$PATH\"
export PATH SCALA_HOME
" > ~/bin/set-scala.sh
$ chmod u+x ~/bin/set-scala.sh
```

(5.1) append env setup to bashrc, profile:
```sh
$ echo "
source $HOME/bin/set-scala.sh
" >> ~/.bashrc
$ echo "
source $HOME/bin/set-scala.sh
" >> ~/.profile
```

(5.2) Check Scala installation:
```sh
$ scala -version
```
```
Scala code runner version 2.11.7 -- Copyright 2002-2013, LAMP/EPFL
```

(6) Download and install maven
```sh
$ cd ~/dev
$ wget http://ftp.byfly.by/pub/apache.org/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz
$ tar xzf apache-maven-3.3.9-bin.tar.gz
$ ln -s apache-maven-3.3.9 apache-maven
```

(6.1) create env-setuper:
```sh
$ echo "
JAVA_HOME=/usr/lib/jvm/java-8-oracle
MAVEN_HOME=\$HOME/dev/apache-maven
PATH=\$MAVEN_HOME/bin:\$PATH
export PATH JAVA_HOME
" >> ~/bin/set-maven.sh
$ chmod u+x ~/bin/set-maven.sh
```

(6.2) Append env setup to bashrc, profile:
```sh
$ echo "
source $HOME/bin/set-maven.sh
" >> ~/.bashrc
$ echo "
source $HOME/bin/set-maven.sh
" >> ~/.profile
```

(6.3) Check maven installation:
```sh
$ mvn -version
```
```
Apache Maven 3.3.9 (bb52d8502b132ec0a5a3f4c09453c07478323dc5; 2015-11-10T16:41:47+00:00)
Maven home: /home/ubuntu/dev/apache-maven
Java version: 1.8.0_72, vendor: Oracle Corporation
Java home: /usr/lib/jvm/java-8-oracle/jre
Default locale: en_US, platform encoding: ANSI_X3.4-1968
OS name: "linux", version: "3.13.0-77-generic", arch: "amd64", family: "unix"
```