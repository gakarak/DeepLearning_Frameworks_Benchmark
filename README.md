# DeepLearning_Frameworks_Benchmark

Benchmark code and guidelines for article ["Deep Learning with Theano, Torch, Caffe, TensorFlow, and Deeplearning4J: Which One Is the Best in Speed and Accuracy?"](https://www.researchgate.net/publication/302955130_Deep_Learning_with_Theano_Torch_Caffe_TensorFlow_and_Deeplearning4J_Which_One_Is_the_Best_in_Speed_and_Accuracy)


This  work  was mainlyfunded  by [Altoros](www.altoros.com), a global provider of big data, cloud computing, and Platform-as-a-Service solutions. You can find more materials at [Altoros research papers](http://www.altoros.com/research-papers)

# DeepLearning Frameworks Benchmark (Main contents)

- [1. Installation instrutions on g2.2xlarge](doc/Step01-Installation-g2.2xlarge/index.md)
- [2. Basic test: Logistic Regression](doc/Step02-BasicTest-LogisticRegression/index.md)
    - [2.1 Logistic Regression with Theano](doc/Step02-BasicTest-LogisticRegression/step-02.01-LogReg-with-Theano.md)
        - [2.1.1 Theano: Blocks Framework](doc/Step02-BasicTest-LogisticRegression/step-02.01.01-Theano-Blocks-Framework.md)
        - [2.1.2 Theano: Lasagne Framework](doc/Step02-BasicTest-LogisticRegression/step-02.01.02-Theano-Lasagne-Framework.md)
        - [2.1.3 Theano: Keras Framework](doc/Step02-BasicTest-LogisticRegression/step-02.01.03-Theano-Keras-Framework.md)
    - [2.2 Logistic Regression with Torch](doc/Step02-BasicTest-LogisticRegression/step-02.02-LogReg-with-Torch.md)
    - [2.3 Logistic Regression with Caffe](doc/Step02-BasicTest-LogisticRegression/step-02.03-LogReg-with-Caffe.md)
    - [2.4 Logisitc Regression with Tensorflow](doc/Step02-BasicTest-LogisticRegression/step-02.04-LogReg-with-Tensorflow.md)
    - [2.5 Logistic Regression with Deeplearning4J](doc/Step02-BasicTest-LogisticRegression/step-02.05-LogReg-with-Deeplearning4J.md)
- [3. FCNN: Fully Connected Neural Network (on CPU)](src/Step03/Readme.md)
    - [3.1 FCNN with Theano (Keras framework)](src/Step03/SubStep-01-FCNN-Theano-Keras/Readme.md)
    - [3.2 FCNN with Torch](src/Step03/SubStep-02-FCNN-Torch/Readme.md)
    - [3.3 FCNN with Caffe](src/Step03/SubStep-03-FCNN-Caffe/Readme.md)
    - [3.3 FCNN with TensorFlow](src/Step03/SubStep-04-FCNN-TensorFlow/Readme.md)
    - [3.4 FCNN with DeepLearning4J](src/Step03/SubStep-05-FCNN-DeepLearning4J/Readme.md)
    - [3.5 Conclusion](src/Step03/Conclusion.md)
- [4. Distibuted TensorFlow: first steps](src/Step04_Distrib_TF_on_VB/Readme00.md)
    - [4.1 Deep Learning Model for expriment with distributed TF](Step04_Distrib_TF_on_VB/Readme01_Model.md)
    - [4.2 Configuration of the virtual environment](Step04_Distrib_TF_on_VB/Readme02_Config_VirtualBox.md)
    - [4.3 Preparing dataset](Step04_Distrib_TF_on_VB/Readme03_Dataset.md)
    - [4.4 Configure and Run distributed TF in VBox Env](Step04_Distrib_TF_on_VB/Readme04_Run_TF_in_VBox.md)
    - [4.5 Check TF scalability on multi-GPU one-Node](Step04_Distrib_TF_on_VB/Readme05_Multi_GPU_one_Node.md)
    
