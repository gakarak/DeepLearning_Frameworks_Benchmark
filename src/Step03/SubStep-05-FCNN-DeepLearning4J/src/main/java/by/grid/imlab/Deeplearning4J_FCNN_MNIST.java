package by.grid.imlab;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Deeplearning4J_FCNN_MNIST
{
    private static Logger log = LoggerFactory.getLogger(Deeplearning4J_FCNN_MNIST.class);

    public static void main( String[] args ) throws IOException {

        int rngSeed     = 123;
        int paramBatchSize=128;
        int paramEpochs=10;
        int paramReps=10;
        if(args.length<1) {
            System.out.println("Usage: $Program {L1:L2:...:Ln} {batchSize} {#Epochs} {numReps}");
            System.exit(1);
        }

        ArrayList<Integer> arrParam = new ArrayList<>();
        String[] arrParamStr = args[0].split(":");
        for (String anArrParamStr : arrParamStr) {
            arrParam.add(Integer.parseInt(anArrParamStr));
        }

        if(args.length>3) {
            paramBatchSize  = Integer.parseInt(args[1]);
            paramEpochs     = Integer.parseInt(args[2]);
            paramReps       = Integer.parseInt(args[3]);
        }

        final int numRows       = 28;
        final int numColumns    = 28;
        final int sizeInput     = numRows*numColumns;
        int sizeOutput          = 10;

        // (1) Configure model (LogisticRegression is equivelent to one-layer perceptron )
        log.info("Build model....");
        int numLayers = arrParam.size() + 1;

        String modelName = "ModelFCN-Deeplearning4J-p" + args[0] + "-b" + paramBatchSize + "-e" + paramEpochs;
        String foutLog = modelName + "-Log.txt";

        File file = new File(foutLog);
        BufferedWriter fileWriter = new BufferedWriter(new FileWriter(file));
        fileWriter.write("model, timeTrain, timeTest, acc\n");
        for(int rr=0; rr<paramReps; rr++) {
            // (2) Prepare Datasets
            log.info("Prepare Datasets....");
            DataSetIterator mnistTrain = new MnistDataSetIterator(paramBatchSize, true, rngSeed);
            DataSetIterator mnistTest = new MnistDataSetIterator(paramBatchSize, false, rngSeed);
            //
            NeuralNetConfiguration.ListBuilder netBulder = new NeuralNetConfiguration.Builder()
                    .seed(rngSeed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .iterations(1)
                    .learningRate(0.01)
                    .updater(Updater.NESTEROVS).momentum(0.9).list(numLayers);
            netBulder = netBulder.layer(0, new DenseLayer.Builder()
                    .nIn(sizeInput)
                    .nOut(arrParam.get(0))
                    .activation("tanh")
                    .weightInit(WeightInit.XAVIER)
                    .build());
            for(int ii=1; ii<arrParam.size(); ii++) {
                netBulder = netBulder.layer(ii, new DenseLayer.Builder()
                        .nIn(arrParam.get(ii-1))
                        .nOut(arrParam.get(ii))
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build());
            }
            MultiLayerConfiguration conf = netBulder.layer(numLayers-1,
                    new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(arrParam.get(arrParam.size()-1))
                            .nOut(sizeOutput)
                            .activation("softmax")
                            .weightInit(WeightInit.XAVIER)
                            .build())
                    .pretrain(false).backprop(true).build();
            // (3) Build Network model and configure iteration-listener
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            // (4) Train model
            log.info("Train model....");
            long t0 = System.currentTimeMillis();
            for( int i=0; i<paramEpochs; i++ ){
                model.fit(mnistTrain);
            }
            double timeTrain = (double)(System.currentTimeMillis() - t0)/1000.0;
            // (5) Calc Test-score
            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(sizeOutput);
            t0 = System.currentTimeMillis();
            while(mnistTest.hasNext()) {
                DataSet next = mnistTest.next();
                INDArray output = model.output(next.getFeatureMatrix());
                eval.eval(next.getLabels(), output);
            }
            double timeTest = (double)(System.currentTimeMillis() - t0)/1000.0;
            log.info(eval.stats());
            double retACC = eval.accuracy();
            String tstr = modelName + ", " + timeTrain + ", " + timeTest + ", " + retACC;
            fileWriter.write(tstr + "\n");
            System.out.println(tstr);
        }
        fileWriter.close();
    }
}
