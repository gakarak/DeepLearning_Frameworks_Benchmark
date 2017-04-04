package by.grid.imlab;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.FilterPanel;
import org.deeplearning4j.plot.PlotFilters;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.deeplearning4j.util.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;

public class LogisticRegressionMNIST
{
    private static Logger log = LoggerFactory.getLogger(LogisticRegressionMNIST.class);

    public static void main( String[] args ) throws IOException {
        final int numRows    = 28;
        final int numColumns = 28;
        int outputNum   = 10;
        int batchSize   = 128;
        int rngSeed     = 123;
        int numEpochs   = 1;

        // (1) Prepare Datasets
        log.info("Prepare Datasets....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        // (2) Configure model (LogisticRegression is equivelent to one-layer perceptron )
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.01)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list(1)
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(numRows * numColumns)
                        .nOut(outputNum)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        // (3) Build Network model and configure iteration-listener
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
//        model.setListeners(new HistogramIterationListener(1));
        model.setListeners(new ScoreIterationListener(100));

        // (4) Train model
        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(mnistTrain);
        }

        // (5) Calc Test-score
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix());
            eval.eval(next.getLabels(), output);
        }
        log.info(eval.stats());

        Map<String, INDArray> tmpParams = model.getLayer(0).paramTable();
        INDArray dataWeights  =tmpParams.get("W");
        PlotFilters plotFilters = new PlotFilters(dataWeights, new int[]{2,5}, new int[]{1,1}, new int[]{28,28});
        plotFilters.setInput(dataWeights.transpose());
        plotFilters.plot();
        INDArray plot = plotFilters.getPlot();
        BufferedImage image = ImageLoader.toImage(plot);
        File fout = new File("DL4J_LogReg_NetworkWeights.png");
        ImageIO.write(image,"png",fout);
        /*
        FilterPanel panel = new FilterPanel(image);
        JFrame frame = new JFrame();
        panel.setPreferredSize(new Dimension(200,100));
        frame.add(panel, BorderLayout.CENTER);
        frame.pack();
        frame.setVisible(true);
        */
    }
}
