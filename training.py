#!python3
import utils
import click
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,Concatenate
from tensorflow.keras.models import Model, Sequential 
import numpy as np
from scipy.stats import pearsonr
import csv

from utils import getBigwigFileList, getMatrixFromCooler, binChromatinFactor, scaleArray
from tqdm import tqdm

from numpy.random import seed
seed(35)

tf.random.set_seed(35)

@click.option("--trainmatrix","-tm",required=True,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Training matrix in cooler format")
@click.option("--chromatinPath","-cp", required=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides")
@click.option("--sequenceFile", "-sf", required=True,
                    type=click.Path(exists=True,readable=True,dir_okay=False),
                    help="Path to DNA sequence in fasta format")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where trained network will be stored")
@click.option("--chromosome", "-chrom", required=True,
              type=str, default="17", help="chromosome to train on")
@click.option("--modelfilepath", "-mfp", required=True, 
              type=click.Path(writable=True,dir_okay=False), 
              default="trainedModel.h5", help="path+filename for trained model")
@click.option("--learningRate", "-lr", required=True,
                type=click.FloatRange(min=1e-10), default=1e-1,
                help="learning rate for stochastic gradient descent")
@click.option("--numberEpochs", "-ep", required=True,
                type=click.IntRange(min=20), default=1000,
                help="number of epochs for training the neural network")
@click.option("--batchsize", "-bs", required=True,
                type=click.IntRange(min=5), default=30,
                help="batch size for training the neural network")
@click.option("--windowsize", "-ws", required=True,
                type=click.IntRange(min=10), default=80,
                help="Window size (in bins) for composing training data")
@click.option("--scaleMatrix", "-scm", required=True,
                type=bool, default=False,
                help="Scale Hi-C matrix to [0...1]")
@click.option("--clampFactors","-cfac", required=False,
                type=bool, default=True,
                help="Clamp outliers in chromatin factor data")
@click.option("--scaleFactors","-scf", required=False,
                type=bool, default=True,
                help="Scale chromatin factor data to range 0...1 (recommended)")
@click.command()
def training(trainmatrix,
            chromatinpath,
            sequencefile,
            outputpath,
            chromosome,
            modelfilepath,
            learningrate,
            numberepochs,
            batchsize,
            windowsize,
            scalematrix,
            clampfactors,
            scalefactors):
    #save the input parameters so they can be written to csv later
    paramDict = locals().copy()

    #load relevant part of Hi-C matrix
    sparseHiCMatrix, binSizeInt  = getMatrixFromCooler(trainmatrix,chromosome)
    if sparseHiCMatrix is None:
        msg = "Could not read HiC matrix {:s} for training, check inputs"
        msg = msg.format(trainmatrix)
        raise SystemExit(msg)
    msg = "Cooler matrix {:s} loaded.\nBin size (resolution) is {:d}bp."
    msg = msg.format(trainmatrix, binSizeInt)
    print(msg)
    print("matrix shape", sparseHiCMatrix.shape)
    paramDict["binSize"] = binSizeInt
    paramDict["train_matrix_shape"] = sparseHiCMatrix.shape
    
    #matrix distance normalization, divide values in each side diagonal by their average
    ##possible and even quite fast, but doesn't look reasonable
    #sparseHiCMatrix = utils.distanceNormalize(sparseHiCMatrix, windowsize)

    #scale matrix, if requested
    if scalematrix:
        print("Scaling Hi-C matrix.")
        print("Current extreme values: min. {:.3f}, max. {:.3f}".format(sparseHiCMatrix.min(), sparseHiCMatrix.max()))
        sparseHiCMatrix = utils.scaleArray(sparseHiCMatrix)
        print("New extreme values: min.{:.3f}, max. {:.3f}".format(sparseHiCMatrix.min(), sparseHiCMatrix.max()))

    #check chromatin files
    bigwigFileList = getBigwigFileList(chromatinpath)
    if len(bigwigFileList) == 0:
        msg = "No bigwig files (*.bigwig, *.bigWig, *.bw) found in {:s}. Aborting."
        msg = msg.format(chromatinpath)
        raise SystemExit(msg)
    msg = "Found {:d} chromatin factors in {:s}:"
    msg = msg.format(len(bigwigFileList),chromatinpath)
    print(msg)
    for factor in bigwigFileList:
        print(" ", factor)
    
    #bin the chromatin factors and create a numpy array from them
    plotFilename = outputpath + "chromFactorBoxplots.png"
    chromatinFactorArray = utils.composeChromatinFactors(pBigwigFileList = bigwigFileList,
                                                         pChromLength_bins = sparseHiCMatrix.shape[0],
                                                         pBinSizeInt = binSizeInt,
                                                         pChromosomeStr = chromosome,
                                                         pPlotFilename = plotFilename,
                                                         pClampArray=clampfactors,
                                                         pScaleArray=scalefactors)

    #read the DNA sequence and do a one-hot encoding
    encodedSequenceArray = utils.buildSequenceArray(sequencefile,binSizeInt)
    
    nr_Factors = chromatinFactorArray.shape[0]
    nr_matrices = sparseHiCMatrix.shape[0] - 3* windowsize + 1
    
    #compute indices for train / validation split 
    trainIndices = np.random.choice(range(nr_matrices), size=(int(0.8*nr_matrices)), replace=False)
    validationIndices = np.setdiff1d(range(nr_matrices),trainIndices)

    #generators for training and validation data
    trainDataGenerator = utils.multiInputGenerator(sparseHiCMatrix,chromatinFactorArray, encodedSequenceArray, trainIndices,batchsize,windowsize,binSizeInt)
    validationDataGenerator = utils.multiInputGenerator(sparseHiCMatrix,chromatinFactorArray,encodedSequenceArray, validationIndices,batchsize,windowsize,binSizeInt)

    #neural network constants as per Farre et al.
    chromLength_bins = 3 * windowsize
    kernelWidth = 1
    #nr_neurons1 = 460
    #nr_neurons2 = 881
    nr_neurons1 = 100
    nr_neurons2 = 200
    #nr_neurons3 = 1690
    nr_neurons4 = int(1/2 * windowsize * (windowsize + 1)) #always an int, even*odd=even

    #build neural network as described by Farre et al.
    model = Sequential()
    model.add(Conv2D(filters=1, 
                     kernel_size=(nr_Factors,kernelWidth), 
                     activation="sigmoid",
                     data_format="channels_last",
                     input_shape=(nr_Factors,chromLength_bins,1)))
    model.add(Flatten())
    model.add(Dense(nr_neurons1,activation="relu",kernel_regularizer="l2"))        
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons2,activation="relu",kernel_regularizer="l2"))
    model.add(Dropout(0.1))
    #model.add(Dense(nr_neurons3,activation="relu",kernel_regularizer="l2"))
    #model.add(Dropout(0.1))
    #model.add(Dense(nr_neurons4,activation="relu",kernel_regularizer="l2"))
    #model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learningrate), 
    #             loss=tf.keras.losses.MeanSquaredError())
    #model.summary()
    
    model2 = Sequential()
    model2.add(Conv2D(filters=1, 
                     kernel_size=(6,5), 
                     padding="valid",
                     activation="sigmoid",
                     data_format="channels_last",
                     input_shape=(6000000,5,1)))
    model2.add(Conv2D(filters=1,
                      kernel_size=(300,1),
                      activation="sigmoid",
                      data_format="channels_last"))
    model2.add(Conv2D(filters=1,kernel_size=(5000,1),
                      activation="sigmoid",
                      data_format="channels_last"))
    model2.add(Conv2D(filters=1,kernel_size=(25000,1),
                      activation="sigmoid",
                      data_format="channels_last"))
    model2.add(Conv2D(filters=1,kernel_size=(100000,1),
                      activation="sigmoid",
                      data_format="channels_last"))
    model2.add(Conv2D(filters=1,kernel_size=(1000000,1),
                      activation="sigmoid",
                      data_format="channels_last"))                              
    model2.add(Flatten())
    model2.add(Dense(20, activation="relu",kernel_regularizer="l2"))
    model2.add(Dropout(0.1))

    combined = Concatenate()([model.output,model2.output])
    x = Dense(nr_neurons4,activation="relu",kernel_regularizer="l2")(combined)
 
    finalModel = Model(inputs=[model.input, model2.input], outputs=x)
    finalModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learningrate), 
                 loss=tf.keras.losses.MeanSquaredError())
    finalModel.summary()


    #callbacks to check the progress etc.
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=outputpath)
    saveFreqInt = int(np.floor(len(trainIndices)/batchsize) * 20)
    checkpointFilename = outputpath + "checkpoint_{epoch:05d}.h5"
    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointFilename,
                                                        monitor="val_loss",
                                                        save_freq=saveFreqInt)
    #earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
    #                                                     min_delta=1e-3,
    #                                                     patience=5,
    #                                                     restore_best_weights=True)

    #save the training parameters to a file before starting to train
    #(allows recovering the parameters even if training is aborted
    # and only intermediate models are available)
    parameterFile = outputpath + "trainParams.csv"    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)

    #plot the model
    tf.keras.utils.plot_model(finalModel,show_shapes=True, to_file=outputpath + "model.png")
    
    #train the neural network
    history = finalModel.fit(trainDataGenerator,
              epochs= numberepochs,
              validation_data= validationDataGenerator,
              callbacks=[tensorboardCallback,
                          checkpointCallback,
                            #earlyStoppingCallback
                           ]
            )

    #store the trained network
    model.save(filepath=modelfilepath,save_format="h5")

    #plot train- and validation loss over epochs
    lossPlotFilename = outputpath + "lossOverEpochs.png"
    utils.plotLoss(history, lossPlotFilename)

    #self-prediction just for testing
    # input_test = np.expand_dims(chromatinFactorArray[0,:,:,:],0)
    # target_test = np.expand_dims(matrixArray[0,:],0)
    # loss = model.evaluate(x=input_test, y=target_test)
    # print("loss: {:.3f}".format(loss))
    # pred = model.predict(x=input_test)
    # predMatrix = np.zeros(shape=(windowsize,windowsize))
    # predMatrix[np.triu_indices(windowsize)] = pred[0] 
    # utils.plotMatrix(predMatrix *1000, outputpath + "predMatrix.png", "pred. matrix")
    # targetMatrix = np.zeros(shape=(windowsize,windowsize))
    # targetMatrix[np.triu_indices(windowsize)] = matrixArray[0]
    # utils.plotMatrix(targetMatrix *1000, outputpath + "targetMatrix.png", "target matrix" )

    # pearson_r, pearson_p = pearsonr(matrixArray[0],pred[0])
    # msg = "Pearson R = {:.3f}, Pearson p = {:.3f}"
    # msg = msg.format(pearson_r, pearson_p)
    # print(msg)

    # mse = (np.square(matrixArray[0] - pred[0])).mean(axis=None)
    # print("MSE", mse)
    
if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter