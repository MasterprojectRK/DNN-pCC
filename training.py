#!python3
import utils
import click
import keras
from keras.layers import Conv2D,Dense,Dropout,Flatten
from keras.models import Sequential
import numpy as np

from utils import getBigwigFileList, getMatrixFromCooler, binChromatinFactor
from tqdm import tqdm

@click.option("--trainmatrix","-tm",required=True,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Training matrix in cooler format")
@click.option("--chromatinPath","-cp", required=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where trained network will be stored")
@click.option("--chromosome", "-chrom", required=True,
              type=str, default="17", help="chromosome to train on")
@click.command()
def training(trainmatrix, chromatinpath, outputpath, chromosome):

    #constants
    windowSize_bins = 80
    chromLength_bins = 3 * windowSize_bins
    matrixSize_bins = int(1/2 * windowSize_bins * (windowSize_bins + 1))
    kernelWidth = 1
    nr_neurons1 = 460
    nr_neurons2 = 881
    nr_neurons3 = 1690
    nr_neurons4 = matrixSize_bins
    nr_epochs = 10
       
    #check inputs
    ##load relevant part of Hi-C matrix
    sparseHiCMatrix, binSizeInt  = getMatrixFromCooler(trainmatrix,chromosome)
    if sparseHiCMatrix == None:
        msg = "Could not read HiC matrix {:s} for training, check inputs"
        msg = msg.format(trainmatrix)
        raise SystemExit(msg)
    msg = "Cooler matrix {:s} loaded.\nBin size (resolution) is {:d}bp."
    msg = msg.format(trainmatrix, binSizeInt)
    print(msg)
    print("matrix shape", sparseHiCMatrix.shape)
    ##check chromatin files
    bigwigFileList = getBigwigFileList(chromatinpath)
    if len(bigwigFileList) == 0:
        msg = "No bigwig files (*.bigwig, *.bigWig, *.bw) found in {:s}. Aborting."
        msg = msg.format(chromatinpath)
        raise SystemExit(msg)
    msg = "found {:d} chromatin factors in {:s}"
    msg = msg.format(len(bigwigFileList),chromatinpath)
    print(msg)
    for factor in bigwigFileList:
        print(factor)
    
    #compose inputs into useful dataset
    ##todo: distance normalization, divide values in each side diagonal by their average
    ##get all possible windowSize x windowSize matrices out of the original one
    startIndex = windowSize_bins
    nr_matrices = int(sparseHiCMatrix.shape[0] - 3*windowSize_bins + 1)
    matrixArray = np.empty(shape=(nr_matrices,matrixSize_bins))
    for i in tqdm(range(nr_matrices),desc="composing training matrices"):
        endIndex = i + startIndex + windowSize_bins
        trainmatrix = sparseHiCMatrix.toarray()[i+startIndex:endIndex,i+startIndex:endIndex][np.triu_indices(windowSize_bins)]
        matrixArray[i] = trainmatrix.reshape(1,matrixSize_bins)
    binnedChromatinFactorArray = np.empty(shape=(len(bigwigFileList),sparseHiCMatrix.shape[0]))
    ##bin the proteins
    for i in tqdm(range(len(bigwigFileList)),desc="binning chromatin factors"):
        binnedChromatinFactorArray[i] = binChromatinFactor(bigwigFileList[i],binSizeInt,chromosome)
    ##compose chromatin factor input for all possible matrices
    chromatinFactorArray = np.empty(shape=(nr_matrices,len(bigwigFileList),3*windowSize_bins))
    for i in tqdm(range(nr_matrices),desc="composing chromatin factors"):
        endIndex = i + 3*windowSize_bins
        chromatinFactorArray[i] = binnedChromatinFactorArray[:,i:endIndex]

    ##random input for now
    nr_Factors = len(bigwigFileList)
    input_train = chromatinFactorArray
    target_train = matrixArray

    #build neural network as described by Farre et al.
    model = Sequential()
    model.add(Conv2D(filters=1, 
                     kernel_size=(nr_Factors,kernelWidth), 
                     activation="sigmoid",
                     input_shape=(nr_Factors,chromLength_bins,1)))
    model.add(Flatten())
    model.add(Dense(nr_neurons1,activation="relu",kernel_regularizer="l2"))        
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons2,activation="relu",kernel_regularizer="l2"))
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons3,activation="relu",kernel_regularizer="l2"))
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons4,activation="relu",kernel_regularizer="l2"))
    model.compile(optimizer=keras.optimizers.SGD(), 
                  loss=keras.losses.MeanSquaredError())
    model.summary()
    
    #train the neural network
    model.fit(input_train, 
              target_train, 
              epochs= nr_epochs)

    #store the trained network

    #random input and output just for testing
    np.random.seed(111)
    input_test = np.random.rand(1, nr_Factors, chromLength_bins, 1)
    target_test = np.random.rand(1, matrixSize_bins)
    loss = model.evaluate(x=input_test, y=target_test)
    print("loss: {:.3f}".format(loss))
    pred = model.predict(x=input_test)
    print(pred[0:10])





if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter