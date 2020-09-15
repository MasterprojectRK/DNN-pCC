#!python3
import utils
import click
import keras
from keras.layers import Conv2D,Dense,Dropout,Flatten
from keras.models import Sequential
import numpy as np
from scipy.stats import pearsonr

from utils import getBigwigFileList, getMatrixFromCooler, binChromatinFactor, normalize1Darray, showMatrix
from tqdm import tqdm

from numpy.random import seed
seed(35)

from tensorflow import set_random_seed
set_random_seed(35)

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
@click.option("--modelfilepath", "-mfp", required=True, 
              type=click.Path(writable=True,dir_okay=False), 
              default="./trainedModel", help="path+filename for trained model")
@click.command()
def training(trainmatrix, chromatinpath, outputpath, chromosome, modelfilepath):

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
    batchSize = 30
       
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
    #nr_matrices = 100
    matrixArray = np.empty(shape=(nr_matrices,matrixSize_bins))
    for i in tqdm(range(nr_matrices),desc="composing training matrices"):
        endIndex = i + startIndex + windowSize_bins
        trainmatrix = sparseHiCMatrix.toarray()[i+startIndex:endIndex,i+startIndex:endIndex][np.triu_indices(windowSize_bins)]
        matrixArray[i] = trainmatrix
    #plotMatrix = np.zeros(shape=(windowSize_bins,windowSize_bins))
    #plotMatrix[np.triu_indices(windowSize_bins)] = matrixArray[0]
    #showMatrix(plotMatrix)
    binnedChromatinFactorArray = np.empty(shape=(len(bigwigFileList),sparseHiCMatrix.shape[0]))
    ##bin the proteins
    for i in tqdm(range(len(bigwigFileList)),desc="binning chromatin factors"):
        binnedChromatinFactorArray[i] = normalize1Darray(binChromatinFactor(bigwigFileList[i],binSizeInt,chromosome))
    binnedChromatinFactorArray = np.expand_dims(binnedChromatinFactorArray, 2)
    ##compose chromatin factor input for all possible matrices
    chromatinFactorArray = np.empty(shape=(nr_matrices,len(bigwigFileList),3*windowSize_bins,1))
    for i in tqdm(range(nr_matrices),desc="composing chromatin factors"):
        endIndex = i + 3*windowSize_bins
        chromatinFactorArray[i] = binnedChromatinFactorArray[:,i:endIndex,:]
    #showMatrix(chromatinFactorArray[0].reshape(len(bigwigFileList),3*windowSize_bins))
    nr_Factors = len(bigwigFileList)
    #input_train = chromatinFactorArray[0:nr_matrices,:,:,:]
    #target_train = matrixArray[0:nr_matrices,:]
    choice = np.random.choice(range(nr_matrices), size=(int(0.8*nr_matrices)), replace=False)
    indices = np.zeros(nr_matrices, dtype=bool)
    indices[choice] = True
    
    input_train = chromatinFactorArray[indices,:,:,:]
    target_train = matrixArray[indices,:]
    input_val = chromatinFactorArray[~indices,:,:,:]
    target_val = matrixArray[~indices,:]

    print("chromatin NANs", np.any(np.isnan(chromatinFactorArray)), np.count_nonzero(np.isnan(chromatinFactorArray)))
    print("matrix NANs", np.any(np.isnan(matrixArray)))
    print("chromatin infs", np.any(np.isinf(chromatinFactorArray)))
    print("matrix infs", np.any(np.isinf(matrixArray)))

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
    model.add(Dense(nr_neurons3,activation="relu",kernel_regularizer="l2"))
    model.add(Dropout(0.1))
    model.add(Dense(nr_neurons4,activation="relu",kernel_regularizer="l2"))
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3), 
                  loss=keras.losses.MeanSquaredError())
    model.summary()
    
    #train the neural network
    model.fit(input_train, 
              target_train, 
              epochs= nr_epochs,
              batch_size=batchSize,
              validation_data=(input_val,target_val))

    #store the trained network
    keras.models.save_model(model,filepath=modelfilepath)

    #self-prediction just for testing
    input_test = np.expand_dims(chromatinFactorArray[0,:,:,:],0)
    target_test = np.expand_dims(matrixArray[0,:],0)
    loss = model.evaluate(x=input_test, y=target_test)
    print("loss: {:.3f}".format(loss))
    pred = model.predict(x=input_test)
    predMatrix = np.zeros(shape=(windowSize_bins,windowSize_bins))
    predMatrix[np.triu_indices(windowSize_bins)] = pred[0] 
    showMatrix(predMatrix)

    pearson_r, pearson_p = pearsonr(matrixArray[0],pred[0])
    msg = "Pearson R = {:.3f}, Pearson p = {:.3f}"
    msg = msg.format(pearson_r, pearson_p)
    print(msg)

    mse = (np.square(matrixArray[0] - pred[0])).mean(axis=None)
    print("MSE", mse)


if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter