#!python3
import utils
import models
import click
import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
import csv
import os
from Bio import SeqIO

from tqdm import tqdm

from numpy.random import seed
seed(35)

tf.random.set_seed(35)

@click.option("--trainMatrices","-tm",required=True,
                    multiple=True,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Training matrix in cooler format")
@click.option("--trainChromatinPaths","-tcp", required=True,
                    multiple=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides (training)")
@click.option("--trainChromosomes", "-tchroms", required=True,
              type=str, default="17", help="chromosome(s) to train on; separate multiple chromosomes by spaces")
@click.option("--validationMatrices", "-vm", required=True,
                    multiple=True,
                    type=click.Path(exists=True,dir_okay=False, readable=True),
                    help="Validation matrix in cooler format")
@click.option("--validationChromatinPaths","-vcp", required=True,
                    multiple=True,
                    type=click.Path(exists=True,file_okay=False,readable=True),
                    help="Path where chromatin factor data in bigwig format resides (validation)")
@click.option("--validationChromosomes", "-vchroms", required=True,
                    type=str, help="chromosomes for validation; separate multiple chromosomes by spaces")
@click.option("--sequenceFile", "-sf", required=False,
                    type=click.Path(exists=True,readable=True,dir_okay=False),
                    default=None,
                    help="Path to DNA sequence in fasta format")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where trained network will be stored")
@click.option("--modelfilepath", "-mfp", required=True, 
              type=click.Path(writable=True,dir_okay=False), 
              default="trainedModel.h5", help="path+filename for trained model")
@click.option("--learningRate", "-lr", required=True,
                type=click.FloatRange(min=1e-10), default=1e-1,
                help="learning rate for stochastic gradient descent")
@click.option("--numberEpochs", "-ep", required=True,
                type=click.IntRange(min=2), default=1000,
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
                type=bool, default=False,
                help="Clamp outliers in chromatin factor data")
@click.option("--scaleFactors","-scf", required=False,
                type=bool, default=True,
                help="Scale chromatin factor data to range 0...1 (recommended)")
@click.option("--modelType", "-mod", required=False,
                type=click.Choice(["initial", "current", "sequence"]),
                default="current",
                help="Type of model to use")
@click.option("--optimizer", "-opt", required=False,
                type=click.Choice(["SGD", "Adam", "RMSprop"]),
                default="Adam",
                help="Optimizer to use (SGD, Adam, RMSprop or cosine similarity)")
@click.option("--loss", "-l", required=False,
                type=click.Choice(["MSE", "MAE", "MAPE", "MSLE", "Cosine"]),
                default="MSE",
                help="Loss function to use, Mean Squared-, Mean Absolute-, Mean Absolute Percentage-, Mean Squared Logarithmic Error or Cosine similarity")
@click.command()
def training(trainmatrices,
            trainchromatinpaths,
            trainchromosomes,
            validationmatrices,
            validationchromatinpaths,
            validationchromosomes,
            sequencefile,
            outputpath,
            modelfilepath,
            learningrate,
            numberepochs,
            batchsize,
            windowsize,
            scalematrix,
            clampfactors,
            scalefactors,
            modeltype,
            optimizer,
            loss):
    #save the input parameters so they can be written to csv later
    paramDict = locals().copy()

    #number of train matrices must match number of chromatin paths
    #this is useful for training on matrices and chromatin factors 
    #from different cell lines
    if len(trainmatrices) != len(trainchromatinpaths):
        msg = "Number of train matrices and chromatin paths must match\n"
        msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
        msg = msg.format(len(trainmatrices), len(trainchromatinpaths))
        raise SystemExit(msg)
    if len(validationmatrices) != len(validationchromatinpaths):
        msg = "Number of validation matrices and chromatin paths must match\n"
        msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
        msg = msg.format(len(validationmatrices), len(validationchromatinpaths))
        raise SystemExit(msg) 

    #extract chromosome names and size from the input
    trainChromNameList = trainchromosomes.rstrip().split(" ")  
    trainChromNameList = [x.lstrip("chr") for x in trainChromNameList]  
    #check if all requested chroms are present in all train matrices
    trainMatricesDict = utils.checkGetChromsPresentInMatrices(trainmatrices,trainChromNameList)
    #check if all requested chroms are present in all bigwig files
    #and check if the chromosome lengths are equal for all bigwig files in each folder
    trainChromFactorsDict = utils.checkGetChromsPresentInFactors(trainchromatinpaths,trainChromNameList)
    #check if the chromosome lengths from the bigwig files match the ones from the matrices
    utils.checkChromSizesMatching(trainMatricesDict, trainChromFactorsDict, trainChromNameList)

    #repeat the steps above for the validation matrices and chromosomes
    validationChromNameList = validationchromosomes.rstrip().split(" ")
    validationChromNameList = [x.lstrip("chr") for x in validationChromNameList]
    validationMatricesDict = utils.checkGetChromsPresentInMatrices(validationmatrices, validationChromNameList)
    validationChromFactorsDict = utils.checkGetChromsPresentInFactors(validationchromatinpaths, validationChromNameList)
    utils.checkChromSizesMatching(validationMatricesDict, validationChromFactorsDict, validationChromNameList)

    #check if chosen model type matches inputs
    modelTypeStr = checkSetModelTypeStr(modeltype, sequencefile)
    paramDict["modeltype"] = modelTypeStr

    #load sparse Hi-C matrices per chromosome
    #scale and normalize, if requested
    print("Loading Training matrix/matrices")
    utils.loadMatricesPerChrom(trainMatricesDict, scalematrix, windowsize)
    print("\nLoading Validation matrix/matrices")
    utils.loadMatricesPerChrom(validationMatricesDict, scalematrix, windowsize)

    #load, bin and aggregate the chromatin factors into a numpy array
    #of size #matrix_size_in_bins x nr_chromatin_factors
    #loading is per corresponding training matrix (per folder)
    #and the bin sizes also correspond to the matrices
    print("\nLoading Chromatin factors for training")
    utils.loadChromatinFactorDataPerMatrix(trainMatricesDict,trainChromFactorsDict, trainChromNameList, pScaleFactors=scalefactors, pClampFactors=clampfactors)
    print("\nLoading Chromatin factors for validation")
    utils.loadChromatinFactorDataPerMatrix(validationMatricesDict, validationChromFactorsDict, validationChromNameList, pScaleFactors=scalefactors, pClampFactors=clampfactors)
    
    #check if DNA sequences for all chroms are there and correspond with matrices/chromatin factors
    #do not load them in memory yet, only store paths and sequence ids in the dicts
    #the generator can then load sequence data as required
    utils.getCheckSequences(trainMatricesDict,trainChromFactorsDict, sequencefile)
    utils.getCheckSequences(validationMatricesDict, validationChromFactorsDict, sequencefile)
    

    #generators for training and validation data
    trainDataGenerator = models.multiInputGenerator(matrixDict=trainMatricesDict,
                                                        factorDict=trainChromFactorsDict,
                                                        batchsize=batchsize,
                                                        windowsize=windowsize,
                                                        shuffle=True)
    validationDataGenerator = models.multiInputGenerator(matrixDict=validationMatricesDict,
                                                        factorDict=validationChromFactorsDict,
                                                        batchsize=batchsize,
                                                        windowsize=windowsize)
    
    #get number and names of chromatin factors 
    #and save to parameters dictionary
    nr_factors = max([trainChromFactorsDict[folder]["nr_factors"] for folder in trainChromFactorsDict])
    paramDict["nr_factors"] = nr_factors
    factorsNameList = []
    for folder in trainChromFactorsDict:
        factorsNameList.extend([name for name in trainChromFactorsDict[folder]["bigwigs"]])
    factorsNameList = sorted(list(set(factorsNameList)))
    for i, factor in enumerate(factorsNameList):
        paramDict["chromFactor_" + str(i)] = factor
    #get binsize and save to parameters dictionary
    binsize = max([trainMatricesDict[mName]["binsize"] for mName in trainMatricesDict])
    paramDict["binsize"] = binsize
    #get number of symbols and save to parameters dictionary
    nr_symbols = None
    if sequencefile is not None:
        nr_symbols =max([len(trainMatricesDict[mName]["seqSymbols"]) for mName in trainMatricesDict])
    paramDict["nr_symbols"] = nr_symbols
    

    #build the requested model
    model = models.buildModel(pModelTypeStr=modelTypeStr, 
                                    pWindowSize=windowsize,
                                    pBinSizeInt=binsize,
                                    pNrFactors=nr_factors,
                                    pNrSymbols=nr_symbols)
    #create optimizer
    kerasOptimizer = None
    if optimizer == "SGD":
        kerasOptimizer = tf.keras.optimizers.SGD(learning_rate=learningrate)
    elif optimizer == "Adam":
        kerasOptimizer = tf.keras.optimizers.Adam(learning_rate=learningrate)
    elif optimizer == "RMSprop":
        kerasOptimizer = tf.keras.optimizers.RMSprop(learning_rate=learningrate)
    else:
        raise NotImplementedError("unknown optimizer")
    #create loss
    loss_fn = None
    if loss == "MSE":
        loss_fn = tf.keras.losses.MeanSquaredError()
    elif loss == "MAE":
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    elif loss == "MAPE":
        loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
    elif loss == "MSLE":
        loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
    elif loss == "Cosine":
        loss_fn = tf.keras.losses.CosineSimilarity()
    else:
        raise NotImplementedError("unknown loss function")
    #compile the model
    model.compile(optimizer=kerasOptimizer, 
                 loss=loss_fn)
    model.summary()

    #callbacks to check the progress etc.
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=outputpath)
    saveFreqInt = len(trainDataGenerator) * 20 #every twenty batches
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
    tf.keras.utils.plot_model(model,show_shapes=True, to_file=outputpath + "model.png")
    
    #don't shuffle when sequencefile present
    #to avoid reloading sequences
    shuffle = (sequencefile is None)
    #shuffle = False

    #train the neural network
    history = model.fit(trainDataGenerator,
              shuffle=shuffle,
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


def checkSetModelTypeStr(pModelTypeStr, pSequenceFile):
    modeltypeStr = pModelTypeStr
    if pModelTypeStr == "sequence" and pSequenceFile is None:
        msg = "Aborting. Cannot use model type >sequence< without providing a sequence file (-sf option)"
        raise SystemExit(msg)
    if pModelTypeStr != "sequence" and pSequenceFile is not None:
        modeltypeStr = "sequence"
        msg = "Sequence file provided, but model type >sequence< not selected.\n" 
        msg += "Changed model type to >sequence<"
        print(msg)
    return modeltypeStr


if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter