import utils
import models
import records
import dataContainer
import click
import tensorflow as tf
import numpy as np
import csv
import os
from tqdm import tqdm
import pydot # implicitly required for plotting models

from numpy.random import seed
seed(35)

tf.random.set_seed(35)

@click.option("--trainMatrices","-tm",required=True,
                    multiple=True,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Training matrices in cooler format (-tm can be specfied multiple times)")
@click.option("--trainChromatinPaths","-tcp", required=True,
                    multiple=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides (training, -tcp can be specified multiple times)")
@click.option("--trainChromosomes", "-tchroms", required=True,
              type=str, help="chromosome(s) to train on; separate multiple chromosomes by spaces, eg. '10 11 12'")
@click.option("--validationMatrices", "-vm", required=True,
                    multiple=True,
                    type=click.Path(exists=True,dir_okay=False, readable=True),
                    help="Validation matrices in cooler format, -vm can be specified multiple times")
@click.option("--validationChromatinPaths","-vcp", required=True,
                    multiple=True,
                    type=click.Path(exists=True,file_okay=False,readable=True),
                    help="Path where chromatin factor data in bigwig format resides (validation, -vcp can be specified multiple times)")
@click.option("--validationChromosomes", "-vchroms", required=True,
                    type=str, help="chromosomes for validation; separate multiple chromosomes by spaces, eg. '10 11 12'")
@click.option("--sequenceFile", "-sf", required=False,
                    type=click.Path(exists=True,readable=True,dir_okay=False),
                    default=None,
                    help="Path to DNA sequence in fasta format. If specified, must contain sequences for all training- and validation chromosomes; seq. lengths must match with matrices and bigwig files")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where trained network, intermediate data and figures will be stored")
@click.option("--modelfilepath", "-mfp", required=True, 
              type=click.Path(writable=True,dir_okay=False), 
              default="trainedModel.h5", show_default=True, 
              help="path+filename for trained model in h5 format")
@click.option("--learningRate", "-lr", required=True,
                type=click.FloatRange(min=1e-10), 
                default=1e-5, show_default=True,
                help="learning rate for optimizer")
@click.option("--numberEpochs", "-ep", required=True,
                type=click.IntRange(min=2), 
                default=1000, show_default=True,
                help="Number of epochs for training the neural network.")
@click.option("--batchsize", "-bs", required=True,
                type=click.IntRange(min=5), 
                default=256, show_default=True,
                help="Batch size for training the neural network.")
@click.option("--recordsize", "-rs", required=False,
                type=click.IntRange(min=100), 
                default=2000, show_default=True,
                help="size (in samples) for training/validation data records")
@click.option("--windowsize", "-ws", required=True,
                type=click.IntRange(min=10), 
                default=80, show_default=True,
                help="Window size (in bins) for composing training data")
@click.option("--flankingsize", "-fs", required=False,
                type=click.IntRange(min=10),
                help="Size of flanking regions left/right of window in bins. Equal to windowsize if not set")
@click.option("--maxdist", "-md", required=False,
                type=click.IntRange(min=1),
                help="Training window can be capped at this distance (in bins). Equal to windowsize if not set")
@click.option("--scaleMatrix", "-scm", required=False,
                type=bool, 
                default=False, show_default=True,
                help="Scale Hi-C matrix to [0...1].")
@click.option("--clampFactors","-cfac", required=False,
                type=bool, 
                default=False, show_default=True,
                help="Clamp outliers in chromatin factor data.")
@click.option("--scaleFactors","-scf", required=False,
                type=bool, default=True,
                help="Scale chromatin factor data to range 0...1 (recommended)")
@click.option("--modelType", "-mod", required=False,
                type=click.Choice(["initial", "wider", "longer", "wider-longer", "sequence"]),
                default="initial", show_default=True,
                help="Type of model to use")
@click.option("--optimizer", "-opt", required=False,
                type=click.Choice(["SGD", "Adam", "RMSprop"]),
                default="SGD", show_default=True,
                help="Optimizer to use: SGD, Adam, RMSprop or cosine similarity.")
@click.option("--loss", "-l", required=False,
                type=click.Choice(["MSE", "Huber", "MAE", "MAPE", "MSLE", "Cosine"]),
                default="MSE", show_default=True,
                help="Loss function to use, Mean Squared-, Mean Absolute-, Mean Absolute Percentage-, Mean Squared Logarithmic Error or Cosine similarity.")
@click.option("--earlyStopping", "-early",
                required=False, type=click.IntRange(min=5),
                help="patience for early stopping, stop after this number of epochs w/o improvement in validation loss")
@click.option("--debugState", "-dbs", 
                required=False, type=click.Choice(["0","Figures"]),
                help="debug state for internal use during development")
@click.option("--figureType", "-ft",
                required=False,
                type=click.Choice(["png","svg","pdf"]),
                default="png", show_default=True,
                help="Figure format for plots")
@click.option("--saveFreq", "-sfreq",
                required=False,type=click.IntRange(min=1, max=1000),
                default=50, show_default=True,
                help="Save the trained model every sfreq batches (1<=sfreq<=1000)")
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
            recordsize,
            windowsize,
            flankingsize,
            maxdist,
            scalematrix,
            clampfactors,
            scalefactors,
            modeltype,
            optimizer,
            loss,
            earlystopping,
            debugstate,
            figuretype,
            savefreq):
    #save the input parameters so they can be written to csv later
    paramDict = locals().copy()

    if debugstate is not None and debugstate != "Figures":
        debugstate = int(debugstate)

    if maxdist is not None:
        maxdist = min(windowsize, maxdist)
        paramDict["maxdist"] = maxdist

    if flankingsize is None:
        flankingsize = windowsize
        paramDict["flankingsize"] = windowsize

    trainChromNameList = trainchromosomes.rstrip().split(" ")  
    trainChromNameList = sorted([x.lstrip("chr") for x in trainChromNameList])  

    validationChromNameList = validationchromosomes.rstrip().split(" ")
    validationChromNameList = sorted([x.lstrip("chr") for x in validationChromNameList])

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

    #check if chosen model type matches inputs
    modelTypeStr = checkSetModelTypeStr(modeltype, sequencefile)
    paramDict["modeltype"] = modelTypeStr

    #prepare the training data containers. No data is loaded yet.
    traindataContainerList = []
    for chrom in trainChromNameList:
        for matrix, chromatinpath in zip(trainmatrices, trainchromatinpaths):
            container = dataContainer.DataContainer(chromosome=chrom,
                                                    matrixfilepath=matrix,
                                                    chromatinFolder=chromatinpath,
                                                    sequencefilepath=sequencefile)
            traindataContainerList.append(container)

    #prepare the validation data containers. No data is loaded yet.
    validationdataContainerList = []
    for chrom in validationChromNameList:
        for matrix, chromatinpath in zip(validationmatrices, validationchromatinpaths):
            container = dataContainer.DataContainer(chromosome=chrom,
                                                    matrixfilepath=matrix,
                                                    chromatinFolder=chromatinpath,
                                                    sequencefilepath=sequencefile)
            validationdataContainerList.append(container)

    #now load the data and write TFRecords, one container at a time.
    if len(traindataContainerList) == 0:
        return #nothing to do
    container0 = traindataContainerList[0]
    tfRecordFilenames = []
    nr_samples_list = []
    for container in traindataContainerList + validationdataContainerList:
        container.loadData()
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
        tfRecordFilenames.append(container.writeTFRecord(pWindowsize=windowsize,
                                                        pOutfolder=outputpath,
                                                        pFlankingsize=flankingsize,
                                                        pMaxdist=maxdist,
                                                        pRecordSize=recordsize))
        if debugstate is not None:
            if isinstance(debugstate, int):
                idx = debugstate
            else:
                idx = None
            container.plotFeatureAtIndex(idx=idx, 
                                         flankingsize=flankingsize, 
                                         windowsize=windowsize, 
                                         maxdist=maxdist,
                                         outpath=outputpath,
                                         figuretype=figuretype)
        nr_samples_list.append(container.getNumberSamples(flankingsize=flankingsize, windowsize=windowsize))
        container.unloadData()
    traindataRecords = [item for sublist in tfRecordFilenames[0:len(traindataContainerList)] for item in sublist]
    validationdataRecords = [item for sublist in tfRecordFilenames[len(traindataContainerList):] for item in sublist]    
    

    #because of compatibility checks above, just use data from first container
    binsize = container0.binsize
    nr_factors = container0.nr_factors
    sequenceSymbols = container0.sequenceSymbols
    nr_symbols = None
    if isinstance(sequenceSymbols, set):
        nr_symbols = len(sequenceSymbols)
    nr_trainingSamples = sum(nr_samples_list[0:len(traindataContainerList)])
    storedFeaturesDict = container0.storedFeatures
    
    #build the requested model
    model = models.buildModel(pModelTypeStr=modelTypeStr, 
                                    pWindowSize=windowsize,
                                    pBinSizeInt=binsize,
                                    pNrFactors=nr_factors,
                                    pNrSymbols=nr_symbols,
                                    pFlankingSize=flankingsize,
                                    pMaxDist=maxdist)
    #define optimizer
    kerasOptimizer = getOptimizer(pOptimizerString=optimizer, pLearningrate=learningrate)
    
    # define loss(es)
    loss_fn = getLosses(pLossStr = loss)

    #compile the model
    model.compile(optimizer=kerasOptimizer, 
                 loss=loss_fn)
    model.summary()

    #callbacks to check the progress etc.
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=outputpath)
    saveFreqInt = int(np.ceil(nr_trainingSamples / batchsize) * savefreq)
    checkpointFilename = os.path.join(outputpath, "checkpoint_{epoch:05d}.h5")
    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointFilename,
                                                        monitor="val_loss",
                                                        save_freq=saveFreqInt)
    earlyStoppingCallback = None
    if earlystopping is not None:
        earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=earlystopping,
                                                            min_delta=0.001,
                                                            restore_best_weights=True)
    callback_fns = [tensorboardCallback,
                checkpointCallback]
    if earlystopping is not None:
        callback_fns.append(earlyStoppingCallback)
    

    #save the training parameters to a file before starting to train
    #(allows recovering the parameters even if training is aborted
    # and only intermediate models are available)
    parameterFile = os.path.join(outputpath, "trainParams.csv")    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)

    #plot the model
    modelPlotName = "model.{:s}".format(figuretype)
    modelPlotName = os.path.join(outputpath, modelPlotName)
    tf.keras.utils.plot_model(model,show_shapes=True, to_file=modelPlotName)
    
    #build input streams
    shuffleBufferSize = 3*recordsize
    trainDs = tf.data.TFRecordDataset(traindataRecords, 
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                        compression_type="GZIP")
    trainDs = trainDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    trainDs = trainDs.shuffle(buffer_size=shuffleBufferSize, reshuffle_each_iteration=True)
    trainDs = trainDs.batch(batchsize)
    trainDs = trainDs.repeat(numberepochs)
    trainDs = trainDs.prefetch(tf.data.experimental.AUTOTUNE)
    validationDs = tf.data.TFRecordDataset(validationdataRecords, 
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                            compression_type="GZIP")
    validationDs = validationDs.map(lambda x: records.parse_function(x, storedFeaturesDict) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validationDs = validationDs.batch(batchsize)
    validationDs = validationDs.prefetch(tf.data.experimental.AUTOTUNE)

    #train the neural network
    history = model.fit(trainDs,
              epochs= numberepochs,
              validation_data= validationDs,
              callbacks= callback_fns,
              steps_per_epoch= int(np.ceil(nr_trainingSamples / batchsize))
            )

    #store the trained network
    model.save(filepath=modelfilepath,save_format="h5")

    #plot train- and validation loss over epochs
    lossPlotFilename = "lossOverEpochs.{:s}".format(figuretype)
    lossPlotFilename = os.path.join(outputpath, lossPlotFilename)
    utils.plotHistory(history, lossPlotFilename)

    #delete train- and validation records, if debugstate not set
    if debugstate is None or debugstate=="Figures":
        for record in tqdm(traindataRecords + validationdataRecords, desc="Deleting TFRecord files"):
            if os.path.exists(record):
                os.remove(record)

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

def getOptimizer(pOptimizerString, pLearningrate):
    kerasOptimizer = None
    if pOptimizerString == "SGD":
        kerasOptimizer = tf.keras.optimizers.SGD(learning_rate=pLearningrate)
    elif pOptimizerString == "Adam":
        kerasOptimizer = tf.keras.optimizers.Adam(learning_rate=pLearningrate)
    elif pOptimizerString == "RMSprop":
        kerasOptimizer = tf.keras.optimizers.RMSprop(learning_rate=pLearningrate)
    else:
        raise NotImplementedError("unknown optimizer")
    return kerasOptimizer

def getLosses(pLossStr):
    loss_fn = None
    if pLossStr == "MSE":
        loss_fn = tf.keras.losses.MeanSquaredError()
    elif pLossStr == "Huber":
        loss_fn = tf.keras.losses.Huber(delta=2.5)
    elif pLossStr == "MAE":
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    elif pLossStr == "MAPE":
        loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
    elif pLossStr == "MSLE":
        loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
    elif pLossStr == "Cosine":
        loss_fn = tf.keras.losses.CosineSimilarity()
    else:
        raise NotImplementedError("unknown loss function")
    return loss_fn


if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter