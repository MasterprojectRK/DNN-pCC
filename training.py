import utils
import models
import records
import click
import tensorflow as tf
import numpy as np
import csv
import os
from tqdm import tqdm

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
                type=click.Choice(["MSE", "MAE", "MAPE", "MSLE", "Cosine"]),
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
    trainChromNameList = sorted([x.lstrip("chr") for x in trainChromNameList])  
    #check if all requested chroms are present in all train matrices
    trainMatricesDict = utils.checkGetChromsPresentInMatrices(trainmatrices,trainChromNameList)
    #check if all requested chroms are present in all bigwig files
    #and check if the chromosome lengths are equal for all bigwig files in each folder
    trainChromFactorsDict = utils.checkGetChromsPresentInFactors(trainchromatinpaths,trainChromNameList)
    #check if the chromosome lengths from the bigwig files match the ones from the matrices
    utils.checkChromSizesMatching(trainMatricesDict, trainChromFactorsDict, trainChromNameList)

    #repeat the steps above for the validation matrices and chromosomes
    validationChromNameList = validationchromosomes.rstrip().split(" ")
    validationChromNameList = sorted([x.lstrip("chr") for x in validationChromNameList])
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
    
    if debugstate is not None:
        for plotType in ["box", "line"]:
            utils.plotChromatinFactors(pFactorDict=trainChromFactorsDict,
                                        pMatrixDict=trainMatricesDict,
                                        pOutputPath=outputpath,
                                        pPlotType=plotType,
                                        pFigureType=figuretype)
            utils.plotChromatinFactors(pFactorDict=validationChromFactorsDict,
                                        pMatrixDict=validationMatricesDict,
                                        pOutputPath=outputpath,
                                        pPlotType=plotType,
                                        pFigureType=figuretype)

    #check if DNA sequences for all chroms are there and correspond with matrices/chromatin factors
    #do not load them in memory yet, only store paths and sequence ids in the dicts
    #the generator can then load sequence data as required
    utils.getCheckSequences(trainMatricesDict,trainChromFactorsDict, sequencefile)
    utils.getCheckSequences(validationMatricesDict, validationChromFactorsDict, sequencefile)
    
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


    #generators for training and validation data
    trainDataGenerator = models.multiInputGenerator(matrixDict=trainMatricesDict,
                                                        factorDict=trainChromFactorsDict,
                                                        batchsize=recordsize,
                                                        windowsize=windowsize,
                                                        binsize=binsize,
                                                        shuffle=False) #done in dataset)
    validationDataGenerator = models.multiInputGenerator(matrixDict=validationMatricesDict,
                                                        factorDict=validationChromFactorsDict,
                                                        batchsize=recordsize,
                                                        windowsize=windowsize,
                                                        binsize=binsize,
                                                        shuffle=False)    

    #write the training data to disk as tfRecord
    trainFilenameList = ["trainfile_{:03d}.tfrecord".format(i+1) for i in range(len(trainDataGenerator))]
    trainFilenameList = [os.path.join(outputpath, x) for x in trainFilenameList]
    nr_samples = 0
    for i, batch in enumerate(tqdm(trainDataGenerator,desc="generating training data")):
        if len(batch[0]) == 1:
            records.writeTFRecord(batch[0][0], None, batch[1], trainFilenameList[i])
        else:
            records.writeTFRecord(batch[0][0], batch[0][1], batch[1], trainFilenameList[i])
        nr_samples += batch[1].shape[0]
    #write the validation data to disk as tfRecord
    validationFilenameList = ["validationfile_{:03d}.tfrecord".format(i+1) for i in range(len(validationDataGenerator))]
    validationFilenameList = [os.path.join(outputpath, x) for x in validationFilenameList]
    for i, batch in enumerate(tqdm(validationDataGenerator, desc="generating validation data")):
        if len(batch[0]) == 1:
            records.writeTFRecord(batch[0][0], None, batch[1], validationFilenameList[i])
        else:
            records.writeTFRecord(batch[0][0], batch[0][1], batch[1], validationFilenameList[i])
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
    saveFreqInt = int(np.ceil(nr_samples / batchsize) * savefreq)
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
    shapeDict = dict()
    shapeDict["feats"] = (3*windowsize,nr_factors)
    shapeDict["targs"] = ( int(windowsize*(windowsize+1)/2), )
    if sequencefile is not None:
        shapeDict["dna"] = (windowsize*binsize,nr_symbols)
    trainDs = tf.data.TFRecordDataset(trainFilenameList, 
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                        compression_type="GZIP")
    trainDs = trainDs.map(lambda x: records.parse_function(x, shapeDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    trainDs = trainDs.shuffle(buffer_size=shuffleBufferSize, reshuffle_each_iteration=True)
    trainDs = trainDs.batch(batchsize)
    trainDs = trainDs.repeat(numberepochs)
    trainDs = trainDs.prefetch(tf.data.experimental.AUTOTUNE)
    validationDs = tf.data.TFRecordDataset(validationFilenameList, 
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                            compression_type="GZIP")
    validationDs = validationDs.map(lambda x: records.parse_function(x, shapeDict) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validationDs = validationDs.batch(batchsize)
    validationDs = validationDs.prefetch(tf.data.experimental.AUTOTUNE)

    #train the neural network
    history = model.fit(trainDs,
              epochs= numberepochs,
              validation_data= validationDs,
              callbacks= callback_fns,
              steps_per_epoch= int(np.ceil(nr_samples / batchsize))
            )

    #store the trained network
    model.save(filepath=modelfilepath,save_format="h5")

    #plot train- and validation loss over epochs
    lossPlotFilename = "lossOverEpochs.{:s}".format(figuretype)
    lossPlotFilename = os.path.join(outputpath, lossPlotFilename)
    utils.plotHistory(history, lossPlotFilename)

    #delete train- and validation records, if debugstate not set
    if debugstate is None or debugstate=="Figures":
        for trainfile in tqdm(trainFilenameList, desc="Deleting training record files"):
            if os.path.exists(trainfile):
                os.remove(trainfile)
        for validationfile in tqdm(validationFilenameList, desc="Deleting validation record files"):
            if os.path.exists(validationfile):
                os.remove(validationfile)

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