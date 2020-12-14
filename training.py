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

np.random.seed(35)
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
@click.option("--scoreWeight", "-scw", required=False,
                type=click.FloatRange(min=0.0), default=0.0, show_default=True,
                help="Weight for insulation score loss (matrix loss is 1.0). Default 0.0 means score loss is not used")
@click.option("--scoreSize", "-ssz", required=False,
                type=click.IntRange(min=1), 
                help="Size for computation of insulation score loss. Must be (much) smaller than windowsize. Only relevant, if scoreWeight > 0")
@click.option("--tvWeight", "-tvw", required=False,
                type=click.FloatRange(min=0.0), default=0.0, show_default=True,
                help="Weight for Total Variation loss. Default 0.0 means TV loss is not used")
@click.option("--structureWeight", "-stw", required=False,
                type=click.FloatRange(min=0.0), default=0.0, show_default=True,
                help="Weight for MS-SSIM loss. Default 0.0 means MS-SSIM loss is not used")
@click.option("--perceptionWeight", "-pcw", required=False,
                type=click.FloatRange(min=0.0), default=0.0, show_default=True,
                help="Weight for perception loss. Default 0.0 means perception loss is not used")
@click.option("--optimizer", "-opt", required=False,
                type=click.Choice(["SGD", "Adam", "RMSprop"]),
                default="SGD", show_default=True,
                help="Optimizer to use: SGD (recommended), Adam or RMSprop")
@click.option("--loss", "-l", required=False,
                type=click.Choice(["MSE", "Huber0.1", "Huber0.5", "Huber1.0", "Huber5.0", "Huber10.0" "Huber100.0", "Huber1000.0", "MAE", "MAPE", "MSLE", "Cosine"]),
                default="MSE", show_default=True,
                help="Loss function to use for per-pixel loss: Mean Squared-, Mean Absolute-, Mean Absolute Percentage-, Mean Squared Logarithmic Error or Cosine similarity.")
@click.option("--pixelLossWeight", "-plw", required=False,
                type=click.FloatRange(min=0.0),
                default=1.0, show_default=True,
                help="Weight for per-pixel loss, default of 1.0 should not be exceeded in most cases")
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
            scoreweight,
            scoresize,
            tvweight,
            structureweight,
            perceptionweight,
            optimizer,
            loss,
            pixellossweight,
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
        paramDict["flankingsize"] = flankingsize
    
    if scoresize is None and scoreweight > 0.0:
        scoresize = int(windowsize * 0.25)
        paramDict["scoresize"] = scoresize

    #workaround for AlreadyExistsException when using perception loss
    #root cause seems to be a bug in grappler?
    if perceptionweight > 0.0:
        tf.config.optimizer.set_experimental_options({"arithmetic_optimization": False})

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

    #select the correct class for the data container
    containerCls = dataContainer.DataContainer
    
    #prepare the training data containers. No data is loaded yet.
    traindataContainerList = []
    for chrom in trainChromNameList:
        for matrix, chromatinpath in zip(trainmatrices, trainchromatinpaths):
            container = containerCls(chromosome=chrom,
                                    matrixfilepath=matrix,
                                    chromatinFolder=chromatinpath,
                                    sequencefilepath=sequencefile)
            traindataContainerList.append(container)

    #prepare the validation data containers. No data is loaded yet.
    validationdataContainerList = []
    for chrom in validationChromNameList:
        for matrix, chromatinpath in zip(validationmatrices, validationchromatinpaths):
            container = containerCls(chromosome=chrom,
                                    matrixfilepath=matrix,
                                    chromatinFolder=chromatinpath,
                                    sequencefilepath=sequencefile)
            validationdataContainerList.append(container)

    #define the load params for the containers
    loadParams = {"scaleFeatures": scalefactors,
                  "clampFeatures": clampfactors,
                  "scaleTargets": scalematrix,
                  "windowsize": windowsize,
                  "flankingsize": flankingsize,
                  "maxdist": maxdist}
    #now load the data and write TFRecords, one container at a time.
    if len(traindataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return #nothing to do
    container0 = traindataContainerList[0]
    tfRecordFilenames = []
    nr_samples_list = []
    for container in traindataContainerList + validationdataContainerList:
        container.loadData(**loadParams)
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
        tfRecordFilenames.append(container.writeTFRecord(pOutfolder=outputpath,
                                                        pRecordSize=recordsize))
        if debugstate is not None:
            if isinstance(debugstate, int):
                idx = debugstate
            else:
                idx = None
            container.plotFeatureAtIndex(idx=idx,
                                         outpath=outputpath,
                                         figuretype=figuretype)
            container.saveMatrix(outputpath=outputpath, index=idx)
        nr_samples_list.append(container.getNumberSamples())
        container.unloadData()
    traindataRecords = [item for sublist in tfRecordFilenames[0:len(traindataContainerList)] for item in sublist]
    validationdataRecords = [item for sublist in tfRecordFilenames[len(traindataContainerList):] for item in sublist]    
    

    #different binsizes are ok, as long as no sequence is used
    #not clear which binsize to use for prediction when they differ during training.
    #For now, store the max. 
    binsize = max([container.binsize for container in traindataContainerList])
    paramDict["binsize"] = binsize
    #because of compatibility checks above, 
    #the following properties are the same with all containers,
    #so just use data from first container
    nr_factors = container0.nr_factors
    paramDict["nr_factors"] = nr_factors
    for i in range(nr_factors):
        paramDict["chromFactor_" + str(i)] = container0.factorNames[i]
    sequenceSymbols = container0.sequenceSymbols
    nr_symbols = 0
    if isinstance(sequenceSymbols, set):
        nr_symbols = len(sequenceSymbols)
    nr_trainingSamples = sum(nr_samples_list[0:len(traindataContainerList)])
    storedFeaturesDict = container0.storedFeatures
    
    #save the training parameters to a file before starting to train
    #(allows recovering the parameters even if training is aborted
    # and only intermediate models are available)
    parameterFile = os.path.join(outputpath, "trainParams.csv")    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(paramDict)
    
    #build input streams
    #train data
    shuffleBufferSize = 3*recordsize
    trainDs = tf.data.TFRecordDataset(traindataRecords, 
                                        num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                        compression_type="GZIP")
    trainDs = trainDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    trainDs = trainDs.shuffle(buffer_size=shuffleBufferSize, reshuffle_each_iteration=True)
    trainDs = trainDs.batch(batchsize, drop_remainder=True)
    trainDs = trainDs.prefetch(tf.data.experimental.AUTOTUNE)
    #validation data
    validationDs = tf.data.TFRecordDataset(validationdataRecords, 
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                            compression_type="GZIP")
    validationDs = validationDs.map(lambda x: records.parse_function(x, storedFeaturesDict) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validationDs = validationDs.batch(batchsize)
    validationDs = validationDs.prefetch(tf.data.experimental.AUTOTUNE)

    #build the requested model
    model = models.ConversionModel(optimizer = optimizer,
                                   learning_rate=learningrate,
                                   windowsize=windowsize,
                                   flankingsize=flankingsize,
                                   nr_factors=nr_factors,
                                   nr_symbols=nr_symbols,
                                   model_type=modelTypeStr,
                                   binsize=binsize,
                                   tv_weight=tvweight,
                                   ssim_loss_weight=structureweight,
                                   pixel_loss_weight=pixellossweight,
                                   score_loss_weight=scoreweight,
                                   perception_loss_weight=perceptionweight,
                                   pixel_loss_function=loss,
                                   outfolder=outputpath,
                                   figure_type=figuretype)
    model.plotModel()
    #fit the model
    weights_before = model.model.trainable_weights[0].numpy()
    model.fit(train_ds=trainDs, 
              validation_ds=validationDs, 
              nr_epochs=numberepochs,
              steps_per_epoch=int(np.floor(nr_trainingSamples/batchsize)),
              save_freq=savefreq)
    weights_after = model.model.trainable_weights[0].numpy()
    print("weight sum before", np.sum(weights_before))
    print("weight sum after", np.sum(weights_after))

    #store the trained network
    model.model.save(filepath=modelfilepath,save_format="h5")

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

if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter