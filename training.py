#!python3
import utils
import models
import click
import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
import csv
import os

from utils import getBigwigFileList, getMatrixFromCooler, binChromatinFactor, scaleArray
from tqdm import tqdm

from numpy.random import seed
seed(35)

tf.random.set_seed(35)

@click.option("--trainMatrices","-tm",required=True,
                    multiple=True,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Training matrix in cooler format")
@click.option("--chromatinPaths","-cp", required=True,
                    multiple=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides")
@click.option("--sequenceFile", "-sf", required=False,
                    type=click.Path(exists=True,readable=True,dir_okay=False),
                    default=None,
                    help="Path to DNA sequence in fasta format")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where trained network will be stored")
@click.option("--chromosomes", "-chroms", required=True,
              type=str, default="17", help="chromosome(s) to train on; separate multiple chromosomes by spaces")
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
                type=bool, default=True,
                help="Clamp outliers in chromatin factor data")
@click.option("--scaleFactors","-scf", required=False,
                type=bool, default=True,
                help="Scale chromatin factor data to range 0...1 (recommended)")
@click.option("--modelType", "-mod", required=False,
                type=click.Choice(["initial", "current", "sequence"]),
                default="current",
                help="Type of model to use")
@click.command()
def training(trainmatrices,
            chromatinpaths,
            sequencefile,
            outputpath,
            chromosomes,
            modelfilepath,
            learningrate,
            numberepochs,
            batchsize,
            windowsize,
            scalematrix,
            clampfactors,
            scalefactors,
            modeltype):
    #save the input parameters so they can be written to csv later
    paramDict = locals().copy()

    #number of train matrices must match number of chromatin paths
    #this is useful for training on matrices and chromatin factors 
    #from different cell lines
    if len(trainmatrices) != len(chromatinpaths):
        msg = "Number of train matrices and chromatin paths must match\n"
        msg += "Current numbers: Train matrices: {:d}; Chromatin Paths: {:d}"
        msg = msg.format(len(trainmatrices), len(chromatinpaths))
        raise SystemExit(msg)

    #extract chromosome names and size from the input
    chromNameList = chromosomes.rstrip().split(" ")  
    chromNameList = [x.lstrip("chr") for x in chromNameList]  
    #check if all requested chroms are present in all train matrices
    trainMatricesDict = checkGetChromsPresentInMatrices(trainmatrices,chromNameList)
    #check if all requested chroms are present in all bigwig files
    #and check if the chromosome lengths are equal for all bigwig files in each folder
    trainChromFactorsDict = checkGetChromsPresentInFactors(chromatinpaths,chromNameList)
    #check if the chromosome lengths from the bigwig files match the ones from the matrices
    checkChromSizesMatching(trainMatricesDict, trainChromFactorsDict, chromNameList)

    #check if chosen model type matches inputs
    modelTypeStr = modeltype
    if modelTypeStr == "sequence" and sequencefile is None:
        msg = "Aborting. Cannot use model type >sequence< without providing a sequence file (-sf option)"
        raise SystemExit(msg)
    if modelTypeStr != "sequence" and sequencefile is not None:
        modelTypeStr = "sequence"
        paramDict["modeltype"] = modelTypeStr
        msg = "Sequence file provided, but model type >sequence< not selected.\n" 
        msg += "Changed model type to >sequence<"
        print(msg)

    #load relevant parts of Hi-C matrices
    for mName in trainMatricesDict:
        dataDict = dict()
        for chromname in trainMatricesDict[mName]["chromsizes"]:
            sparseHiCMatrix, binSizeInt = getMatrixFromCooler(mName,chromname)
            if sparseHiCMatrix is None:
                msg = "Could not read Hi-C matrix {:s} for training, check inputs"
                msg = msg.format(mName)
                raise SystemExit(msg)
            if scalematrix: #scale matrix to 0..1, if requested
                sparseHiCMatrix = utils.scaleArray(sparseHiCMatrix)
            dataDict[chromname] = sparseHiCMatrix
            trainMatricesDict[mName]["binsize"] = binSizeInt #is the same for all chroms anyway
        trainMatricesDict[mName]["data"] = dataDict
        msg = "Cooler matrix {:s} loaded.\nBin size (resolution) is {:d}bp.\n"
        msg = msg.format(mName, trainMatricesDict[mName]["binsize"])
        chromList = [name for name in trainMatricesDict[mName]["chromsizes"]]
        chromSizeList = [size for size in [trainMatricesDict[mName]["chromsizes"][name] for name in chromList]]
        matShapeList = [mat.shape for mat in [trainMatricesDict[mName]["data"][name] for name in chromList]]
        minList = [mat.min() for mat in [trainMatricesDict[mName]["data"][name] for name in chromList]]
        maxList = [mat.max() for mat in [trainMatricesDict[mName]["data"][name] for name in chromList]]
        shapeMsg = []
        for name, size, shapeTuple, minVal, maxVal in zip(chromList, chromSizeList, matShapeList, minList, maxList):
            s = "Chromosome: {:s} - Length {:d} - Matrix shape ({:s}) - min. {:.1f} - max. {:.1f}"
            s = s.format(str(name), size, ", ".join(str(s) for s in shapeTuple), minVal, maxVal)
            shapeMsg.append(s)
        msg += "\n".join(shapeMsg)
        print(msg)

    #matrix distance normalization, divide values in each side diagonal by their average
    ##possible and even quite fast, but doesn't look reasonable
    #sparseHiCMatrix = utils.distanceNormalize(sparseHiCMatrix, windowsize)


    #check chromatin files
    bigwigFileList = getBigwigFileList(chromatinpath)
    if len(bigwigFileList) == 0:
        msg = "No bigwig files (*.bigwig, *.bigWig, *.bw) found in {:s}. Aborting."
        msg = msg.format(chromatinpath)
        raise SystemExit(msg)
    msg = "Found {:d} chromatin factors in {:s}:"
    msg = msg.format(len(bigwigFileList),chromatinpath)
    print(msg)
    for i, factor in enumerate(bigwigFileList):
        print(" ", factor)
        paramDict["chromFactor_" + str(i)] = str(factor)
    paramDict["nr_Factors"] = len(bigwigFileList)

    chromSizeFactors_List = [utils.getChromLengthFromBigwig(fn, chromosome) for fn in bigwigFileList]
    if min(chromSizeFactors_List) != max(chromSizeFactors_List):
        msg = "Warning: Chromosome lengths are not equal in bigwig files\n"
        msg += "Recorded lengths: "
        msg += ", ".join(str(l) for l in chromSizeFactors_List) + "\n"
        msg += "Using the max. value"
        print(msg)
    chromSizeFactorsInt = max(chromSizeFactors_List)

    if chromSizeMatrixInt != chromSizeFactorsInt:
        msg = "Warning: Matrix and chromatin factors have different chrom. sizes\n"
        msg += "Matrix: {:d}; Factors: {:d}"
        msg = msg.format(chromSizeMatrixInt, chromSizeFactorsInt)
        print(msg)
        binSizeFactors = int(np.ceil(chromSizeFactorsInt/binSizeInt))
        if binSizeFactors != sparseHiCMatrix.shape[0]:
            msg = "Aborting. Binned size also differs."
            msg += "Matrix: {:d}; Factors: {:d}\n"
            msg += "Matrix and bigwig files must correspond to same ref. genome"
            msg = msg.format(sparseHiCMatrix.shape[0], binSizeFactors)
            raise SystemExit(msg)
        else:
            msg = "The number of bins is the same. Continuing..."
            print(msg)
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
    encodedSequenceArray = None
    if modelTypeStr == "sequence" and sequencefile is not None:
        encodedSequenceArray = utils.buildSequenceArray(sequencefile,binSizeInt)
    
    nr_Factors = chromatinFactorArray.shape[1]
    nr_matrices = sparseHiCMatrix.shape[0] - 3* windowsize + 1
    
    #compute indices for train / validation split 
    trainIndices = np.random.choice(range(nr_matrices), size=(int(0.8*nr_matrices)), replace=False)
    validationIndices = np.setdiff1d(range(nr_matrices),trainIndices)

    #generators for training and validation data
    trainDataGenerator = models.multiInputGenerator(sparseMatrix=sparseHiCMatrix,
                                                   chromatinFactorArray=chromatinFactorArray, 
                                                   encodedDNAarray=encodedSequenceArray, 
                                                   indices=trainIndices,
                                                   batchsize=batchsize,
                                                   windowsize=windowsize,
                                                   binsize=binSizeInt)
    validationDataGenerator = models.multiInputGenerator(sparseMatrix=sparseHiCMatrix,
                                                        chromatinFactorArray=chromatinFactorArray,
                                                        encodedDNAarray=encodedSequenceArray, 
                                                        indices=validationIndices,
                                                        batchsize=batchsize,
                                                        windowsize=windowsize,
                                                        binsize=binSizeInt)

    #get the number of symbols in the DNA sequence (usually 5 or 4)
    nr_symbols = None
    if encodedSequenceArray is not None:
        nr_symbols = encodedSequenceArray.shape[1]
    paramDict["nr_symbols"] = nr_symbols

    #build the requested model
    model = models.buildModel(pModelTypeStr=modelTypeStr, 
                                    pWindowSize=windowsize,
                                    pBinSizeInt=binSizeInt,
                                    pNrFactors=nr_Factors,
                                    pNrSymbols=nr_symbols)
    kerasOptimizer = tf.keras.optimizers.Adam(learning_rate=learningrate)
    model.compile(optimizer=kerasOptimizer, 
                 loss=tf.keras.losses.MeanSquaredError())
    model.summary()

    #callbacks to check the progress etc.
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=outputpath)
    saveFreqInt = int(np.ceil(len(trainIndices)/batchsize) * 20)
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
    
    #train the neural network
    history = model.fit(trainDataGenerator,
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

def checkGetChromsPresentInMatrices(pTrainmatrices, pChromNameList):
    chromSizesInMatrices = dict()
    #get the chrom names and sizes present in the matrices
    for matrixName in pTrainmatrices:
        tmpMatDict = dict()
        tmpSizeDict = utils.getChromSizesFromCooler(matrixName)
        print(tmpSizeDict)
        if len(tmpSizeDict) == 0:
            msg = "Aborting. No chromosomes found in matrix {:s}"
            msg = msg.format(matrixName)
            raise SystemExit(msg)
        tmpMatDict["chromsizes"] = tmpSizeDict
        firstKey = str(list(tmpSizeDict.keys())[0])
        if firstKey.startswith("chr"):
            tmpMatDict["namePrefix"] = "chr"
        else:
            tmpMatDict["namePrefix"] = ""
        chromSizesInMatrices[matrixName] = tmpMatDict
    #check if all requested chromosomes are present in all matrices
    missingChroms = dict()
    for matrixName in chromSizesInMatrices:
        fullChromNameList = [chromSizesInMatrices[matrixName]["namePrefix"] + name for name in pChromNameList]
        missingChromList = [x for x in fullChromNameList if x not in chromSizesInMatrices[matrixName]["chromsizes"]]
        if len(missingChromList) > 0:
            missingChroms[matrixName] = missingChromList
    if len(missingChroms) > 0:
        msg = "Aborting. Problem with cooler matrices. The following chromosomes are missing:\n"
        for entry in missingChroms:
            msg += " Matrix: {:s} - Chrom(s): {:s}\n"
            msg = msg.format(entry, ", ".join(missingChroms[entry]))
        raise SystemExit(msg)
    #check if the sizes of the requested chromosomes are equal in all matrices
    sizeDict = dict()
    for chromName in pChromNameList:
        sizeList = []
        for matrixName in chromSizesInMatrices:
            fullname = chromSizesInMatrices[matrixName]["namePrefix"] + chromName
            sizeList.append(chromSizesInMatrices[matrixName]["chromsizes"][fullname])
        if len(set(sizeList)) > 1:
            sizeDict[chromName] = sizeList     
    if len(sizeDict) > 0:
        msg = "Warning: different chrom sizes in matrices.\n"
        msg = "Check input matrices if this is not intended.\n"
        msg = "Probably different reference genome.\n"
        msg += "\n".join(["chrom. " + str(chrom) + "- sizes:" + " ".join(sizeDict[chrom]) for chrom in sizeDict]) 
        print(msg)
    #restrict the output to the requested chromosomes
    for matrixName in chromSizesInMatrices:
        fullChromNameList = [chromSizesInMatrices[matrixName]["namePrefix"] + name for name in pChromNameList]
        chromSizesInMatrices[matrixName]["chromsizes"] = {k:chromSizesInMatrices[matrixName]["chromsizes"][k] for k in fullChromNameList}
    return chromSizesInMatrices

def checkGetChromsPresentInFactors(pChromatinpaths, pChromNameList):
    #load size data from all chromatin factors into a dict with the following structure:
    #folder1 - bw1 - name:size dict
    #              - namePrefix (e.g. "chr")
    #        - bw2 - name:size dict
    #              - namePrefix
    #etc.
    chromFactorDict = dict()
    for folder in pChromatinpaths:
        folderDict = dict()
        for bigwigfile in utils.getBigwigFileList(folder):
            bwDict = dict()
            bwDict["chromsizes"] = utils.getChromSizesFromBigwig(bigwigfile)
            if str(list(bwDict["chromsizes"].keys())[0]).startswith("chr"):
                bwDict["namePrefix"] = "chr"
            else:
                bwDict["namePrefix"] = ""
            folderDict[os.path.basename(bigwigfile)] = bwDict
        chromFactorDict[folder] = folderDict
    #check if the same number of chromatin factors is present in each folder
    if len(chromFactorDict) == 0:
        msg = "Aborting. Error loading bigwig files. Wrong format?"
        raise SystemExit(msg)
    nr_factorsInFolder = [len(y) for y in [chromFactorDict[x] for x in chromFactorDict]]
    if min(nr_factorsInFolder) != max(nr_factorsInFolder):
        msg = "Aborting. Number of chromatin factors in folders not equal"
        raise SystemExit(msg)
    nr_factorsInFolder = max(nr_factorsInFolder)
    #issue a warning if the file names are different
    #this is the case when there are more filenames than chromatin factors in each single folder
    fileNameSet = set()
    for folder in chromFactorDict:
        for bigwigfile in chromFactorDict[folder]:
            fileNameSet.add(bigwigfile)
    if len(fileNameSet) > nr_factorsInFolder:
        msg = "Warning: the names of the chromatin factors are not equal in each folder\n"
        msg += "Filenames:" + ", ".join(sorted(list(fileNameSet)))
        print(msg)
    #check if chromosomes are missing or have different lengths within the same folder
    #different lengths across folders is permitted, provided that the lengths are
    #equal to the ones from the corresponding matrices (to be checked separately)
    missingChromList = []
    lengthErrorList = []
    for chrom in pChromNameList:
        for folder in chromFactorDict:
            folderChromLengthList = []
            for bwfile in chromFactorDict[folder]:
                csDict = chromFactorDict[folder][bwfile]["chromsizes"]
                csPrefix = chromFactorDict[folder][bwfile]["namePrefix"]
                fullChromName = csPrefix + chrom
                if fullChromName not in csDict:
                    missingChromList.append([folder, bwfile, fullChromName])
                else:
                    folderChromLengthList.append(csDict[fullChromName])
            if len(folderChromLengthList) >0 and min(folderChromLengthList) != max(folderChromLengthList):
                lengthErrorList.append([folder, fullChromName])
    if len(missingChromList) > 0:
        msg = "Aborting. Following chromosomes are missing:\n"
        msg += "\n".join(["File: " + f[0]+f[1]+ "; Chrom: " + f[2] for f in missingChromList])
        raise SystemExit(msg)
    if len(lengthErrorList) > 0:
        msg = "Aborting. Following chromosomes differ in length:\n"
        msg += "\n".join(["Folder: " + f[0] + "; Chrom: " + f[1] for f in lengthErrorList])
        raise SystemExit(msg)
    #restrict the output to just the requested chromosomes.
    #we now know that they are all there and have the same length in each folder
    for folder in chromFactorDict:
        for bigwigfile in chromFactorDict[folder]:
            fullChromNameList = [chromFactorDict[folder][bigwigfile]["namePrefix"] + chromName for chromName in pChromNameList]
            chromFactorDict[folder][bigwigfile]["chromsizes"] = {k:chromFactorDict[folder][bigwigfile]["chromsizes"][k] for k in fullChromNameList}    
    return chromFactorDict

def checkChromSizesMatching(pMatrixChromSizesDict, pFactorsChromSizesDict, pChromNameList):
    #check if the matrices and the chromatin factors (bigwig files) in the corresponding folder
    #have the same chromosome length
    for mName,fFolder in zip(pMatrixChromSizesDict, pFactorsChromSizesDict):
        for chromName in pChromNameList:
            #get the full names and lengths of the relevant chromosomes
            #it has already been checked that the bigwig files have equal
            #chrom lengths within each folder, so looking at the first 
            #one in each folder is enough
            fullChromName_matrix = pMatrixChromSizesDict[mName]["namePrefix"] + chromName
            firstKey = str(list(pFactorsChromSizesDict[fFolder].keys())[0])
            fullChromName_factor1 = pFactorsChromSizesDict[fFolder][firstKey]["namePrefix"] + chromName
            chromLengthMatrix = pMatrixChromSizesDict[mName]["chromsizes"][fullChromName_matrix]
            chromLengthFactors = pFactorsChromSizesDict[fFolder][firstKey]["chromsizes"][fullChromName_factor1]
            if chromLengthFactors != chromLengthMatrix:
                msg = "Aborting. Chromosome length difference between matrix and chromatin factors\n"
                msg += "Matrix {:s} - Chrom {:s} - Length {:d} \n"
                msg = msg.format(mName, fullChromName_matrix, chromLengthMatrix)
                msg += "Chromatin factors in folder {:s} - Chrom {:s} - Length {:s}"
                msg = msg.format(fFolder, fullChromName_factor1, chromLengthFactors)
                raise SystemExit(msg)

if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter