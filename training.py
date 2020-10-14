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

from utils import getBigwigFileList, getMatrixFromCooler, binChromatinFactor, scaleArray
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
            modeltype):
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
    trainMatricesDict = checkGetChromsPresentInMatrices(trainmatrices,trainChromNameList)
    #check if all requested chroms are present in all bigwig files
    #and check if the chromosome lengths are equal for all bigwig files in each folder
    trainChromFactorsDict = checkGetChromsPresentInFactors(trainchromatinpaths,trainChromNameList)
    #check if the chromosome lengths from the bigwig files match the ones from the matrices
    checkChromSizesMatching(trainMatricesDict, trainChromFactorsDict, trainChromNameList)

    #repeat the steps above for the validation matrices and chromosomes
    validationChromNameList = validationchromosomes.rstrip().split(" ")
    validationChromNameList = [x.lstrip("chr") for x in validationChromNameList]
    validationMatricesDict = checkGetChromsPresentInMatrices(validationmatrices, validationChromNameList)
    validationChromFactorsDict = checkGetChromsPresentInFactors(validationchromatinpaths, validationChromNameList)
    checkChromSizesMatching(validationMatricesDict, validationChromFactorsDict, validationChromNameList)

    #check if chosen model type matches inputs
    modelTypeStr = checkSetModelTypeStr(modeltype, sequencefile)
    paramDict["modeltype"] = modelTypeStr

    #load sparse Hi-C matrices per chromosome
    #scale and normalize, if requested
    loadMatricesPerChrom(trainMatricesDict, scalematrix, windowsize)
    loadMatricesPerChrom(validationMatricesDict, scalematrix, windowsize)

    #load, bin and aggregate the chromatin factors into a numpy array
    #of size #matrix_size_in_bins x nr_chromatin_factors
    #loading is per corresponding training matrix (per folder)
    #and the bin sizes also correspond to the matrices
    loadChromatinFactorDataPerMatrix(trainMatricesDict,trainChromFactorsDict, trainChromNameList)
    loadChromatinFactorDataPerMatrix(validationMatricesDict, validationChromFactorsDict, validationChromNameList)
    
    #check if DNA sequences for all chroms are there and correspond with matrices/chromatin factors
    #do not load them in memory yet, only store paths and sequence ids in the dicts
    #the generator can then load sequence data as required
    getCheckSequences(trainMatricesDict,trainChromFactorsDict, sequencefile)
    getCheckSequences(validationMatricesDict, validationChromFactorsDict, sequencefile)
    

    #generators for training and validation data
    trainDataGenerator = models.multiInputGenerator(matrixDict=trainMatricesDict,
                                                        factorDict=trainChromFactorsDict,
                                                        batchsize=batchsize,
                                                        windowsize=windowsize)
    validationDataGenerator = models.multiInputGenerator(matrixDict=validationMatricesDict,
                                                        factorDict=validationChromFactorsDict,
                                                        batchsize=batchsize,
                                                        windowsize=windowsize)
    #get the important parameters for the models
    nr_factors = max([trainChromFactorsDict[folder]["nr_factors"] for folder in trainChromFactorsDict])
    binsize = max([trainMatricesDict[mName]["binsize"] for mName in trainMatricesDict])
    nr_symbols =max([len(trainMatricesDict[mName]["seqSymbols"]) for mName in trainMatricesDict])

    #build the requested model
    model = models.buildModel(pModelTypeStr=modelTypeStr, 
                                    pWindowSize=windowsize,
                                    pBinSizeInt=binsize,
                                    pNrFactors=nr_factors,
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
    matrixDict = dict()
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
        matrixDict[matrixName] = tmpMatDict
    #check if all requested chromosomes are present in all matrices
    missingChroms = dict()
    for matrixName in matrixDict:
        fullChromNameList = [matrixDict[matrixName]["namePrefix"] + name for name in pChromNameList]
        missingChromList = [x for x in fullChromNameList if x not in matrixDict[matrixName]["chromsizes"]]
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
        for matrixName in matrixDict:
            fullname = matrixDict[matrixName]["namePrefix"] + chromName
            sizeList.append(matrixDict[matrixName]["chromsizes"][fullname])
        if len(set(sizeList)) > 1:
            sizeDict[chromName] = sizeList     
    if len(sizeDict) > 0:
        msg = "Warning: different chrom sizes in matrices.\n"
        msg = "Check input matrices if this is not intended.\n"
        msg = "Probably different reference genome.\n"
        msg += "\n".join(["chrom. " + str(chrom) + "- sizes:" + " ".join(sizeDict[chrom]) for chrom in sizeDict]) 
        print(msg)
    #restrict the output to the requested chromosomes
    for matrixName in matrixDict:
        fullChromNameList = [matrixDict[matrixName]["namePrefix"] + name for name in pChromNameList]
        matrixDict[matrixName]["chromsizes"] = {k:matrixDict[matrixName]["chromsizes"][k] for k in fullChromNameList}
    return matrixDict

def checkGetChromsPresentInFactors(pChromatinpaths, pChromNameList):
    #load size data from all chromatin factors into a dict with the following structure:
    #folder1 - bigwigs - bw1 - chromsizes - name:size dict
    #                        - namePrefix (e.g. "chr")
    #                  - bw2 - chromsizes - name:size dict
    #                        - namePrefix
    #                 - ...
    #        - nr_factors
    #etc.
    chromFactorDict = dict()
    for folder in pChromatinpaths:
        folderDict = dict()
        folderDict["bigwigs"] = dict()
        for bigwigfile in utils.getBigwigFileList(folder):
            bwDict = dict()
            bwDict["chromsizes"] = utils.getChromSizesFromBigwig(bigwigfile)
            if str(list(bwDict["chromsizes"].keys())[0]).startswith("chr"):
                bwDict["namePrefix"] = "chr"
            else:
                bwDict["namePrefix"] = ""
            folderDict["bigwigs"][os.path.basename(bigwigfile)] = bwDict
        chromFactorDict[folder] = folderDict
        chromFactorDict[folder]["nr_factors"] = len(chromFactorDict[folder]["bigwigs"])
    #check if the same number of chromatin factors is present in each folder
    if len(chromFactorDict) == 0:
        msg = "Aborting. Error loading bigwig files. Wrong format?"
        raise SystemExit(msg)
    nr_factorsInFolder = [chromFactorDict[folder]["nr_factors"] for folder in chromFactorDict]
    if min(nr_factorsInFolder) != max(nr_factorsInFolder):
        msg = "Aborting. Number of chromatin factors in folders not equal"
        raise SystemExit(msg)
    nr_factorsInFolder = max(nr_factorsInFolder)
    #Abort if the file names are different
    #this is the case when there are more filenames than chromatin factors in each single folder
    fileNameSet = set()
    for folder in chromFactorDict:
        for bigwigfile in chromFactorDict[folder]["bigwigs"]:
            fileNameSet.add(bigwigfile)
    if len(fileNameSet) > nr_factorsInFolder:
        msg = "Aborting. The names of the chromatin factors are not equal in each folder\n"
        msg += "Filenames:" + ", ".join(sorted(list(fileNameSet)))
        raise SystemExit(msg)
    #check if chromosomes are missing or have different lengths within the same folder
    #different lengths across folders is permitted, provided that the lengths are
    #equal to the ones from the corresponding matrices (to be checked separately)
    missingChromList = []
    lengthErrorList = []
    for chrom in pChromNameList:
        for folder in chromFactorDict:
            folderChromLengthList = []
            for bwfile in chromFactorDict[folder]["bigwigs"]:
                csDict = chromFactorDict[folder]["bigwigs"][bwfile]["chromsizes"]
                csPrefix = chromFactorDict[folder]["bigwigs"][bwfile]["namePrefix"]
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
        for bigwigfile in chromFactorDict[folder]["bigwigs"]:
            fullChromNameList = [chromFactorDict[folder]["bigwigs"][bigwigfile]["namePrefix"] + chromName for chromName in pChromNameList]
            chromFactorDict[folder]["bigwigs"][bigwigfile]["chromsizes"] = {k:chromFactorDict[folder]["bigwigs"][bigwigfile]["chromsizes"][k] for k in fullChromNameList}    
    return chromFactorDict

def checkChromSizesMatching(pMatricesDict, pFactorsDict, pChromNameList):
    #check if the matrices and the chromatin factors (bigwig files) in the corresponding folder
    #have the same chromosome length
    for mName,fFolder in zip(pMatricesDict, pFactorsDict):
        for chromName in pChromNameList:
            #get the full names and lengths of the relevant chromosomes
            #it has already been checked that the bigwig files have equal
            #chrom lengths within each folder, so looking at the first 
            #one in each folder is enough
            fullChromName_matrix = pMatricesDict[mName]["namePrefix"] + chromName
            firstBigwigFilename = str(list(pFactorsDict[fFolder]["bigwigs"].keys())[0])
            fullChromName_factor1 = pFactorsDict[fFolder]["bigwigs"][firstBigwigFilename]["namePrefix"] + chromName
            chromLengthMatrix = pMatricesDict[mName]["chromsizes"][fullChromName_matrix]
            chromLengthFactors = pFactorsDict[fFolder]["bigwigs"][firstBigwigFilename]["chromsizes"][fullChromName_factor1]
            if chromLengthFactors != chromLengthMatrix:
                msg = "Aborting. Chromosome length difference between matrix and chromatin factors\n"
                msg += "Matrix {:s} - Chrom {:s} - Length {:d} \n"
                msg = msg.format(mName, fullChromName_matrix, chromLengthMatrix)
                msg += "Chromatin factors in folder {:s} - Chrom {:s} - Length {:s}"
                msg = msg.format(fFolder, fullChromName_factor1, chromLengthFactors)
                raise SystemExit(msg)
        pFactorsDict[fFolder]["matrixName"] = mName
        pMatricesDict[mName]["chromatinFolder"] = fFolder

def loadMatricesPerChrom(pMatricesDict, pScaleMatrix, pWindowsize, pDistanceCorrection=False):
    #load relevant parts of Hi-C matrices
    for mName in pMatricesDict:
        dataDict = dict()
        for chromname in pMatricesDict[mName]["chromsizes"]:
            sparseHiCMatrix, binSizeInt = getMatrixFromCooler(mName,chromname)
            if sparseHiCMatrix is None:
                msg = "Could not read Hi-C matrix {:s} for training, check inputs"
                msg = msg.format(mName)
                raise SystemExit(msg)
            if pScaleMatrix: #scale matrix to 0..1, if requested
                sparseHiCMatrix = utils.scaleArray(sparseHiCMatrix)
            #matrix distance normalization, divide values in each side diagonal by their average
            ##possible and even quite fast, but doesn't look reasonable
            if pDistanceCorrection:
                sparseHiCMatrix = utils.distanceNormalize(sparseHiCMatrix, pWindowsize)
            dataDict[chromname] = sparseHiCMatrix
            pMatricesDict[mName]["binsize"] = binSizeInt #is the same for all chroms anyway
        pMatricesDict[mName]["data"] = dataDict
        msg = "Cooler matrix {:s} loaded.\nBin size (resolution) is {:d}bp.\n"
        msg = msg.format(mName, pMatricesDict[mName]["binsize"])
        chromList = [name for name in pMatricesDict[mName]["chromsizes"]]
        chromSizeList = [size for size in [pMatricesDict[mName]["chromsizes"][name] for name in chromList]]
        matShapeList = [mat.shape for mat in [pMatricesDict[mName]["data"][name] for name in chromList]]
        minList = [mat.min() for mat in [pMatricesDict[mName]["data"][name] for name in chromList]]
        maxList = [mat.max() for mat in [pMatricesDict[mName]["data"][name] for name in chromList]]
        shapeMsg = []
        for name, size, shapeTuple, minVal, maxVal in zip(chromList, chromSizeList, matShapeList, minList, maxList):
            s = "Chromosome: {:s} - Length {:d} - Matrix shape ({:s}) - min. {:.1f} - max. {:.1f}"
            s = s.format(str(name), size, ", ".join(str(s) for s in shapeTuple), minVal, maxVal)
            shapeMsg.append(s)
        msg += "\n".join(shapeMsg)
        print(msg)

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

def loadChromatinFactorDataPerMatrix(pMatricesDict,pChromFactorsDict,pChromosomes):
    #note the name of the corresponding matrices in the chromFactor dictionary
    for fFolder, mName in zip(pChromFactorsDict,pMatricesDict):
        bigwigFileList = [os.path.basename(x) for x in utils.getBigwigFileList(fFolder)] #ensures sorted order of files
        binsize = pMatricesDict[mName]["binsize"]
        dataPerChromDict = dict()
        for chrom in pChromosomes:
            chromName_matrix = pMatricesDict[mName]["namePrefix"] + chrom
            chromLength_bins = pMatricesDict[mName]["data"][chromName_matrix].shape[0]
            binnedChromFactorArray = np.empty(shape=(len(bigwigFileList),chromLength_bins))
            for i, bigwigFile in enumerate(bigwigFileList):
                chromName_bigwig = pChromFactorsDict[fFolder]["bigwigs"][bigwigFile]["namePrefix"] + chrom
                binnedFactor = utils.binChromatinFactor(fFolder+bigwigFile, binsize, chromName_bigwig)
                binnedChromFactorArray[i] = binnedFactor
            dataPerChromDict[chromName_matrix] = np.transpose(binnedChromFactorArray) #use matrix chrom name for easier access later on        
        pChromFactorsDict[fFolder]["data"] = dataPerChromDict

def getCheckSequences(pMatrixDict, pFactorsDict, pSequenceFile):
    if pSequenceFile is None:
        return
    #check if the binsize is the same for all matrices
    #sequence-based models won't work otherwise and we can stop right here
    #before loading any sequence
    binSizeList = [pMatrixDict[mName]["binsize"] for mName in pMatrixDict]
    if len(set(binSizeList)) > 1:
        msg = "Aborting. Bin size must be equal for all matrices\n"
        msg += "Current sizes: " + ", ".join(str(x) for x in binSizeList)
        raise SystemExit(msg)
    try:
        records = SeqIO.index(pSequenceFile, format="fasta")
    except Exception as e:
        print(e)
        msg = "Could not read sequence file. Wrong format?"
        raise SystemExit(msg)
    #find number of symbols in DNA (usually A,C,G,T and possibly N)
    symbolList = []
    for record in records:
        seqStr = records[record].seq.upper()
        symbolList.extend(set(list(seqStr)))
    del seqStr
    #check if all chromosomes are in the sequence file
    #and if they have the appropriate length
    seqIdList = list(records)
    for mName in pMatrixDict:
        seqIdDict = dict()
        chromNameList = list(pMatrixDict[mName]["chromsizes"].keys())
        for chrom in chromNameList:
            if chrom in seqIdList:
                seqIdDict[chrom] = chrom
            elif "chr" + chrom in seqIdList:
                seqIdDict[chrom] = "chr" + chrom
            else:
                msg = "Aborting. Chromsome {:s} is missing in sequence file {:s}"
                msg = msg.format(chrom, pSequenceFile)
                raise SystemExit(msg)
            #length check
            chromLengthSequence = len(records[ seqIdDict[chrom] ])
            chromLengthMatrix = pMatrixDict[mName]["chromsizes"][chrom]
            if chromLengthSequence != chromLengthMatrix:
                msg = "Aborting. Chromosome {:s} in sequence file {:s} has bad length\n"
                msg += "Matrix and chrom. factors: {:d} - Sequence File {:d}"    
                msg = msg.format(seqIdDict[chrom], pSequenceFile, chromLengthSequence, chromLengthMatrix)
                raise SystemExit(msg)
        pMatrixDict[mName]["seqID"] = seqIdDict
        pMatrixDict[mName]["seqFile"] = pSequenceFile
        folderName = pMatrixDict[mName]["chromatinFolder"]
        pFactorsDict[folderName]["seqID"] = seqIdDict
        pFactorsDict[folderName]["seqFile"] = pSequenceFile
        #add number of symbols
        pMatrixDict[mName]["seqSymbols"] = sorted(list(set(symbolList)))
        pFactorsDict[folderName]["seqSymbols"] = sorted(list(set(symbolList)))   
    records.close()




if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter