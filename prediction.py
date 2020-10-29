import utils
import models
import click
import tensorflow as tf
import numpy as np
import csv
import ast
import os
from scipy import sparse


@click.option("--validationmatrix","-vm", required=False,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Target matrix in cooler format for statistical result evaluation, if available")
@click.option("--chromatinPath","-cp", required=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides")
@click.option("--sequenceFile", "-sf", required=False,
                    type=click.Path(exists=True,readable=True,dir_okay=False),
                    default=None,
                    help="Path to DNA sequence in fasta format")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where results will be stored")
@click.option("--trainedmodel", "-trm", required=True,
                    type=click.Path(exists=True, dir_okay=False),
                    help="Path+Filename of trained model")
@click.option("--chromosome", "-chrom", required=True,
                    type=str, default="17", help="chromosome to predict")
@click.option("--multiplier", "-mul", required=False,
                    type=click.FloatRange(min=1.0, max=50000), default=1.0,
                    help="Predicted matrices are scaled to value range 0...1.\n Use --multiplier mmm to get range 0...mmm e.g. for better visualization")
@click.option("--trainParamFile", "-tpf", required=True,
                type=click.Path(exists=True, dir_okay=False,readable=True),
                help="train parameter file (csv format)")
@click.command()
def prediction(validationmatrix, 
                chromatinpath, 
                sequencefile,
                outputpath, 
                trainedmodel,
                chromosome,
                multiplier,
                trainparamfile):
    
    predParamDict = locals().copy()

    #load the trained model
    try:
        trainedModel = tf.keras.models.load_model(trainedmodel)
    except Exception as e:
        print(e)
        msg = "Could not load trained model {:s}. Wrong file?"
        msg = msg.format(trainedmodel)
        raise SystemExit(msg)

    #load the param file and extract params;
    #required to decide about bin size 
    #and whether chromatin factors should be clamped and scaled
    try:
        with open(trainparamfile) as csvfile:
            reader = csv.DictReader(csvfile)
            trainParamDict =  reader.__next__()   #ignore anything but first line after header
    except Exception as e:
        msg = "Error: {:s}.\nCould not read train param file.".format(str(e))
        raise SystemExit(msg)
    try:
        windowsize = int(trainParamDict["windowsize"])
        binSizeInt = int(trainParamDict["binsize"])
        batchSizeInt = int(trainParamDict["batchsize"])
        clampfactors = trainParamDict["clampfactors"] == "True"
        scalefactors = trainParamDict["scalefactors"] == "True"
        scalematrix = trainParamDict["scalematrix"] == "True"
        modelType = str(trainParamDict["modeltype"])
        nr_Factors = int(trainParamDict["nr_factors"])
        factorNameSet = set([os.path.basename(trainParamDict["chromFactor_" + str(i)]) for i in range(nr_Factors)])
    except Exception as e:
        msg = "Aborting. Parameter not in param file or wrong data type:\n{:s}"
        msg = msg.format(str(e))
        raise SystemExit(msg)

    if modelType == "sequence" and sequencefile is None:
        msg = "Aborting. Model was trained with sequence, but no sequence file provided (option -sf)"
        raise SystemExit(msg)

    #extract chromosome names and size from the input
    chromNameList = chromosome.rstrip().split(" ")  
    chromNameList = sorted([x.lstrip("chr") for x in chromNameList])

    #check chromatin files first
    #if there are too few or too much, 
    #we can already stop here.
    chromFactorsDict = utils.checkGetChromsPresentInFactors([chromatinpath],chromNameList)
    if chromFactorsDict[chromatinpath]["nr_factors"] != nr_Factors:
        msg = "Too few or too many chromatin factors\n"
        msg += "Folder {:s} - {:d}\n"
        msg += "Trained model - {:d}"
        msg = msg.format(chromatinpath,chromFactorsDict[chromatinpath]["nr_factors"],nr_Factors)
        raise SystemExit(msg)
    factorsInFolderSet = set([bigwigfile for bigwigfile in chromFactorsDict[chromatinpath]["bigwigs"]])
    factorsDiff1 = factorNameSet - factorsInFolderSet
    factorsDiff2 = factorsInFolderSet - factorNameSet
    if len(factorsDiff1) > 0 or len(factorsDiff2) > 0:
        msg = "Aborting. Different chromatin factor (bigwig-)filenames\n"
        if len(factorsDiff1) > 0:
            msg += "The following factors were in the training set, but are not in the folder {:s}\n"
            msg = msg.format(chromatinpath)
            msg += ", ".join(list(factorsDiff1))
        if len(factorsDiff2) > 0:
            msg += "The following factors are in the folder {:s}, but were not in the training set\n"
            msg = msg.format(chromatinpath)
            msg += ", ".join(list(factorsDiff2))
        raise SystemExit(msg)
 
    if validationmatrix is not None:
        #load matrix for evaluating the predictions, if present 
        matricesDict = utils.checkGetChromsPresentInMatrices([validationmatrix],chromNameList)
        utils.checkChromSizesMatching(matricesDict,chromFactorsDict,chromNameList)
        utils.loadMatricesPerChrom(matricesDict,scalematrix,windowsize)
    else:
        #otherwise use a dummy dictionary so that the utils functions can be used
        matricesDict = buildDummyMatrixDict(chromatinpath,chromNameList,binSizeInt,chromFactorsDict)

    #load chromatin factor data
    utils.loadChromatinFactorDataPerMatrix(pMatricesDict=matricesDict,
                                            pChromFactorsDict=chromFactorsDict,
                                            pChromosomes=chromNameList,
                                            pScaleFactors=scalefactors,
                                            pClampFactors=clampfactors)

    #read the DNA sequence and do a one-hot encoding
    utils.getCheckSequences(matricesDict,chromFactorsDict, sequencefile)
    
    predictionDataGenerator = models.multiInputGenerator(matrixDict=None,
                                                factorDict=chromFactorsDict,
                                                batchsize=batchSizeInt,
                                                windowsize=windowsize,
                                                binsize=binSizeInt,
                                                shuffle=False)  

    #feed the chromatin factors through the trained model
    predMatrixArray = trainedModel.predict(predictionDataGenerator,batch_size=batchSizeInt)
    
    #the predicted matrices are overlapping submatrices of the actual target Hi-C matrices
    #they are ordered by chromosome names
    #first find the chrom lengths in bins
    chrLengthInBinsList = [chromFactorsDict[chromatinpath]["data"][chrom].shape[0] - 3*windowsize + 1  for chrom in chromNameList]
    if sum(chrLengthInBinsList) != predMatrixArray.shape[0]:
        msg = "Aborting. Failed separating prediction into single chromosomes"
        raise SystemExit(msg)
    #now split the prediction up into arrays of submatrices for each chromosome
    #scale predicted submatrices to 0...1
    indicesList = [sum(chrLengthInBinsList[0:i]) for i in range(len(chrLengthInBinsList)+1)]
    matrixPerChromList = []
    for i,j in zip(indicesList, indicesList[1:]):
        matrixPerChromList.append( utils.scaleArray(predMatrixArray[i:j,:]) * multiplier)
    
    #rebuild the cooler matrices from the overlapping 
    #submatrices for each chromosome and write to disk
    for i, matrix in enumerate(matrixPerChromList):
        matrixPerChromList[i] = utils.rebuildMatrix(matrix, windowsize)
    coolerMatrixName = outputpath + "predMatrix.cool"
    utils.writeCooler(pMatrixList=matrixPerChromList,
                     pBinSizeInt=binSizeInt,
                     pOutfile=coolerMatrixName,
                     pChromosomeList=chromNameList)

    #If target matrix provided, compute loss 
    #to assess prediction quality
    if validationmatrix is not None:
        evalGenerator = models.multiInputGenerator(matrixDict=matricesDict,
                                                  factorDict=chromFactorsDict,
                                                  batchsize=batchSizeInt,
                                                  windowsize=windowsize,
                                                  binsize=binSizeInt,
                                                  shuffle=False)
        loss = trainedModel.evaluate(evalGenerator)
        print("loss: {:.3f}".format(loss))

    #store results
    parameterFile = outputpath + "predParams.csv"    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(predParamDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(predParamDict)



def buildDummyMatrixDict(pChromPath, pChromNameList, pBinsize, pFactorDict):
    firstBigwigFile = list(pFactorDict[pChromPath]["bigwigs"].keys())[0]
    chromName = pFactorDict[pChromPath]["bigwigs"][firstBigwigFile]["namePrefix"] + pChromNameList[0]
    chromSize = pFactorDict[pChromPath]["bigwigs"][firstBigwigFile]["chromsizes"][chromName]
    chromSize = int(np.ceil(chromSize/pBinsize))
    dummyDict = dict()
    dummyDict["dummy"] = dict()
    dummyDict["dummy"]["data"] = dict()
    dummyDict["dummy"]["binsize"] = pBinsize
    dummyDict["dummy"]["namePrefix"] = ""
    dummyDict["dummy"]["data"][pChromNameList[0]] = sparse.csr_matrix((chromSize,chromSize),dtype=bool)
    return dummyDict





if __name__ == "__main__":
    prediction() #pylint: disable=no-value-for-parameter