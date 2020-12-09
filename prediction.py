import utils
import models
import dataContainer
import records
import click
import tensorflow as tf
import numpy as np
import csv
import ast
import os
from scipy import sparse
from tqdm import tqdm


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
        factorNameList = [os.path.basename(trainParamDict["chromFactor_" + str(i)]) for i in range(nr_Factors)]
    except Exception as e:
        msg = "Aborting. Parameter not in param file or wrong data type:\n{:s}"
        msg = msg.format(str(e))
        raise SystemExit(msg)

    #backward compatibility with older param files
    #flankingsize used to be equal to windowsize and maxdist was not used
    flankingsize = None
    try:
        flankingsize = int(trainParamDict["flankingsize"])
    except:
        flankingsize = windowsize
    maxdist = None
    try:
        maxdist = int(trainParamDict["maxdist"])
    except:
        maxdist = None
    if maxdist is not None and maxdist > windowsize:
        msg = "Aborting. Parameters maxdist and windowsize from train parameter file colliding. Maxdist cannot be larger than windowsize."
        raise SystemExit(msg)
    #score was not used previously
    try:
        scoreSize = int(trainParamDict["scoresize"])
        scoreWeight = int(trainParamDict["scoreweight"])    
    except:
        scoreSize = None
        scoreWeight = 0.0
    
    #load the trained model
    modelLoadParams = {"filepath": trainedmodel}
    try:
        trainedModel = tf.keras.models.load_model(**modelLoadParams)
        weightsFirstLayer = trainedModel.layers[1].weights[0].numpy()
        print("weight sum {:.3f}".format(np.sum(weightsFirstLayer)))
    except Exception as e:
        print(e)
        msg = "Could not load trained model {:s} - Wrong file or format?"
        msg = msg.format(trainedmodel)
        raise SystemExit(msg)
    #check if a DNA sequence data is required as model input
    if modelType == "sequence" and sequencefile is None:
        msg = "Aborting. Model was trained with sequence, but no sequence file provided (option -sf)"
        raise SystemExit(msg)

    #extract chromosome names from the input
    chromNameList = chromosome.replace(",", " ").rstrip().split(" ")  
    chromNameList = sorted([x.lstrip("chr") for x in chromNameList])
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    for chrom in chromNameList:
        testdataContainerList.append(containerCls(chromosome=chrom,
                                                  matrixfilepath=validationmatrix,
                                                  chromatinFolder=chromatinpath,
                                                  sequencefilepath=sequencefile,
                                                  binsize=binSizeInt)) 
    #define the load params for the containers
    loadParams = {"scaleFeatures": scalefactors,
                  "clampFeatures": clampfactors,
                  "scaleTargets": scalematrix,
                  "windowsize": windowsize,
                  "flankingsize": flankingsize,
                  "maxdist": maxdist}
    #now load the data and write TFRecords, one container at a time.
    if len(testdataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return #nothing to do
    container0 = testdataContainerList[0]
    tfRecordFilenames = []
    for container in testdataContainerList:
        container.loadData(**loadParams)
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
        tfRecordFilenames.append(container.writeTFRecord(pOutfolder=outputpath,
                                                        pRecordSize=None)[0]) #list with 1 entry
        container.unloadData()    
    
    #input check - chromatin factors must have the same names
    #sufficient to compare against container0 due to above compatibility check
    if container0.factorNames != factorNameList:
        msg = "Aborting. The names of the chromatin factors are not equal\n"
        msg += "Trained model:\n"
        msg += "\n".join(factorNameList)
        msg += "Bigwig files in folder {:s}:\n".format(chromatinpath)
        msg += "\n".join(container0.factorNames)
        raise SystemExit(msg)
    
    #build the TFData input stream for prediction
    storedFeaturesDict = container0.storedFeatures
    testDs = tf.data.TFRecordDataset(tfRecordFilenames, 
                                        num_parallel_reads=None,
                                        compression_type="GZIP")
    testDs = testDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    testDs = testDs.batch(batchSizeInt) #do NOT drop the last batch (maybe incomplete, i.e. smaller, because batch size doesn't integer divide chrom size)
    if validationmatrix is not None:
        testDs = testDs.map(lambda x, y: x) #drop the target matrices (they are for evaluation)
    testDs = testDs.prefetch(tf.data.experimental.AUTOTUNE)

    #feed the chromatin factors through the trained model
    predList = []
    for x in testDs:
        predBatch = predStep(trainedModel, x).numpy()
        for i in range(predBatch.shape[0]):
            predList.append(predBatch[i])        
    predMatrixArray = np.array(predList)
    #the predicted matrices are overlapping submatrices of the actual target Hi-C matrices
    #they are ordered by chromosome names
    #first find the chrom lengths in bins
    chrLengthList = [container.chromSize_factors for container in testdataContainerList]
    chrLengthList = [int(np.ceil(entry / binSizeInt)) - (2*flankingsize + windowsize) + 1 for entry in chrLengthList]
    if sum(chrLengthList) != predMatrixArray.shape[0]:
        msg = "Aborting. Failed separating prediction into single chromosomes"
        raise SystemExit(msg)
    #now split the prediction up into arrays of submatrices for each chromosome
    indicesList = [sum(chrLengthList[0:i]) for i in range(len(chrLengthList)+1)]
    matrixPerChromList = []
    for i,j in zip(indicesList, indicesList[1:]):
        matrixPerChromList.append(predMatrixArray[i:j,:])
    #rebuild the matrices from the overlapping 
    #submatrices for each chromosome
    for i, matrix in enumerate(matrixPerChromList):
        matrixPerChromList[i] = utils.rebuildMatrix(pArrayOfTriangles=matrix, 
                                                    pWindowSize=windowsize,
                                                    pFlankingSize=flankingsize,
                                                    pMaxDist=maxdist )
    #scale the re-assembled matrices into range [0..multiplier]
    matrixPerChromList = [utils.scaleArray(matrix) * multiplier for matrix in matrixPerChromList]
    #write predicted chromosomes into a single cooler file
    coolerMatrixName = os.path.join(outputpath, "predMatrix.cool")
    metadata = {"trainParams": trainParamDict, "predParams": predParamDict}
    utils.writeCooler(pMatrixList=matrixPerChromList,
                     pBinSizeInt=binSizeInt,
                     pOutfile=coolerMatrixName,
                     pChromosomeList=chromNameList,
                     pMetadata=metadata)
    #compute scores, if scores were used during the training process
    if scoreWeight > 0.0 and isinstance(scoreSize, int):
        bedgraphFileName = "scorePrediction_ds{:d}.bedgraph".format(scoreSize)            
        bedgraphFileName = os.path.join(outputpath, bedgraphFileName)
        scoreList = [utils.computeScore(pMatrix=i, pDiamondsize=scoreSize) for i in tqdm(matrixPerChromList, desc="computing scores")]
        chromSizeList = [i.shape[0] * binSizeInt for i in matrixPerChromList]
        utils.saveInsulationScoreToBedgraph(scoreArrayList=scoreList, 
                                            chromSizeList=chromSizeList, 
                                            binsize=binSizeInt,
                                            diamondsize=scoreSize,
                                            chromosomeList=chromNameList,
                                            filename=bedgraphFileName)

    #If target matrix provided, compute loss 
    #to assess prediction quality
    if validationmatrix is not None:
        evalDs = tf.data.TFRecordDataset(tfRecordFilenames, 
                                        num_parallel_reads=None, #otherwise samples will be interleaved
                                        compression_type="GZIP")
        evalDs = evalDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        evalDs = evalDs.batch(batchSizeInt)
        evalDs = evalDs.prefetch(tf.data.experimental.AUTOTUNE)
        #compute loss for all samples
        lossList = []
        for x,y in evalDs:
            loss = evalStep(trainedModel, x, y)
            lossList.append(loss)
        #(approximately) split loss per chromosomes
        batchIndexList = [int(np.ceil(i/batchSizeInt)) for i in indicesList]
        lossPerChromList = [np.mean(lossList[i:j]) for i, j in zip(batchIndexList, batchIndexList[1:])]
        chromLossStrList = ["Chrom {:s}: {:.3f}".format(chrom, loss) for chrom, loss in zip(chromNameList, lossPerChromList)] 
        msg = "Mean loss(es):\n{:s}".format("\n".join(chromLossStrList))
        print(msg)
    
    #store prediction parameters
    parameterFile = os.path.join(outputpath, "predParams.csv")    
    with open(parameterFile, "w") as csvfile:
        dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(predParamDict.keys())))
        dictWriter.writeheader()
        dictWriter.writerow(predParamDict)
    #remove TFRecords (they can be large files)
    for record in tqdm(tfRecordFilenames, "removing TFRecords"):
        if os.path.exists(record):
            os.remove(record)

@tf.function
def predStep(trainedModel, inputBatch):
    pred_vals = trainedModel(inputBatch, training=False)
    return pred_vals

@tf.function
def evalStep(trainedModel, inputBatch, targetBatch, lossFn=tf.keras.losses.MeanSquaredError()):
    pred_vals = trainedModel(inputBatch, training=False)
    true_vals = targetBatch["out_matrixData"]
    loss = lossFn(true_vals, pred_vals)
    return loss



if __name__ == "__main__":
    prediction() #pylint: disable=no-value-for-parameter