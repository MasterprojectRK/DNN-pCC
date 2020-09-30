import utils
import click
import tensorflow as tf
import numpy as np
import csv
import ast


@click.option("--validationmatrix","-vm", required=False,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Target matrix in cooler format for statistical result evaluation, if available")
@click.option("--chromatinPath","-cp", required=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides")
@click.option("--sequenceFile", "-sf", required=True,
                    type=click.Path(exists=True,readable=True,dir_okay=False),
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
        binSizeInt = int(trainParamDict["binSize"])
        batchSizeInt = int(trainParamDict["batchsize"])
        clampfactors = bool(trainParamDict["clampfactors"])
        scalefactors = bool(trainParamDict["scalefactors"])
        trainmatshape = ast.literal_eval(trainParamDict["train_matrix_shape"])
        #for now, only allow same chrom length as in training
        chromLength_bins = int(trainmatshape[0])
    except Exception as e:
        msg = "Aborting. Parameter not in param file or wrong data type:\n{:s}"
        msg = msg.format(str(e))
        raise SystemExit(msg)

    #check whether the window size of the model fits the train params   
    inputlength = trainedModel.layers[0].input_shape[2]
    if inputlength % 3 != 0:
        msg = "Error. Expected input length is {:d} and does not divide by 3"
        msg = msg.format(inputlength)
        raise SystemExit(msg)
    modelWindowSize = int(inputlength / 3)
    if modelWindowSize != windowsize:
        msg = "Error. Windowsize in trainParam file does not match trained model\n."
        msg += "Trained Model: {:d}, Param File {:d}"
        msg = msg.format(modelWindowSize, windowsize)
        raise SystemExit(msg)
    
    #load chromatin files first
    #if there are too few or too much, 
    #we can already stop here.
    bigwigFileList = utils.getBigwigFileList(chromatinpath)
    nr_chromatinFactors = trainedModel.layers[0].input_shape[1]
    if len(bigwigFileList) != nr_chromatinFactors:
        msg = "Aborting.\n"
        msg += "Did not find the required number of bigwig files in {:s}\n"
        msg += "Required: {:d}, Found: {:d}"
        msg = msg.format(chromatinpath, nr_chromatinFactors, len(bigwigFileList))
        raise SystemExit(msg)
    msg = "Found {:d} chromatin factors in {:s}."
    msg = msg.format(len(bigwigFileList),chromatinpath)
    print(msg)
    for factor in bigwigFileList:
        print(factor)

    #read the DNA sequence and do a one-hot encoding
    encodedSequenceArray = utils.buildSequenceArray(sequencefile,binSizeInt)

    #now load relevant part of Hi-C matrix, if provided,
    #since the bin size will be taken from there
    if validationmatrix is not None:
        sparseHiCMatrix, binSizeInt2  = utils.getMatrixFromCooler(validationmatrix,chromosome)
        if sparseHiCMatrix is None:
            msg = "Could not read HiC matrix {:s} for training, check inputs"
            msg = msg.format(validationmatrix)
            raise SystemExit(msg)
        msg = "Cooler matrix {:s} loaded.\nBin size (resolution) is {:d}bp."
        msg = msg.format(validationmatrix, binSizeInt2)
        print(msg)
        print("matrix shape", sparseHiCMatrix.shape)
        if binSizeInt != binSizeInt2:
            msg = "Warning. Bin size in training file and parameter file do not match"
            print(msg)
    
    #compose chromatin factors
    chromatinFactorArray = utils.composeChromatinFactors(bigwigFileList,
                                                           pChromLength_bins=chromLength_bins, 
                                                           pBinSizeInt=binSizeInt,
                                                           pChromosomeStr=chromosome,
                                                           pClampArray=clampfactors,
                                                           pScaleArray=scalefactors)

    predIndices = np.arange(chromatinFactorArray.shape[1] - 3*windowsize + 1)
    predictionDataGenerator = utils.multiInputGenerator(None,chromatinFactorArray,encodedSequenceArray, predIndices,batchSizeInt,windowsize, binSizeInt, shuffle=False)  

    #feed the chromatin factors through the trained model
    predMatrixArray = trainedModel.predict(predictionDataGenerator,batch_size=batchSizeInt)
    
    #Scale predicted matrix to 0...1 
    #and multiply with given multiplier for better visualization.
    predMatrixArray = utils.scaleArray(predMatrixArray) * multiplier

    #rebuild the cooler matrix from the predictions and write out
    meanMatrix = utils.rebuildMatrix(predMatrixArray,windowsize)
    coolerMatrixName = outputpath + "predMatrix.cool"
    utils.writeCooler(meanMatrix,binSizeInt,coolerMatrixName,chromosome)

    #If target matrix provided, compute some figures 
    #to assess prediction quality
    if validationmatrix is not None:
        evalGenerator = utils.multiInputGenerator(sparseHiCMatrix,chromatinFactorArray,encodedSequenceArray,predIndices,batchSizeInt,windowsize,binSizeInt,shuffle=False)
        loss = trainedModel.evaluate(evalGenerator)
        print("loss: {:.3f}".format(loss))

    #store results
        #to be implemented


if __name__ == "__main__":
    prediction() #pylint: disable=no-value-for-parameter