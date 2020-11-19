#!python3
import os
import cooler
import pyBigWig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from scipy import sparse
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics as metrics

def getBigwigFileList(pDirectory):
    #returns a list of bigwig files in pDirectory
    retList = []
    for file in sorted(os.listdir(pDirectory)):
        if file.endswith(".bigwig") or file.endswith("bigWig") or file.endswith(".bw"):
            retList.append(pDirectory + file)
    return retList

def getChromSizesFromBigwig(pBigwigFileName):
    #returns the chrom sizes from a bigwig file in form of a dict
    chromSizeDict = dict()
    try:
        bigwigFile = pyBigWig.open(pBigwigFileName)
        chromSizeDict = bigwigFile.chroms()
        for entry in chromSizeDict:
            chromSizeDict[entry] = int(chromSizeDict[entry])
    except Exception as e:
        print(e) 
    return chromSizeDict
         
def getMatrixFromCooler(pCoolerFilePath, pChromNameStr):
    #returns sparse csr matrix from cooler file for given chromosome name
    sparseMatrix = None
    binSizeInt = 0
    try:
        coolerMatrix = cooler.Cooler(pCoolerFilePath)
        sparseMatrix = coolerMatrix.matrix(sparse=True,balance=False).fetch(pChromNameStr)
        sparseMatrix = sparseMatrix.tocsr() #so it can be sliced later
        binSizeInt = int(coolerMatrix.binsize)
    except Exception as e:
        print(e)
    return sparseMatrix, binSizeInt

def getChromSizesFromCooler(pCoolerFilePath):
    #get the sizes of the chromosomes present in a cooler matrix
    chromSizes = dict()
    try:
        coolerMatrix = cooler.Cooler(pCoolerFilePath) 
        chromSizes = coolerMatrix.chromsizes.to_dict()
    except Exception as e:
        print(e)
    return chromSizes

def binChromatinFactor(pBigwigFileName, pBinSizeInt, pChromStr, pChromSize=None):
    #bin chromatin factor loaded from bigwig file pBigwigFileName with bin size pBinSizeInt for chromosome pChromStr
    binArray = None
    properFileType = False
    try:
        bigwigFile = pyBigWig.open(pBigwigFileName)
        properFileType = bigwigFile.isBigWig()
    except Exception as e:
        print(e)
    if properFileType:
        chrom = pChromStr
        if chrom not in bigwigFile.chroms():
            msg = "Chromosome {:s} not present in bigwigfile {:s}"
            msg = msg.format(chrom, pBigwigFileName)
            raise SystemExit(msg)
        #compute signal values (stats) over resolution-sized bins
        if pChromSize is None:
            chromsize = bigwigFile.chroms(chrom)
        else:
            chromsize = pChromSize
        chromStartList = list(range(0,chromsize,pBinSizeInt))
        chromEndList = list(range(pBinSizeInt,chromsize,pBinSizeInt))
        chromEndList.append(chromsize)
        mergeType = 'mean'
        binArray = np.array(bigwigFile.stats(chrom, 0, chromsize, nBins=len(chromStartList), type=mergeType)).astype("float32")
        nr_nan = np.count_nonzero(np.isnan(binArray))
        nr_inf = np.count_nonzero(np.isinf(binArray))
        if nr_inf != 0 or nr_nan != 0:
            binArray = np.nan_to_num(binArray, nan=0.0, posinf=np.nanmax(binArray[binArray != np.inf]),neginf=0.0)
        if nr_inf != 0:
            msg_inf = "Warning: replaced {:d} infinity values in {:s} by 0/max. numeric value in data"
            msg_inf = msg_inf.format(nr_inf, pBigwigFileName)
            print(msg_inf)
        if nr_nan != 0:
            msg_nan = "Warning: replaced {:d} NANs in {:s} by 0."
            msg_nan = msg_nan.format(nr_nan, pBigwigFileName)
            print(msg_nan)
    return binArray

def scaleArray(pArray):
    '''
    min-max scaling for numpy arrays and sparse csr matrices

    Parameters:
    pArray (np.ndarray or sparse.csr_matrix): array to scale

    Returns:
    array scaled to value range [0..1]
    ''' 
    if pArray is None or pArray.size == 0:
        msg = "cannot normalize empty array"
        print(msg)
        return pArray
    if pArray.max() - pArray.min() != 0:
        normArray = (pArray - pArray.min()) / (pArray.max() - pArray.min())
    elif pArray.max() > 0: #min = max >0
        normArray = pArray / pArray.max()
    else: #min=max <= 0
        normArray = np.zeros_like(pArray)
    return normArray

def showMatrix(pMatrix):
    #test function to show matrices
    #debug only, not for production use
    print(pMatrix.max())
    plotmatrix = pMatrix + 1
    plt.matshow(plotmatrix, cmap="Reds", norm=colors.LogNorm())
    plt.show()

def plotMatrix(pMatrix, pFilename, pTitle):
    '''
    helper function to plot dense numpy 2D matrices in logscale to a file
    
    Parameters:
    pMatrix (numpy.ndarray): The matrix to plot, must be 2D
    pFilename (str): The filename for the plot, should have file extension .png, .pdf or .svg
    pTitle (str): A title that will appear on the plot

    Returns:
    None
    '''
    if not isinstance(pMatrix, np.ndarray) \
            or len(pMatrix.shape) != 2:
        return
    fig1, ax1 = plt.subplots()
    cs = ax1.matshow(pMatrix, cmap="RdYlBu_r", norm=colors.LogNorm())
    ax1.set_title(str(pTitle))
    fig1.colorbar(cs)
    fig1.savefig(pFilename)

def plotHistory(pKerasHistoryObject, pFilename):
    #plot loss and validation loss over epoch numbers
    fig1, ax1 = plt.subplots(figsize=(6,4.5))
    nr_epochs = len(pKerasHistoryObject.history['loss'])
    x_vals = np.arange(nr_epochs) + 1
    ax1.plot(x_vals, pKerasHistoryObject.history['loss'])
    ax1.plot(x_vals, pKerasHistoryObject.history['val_loss'])
    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.set_yscale('log')
    locVal = 0
    if nr_epochs <= 25:
        locVal = 1
    elif nr_epochs <= 50:
        locVal = 5
    elif nr_epochs <= 100:
        locVal = 10
    elif nr_epochs <= 500:
        locVal = 50
    elif nr_epochs <= 1000:
        locVal = 100
    elif nr_epochs <= 3000:
        locVal = 500
    elif nr_epochs <= 5000:
        locVal = 600
    else:
        locVal = 1000
    ax1.xaxis.set_major_locator(MultipleLocator(locVal))
    ax1.grid(True, which="both")
    ax1.legend(['train', 'val'], loc='upper left')
    fig1.savefig(pFilename)

def rebuildMatrix(pArrayOfTriangles, pWindowSize, pFlankingSize=None, pMaxDist=None, pStepsize=1):
    #rebuilds the interaction matrix (a trapezoid along its diagonal)
    #by taking the mean of all overlapping triangles
    #returns an interaction matrix as a numpy ndarray
    if pFlankingSize == None:
        flankingSize = pWindowSize
    else:
        flankingSize = pFlankingSize
    nr_matrices = pArrayOfTriangles.shape[0]
    sum_matrix = np.zeros( (nr_matrices - 1 + (pWindowSize+2*flankingSize), nr_matrices - 1 + (pWindowSize+2*flankingSize)) )
    count_matrix = np.zeros_like(sum_matrix,dtype=int)    
    mean_matrix = np.zeros_like(sum_matrix,dtype="float32")
    if pMaxDist is None or pMaxDist == pWindowSize:
        stepsize = 1
    else:
        #trapezoid, compute the stepsize such that the overlap is minimized
        stepsize = max(pStepsize, 1)
        stepsize = min(stepsize, pWindowSize - pMaxDist + 1) #the largest possible value such that predictions are available for all bins
    #sum up all the triangular or trapezoidal matrices, shifting by one along the diag. for each matrix
    for i in tqdm(range(0, nr_matrices, stepsize), desc="rebuilding matrix"):
        j = i + flankingSize
        k = j + pWindowSize
        if pMaxDist is None or pMaxDist == pWindowSize: #triangles
            sum_matrix[j:k,j:k][np.triu_indices(pWindowSize)] += pArrayOfTriangles[i]
        else: #trapezoids
            sum_matrix[j:k,j:k][np.mask_indices(pWindowSize, maskFunc, pMaxDist)] += pArrayOfTriangles[i]
        count_matrix[j:k,j:k] += np.ones((pWindowSize,pWindowSize),dtype=int) #keep track of how many matrices have contributed to each position
    mean_matrix[count_matrix!=0] = sum_matrix[count_matrix!=0] / count_matrix[count_matrix!=0]
    return mean_matrix

def writeCooler(pMatrixList, pBinSizeInt, pOutfile, pChromosomeList, pChromSizeList=None,  pMetadata=None):
    #takes a matrix as numpy array or sparse matrix and writes a cooler matrix from it
    #modified from study project such that multiple chroms can be written to a single matrix

    if pMatrixList is None or pChromosomeList is None or pBinSizeInt is None or pOutfile is None:
        msg = "input empty. No cooler matrix written"
        print(msg)
        return
    if len(pMatrixList) != len(pChromosomeList):
        msg = "number of input arrays and chromosomes must be the same"
        print(msg)
        return
    if pChromSizeList is not None and len(pChromSizeList) != len(pChromosomeList):
        msg = "if chrom sizes are given, they must be provided for ALL chromosomes"
        print(msg)
        return
    bins = pd.DataFrame(columns=['chrom','start','end'])
    pixels = pd.DataFrame(columns=['bin1_id','bin2_id','count']) 
    
    for i, (matrix, chrom) in enumerate(zip(pMatrixList,pChromosomeList)):
        #the chromosome size may not be integer-divisible by the bin size
        #so specifying the real chrom size is possible, but the
        #number of bins must still correspond to the matrix size
        chromSizeInt = int(matrix.shape[0] * pBinSizeInt)
        if pChromSizeList is not None \
                and pChromSizeList[i] is not None \
                and pChromSizeList[i] > (chromSizeInt - pBinSizeInt)\
                and pChromSizeList[i] < chromSizeInt:
            chromSizeInt = int(pChromSizeList[0])
        
        #store offset for later
        offset = bins.shape[0]

        #create the bins for cooler
        bins_tmp = pd.DataFrame(columns=['chrom','start','end'])
        binStartList = list(range(0, chromSizeInt, int(pBinSizeInt)))
        binEndList = list(range(int(pBinSizeInt), chromSizeInt, int(pBinSizeInt)))
        binEndList.append(chromSizeInt)
        bins_tmp['start'] = np.uint32(binStartList)
        bins_tmp['end'] = np.uint32(binEndList)
        bins_tmp["chrom"] = str(chrom)
        bins = bins.append(bins_tmp, ignore_index=True)

        #create the pixels for cooler
        triu_Indices = np.triu_indices(matrix.shape[0])
        pixels_tmp = pd.DataFrame(columns=['bin1_id','bin2_id','count'])
        pixels_tmp['bin1_id'] = (triu_Indices[0] + offset).astype("uint32")
        pixels_tmp['bin2_id'] = (triu_Indices[1] + offset).astype("uint32")
        readCounts = matrix[triu_Indices]
        if sparse.isspmatrix_csr(matrix): #for sparse matrices, slicing is different
            readCounts = np.transpose(readCounts)
        pixels_tmp['count'] = np.float64(readCounts)
        pixels = pixels.append(pixels_tmp, ignore_index=True)

    #convert the data types
    pixels["bin1_id"] = pixels["bin1_id"].astype("uint32")
    pixels["bin2_id"] = pixels["bin2_id"].astype("uint32")
    bins["start"] = bins["start"].astype("uint32")
    bins["end"] = bins["end"].astype("uint32")

    pixels.sort_values(by=['bin1_id','bin2_id'],inplace=True)

    #write out the cooler
    cooler.create_cooler(pOutfile, bins=bins, pixels=pixels, dtypes={'count': np.float64}, ordered=True, metadata=pMetadata)

def distanceNormalize(pSparseCsrMatrix, pWindowSize_bins):
    #compute the means along the diagonals (= same distance)
    #and divide all values on the diagonals by their respective mean
    diagList = []
    for i in range(pWindowSize_bins):
        diagArr = sparse.csr_matrix.diagonal(pSparseCsrMatrix,i)
        diagList.append(diagArr/diagArr.mean())
    distNormalizedMatrix = sparse.diags(diagList,np.arange(pWindowSize_bins),format="csr")
    return distNormalizedMatrix

def plotChromatinFactors(pFactorArray, pFeatureNameList, 
                            pChromatinFolder, pChrom, pBinsize, pStartbin,
                            pOutputPath, pPlotType, pFigureType="png"):
    #plot box- or line plots of the chromatin factors stored in pFactorDict
    #the matrices are required to determine the binsize for the line plots
    if pPlotType == "box":
        plotFn = plotChromatinFactors_boxplots
    elif pPlotType == "line":
        plotFn = plotChromatinFactors_lineplots
    else:
        return
 
    filename = "chromFactors_{:s}_{:s}_{:s}.{:s}".format(pPlotType, pChromatinFolder.rstrip("/").replace("/","-"), str(pChrom), pFigureType)
    filename = os.path.join(pOutputPath,filename)
    plotTitle = "Chromosome {:s} | Dir. {:s}".format(str(pChrom),pChromatinFolder)
    plotFn(pChromFactorArray=pFactorArray,
                    pFilename=filename, 
                    pBinSize=pBinsize,
                    pStartbin=pStartbin,
                    pAxTitle=plotTitle, 
                    pFactorNames=pFeatureNameList)

def plotChromatinFactors_boxplots(pChromFactorArray, pFilename, pBinSize=None, pStartbin=None, pAxTitle=None, pFactorNames=None):
    #store box plots of the chromatin factors in the array
    fig1, ax1 = plt.subplots()
    toPlotList = []
    for i in range(pChromFactorArray.shape[1]):
        toPlotList.append(pChromFactorArray[:,i])
    ax1.boxplot(toPlotList)
    fig1.suptitle("Chromatin factor boxplots")
    if pAxTitle is not None:
        ax1.set_title(str(pAxTitle))
    if pFactorNames is not None \
            and isinstance(pFactorNames,list) \
            and len(pFactorNames) == pChromFactorArray.shape[1]:
        ax1.set_xticklabels(pFactorNames, rotation=90)
    ax1.set_xlabel("Chromatin factor")
    ax1.set_ylabel("Chromatin factor signal value")
    fig1.tight_layout()
    fig1.savefig(pFilename)

def plotChromatinFactors_lineplots(pChromFactorArray, pFilename, pBinSize, pStartbin, pAxTitle=None, pFactorNames=None):
    #plot chromatin factors line plots
    #for debugging purposes only, not for production use
    winsize = pChromFactorArray.shape[0]
    nr_subplots = pChromFactorArray.shape[1]
    x_axis_values = np.arange(winsize) * pBinSize
    figsizeX = max(30, int(max(x_axis_values)/2000000))
    figsizeX = min(100, figsizeX)
    figsizeY = max(6, 3*nr_subplots)
    figsizeY = min(100, figsizeY)
    if isinstance(pStartbin, int):
        x_axis_values += pStartbin * pBinSize
    fig1, axs1 = plt.subplots(nr_subplots, 1, sharex = True, figsize=(figsizeX, figsizeY))
    for i in range(nr_subplots):
        axs1[i].plot(x_axis_values, pChromFactorArray[:,i])
        axs1[i].grid(True)
        #try to plot a reasonable number of major x-axis ticks
        if max(x_axis_values) < 1000000:
            locVal = 50000
        elif max(x_axis_values) < 10000000:
            locVal = 500000
        elif max(x_axis_values) < 50000000:
            locVal = 2500000
        elif max(x_axis_values) < 100000000:
            locVal = 5000000
        else:
            locVal = 10000000
        axs1[i].xaxis.set_major_locator(MultipleLocator(locVal))
        if pFactorNames is not None \
                and isinstance(pFactorNames,list) \
                and len(pFactorNames) == nr_subplots:
            axs1[i].set_xlabel(pFactorNames[i])
    if pAxTitle is not None:
        fig1.text(0.5, 0.04, str(pAxTitle), ha='center')
    axs1[0].set_xlim([min(x_axis_values), max(x_axis_values)])
    fig1.tight_layout()
    fig1.text(0.04, 0.5, 'signal value', va='center', rotation='vertical')
    fig1.suptitle("Chromatin factors")
    fig1.savefig(pFilename)

def clampArray(pArray):
    #clamp all values in pArray to be within 
    #lowerQuartile - 1.5xInterquartile ... upperQuartile + 1.5xInterquartile
    clampedArray = pArray.copy()
    upperQuartile = np.quantile(pArray,0.75)
    lowerQuartile = np.quantile(pArray,0.25)
    interQuartile = upperQuartile - lowerQuartile
    if interQuartile > 1.0:
        upperClampingBound = upperQuartile + 1.5*interQuartile
        lowerClampingBound = lowerQuartile - 1.5*interQuartile
        clampedArray[clampedArray < lowerClampingBound] = lowerClampingBound
        clampedArray[clampedArray > upperClampingBound] = upperClampingBound
    return clampedArray

def readSequencesPerId(pDNAFastaFileStr, pIdentifier):
    #read DNA sequence from a Fasta file based using its record identifier
    sequenceStr = ""
    try:
        seqDict = SeqIO.index(pDNAFastaFileStr,"fasta")
        sequenceStr = str(seqDict[pIdentifier].seq.upper())
    except Exception as e:
        msg = "Could not read DNA sequence for chrom {:s} from fasta file {:s}\n"
        msg += str(e)
        msg = msg.format(pIdentifier, pDNAFastaFileStr)
        print(msg)
    return sequenceStr

def encodeSequence(pSequenceStr, pClasses=None):
    #one-hot encoding for DNA sequences
    if pSequenceStr is None or pSequenceStr == "":
        msg = "Aborting. DNA sequence is empty"
        raise SystemExit(msg)
    mlb = MultiLabelBinarizer(classes=pClasses)
    encodedSequenceArray = mlb.fit_transform(pSequenceStr).astype("uint8")
    if encodedSequenceArray.shape[1] != 4:
        msg = "Warning: DNA sequence contains more than the 4 nucleotide symbols A,C,G,T\n"
        msg += "Check your input sequence, if this is not intended."
        print(msg)
        print("Contained symbols:", ", ".join(mlb.classes_))
    return encodedSequenceArray

def fillEncodedSequence(pEncodedSequenceArray, pBinSizeInt):
    #fill one-hot encoded sequence array with zero vectors such that
    #the length matches the number of bins
    if pBinSizeInt is None or not isinstance(pBinSizeInt, int):
        return
    actualLengthInt = pEncodedSequenceArray.shape[0] #here, length in basepairs
    targetLengthInt = int(np.ceil(actualLengthInt/pBinSizeInt))*pBinSizeInt #in basepairs
    returnArray = None
    if targetLengthInt > actualLengthInt:
        #append zero vectors to the array to fill the last bin
        #in case the chromosome length is not divisible by bin size (as is normal)
        toAppendArray = np.zeros((targetLengthInt-actualLengthInt,pEncodedSequenceArray.shape[1]),dtype="uint8")
        returnArray = np.append(pEncodedSequenceArray,toAppendArray,axis=0)
    else:
        msg = "Warning: could not append zeros to end of array.\n"
        msg += "Target length {:d}, actual length {:d}\n"
        msg += "Array left unchanged."
        msg = msg.format(targetLengthInt, actualLengthInt)
        print(msg)
        returnArray = pEncodedSequenceArray
    return returnArray

def checkGetChromsPresentInMatrices(pTrainmatricesList, pChromNameList):
    matrixDict = dict()
    #get the chrom names and sizes present in the matrices
    for matrixName in pTrainmatricesList:
        tmpMatDict = dict()
        tmpSizeDict = getChromSizesFromCooler(matrixName)
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
                sparseHiCMatrix = scaleArray(sparseHiCMatrix)
            #matrix distance normalization, divide values in each side diagonal by their average
            ##possible and even quite fast, but doesn't look reasonable
            if pDistanceCorrection:
                sparseHiCMatrix = distanceNormalize(sparseHiCMatrix, pWindowsize)
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

def loadChromatinFactorDataPerMatrix(pMatricesDict,pChromFactorsDict,pChromosomes,pScaleFactors=True,pClampFactors=False):
    #note the name of the corresponding matrices in the chromFactor dictionary
    for fFolder, mName in zip(pChromFactorsDict,pMatricesDict):
        msg = "Binning chromatin factors (bigwigs) in folder {:s}".format(fFolder)
        print(msg)
        bigwigFileList = [os.path.basename(x) for x in getBigwigFileList(fFolder)] #ensures sorted order of files
        binsize = pMatricesDict[mName]["binsize"]
        dataPerChromDict = dict()
        for chrom in pChromosomes:
            print("Chromosome", chrom)
            chromName_matrix = pMatricesDict[mName]["namePrefix"] + chrom
            chromLength_bins = pMatricesDict[mName]["data"][chromName_matrix].shape[0]
            binnedChromFactorArray = np.empty(shape=(len(bigwigFileList),chromLength_bins))
            for i, bigwigFile in enumerate(bigwigFileList):
                chromName_bigwig = pChromFactorsDict[fFolder]["bigwigs"][bigwigFile]["namePrefix"] + chrom
                binnedFactor = binChromatinFactor(os.path.join(fFolder,bigwigFile), binsize, chromName_bigwig)
                if pScaleFactors:
                    binnedFactor = scaleArray(binnedFactor)
                if pClampFactors:
                    binnedFactor = clampArray(binnedFactor)
                binnedChromFactorArray[i] = binnedFactor
                msg = "{:s} - min {:.3f} - max {:.3f}"
                msg = msg.format(bigwigFile, binnedChromFactorArray[i].min(), binnedChromFactorArray[i].max())
                print(msg)
            dataPerChromDict[chromName_matrix] = np.transpose(binnedChromFactorArray) #use matrix chrom name for easier access later on        
        pChromFactorsDict[fFolder]["data"] = dataPerChromDict

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
        for bigwigfile in getBigwigFileList(folder):
            bwDict = dict()
            bwDict["chromsizes"] = getChromSizesFromBigwig(bigwigfile)
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
        print("Checking sequences in {:s}. This may take a while.".format(pSequenceFile))
        records = SeqIO.index(pSequenceFile, format="fasta")
    except Exception as e:
        print(e)
        msg = "Could not read sequence file. Wrong format?"
        raise SystemExit(msg)
    print("Sequence file loaded...")
    symbolsDict = dict()
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
                msg = "Aborting. Chromosome {:s} in sequence file {:s} has wrong length\n"
                msg += "Matrix and chrom. factors: {:d} - Sequence File {:d}"    
                msg = msg.format(seqIdDict[chrom], pSequenceFile, chromLengthSequence, chromLengthMatrix)
                raise SystemExit(msg)
            if chrom not in symbolsDict:
                print("Checking symbols in DNA seq. of chrom {:s}".format(chrom))
                seqStr = records[seqIdDict[chrom]].seq.upper()
                symbolsDict[chrom] = set(list(seqStr))
                del seqStr
        pMatrixDict[mName]["seqID"] = seqIdDict
        pMatrixDict[mName]["seqFile"] = pSequenceFile
        folderName = pMatrixDict[mName]["chromatinFolder"]
        pFactorsDict[folderName]["seqID"] = seqIdDict
        pFactorsDict[folderName]["seqFile"] = pSequenceFile   
    records.close()
    
    #store symbols, interesting for later one-hot encoding
    symbolsSet = set()
    for key in symbolsDict:
        symbolsSet = symbolsSet.union(symbolsDict[key])
    for mName in pMatrixDict:
        pMatrixDict[mName]["seqSymbols"] = sorted(list(symbolsSet))
        pFactorsDict[folderName]["seqSymbols"] = sorted(list(symbolsSet))

def computePearsonCorrelation(pCoolerFile1, pCoolerFile2, 
                              pWindowsize_bp,
                              pModelChromList, pTargetChromStr,
                              pModelCellLineList, pTargetCellLineStr,
                              pPlotOutputFile=None, pCsvOutputFile=None):
    '''
    compute distance-stratified pearson correlation for target chromosome
    directly from cooler files and plot or write to file

    Parameters:
        pCoolerFile1 (str): Path to cooler file 1
        pCoolerFile2 (str): Path to cooler file 2
        pWindowsize_bp (int): Windowsize in basepairs for which correlations shall be computed
        pModelChromList (list): List of strings, will appear in plot title
        pModelCellLineList (list): List of strings, will appear in plot title
        pTargetChromStr (str): the target chromosome, e.g. >chr10< or >10<
        pTargetCellLineStr (str): the target cell line, will appear in plot title
        pPlotOutputFile (str): filename of correlation plot
        pCsvOutputFile (str): filename of correlation csv file
    
    Returns:
        None
    ''' 

    sparseMatrix1, binsize1 = getMatrixFromCooler(pCoolerFile1, pTargetChromStr)
    sparseMatrix2, binsize2 = getMatrixFromCooler(pCoolerFile2, pTargetChromStr)
    errorMsg = ""
    if sparseMatrix1 is None:
        errorMsg += "Chrom {:s} could not be loaded from {:s}\n"
        errorMsg = errorMsg.format(str(pTargetChromStr), pCoolerFile1)
    if sparseMatrix2 is None:
        errorMsg += "Chrom {:s} could not be loaded from {:s}\n"
        errorMsg = errorMsg.format(str(pTargetChromStr), pCoolerFile2)
    if errorMsg != "":
        errorMsg += "Potential reasons: Wrong file format, wrong chromosome naming scheme or chromosome missing"
        raise SystemExit(errorMsg)
    if binsize1 != binsize2:
        errorMsg = "Aborting. Binsizes of matrices are not equal\n"
        errorMsg += "{:s} -- {:d}bp\n"
        errorMsg += "{:s} -- {:d}bp\n"
        errorMsg = errorMsg.format(pCoolerFile1,binsize1, pCoolerFile2, binsize2)
        raise SystemExit(errorMsg)
    resultsDf = computePearsonCorrelationSparse(pSparseCsrMatrix1= sparseMatrix1,
                                                pSparseCsrMatrix2= sparseMatrix2,
                                                pBinsize= binsize1,
                                                pWindowsize_bp= pWindowsize_bp,
                                                pModelChromList= pModelChromList,
                                                pTargetChromStr= pTargetChromStr,
                                                pModelCellLineList= pModelCellLineList,
                                                pTargetCellLineStr= pTargetCellLineStr)
    if pCsvOutputFile is not None:
        resultsDf.to_csv(pCsvOutputFile)
    if pPlotOutputFile is not None:
        plotPearsonCorrelationDf(pResultsDfList=[resultsDf], 
                                 pLegendList=["Pearson corr."],
                                 pOutfile=pPlotOutputFile,
                                 pMethod="pearson")
    return resultsDf

def computePearsonCorrelationSparse(pSparseCsrMatrix1, pSparseCsrMatrix2, 
                                    pBinsize, pWindowsize_bp, 
                                    pModelChromList, pTargetChromStr, 
                                    pModelCellLineList, pTargetCellLineStr):
    '''
    compute distance-stratified Pearson correlation from two sparse matrices

    Parameters:
        pSparseCsrMatrix1 (scipy.sparse.csr_matrix): sparse csr matrix 1
        pSparseCsrMatrix2 (scipy.sparse.csr_matrix): sparse csr matrix 2
        pBinsize (int): the binsize of each bin in the sparse matrices
        pWindowsize_bp (int): the windowsize in basepairs for which correlations shall be computed
        pModelChromList (list): list of strings, will appear in plot title
        pTargetChromStr (str): the target chromosome, e.g. >chr10< or >10<
        pTargetCellLineStr (str): the target cell line, will appear in plot title
        pModelCellLineList (list): List of strings, will appear in plot title

    Returns:
        (pandas.DataFrame): Pandas dataframe containing the correlations per distance 
    '''
    numberOfDiagonals = int(np.round(pWindowsize_bp/pBinsize))
    if numberOfDiagonals < 1:
        msg = "Window size must be larger than bin size of matrices.\n"
        msg += "Remember to specify window in basepairs, not bins."
        raise SystemExit(msg)
    shape1 = pSparseCsrMatrix1.shape
    shape2 = pSparseCsrMatrix2.shape
    if shape1 != shape2:
        msg = "Aborting. Shapes of matrices are not equal.\n"
        msg += "Shape 1: ({:d},{:d}); Shape 2: ({:d},{:d})"
        msg = msg.format(shape1[0],shape1[1],shape2[0],shape2[1])
        raise SystemExit(msg)
    if numberOfDiagonals > shape1[0]-1:
        msg = "Aborting. Window size {0:d} larger than matrix size {:d}"
        msg = msg.format(numberOfDiagonals, shape1[0]-1)
        raise SystemExit(msg)
    
    trapezIndices = np.mask_indices(shape1[0],maskFunc,k=numberOfDiagonals)
    reads1 = np.array(pSparseCsrMatrix1[trapezIndices])[0]
    reads2 = np.array(pSparseCsrMatrix2[trapezIndices])[0]

    matrixDf = pd.DataFrame(columns=['first','second','distance','reads1','reads2'])
    matrixDf['first'] = np.uint32(trapezIndices[0])
    matrixDf['second'] = np.uint32(trapezIndices[1])
    matrixDf['distance'] = np.uint32(matrixDf['second'] - matrixDf['first'])
    matrixDf['reads1'] = np.float32(reads1)
    matrixDf['reads2'] = np.float32(reads2)
    matrixDf.fillna(0, inplace=True)

    pearsonAucIndices, pearsonAucValues = getCorrelation(matrixDf,'distance', 'reads1', 'reads2', 'pearson')
    pearsonAucScore = metrics.auc(pearsonAucIndices, pearsonAucValues)
    spearmanAucIncides, spearmanAucValues = getCorrelation(matrixDf,'distance', 'reads1', 'reads2', 'spearman')
    spearmanAucScore = metrics.auc(spearmanAucIncides, spearmanAucValues)
    print("PearsonAUC: {:.3f}".format(pearsonAucScore))
    print("SpearmanAUC: {:.3f}".format(spearmanAucScore))

    columns = ["corrMeth", "modelChroms", "targetChrom", 
                           "modelCellLines", "targetCellLine", 
                           "R2", "MSE", "MAE", "MSLE", "AUC",
                           "binsize", "windowsize"]
    columns.extend(sorted(list(matrixDf.distance.unique())))
    resultsDf = pd.DataFrame(columns=columns)
    resultsDf["corrMeth"] = ["pearson", "spearman"]
    resultsDf.set_index("corrMeth", inplace=True)
    resultsDf.loc[:, 'modelChroms'] = ", ".join([str(x) for x in pModelChromList])
    resultsDf.loc[:, 'targetChrom'] = pTargetChromStr
    resultsDf.loc[:, 'modelCellLines'] = ", ".join([str(x) for x in pModelCellLineList])
    resultsDf.loc[:, 'targetCellLine'] = pTargetCellLineStr
    resultsDf.loc[:, "R2"] = metrics.r2_score(matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[:, 'MSE'] = metrics.mean_squared_error( matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[:, 'MAE'] = metrics.mean_absolute_error( matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[:, 'MSLE'] = metrics.mean_squared_log_error(matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc['pearson', 'AUC'] = pearsonAucScore 
    resultsDf.loc['spearman', 'AUC'] = spearmanAucScore
    resultsDf.loc[:, 'binsize'] = pBinsize
    resultsDf.loc[:, 'windowsize'] = pWindowsize_bp
    
    for pearsonIndex, corrValue in zip(pearsonAucIndices,pearsonAucValues):
        columnName = int(round(pearsonIndex * matrixDf.distance.max()))
        resultsDf.loc["pearson", columnName] = corrValue
    for spearmanIndex, corrValue in zip(spearmanAucIncides,spearmanAucValues):
        columnName = int(round(spearmanIndex * matrixDf.distance.max()))
        resultsDf.loc["spearman", columnName] = corrValue
    return resultsDf
    
def plotPearsonCorrelationDf(pResultsDfList, pLegendList, pOutfile, pMethod="pearson"):
    #helper function to plot distance-stratified Pearson correlation stored in pandas dataframes
    if pMethod not in ["pearson", "spearman"]:
        print("plotting only supported for 'pearson' and 'spearman' correlation methods")
        return
    if pResultsDfList is None or pLegendList is None:
        return
    if not isinstance(pResultsDfList,list) or not isinstance(pLegendList,list):
        return
    legendStrList = [str(x) for x in pLegendList]
    if len(pResultsDfList) != len(legendStrList):
        msg = "can't plot, too many / too few legends\n"
        msg += "no. of legend entries should be: {:d}, given {:d}"
        msg = msg.format(len(pResultsDfList), len(legendStrList))
        print(msg)
        return
    
    fig1, ax1 = plt.subplots()
    ax1.set_ylabel("{:s} correlation".format(pMethod))
    ax1.set_xlabel("Genomic distance / Mbp")
    trainChromSet = set()
    targetChromSet = set()
    trainCellLineSet = set()
    targetCellLineSet = set()
    maxXVal = 0
    for i, resultsDf in enumerate(pResultsDfList):
        try:
            resolutionInt = int(resultsDf.loc[pMethod, 'binsize'])
            windowsize_bp = int(resultsDf.loc[pMethod, 'windowsize'])
            trainChromSet.add(resultsDf.loc[pMethod, 'modelChroms'])
            targetChromSet.add(resultsDf.loc[pMethod, 'targetChrom'])
            trainCellLineSet.add(resultsDf.loc[pMethod, 'modelCellLines'])
            targetCellLineSet.add(resultsDf.loc[pMethod, 'targetCellLine'])
            area_under_corr_curve = resultsDf.loc[pMethod, 'AUC']
            maxDist_bp = int(windowsize_bp / resolutionInt)
            columnNameList = [x for x in range(maxDist_bp)]
            corrXValues = np.arange(maxDist_bp) * resolutionInt / 1000000
            corrYValues = resultsDf.loc[pMethod, columnNameList].values.astype("float32")
        except Exception as e:
            msg = str(e) + "\n"
            msg += "results dataframe {:d} does not contain all relevant fields (binsize, distance stratified pearson correlation data etc.)"
            msg = msg.format(i)
            print(msg)
        label = pLegendList[i]
        if label is None:
            label = pMethod + " / AUC: {:.3f}".format(area_under_corr_curve)
        else:
            label = label + " / AUC: {:.3f}".format(area_under_corr_curve)
        ax1.plot(corrXValues, corrYValues, label = label)
        maxXVal = max(maxXVal, corrXValues[-1])
    titleStr = "Pearson correlation vs. genomic distance"
    if len(trainChromSet) == len(targetChromSet) == len(trainCellLineSet) == len(targetCellLineSet) == 1:
        titleStr += "\n {:s}, {:s} on {:s}, {:s}"
        titleStr = titleStr.format(list(trainCellLineSet)[0], list(trainChromSet)[0], list(targetCellLineSet)[0], list(targetChromSet)[0])
    ax1.set_title(titleStr)
    ax1.set_ylim([0,1])
    ax1.set_xlim([0,maxXVal])

    ax1.legend(frameon=False)
    
    if pOutfile is None:
        outfile = "correlation.png"
        fig1.savefig(outfile)
    else:
        outfile = pOutfile
        if os.path.splitext(outfile)[1] not in ['.png', '.svg', '.pdf']:
            outfile = os.path.splitext(pOutfile)[0] + '.png'
            msg = "Outfile must have png, pdf or svg file extension.\n"
            msg += "Renamed outfile to {:s}".format(outfile)
            print(msg)
        fig1.savefig(outfile)

def maskFunc(pArray, pWindowSize=0):
    #mask a trapezoid along the (main) diagonal of a 2D array
    #this code is copied from the study project by Ralf Krauth
    #https://github.com/MasterprojectRK/HiCPrediction/blob/master/hicprediction/createTrainingSet.py
    maskArray = np.zeros(pArray.shape)
    upperTriaInd = np.triu_indices(maskArray.shape[0]) # pylint: disable=unsubscriptable-object
    notRequiredTriaInd = np.triu_indices(maskArray.shape[0], k=pWindowSize) # pylint: disable=unsubscriptable-object
    maskArray[upperTriaInd] = 1
    maskArray[notRequiredTriaInd] = 0
    return maskArray

def getCorrelation(pData, pDistanceField, pTargetField, pPredictionField, pCorrMethod):
    """
    Helper method to calculate correlation
    This method has originally been written by Andre Bajorat during his study project,
    licensed under the MIT License: 
    https://github.com/abajorat/HiCPrediction/blob/master/hicprediction/predict.py
    It has been adapted by Ralf Krauth during his study project:
    https://github.com/MasterprojectRK/HiCPrediction/blob/master/hicprediction/predict.py
    
    Parameters:
        pData (pandas.DataFrame): Pandas dataframe with read counts / distances
        pDistanceField (str): the column name of the distance Field in the dataframe
        pTargetField (str): the column name of the target read counts in the dataframe
        pPredictionField (str): column name of the predicted read counts in the dataframe
        pCorrMethod (str): any of the correlation methods supported by pandas DataFrame corr method
    
    Returns:
        indices (list): integer list of index values 
        values (list): float list of correlation values
    """

    new = pData.groupby(pDistanceField, group_keys=False)[[pTargetField,
        pPredictionField]].corr(method=pCorrMethod)
    new = new.iloc[0::2,-1]
    #sometimes there is no variation in prediction / target per distance, then correlation is NaN
    #need to drop these, otherwise AUC will be NaN, too.
    new.dropna(inplace=True) 
    values = new.values
    indices = new.index.tolist()
    indices = list(map(lambda x: x[0], indices))
    indices = np.array(indices)
    div = pData[pDistanceField].max()
    indices = indices / div 
    return indices, values

def getChromPrefixBigwig(pBigwigFileName):
    '''
    check if the chromosome names in the bigwig file 
    start with 'chr' or not; e.g. 'chr10' vs. '10'
    '''
    try:
        bigwigFile = pyBigWig.open(pBigwigFileName)
        chromSizeDict = bigwigFile.chroms()
        chromNameList = [entry for entry in chromSizeDict]
    except Exception as e:
        raise(e) 
    prefix = None
    if chromNameList is not None and len(chromNameList) > 0 and str(chromNameList[0]).startswith("chr"):
        prefix = "chr"
    elif chromNameList is not None and len(chromNameList) > 0:
        prefix = ""
    else:
        msg = "No valid entries found in bigwig file {:s}"
        msg = msg.format(pBigwigFileName)
        raise ValueError(msg)
    return prefix

def getChromPrefixCooler(pCoolerFileName):
    '''
    check if the chromosomes in the cooler file 
    start with 'chr' or not; e.g. 'chr10' vs. '10'
    '''
    try:
        coolerMatrix = cooler.Cooler(pCoolerFileName) 
        chromSizes = coolerMatrix.chromsizes.to_dict()
        chromNameList = [entry for entry in chromSizes]
    except Exception as e:
        raise(e)
    prefix = None
    if chromNameList is not None and len(chromNameList) > 0 and str(chromNameList[0]).startswith("chr"):
        prefix = "chr"
    elif  chromNameList is not None and len(chromNameList) > 0:
        prefix = ""
    else:
        msg = "No valid entries found in cooler file {:s}"
        msg = msg.format(pCoolerFileName)
        raise ValueError(msg) 
    return prefix

def getDiamondIndices(pMatsize, pDiamondsize):
    nr_diamonds = pMatsize - 2*pDiamondsize
    if nr_diamonds <= 1:
        msg = "Diamondsize too large for Matsize"
        raise ValueError(msg)
    start_offset = pDiamondsize
    rowEndList = [i + start_offset for i in range(nr_diamonds)]
    rowStartList = [i-pDiamondsize for i in rowEndList] 
    columnStartList = [i+1 for i in rowEndList]
    columnEndList = [i+pDiamondsize for i in columnStartList]
    return rowStartList, rowEndList, columnStartList, columnEndList

def plotInsulationScore(pScoreArray, pFilename, pTitle=None, pStartbin=None, pBinsize=None):
    if len(pScoreArray.shape) != 1:
        return
    if not isinstance(pFilename, str):
        return
    x_values = np.arange(pScoreArray.shape[0])
    if isinstance(pStartbin, int) and isinstance(pBinsize, int):
        x_values += pStartbin
        x_values *= pBinsize
    fig1, ax1 = plt.subplots()
    ax1.plot(x_values, pScoreArray)
    ax1.set_xlabel("genomic position")
    ax1.set_ylabel("insulation score")
    ax1.grid(True)
    if pTitle is not None:
        ax1.set_title(str(pTitle))
    fig1.savefig(pFilename)
