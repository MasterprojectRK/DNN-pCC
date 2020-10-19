#!python3
import os
import cooler
import pyBigWig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from scipy import sparse
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer

def getBigwigFileList(pDirectory):
    #returns a list of bigwig files in pDirectory
    retList = []
    for file in sorted(os.listdir(pDirectory)):
        if file.endswith(".bigwig") or file.endswith("bigWig") or file.endswith(".bw"):
            retList.append(pDirectory + file)
    return retList

def getChromSizesFromBigwig(pBigwigFileName):
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
    #returns sparse matrix from cooler file for given chromosome name
    sparseMatrix = None
    binSizeInt = 0
    try:
        coolerMatrix = cooler.Cooler(pCoolerFilePath)
        sparseMatrix = coolerMatrix.matrix(sparse=True,balance=False).fetch(pChromNameStr)
        sparseMatrix = sparseMatrix.tocsr() #so it can be sliced later
        binSizeInt = coolerMatrix.binsize
    except Exception as e:
        print(e)
    return sparseMatrix, binSizeInt


def getChromSizesFromCooler(pCoolerFilePath):
    chromSizes = dict()
    try:
        coolerMatrix = cooler.Cooler(pCoolerFilePath) 
        chromSizes = coolerMatrix.chromsizes.to_dict()
    except Exception as e:
        print(e)
    return chromSizes


def binChromatinFactor(pBigwigFileName, pBinSizeInt, pChromStr):
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
        if not chrom.startswith("chr"):
            chrom = "chr" + pChromStr
        #compute signal values (stats) over resolution-sized bins
        chromsize = bigwigFile.chroms(chrom)
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
    # min-max scaling (0...1) for 1D arrays 
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
    #not for production use
    print(pMatrix.max())
    plotmatrix = pMatrix + 1
    plt.matshow(plotmatrix, cmap="Reds", norm=colors.LogNorm())
    plt.show()

def plotMatrix(pMatrix, pFilename, pTitle):
    #test function to plot matrices
    #not for production use
    fig1, ax1 = plt.subplots()
    ax1.matshow(pMatrix, cmap="Reds", norm=colors.LogNorm())
    ax1.set_title(str(pTitle))
    fig1.savefig(pFilename)

def plotLoss(pKerasHistoryObject, pFilename):
    fig1, ax1 = plt.subplots()
    ax1.plot(pKerasHistoryObject.history['loss'])
    ax1.plot(pKerasHistoryObject.history['val_loss'])
    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'val'], loc='upper left')
    fig1.savefig(pFilename)


def rebuildMatrix(pArrayOfTriangles, pWindowSize):
    #rebuilds the interaction matrix (a trapezoid along its diagonal)
    #by taking the mean of all overlapping triangles
    #returns an interaction matrix as a numpy ndarray
    nr_matrices = pArrayOfTriangles.shape[0]
    sum_matrix = np.zeros((nr_matrices-1+3*pWindowSize,nr_matrices-1+3*pWindowSize))
    count_matrix = np.zeros_like(sum_matrix,dtype=int)    
    mean_matrix = np.zeros_like(sum_matrix,dtype="float32")
    #sum up all the triangular matrices, shifting by one along the diag. for each matrix
    for i in tqdm(range(nr_matrices), desc="rebuilding matrix"):
        j = i + pWindowSize
        k = j + pWindowSize
        sum_matrix[j:k,j:k][np.triu_indices(pWindowSize)] += pArrayOfTriangles[i]
        count_matrix[j:k,j:k] += np.ones((pWindowSize,pWindowSize),dtype=int) #keep track of how many matrices have contributed to each position
    mean_matrix[count_matrix!=0] = sum_matrix[count_matrix!=0] / count_matrix[count_matrix!=0]
    return mean_matrix

def buildMatrixArray(pSparseMatrix, pWindowSize_bins):
    #get all possible (overlapping) windowSize x windowSize matrices out of the original one
    #and put them into a numpy array
    #ignore the first windowSize matrices because of the window approach by Farre et al.
    matrixSize_bins = int(1/2 * pWindowSize_bins * (pWindowSize_bins + 1)) #always an integer because even*odd=even
    nr_matrices = int(pSparseMatrix.shape[0] - 3*pWindowSize_bins + 1)
    #nr_matrices = 100
    matrixArray = np.empty(shape=(nr_matrices,matrixSize_bins))
    for i in tqdm(range(nr_matrices),desc="composing matrices"):
        j = i + pWindowSize_bins
        k = j + pWindowSize_bins
        trainmatrix = pSparseMatrix[j:k,j:k].todense()[np.triu_indices(pWindowSize_bins)]
        matrixArray[i] = trainmatrix
    return matrixArray

def writeCooler(pMatrix, pBinSizeInt, pOutfile, pChromosome, pChromSize=None,  pMetadata=None):
    #takes a matrix as numpy array and writes a cooler matrix from it
    #widely copied from study project
    
    #the chromosome size may not be integer-divisible by the bin size
    #so specifying the real chrom size is possible, but the
    #number of bins must still correspond to the matrix size
    chromSizeInt = int(pMatrix.shape[0] * pBinSizeInt)
    if pChromSize is not None \
                and pChromSize > (chromSizeInt - pBinSizeInt)\
                and pChromSize < chromSizeInt:
        chromSizeInt = int(pChromSize)
        
    #create the bins for cooler
    bins = pd.DataFrame(columns=['chrom','start','end'])
    binStartList = list(range(0, chromSizeInt, int(pBinSizeInt)))
    binEndList = list(range(int(pBinSizeInt), chromSizeInt, int(pBinSizeInt)))
    binEndList.append(chromSizeInt)
    bins['start'] = binStartList
    bins['end'] = binEndList
    bins['chrom'] = str(pChromosome)

    #create the pixels for cooler
    triu_Indices = np.triu_indices(pMatrix.shape[0])
    pixels = pd.DataFrame(columns=['bin1_id','bin2_id','count'])
    pixels['bin1_id'] = triu_Indices[0]
    pixels['bin2_id'] = triu_Indices[1]
    readCounts = pMatrix[triu_Indices]
    if sparse.isspmatrix_csr(pMatrix): #for sparse matrices, slicing is different
        readCounts = np.transpose(readCounts)
    pixels['count'] = np.float64(readCounts)
    pixels.sort_values(by=['bin1_id','bin2_id'],inplace=True)

    #write out the cooler
    cooler.create_cooler(pOutfile, bins=bins, pixels=pixels, dtypes={'count': np.float64}, metadata=pMetadata)

def distanceNormalize(pSparseCsrMatrix, pWindowSize_bins):
    #compute the means along the diagonals (= same distance)
    #and divide all values on the diagonals by their respective mean
    diagList = []
    for i in range(pWindowSize_bins):
        diagArr = sparse.csr_matrix.diagonal(pSparseCsrMatrix,i)
        diagList.append(diagArr/diagArr.mean())
    distNormalizedMatrix = sparse.diags(diagList,np.arange(pWindowSize_bins),format="csr")
    return distNormalizedMatrix

def composeChromatinFactors(pBigwigFileList, pChromLength_bins, pBinSizeInt, pChromosomeStr, pPlotFilename=None, pClampArray=True, pScaleArray=True):
    binnedChromatinFactorArray = np.empty(shape=(len(pBigwigFileList),pChromLength_bins))
    ##bin the single proteins
    for i in tqdm(range(len(pBigwigFileList)),desc="binning chromatin factors"):
        binnedFactor = binChromatinFactor(pBigwigFileList[i],pBinSizeInt,pChromosomeStr)
        if pClampArray: #clamping outliers before scaling
            binnedFactor = clampArray(binnedFactor)
        if pScaleArray:
            binnedFactor = scaleArray(binnedFactor)
        binnedChromatinFactorArray[i] = binnedFactor
    #print boxplots, if requested
    if pPlotFilename is not None:
        plotChromatinFactorStats(binnedChromatinFactorArray, pFilename=pPlotFilename)
    #return the transpose
    return np.transpose(binnedChromatinFactorArray)

def plotChromatinFactorStats(pChromFactorArray, pFilename):
    #store box plots of the chromatin factors in the array
    fig1, ax1 = plt.subplots()
    toPlotList = []
    for i in range(pChromFactorArray.shape[0]):
        toPlotList.append(pChromFactorArray[i].flatten())
    ax1.boxplot(toPlotList)
    ax1.set_title("Chromatin factor boxplots")
    ax1.set_xlabel("Chromatin factor number")
    ax1.set_ylabel("Chromatin factor signal value")
    fig1.savefig(pFilename)

def clampArray(pArray):
    #clamp all values array to be within 
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


def readSequences(pDNAFastaFileStr):
    sequenceStr = ""
    try:
        with open(pDNAFastaFileStr) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequenceStr += str(record.seq.upper())
    except Exception as e:
        msg = "Could not read fasta file {:s}\n"
        msg += str(e)
        msg = msg.format(pDNAFastaFileStr)
        print(msg)
    if len(sequenceStr) > 0:
        msg = "Successfully read sequence of total length {:d}\n"
        msg += "from file {:s}."
        msg = msg.format(len(sequenceStr), pDNAFastaFileStr)
        print(msg)
    return sequenceStr

def encodeSequence(pSequenceStr):
    if pSequenceStr is None or pSequenceStr == "":
        msg = "Aborting. DNA sequence is empty"
        raise SystemExit(msg)
    mlb = MultiLabelBinarizer()
    encodedSequenceArray = mlb.fit_transform(pSequenceStr).astype("uint8")
    if encodedSequenceArray.shape[1] != 4:
        msg = "Warning: DNA sequence contains more than the 4 nucleotide symbols A,C,G,T\n"
        msg += "Check your input sequence, if this is not intended."
        print(msg)
        print("Contained symbols:", ", ".join(mlb.classes_))
    return encodedSequenceArray

def fillEncodedSequence(pEncodedSequenceArray, pBinSizeInt):
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

def buildSequenceArray(pDNAFastaFileStr, pBinSizeInt):
    sequenceStr = readSequences(pDNAFastaFileStr)
    sequenceArr = encodeSequence(sequenceStr)
    sequenceArr = fillEncodedSequence(sequenceArr,pBinSizeInt)
    return sequenceArr


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
                binnedFactor = binChromatinFactor(fFolder+bigwigFile, binsize, chromName_bigwig)
                if pScaleFactors:
                    binnedFactor = scaleArray(binnedFactor)
                if pClampFactors:
                    binnedFactor = clampArray(binnedFactor)
                binnedChromFactorArray[i] = binnedFactor
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


def plotChromatinFactors(pChromSequenceArray, pBinSize, pChrom, pFolder, pFilename):
    #plot chromatin factors
    #for debugging purposes only, not for production use
    winsize = pChromSequenceArray.shape[0]
    nr_subplots = pChromSequenceArray.shape[1]
    x_axis_values = np.arange(winsize) * pBinSize
    fig1, axs1 = plt.subplots(nr_subplots, 1, sharex = True)
    for i in range(nr_subplots):
        axs1[i].plot(x_axis_values, pChromSequenceArray[:,i])
        axs1[i].grid(True)
    axs1[0].set_ylabel("signal val.")    
    axs1[0].set_xlabel("chromosome" + str(pChrom))
    axs1[0].set_title("chrom. factors from " + str(pFolder) + " chrom " + str(pChrom))
    fig1.savefig(pFilename)
