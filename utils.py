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
import tensorflow.keras

def getBigwigFileList(pDirectory):
    #returns a list of bigwig files in pDirectory
    retList = []
    for file in sorted(os.listdir(pDirectory)):
        if file.endswith(".bigwig") or file.endswith("bigWig") or file.endswith(".bw"):
            retList.append(pDirectory + file)
    return retList


def getMatrixFromCooler(pCoolerFilePath, pChromNameStr):
    #returns sparse matrix from cooler file for given chromosome name
    sparseMatrix = None
    binSizeInt = 0
    try:
        coolerMatrix = cooler.Cooler(pCoolerFilePath)
        sparseMatrix = coolerMatrix.matrix(sparse=True,balance=False).fetch(pChromNameStr)
        binSizeInt = coolerMatrix.binsize
    except Exception as e:
        print(e)
    sparseMatrix = sparseMatrix.tocsr() #so it can be sliced later
    return sparseMatrix, binSizeInt

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
            msg_inf = "Warning: replaced {:d} infinity values in chromatin factor data by 0/max. numeric value in data"
            msg_inf = msg_inf.format(nr_inf)
            print(msg_inf)
        if nr_nan != 0:
            msg_nan = "Warning: replaced {:d} NANs in chromatin factor data by 0."
            msg_nan = msg_nan.format(nr_nan)
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

def buildVectorArray(pSparseCsrMatrix, pWindowSize_bins):
    #extract target values in form of vectors (i.e. anti-diagonals of submatrices) from the sparse matrix
    #this concept is used e.g. in DeepC by Schwessinger et al.
    #note that the antidiagonal is a "zig-zag" between the actual anti-diagonal and the first side anti-diagonal
    nr_vectors = int(pSparseCsrMatrix.shape[0] - 3*pWindowSize_bins + 1)
    vectorArray = np.empty(shape=(nr_vectors,pWindowSize_bins))
    for i in tqdm(range(nr_vectors), desc="composing target vector array"):
        j = i + pWindowSize_bins
        k = j + pWindowSize_bins
        #get the two first anti-diagonals of the submatrix by taking the two first diagonals of the flipped submatrix
        flippedTriuMat = np.flipud(pSparseCsrMatrix[j:k,j:k].todense())
        upperSideIndex = int(np.floor(pWindowSize_bins/2))
        #actual anti-diagonal, only upper triangular part needs be kept
        vectorFront = np.diagonal(flippedTriuMat)[upperSideIndex:]
        #1st side anti-diagonal, only upper triangluar part needs be kept
        if pWindowSize_bins % 2 == 0:
            vectorRear = np.diagonal(flippedTriuMat,-1)[upperSideIndex-1:]
        else:
            vectorRear = np.diagonal(flippedTriuMat,-1)[upperSideIndex:]
        #stack the values into one single target vector
        vectorArray[i] = np.hstack((vectorFront, vectorRear))
    return vectorArray

def rebuildMatrixFromVector(pVectorArray):
    #rebuild the target matrix from predicted "anti-diagonal" vectors
    #i.e. the actual predictions resemble the upper triangluar part of anti-diagonals of the predicted matrix
    #put them together to get a matrix
    windowsize = pVectorArray.shape[1]
    offset = windowsize
    targetSize = pVectorArray.shape[0] + 2*offset + windowsize - 1
    targetMatrix = np.zeros((targetSize,targetSize),dtype="float32")
    sumMat = np.zeros((windowsize,windowsize),dtype="float32")
    for i in tqdm(range(pVectorArray.shape[0]), 
                        desc="rebuilding target matrix from predicted vectors"):
        #the predicted vectors correspond to upper-triangluar parts of anti-diagonals in the target matrix
        #note that the "anti-diagonal" here is composed of the actual anti-diagonal and 
        #the first side anti-diagonal, concatenated into the target vector.
        #the actual anti-diagonal comes first (:splitIndex), then the side anti-diag (splitIndex:)
        firstAntiDiag = np.empty(shape=(windowsize,))
        secondAntiDiag = np.empty(shape=(windowsize-1,))
        splitIndex = int(np.ceil(windowsize/2))#index vector where to split between anti-diag and first side anti-diag
        diagIndex = int(np.floor(windowsize/2))#index vector where to put the upper-triangular parts
        firstAntiDiag[diagIndex:] = pVectorArray[i][:splitIndex]
        if windowsize % 2 == 0:
            secondAntiDiag[diagIndex-1:] = pVectorArray[i][splitIndex:]
        else:
            secondAntiDiag[diagIndex:] = pVectorArray[i][splitIndex:]
        #get indices of first lower side diagonal
        diagIndices2 = (np.diag_indices_from(sumMat)[0][1:], np.diag_indices_from(sumMat)[1][:-1])
        sumMat[np.diag_indices_from(sumMat)] = firstAntiDiag
        sumMat[diagIndices2] = secondAntiDiag
        #set target sub-matrix
        #note the flip, because the anti-diagonals are stored as diagonals in sumMat
        j = i + offset
        k = j + windowsize
        targetMatrix[j:k,j:k] += np.flipud(sumMat)
    return targetMatrix


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
    return binnedChromatinFactorArray

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


##the following class is an adapted version from a tutorial at Stanford University
##https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class multiInputGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, vectorArray, chromatinFactorArray, indices, batchsize, windowsize, shuffle=True):
        self.vectorArray = vectorArray
        self.chromatinFactorArray = chromatinFactorArray
        self.indices = indices
        self.batchsize = batchsize
        self.windowsize = windowsize
        self.shuffle = shuffle
        if self.vectorArray is None:
            self.shuffle = False
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices)  / self.batchsize))

    def __getitem__(self, index):
        indices = self.indices[index*self.batchsize : (index+1)*self.batchsize]
        return self.__generateData(indices)

    def __generateData(self, indices):
        factorArray = np.empty((len(indices), self.chromatinFactorArray.shape[0], 3*self.windowsize, 1))
        vectorArray = np.empty((len(indices), self.windowsize))
        for b,i in enumerate(indices):
            if self.vectorArray is not None:
                vectorArray[b] = np.nan_to_num(self.vectorArray[i])
            #the chromatin factors have no offset
            factorMat = self.chromatinFactorArray[:,i:i+3*self.windowsize]
            factorArray[b] = np.expand_dims(factorMat,2)
        if self.vectorArray is not None:
            return factorArray, vectorArray
        else:
            return factorArray

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices) #in-place permutation
