#!python3
import os
import cooler
import pyBigWig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm

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

def scale1Darray(pArray):
    # min-max scaling (0...1) for 1D arrays 
    if pArray is None or pArray.ndim != 1 or pArray.size == 0:
        msg = "cannot normalize empty array"
        print(msg)
        return pArray
    normArray = (pArray - pArray.min()) / (pArray.max() - pArray.min())
    return normArray


def showMatrix(pMatrix):
    #test function to plot matrices
    #not for production use
    print(pMatrix.max())
    plotmatrix = pMatrix + 1
    plt.matshow(plotmatrix, cmap="Reds", norm=colors.LogNorm())
    plt.show()

def plotLoss(pKerasHistoryObject):
    plt.plot(pKerasHistoryObject.history['loss'])
    plt.plot(pKerasHistoryObject.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def rebuildMatrix(pArrayOfTriangles, pWindowSize):
    #rebuilds the interaction matrix (a trapezoid along its diagonal)
    #by taking the mean of all overlapping triangles
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
