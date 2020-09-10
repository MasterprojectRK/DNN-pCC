#!python3
import os
import cooler
import pyBigWig
import numpy as np

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
    return binArray

def normalize1Darray(pArray):
    # min-max normalization (0...1) for 1D arrays 
    if pArray is None or pArray.ndim != 1 or pArray.size == 0:
        msg = "cannot normalize empty array"
        print(msg)
        return pArray
    normArray = (pArray - pArray.min()) / (pArray.max() - pArray.min())
    return normArray