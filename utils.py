import os
import cooler

def getBigwigFileList(pDirectory):
    #returns a list of bigwig files in pDirectory
    retList = []
    for file in os.listdir(pDirectory):
        if file.endswith(".bigwig") or file.endswith("bigWig") or file.endswith(".bw"):
            retList += file
    return retList


def getMatrixFromCooler(pCoolerFilePath, pChromNameStr):
    #returns sparse matrix from cooler file for given chromosome name
    sparseMatrix = None
    try:
        coolerMatrix = cooler.Cooler(pCoolerFilePath)
        sparseMatrix = coolerMatrix.matrix(sparse=True,balance=False).fetch(pChromNameStr)
    except Exception as e:
        print(e)
    return sparseMatrix

