import utils
import records
import os
import numpy as np
from Bio import SeqIO
from tensorflow import dtypes as tfdtypes
from scipy.sparse import save_npz, csr_matrix
from tqdm import tqdm
import pandas as pd

class DataContainer():
    def __init__(self, chromosome, matrixfilepath, chromatinFolder, sequencefilepath, binsize=None):
        self.chromosome = str(chromosome)
        self.matrixfilepath = matrixfilepath
        self.chromatinFolder = chromatinFolder
        self.sequencefilepath = sequencefilepath
        self.FactorDataArray = None
        self.nr_factors = None
        self.sparseHiCMatrix = None
        self.sequenceArray = None
        self.binsize = None
        if matrixfilepath is None: #otherwise it will be defined by the Hi-C matrix itself upon loading
            self.binsize = binsize
        self.factorNames = None
        self.prefixDict_factors = None
        self.prefixDict_matrix = None
        self.prefixDict_sequence = None
        self.chromSize_factors = None
        self.chromSize_matrix = None
        self.chromSize_sequence = None
        self.sequenceSymbols = None
        self.storedFeatures = None
        self.storedFiles = None
        self.windowsize = None
        self.flankingsize = None
        self.maxdist = None
        self.data_loaded = False

    def __loadFactorData(self, ignoreChromLengths=False, scaleFeatures=False, clampFeatures=False):
        #load chromatin factor data from bigwig files
        if self.chromatinFolder is None:
            return
        #ensure that binsizes for matrix (if given) and factors match
        if self.binsize is None:
            msg = "No binsize given; use a Hi-C matrix or explicitly specify binsize for the container"   
            raise TypeError(msg)
        ###load data for a specific chromsome
        #get the names of the bigwigfiles
        bigwigFileList = utils.getBigwigFileList(self.chromatinFolder)
        bigwigFileList = sorted(bigwigFileList)
        if len(bigwigFileList) is None:
            msg = "Warning: folder {:s} does not contain any bigwig files"
            msg = msg.format(self.chromatinFolder)
            print(msg)
            return
        #check the chromosome name prefixes (e.g. "" or "chr") and sizes
        chromSizeList = []
        prefixDict_factors = dict()
        for bigwigFile in bigwigFileList:
            try:
                prefixDict_factors[bigwigFile] = utils.getChromPrefixBigwig(bigwigFile)
                chromname = prefixDict_factors[bigwigFile] + self.chromosome
                chromSizeList.append( utils.getChromSizesFromBigwig(bigwigFile)[chromname] )
            except Exception as e:
                msg = str(e) + "\n"
                msg += "Could not load data from bigwigfile {}".format(bigwigFile) 
                raise IOError(msg)
        #the chromosome lengths should be equal in all bigwig files
        if len(set(chromSizeList)) != 1 and not ignoreChromLengths:
            msg = "Invalid data. Chromosome lengths differ in bigwig files:"
            for i, filename in enumerate(bigwigFileList):
                msg += "{:s}: {:d}\n".format(filename, chromSizeList[i])
            raise IOError(msg)
        elif len(set(chromSizeList)) != 1 and ignoreChromLengths:
            chromSize_factors = min(chromSizeList)
        else:
            chromSize_factors = chromSizeList[0]
        #the chromosome lengths of matrices and bigwig files must be equal
        if self.chromSize_matrix is not None \
                and self.chromSize_matrix != chromSize_factors:
            msg = "Chrom lengths not equal between matrix and bigwig files\n"
            msg += "Matrix: {:d} -- Factors: {:d}".format(self.chromSize_matrix, chromSize_factors)
            raise IOError(msg)
        if self.chromSize_sequence is not None \
                and self.chromSize_sequence != chromSize_factors:
            msg = "Chrom lengths not equal between sequence and bigwig files\n"
            msg += "Sequence: {:d} -- Factors: {:d}".format(self.chromSize_sequence, chromSize_factors)
            raise IOError(msg)
        #load the data into memory now
        self.factorNames = [os.path.splitext(os.path.basename(name))[0] for name in bigwigFileList]
        self.nr_factors = len(self.factorNames)
        self.prefixDict_factors = prefixDict_factors
        self.chromSize_factors = chromSize_factors
        nr_bins = int( np.ceil(self.chromSize_factors / self.binsize) )
        self.FactorDataArray = np.empty(shape=(len(bigwigFileList),nr_bins))
        msg = "Loaded {:d} chromatin features from folder {:s}\n"
        msg = msg.format(self.nr_factors, self.chromatinFolder)
        featLoadedMsgList = [] #pretty printing for features loaded
        for i, bigwigFile in enumerate(bigwigFileList):
            chromname = self.prefixDict_factors[bigwigFile] + self.chromosome
            tmpArray = utils.binChromatinFactor(pBigwigFileName=bigwigFile,
                                                pBinSizeInt=self.binsize,
                                                pChromStr=chromname,
                                                pChromSize=self.chromSize_factors)
            if clampFeatures:
                tmpArray = utils.clampArray(tmpArray)
            if scaleFeatures:
                tmpArray = utils.scaleArray(tmpArray)
            self.FactorDataArray[i] = tmpArray
            nr_nonzero_abs = np.count_nonzero(tmpArray)
            nr_nonzero_perc = nr_nonzero_abs / tmpArray.size * 100
            msg2 = "{:s} - min. {:.3f} - max. {:.3f} - nnz. {:d} ({:.2f}%)"
            msg2 = msg2.format(bigwigFile, tmpArray.min(), tmpArray.max(), nr_nonzero_abs, nr_nonzero_perc)
            featLoadedMsgList.append(msg2)
        self.FactorDataArray = np.transpose(self.FactorDataArray)
        print(msg + "\n".join(featLoadedMsgList))
            
    def __loadMatrixData(self, scaleMatrix=False):
        #load Hi-C matrix from cooler file
        if self.matrixfilepath is None:
            return
        try:
            prefixDict_matrix = {self.matrixfilepath: utils.getChromPrefixCooler(self.matrixfilepath)}
            chromname = prefixDict_matrix[self.matrixfilepath] + self.chromosome
            chromsize_matrix = utils.getChromSizesFromCooler(self.matrixfilepath)[chromname]
            sparseHiCMatrix, binsize = utils.getMatrixFromCooler(self.matrixfilepath, chromname)
        except:
            msg = "Error: Could not load data from Hi-C matrix {:s}"
            msg = msg.format(self.matrixfilepath)
            raise IOError(msg)
        #scale to 0..1, if requested
        if scaleMatrix:
            sparseHiCMatrix = utils.scaleArray(sparseHiCMatrix)       
        #ensure that chrom sizes for matrix and factors are the same
        if self.chromSize_factors is not None and self.chromSize_factors != chromsize_matrix:
            msg = "Chromsize of matrix does not match bigwig files\n"
            msg += "Matrix: {:d} -- Bigwig files: {:d}"
            msg = msg.format(chromsize_matrix, self.chromSize_factors)
            raise IOError(msg)
        self.chromSize_matrix = chromsize_matrix
        #ensure that binsizes for matrix and factors (if given) match
        if self.binsize is None or self.binsize == binsize:
            self.binsize = binsize
            self.sparseHiCMatrix = sparseHiCMatrix
        elif self.binsize is not None and self.binsize != binsize:
            msg = "Matrix has wrong binsize\n"
            msg += "Matrix: {:d} -- Binned chromatin factors {:d}"
            msg = msg.format(binsize, self.binsize)
            raise IOError(msg)
        msg = "Loaded cooler matrix {:s}\n".format(self.matrixfilepath)
        msg += "chr. {:s}, matshape {:d}*{:d} -- min. {:d} -- max. {:d} -- nnz. {:d}"
        msg = msg.format(self.chromosome, self.sparseHiCMatrix.shape[0], self.sparseHiCMatrix.shape[1], int(self.sparseHiCMatrix.min()), int(self.sparseHiCMatrix.max()), self.sparseHiCMatrix.getnnz() )
        print(msg)

    def __loadSequenceData(self):
        #load DNA sequence from Fasta file
        if self.sequencefilepath is None:
            return
        try:
            records = SeqIO.index(self.sequencefilepath, format="fasta")
        except Exception as e:
            print(e)
            msg = "Could not read sequence file {:s}. Wrong format?"
            msg = msg.format(self.sequencefilepath)
            raise IOError(msg)
        seqIdList = list(records)
        chromname = ""
        if "chr" + self.chromosome in seqIdList:
            self.prefixDict_sequence = {self.sequencefilepath: "chr"}
            chromname = "chr" + self.chromosome
        elif self.chromosome in seqIdList:
            self.prefixDict_sequence = {self.sequencefilepath: ""}
            chromname = self.chromosome
        else:
            msg = "Chromsome {:s} is missing in sequence file {:s}"
            msg = msg.format(self.chromosome, self.sequencefilepath)
            raise IOError(msg)
        #length check
        chromLengthSequence = len(records[ chromname ])
        if self.chromSize_factors is not None and self.chromSize_factors != chromLengthSequence:
            msg = "Chromosome {:s} in sequence file {:s} has wrong length\n"
            msg += "Chrom. factors: {:d} - Sequence File {:d}"    
            msg = msg.format(chromname, self.sequencefilepath, self.chromSize_factors, chromLengthSequence)
            raise RuntimeError(msg)
        elif self.chromSize_matrix is not None and self.chromSize_matrix != chromLengthSequence:
            msg = "Chromosome {:s} in sequence file {:s} has wrong length\n"
            msg += "Matrix: {:d} - Sequence File {:d}"    
            msg = msg.format(chromname, self.sequencefilepath, self.chromSize_matrix, chromLengthSequence)
            raise RuntimeError(msg)    
        else:
            self.chromSize_sequence = chromLengthSequence
        #load the data    
        seqStr = records[chromname].seq.upper()
        self.sequenceSymbols = set(list(seqStr))
        originalLength = len(seqStr)
        self.sequenceArray = utils.fillEncodedSequence(utils.encodeSequence(seqStr, self.sequenceSymbols), self.binsize)
        del seqStr
        records.close()
        msg = "Loaded DNA sequence from {:s} -- Len {:d} -- Symbols {:s}"
        msg = msg.format(self.sequencefilepath, originalLength, ", ".join(list(self.sequenceSymbols)))
        print(msg)
    
    def __unloadFactorData(self):
        #unload chromatin factor data to save memory, but do not touch metadata 
        self.FactorDataArray = None
        
    def __unloadMatrixData(self):
        #unload matrix data to save memory, but do not touch metadata
        self.sparseHiCMatrix = None

    def __unloadSequenceData(self):
        #unload the DNA sequence to save memory, but do not touch metadata
        self.sequenceArray = None

    def unloadData(self):
        #unload all data to save memory, but do not touch metadata
        self.__unloadFactorData
        self.__unloadMatrixData
        self.__unloadSequenceData
        self.windowsize = None
        self.flankingsize = None
        self.maxdist = None
        self.data_loaded = False
    
    def loadData(self, windowsize, flankingsize=None, maxdist=None, scaleFeatures=False, clampFeatures=False, scaleTargets=False):
        if not isinstance(windowsize, int):
            msg = "windowsize must be integer"
            raise TypeError(msg)
        if isinstance(maxdist, int):
            maxdist = np.clip(maxdist, a_min=1, a_max=self.windowsize)
        self.__loadMatrixData(scaleMatrix=scaleTargets)
        self.__loadFactorData(scaleFeatures=scaleFeatures, clampFeatures=clampFeatures)
        self.__loadSequenceData()
        self.windowsize = windowsize
        self.flankingsize = flankingsize
        self.maxdist = maxdist
        self.data_loaded = True

    def checkCompatibility(self, containerIterable):
        ret = []
        try:
           for container in containerIterable:
               ret.append(self.__checkCompatibility(container))
        except:
            ret = [self.__checkCompatibility(containerIterable)]
        return np.all(ret)
        
    def __checkCompatibility(self, container):
        if not isinstance(container, DataContainer):
            return False
        if not self.data_loaded or not container.data_loaded:
            return False
        #check if the same kind of data is available for all containers
        factorsOK = type(self.FactorDataArray) == type(container.FactorDataArray)
        matrixOK = type(self.sparseHiCMatrix) == type(container.sparseHiCMatrix)
        sequenceOK = type(self.sequenceArray) == type(container.sequenceArray)
        #check if windowsize, flankingsize and maxdist match
        windowsizeOK = self.windowsize == container.windowsize
        flankingsizeOK = self.flankingsize == container.flankingsize
        maxdistOK = self.maxdist == container.maxdist
        #sanity check loading of bigwig files
        if self.chromatinFolder is not None and self.nr_factors is None:
            return False
        if container.chromatinFolder is not None and container.nr_factors is None:
            return False
        #if chromatin factors are present, the numbers and names of chromatin factors must match
        factorsOK = factorsOK and (self.nr_factors == container.nr_factors)
        factorsOK = factorsOK and (self.factorNames == container.factorNames)
        #sanity check loading of DNA sequences
        if self.sequencefilepath is not None and self.sequenceSymbols is None:
            return False
        if container.sequencefilepath is not None and container.sequenceSymbols is None:
            return False
        #if DNA sequences are present, the number of symbols must match
        sequenceOK = sequenceOK and (self.sequenceSymbols == container.sequenceSymbols)
        #if DNA sequences are present, the binsizes must match
        #because the input shape of the network depends on it
        if self.sequencefilepath is not None:
            sequenceOK = sequenceOK and (self.binsize == container.binsize)
        return factorsOK and matrixOK and sequenceOK and windowsizeOK and flankingsizeOK and maxdistOK
        
    def writeTFRecord(self, pOutfolder, pRecordSize=None):
        '''
        Write a dataset to disk in tensorflow TFRecord format
        
        Parameters:
            pWindowsize (int): size of submatrices
            pOutfolder (str): directory where TFRecords will be written
            pFlankingsize (int): size of flanking regions left/right of submatrices
            pMaxdist (int): cut the matrices off at this distance (in bins)
            pRecordsize (int): split the TFRecords into multiple files containing approximately this number of samples
        
        Returns:
            list of filenames written
        '''

        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to write"
            print(msg)
            return None
        nr_samples = self.getNumberSamples()
        #adjust record size (yields smaller files and reduces memory load)
        recordsize = nr_samples
        if pRecordSize is not None and pRecordSize < recordsize:
            recordsize = pRecordSize
        #compute number of record files, number of samples 
        #in each file and corresponding indices
        nr_files = int( np.ceil(nr_samples/recordsize) )
        target_ct = int( np.floor(nr_samples/nr_files) )
        samples_per_file = [target_ct]*(nr_files-1) + [nr_samples-(nr_files-1)*target_ct]
        sample_indices = [sum(samples_per_file[0:i]) for i in range(len(samples_per_file)+1)] 
        #write the single files
        folderName = self.chromatinFolder.rstrip("/").replace("/","-")
        recordfiles = [os.path.join(pOutfolder, "{:s}_{:s}_{:03d}.tfrecord".format(folderName, str(self.chromosome), i + 1)) for i in range(nr_files)]
        for recordfile, firstIndex, lastIndex in tqdm(zip(recordfiles, sample_indices, sample_indices[1:]), desc="Storing TFRecord files", total=len(recordfiles)):
            recordDict, storedFeaturesDict = self.__prepareWriteoutDict(pFirstIndex=firstIndex, 
                                                                        pLastIndex=lastIndex, 
                                                                        pOutfolder=pOutfolder)
            records.writeTFRecord(pFilename=recordfile, pRecordDict=recordDict)
        self.storedFiles = recordfiles
        self.storedFeatures = storedFeaturesDict
        return recordfiles

    def getNumberSamples(self):
        if not self.data_loaded:
            return None
        featureArrays = [self.FactorDataArray, self.sparseHiCMatrix, self.sequenceArray]
        cutouts = [self.windowsize+2*self.flankingsize, self.windowsize+2*self.flankingsize, (self.windowsize+2*self.flankingsize)*self.binsize]
        nr_samples_list = []
        for featureArray, cutout in zip(featureArrays, cutouts):
            if featureArray is not None:
                nr_samples_list.append(featureArray.shape[0] - cutout + 1)
            else:
                nr_samples_list.append(0)
        #check if all features have the same number of samples
        if len(set( [x for x in nr_samples_list if x>0] )) != 1:
            msg = "Error: sample binning / DNA sequence encoding went wrong"
            raise RuntimeError(msg)
        return max(nr_samples_list)

    def __getMatrixData(self, idx):
        if self.matrixfilepath is None:
            return None # this can't work
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        #the 0-th matrix starts flankingsize away from the boundary
        windowsize = self.windowsize
        flankingsize = self.flankingsize
        if flankingsize is None:
            flankingsize = windowsize
        startInd = idx + flankingsize
        stopInd = startInd + windowsize
        trainmatrix = None
        if isinstance(self.maxdist, int) and self.maxdist < windowsize and self.maxdist > 0: #trapezoids, i.e. distance limited submatrices
            trainmatrix = self.sparseHiCMatrix[startInd:stopInd,startInd:stopInd].todense()[np.mask_indices(windowsize, utils.maskFunc, self.maxdist)]
        else: #triangles, i. e. full submatrices
            trainmatrix = self.sparseHiCMatrix[startInd:stopInd,startInd:stopInd].todense()[np.triu_indices(windowsize)]
        trainmatrix = np.nan_to_num(trainmatrix)
        return trainmatrix
    
    def __getSequenceData(self, idx):
        if self.sequencefilepath is None:
            return None
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        windowsize = self.windowsize
        flankingsize = self.flankingsize
        if flankingsize is None:
            flankingsize = windowsize
        startInd = idx + flankingsize * self.binsize
        stopInd = startInd + windowsize * self.binsize
        seqArray = self.sequenceArray[startInd:stopInd,:]
        return seqArray

    def __getFactorData(self, idx):
        if self.chromatinFolder is None:
            return None
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        #the 0-th feature matrix starts at position 0
        windowsize = self.windowsize
        flankingsize = self.flankingsize
        if flankingsize is None:
            flankingsize = windowsize
        startInd = idx
        stopInd = startInd + 2*flankingsize + windowsize
        factorArray = self.FactorDataArray[startInd:stopInd,:]
        return factorArray

    def getSampleData(self, idx):
        if not self.data_loaded:
            return None
        factorArray = self.__getFactorData(idx)
        matrixArray = self.__getMatrixData(idx)
        sequenceArray = self.__getSequenceData(idx)
        return {"factorData": factorArray, 
                "out_matrixData": matrixArray, 
                "sequenceData": sequenceArray}
        
    def plotFeatureAtIndex(self, idx, outpath, figuretype="png"):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to plot"
            print(msg)
            return
        if isinstance(idx, int) and (idx >= self.FactorDataArray.shape[0] or idx < 0):
            msg = "Error: Invalid index {:d}; must be None or integer in 0..{:d}".format(idx, self.FactorDataArray.shape[0]-1)
            raise ValueError(msg)
        if isinstance(idx, int):
            factorArray = self.__getFactorData(idx)
            startBin = idx
        else:
            factorArray = self.FactorDataArray 
            startBin = None
        for plotType in ["box", "line"]:   
            utils.plotChromatinFactors(pFactorArray=factorArray, 
                                        pFeatureNameList=self.factorNames,
                                        pChromatinFolder=self.chromatinFolder,
                                        pChrom=self.chromosome,
                                        pBinsize=self.binsize,
                                        pStartbin=startBin,
                                        pOutputPath=outpath,
                                        pPlotType=plotType,
                                        pFigureType=figuretype)
    
    def plotFeaturesAtPosition(self, position, outpath, figuretype="png"):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to plot"
            print(msg)
            return
        if isinstance(position, int) and position > self.chromSize_factors:
            msg = "Error: Invalid position {:d}; must be in 0..{:d}"
            msg = msg.format(position, self.chromSize_factors)
            raise ValueError(msg)
        #compute the bin index from the position
        elif isinstance(position, int):
            idx = int(np.floor(position / self.binsize))
        else:
            idx = None
        return self.plotFeatureAtIndex(idx=idx,
                                        outpath=outpath,
                                        figuretype=figuretype)

    def saveMatrix(self, outputpath, index=None):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to save"
            print(msg)
            return
        sparseMatrix = None
        windowsize = self.windowsize
        flankingsize = self.flankingsize
        if not isinstance(flankingsize, int):
            flankingsize = windowsize
        if isinstance(self.maxdist, int) and self.maxdist < windowsize and self.maxdist > 0:
            maxdist = self.maxdist
        else:
            maxdist = windowsize
        if isinstance(index, int) and index < self.getNumberSamples():
            tmpMat = np.zeros(shape=(windowsize, windowsize))
            indices = np.mask_indices(windowsize, utils.maskFunc, k=maxdist)
            tmpMat[indices] = self.__getMatrixData(idx=index)
            sparseMatrix = csr_matrix(tmpMat)
        else:
            sparseMatrix = self.sparseHiCMatrix
        folderName = self.chromatinFolder.rstrip("/").replace("/","-")
        filename = "matrix_{:s}_chr{:s}_{:s}".format(folderName, str(self.chromosome), str(index))
        filename = os.path.join(outputpath, filename)
        save_npz(file=filename, matrix=sparseMatrix)

    def __prepareWriteoutDict(self, pFirstIndex, pLastIndex, pOutfolder):
        if not self.data_loaded:
            msg = "Error: no data loaded, nothing to prepare"
            raise RuntimeError(msg)
        data = [ self.getSampleData(idx=i) for i in range(pFirstIndex, pLastIndex) ]
        recordDict = dict()
        storedFeaturesDict = dict()
        if len(data) < 1:
            msg = "Error: No data to write"
            raise RuntimeError(msg)
        for key in data[0]:
            featData = [feature[key] for feature in data]
            if not any(elem is None for elem in featData):
                recordDict[key] = np.array(featData)
                storedFeaturesDict[key] = {"shape": recordDict[key].shape[1:], "dtype": tfdtypes.as_dtype(recordDict[key].dtype)}
        return recordDict, storedFeaturesDict


class DataContainerWithScores(DataContainer):
    def __init__(self, chromosome, matrixfilepath, chromatinFolder, sequencefilepath, binsize=None):
        if not isinstance(matrixfilepath, str):
            msg = "This container only works when a (path to a) cooler matrix is given"
            raise TypeError(msg)
        super(DataContainerWithScores, self).__init__(chromosome, matrixfilepath, chromatinFolder, sequencefilepath, binsize=None)
        self.scoreArray = None
        self.diamondsize = None
        
    def loadData(self, windowsize, flankingsize=None, maxdist=None, scaleFeatures=False, clampFeatures=False, scaleTargets=False, diamondsize=0):
        super().loadData(windowsize=windowsize, flankingsize=flankingsize, maxdist=maxdist, scaleFeatures=scaleFeatures, clampFeatures=clampFeatures, scaleTargets=scaleTargets)
        self.scoreArray = self.__computeScores(diamondsize=diamondsize)
        self.diamondsize = diamondsize

    def unloadData(self):
        super().unloadData()
        self.scoreArray = None
        self.diamondsize = None

    def __checkCompatibility(self, container):
        compatible = super().__checkCompatibility(container)
        compatible = compatible and ( type(self.scoreArray) == type(container.scoreArray) )
        compatible = compatible and ( self.diamondsize == container.diamondsize )
        return compatible

    def __computeScores(self, diamondsize):
        if self.sparseHiCMatrix is None:
            msg = "Error: cannot compute scores when Hi-C matrix is missing"
            raise RuntimeError(msg)
        if not isinstance(diamondsize, int):
            msg = "Error: size for insulation score computation must be integer"
            raise TypeError(msg)
        if self.sparseHiCMatrix.shape[0] - 2*diamondsize <= 1:
            msg = "Error: Size for insulation score computation is too large for Hi-C matrix"
            raise ValueError(msg)
        rowStartList, rowEndList, columnStartList, columnEndList = utils.getDiamondIndices(pMatsize=self.sparseHiCMatrix.shape[0], pDiamondsize=diamondsize)
        l = [ self.sparseHiCMatrix[i:j,k:l].todense() for i,j,k,l in zip(rowStartList,rowEndList,columnStartList,columnEndList) ]
        return np.array([ np.mean(i) for i in l ]).astype("float32")

    def __getScoreData(self, idx):
        if self.scoreArray is None or self.diamondsize is None or not self.data_loaded:
            msg = "Error: Load Data first"
            raise RuntimeError(msg)
        nr_diamonds = self.windowsize - 2*self.diamondsize
        flankingsize = self.flankingsize
        if not isinstance(flankingsize, int) or flankingsize < 0:
            flankingsize = self.windowsize
        if nr_diamonds <= 1:
            msg = "Error: Size for computing insulation scores is too large / Windowsize too small"
            raise ValueError(msg)
        startInd = idx + flankingsize
        endInd = startInd + nr_diamonds
        return self.scoreArray[startInd:endInd]

    def getSampleData(self, idx):
        if not self.data_loaded:
            msg = "Warning: No data loaded"
            print(msg)
            return None
        retDict = super().getSampleData(idx=idx)
        retDict["out_scoreData"] = self.__getScoreData(idx=idx)
        return retDict

    def plotInsulationScore(self, outpath, figuretype="png", index=None):
        if not self.data_loaded:
            msg = "Warning: No Data loaded, nothing to plot"
            print(msg)
            return
        if self.scoreArray is None or self.diamondsize is None:
            return
        if isinstance(index, int):
            tmpArray = self.__getScoreData(idx=index)
        else:
            tmpArray = self.scoreArray
        matrixName = self.matrixfilepath.lstrip("/").replace("/","-")
        filename = "scores_{:s}_chr{:s}_{:s}.{:s}".format(matrixName, str(self.chromosome), str(index), figuretype)
        filename = os.path.join(outpath, filename)
        titleStr = "Insulation score for\n{:s},\nchr{:s} at ws{:d}, ds{:d}".format(self.matrixfilepath, self.chromosome, self.windowsize, self.diamondsize)
        startbin = self.flankingsize + self.diamondsize
        if isinstance(index, int):
            startbin += index
        utils.plotInsulationScore(pScoreArray=tmpArray, pFilename=filename, pTitle=titleStr, pStartbin=startbin, pBinsize=self.binsize)

    def saveInsulationScoreToBedgraph(self, outpath, index=None):
        if not self.data_loaded:
            msg = "Warning: No Data loaded, nothing to plot"
            print(msg)
            return
        if self.scoreArray is None or self.diamondsize is None:
            return
        if self.windowsize is None or self.flankingsize is None:
            return
        if isinstance(index, int) and index < self.getNumberSamples():
            startbin = index
            tmp_array = self.__getScoreData(index)
            chromsize = self.windowsize * self.binsize
        else:
            startbin = 0
            tmp_array = self.scoreArray
            chromsize = self.chromSize_matrix
        matrixName = self.matrixfilepath.lstrip("/").replace("/","-")
        filename = "scores_{:s}_chr{:s}_ds{:d}_{:s}.bedgraph".format(matrixName, str(self.chromosome), self.diamondsize, str(index))
        filename = os.path.join(outpath, filename)
        utils.saveInsulationScoreToBedgraph(scoreArray=tmp_array,
                                            chromSize_matrix=chromsize,
                                            binsize=self.binsize,
                                            diamondsize=self.diamondsize,
                                            chromosome=self.chromosome,
                                            filename=filename,
                                            startbin=startbin)