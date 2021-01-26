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
    def __init__(self, chromosome: str, matrixfilepath: str, chromatinFolder: str, sequencefilepath: str, mode: str = "prediction"):
        self.chromosome = chromosome
        self.matrixfilepath = matrixfilepath
        self.chromatinFolder = chromatinFolder
        self.sequencefilepath = sequencefilepath
        self.FactorDataArray = None
        self.nr_factors = None
        self.sparseHiCMatrix = None
        self.sequenceArray = None
        self.feature_binsize = None
        self.matrix_binsize = None
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
        self.mode = mode

    def __loadFactorData(self, featureBinsize, ignoreChromLengths=False, scaleFeatures=False, clampFeatures=False):
        #load chromatin factor data from bigwig files
        if self.chromatinFolder is None:
            return

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
        self.feature_binsize = featureBinsize
        if self.matrix_binsize is not None and self.matrix_binsize % self.feature_binsize != 0:
            msg = "Error: Matrix binsize must be an integer multiple of feature binsize"
            raise ValueError(msg)
        elif self.matrix_binsize is not None and self.matrix_binsize < self.feature_binsize:
            msg = "Error: Matrix binsize must be greater than or equal to feature binsize"
            raise ValueError(msg)
        #if no matrix loaded yet, we can set the binsize the same
        #upon loading matrix, this will be overwritten
        elif self.matrix_binsize is None:
            self.matrix_binsize = featureBinsize
        nr_bins = int( np.ceil(self.chromSize_factors / self.feature_binsize) )
        self.FactorDataArray = np.empty(shape=(len(bigwigFileList),nr_bins))
        msg = "Loaded {:d} chromatin features from folder {:s}\n"
        msg = msg.format(self.nr_factors, self.chromatinFolder)
        featLoadedMsgList = [] #pretty printing for features loaded
        for i, bigwigFile in enumerate(bigwigFileList):
            chromname = self.prefixDict_factors[bigwigFile] + self.chromosome
            tmpArray = utils.binChromatinFactor(pBigwigFileName=bigwigFile,
                                                pBinSizeInt=self.feature_binsize,
                                                pChromStr=chromname,
                                                pChromSize=self.chromSize_factors)
            if clampFeatures:
                tmpArray = utils.clampArray(tmpArray)
            if scaleFeatures:
                pass
                #tmpArray = utils.scaleArray(tmpArray)
            self.FactorDataArray[i] = tmpArray
            nr_nonzero_abs = np.count_nonzero(tmpArray)
            nr_nonzero_perc = nr_nonzero_abs / tmpArray.size * 100
            msg2 = "{:s} - min. {:.3f} - max. {:.3f} - nnz. {:d} ({:.2f}%)"
            msg2 = msg2.format(bigwigFile, tmpArray.min(), tmpArray.max(), nr_nonzero_abs, nr_nonzero_perc)
            featLoadedMsgList.append(msg2)
        self.FactorDataArray = utils.standardizeArray(self.FactorDataArray, axis=1)
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
        self.matrix_binsize = binsize
        self.sparseHiCMatrix = sparseHiCMatrix
        if self.feature_binsize is not None and self.matrix_binsize % self.feature_binsize != 0:
            msg = "Error: matrix binsize must be an integer multiple of factor binsize"
            raise ValueError(msg)
        elif self.feature_binsize is not None and self.matrix_binsize < self.feature_binsize:
            msg = "Error: matrix binsize must be greater than or equal to factor binsize"
            raise ValueError(msg)
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
        self.sequenceArray = utils.fillEncodedSequence(utils.encodeSequence(seqStr, self.sequenceSymbols), self.matrix_binsize)
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
    
    def loadData(self, windowsize, featureBinsize, flankingsize=None, maxdist=None, scaleFeatures=False, clampFeatures=False, scaleTargets=False):
        if not isinstance(windowsize, int):
            msg = "windowsize must be integer"
            raise TypeError(msg)
        if isinstance(maxdist, int):
            maxdist = np.clip(maxdist, a_min=1, a_max=self.windowsize)
        self.__loadMatrixData(scaleMatrix=scaleTargets)
        self.__loadFactorData(featureBinsize=featureBinsize, scaleFeatures=scaleFeatures, clampFeatures=clampFeatures)
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
        if isinstance(self.matrix_binsize, int):
            factorsOK = factorsOK and (self.matrix_binsize//self.feature_binsize == container.matrix_binsize//container.feature_binsize)
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
            sequenceOK = sequenceOK and (self.matrix_binsize == container.matrix_binsize)
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
        recordfiles = [os.path.join(pOutfolder, "{:s}_{:s}_{:d}_{:d}_{:03d}.tfrecord".format(folderName, str(self.chromosome), self.matrix_binsize, self.feature_binsize, i + 1)) for i in range(nr_files)]
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
        binsizeDiffFactor = 1
        if isinstance(self.feature_binsize, int) and isinstance(self.matrix_binsize, int):
            binsizeDiffFactor = self.matrix_binsize // self.feature_binsize
        featureArrays = [self.FactorDataArray, self.sparseHiCMatrix, self.sequenceArray]
        winsizes = [(self.windowsize+2*self.flankingsize)*binsizeDiffFactor, self.windowsize+2*self.flankingsize, (self.windowsize+2*self.flankingsize)*self.matrix_binsize]
        stepsizes = [binsizeDiffFactor, 1, 1]
        nr_samples_list = []
        for featureArray, winsize, stepsize in zip(featureArrays, winsizes, stepsizes):
            if featureArray is not None:
                nr_samples_list.append((featureArray.shape[0] - winsize + stepsize)//stepsize)
            else:
                nr_samples_list.append(0)
        #check if all features have the same number of samples
        if nr_samples_list[0] > 0 and nr_samples_list[1] > 0 and nr_samples_list[0] == nr_samples_list[1]:
            pass
        #if the proteins are binned with different resolution than the matrix, 
        #then the number of samples might differ by one
        #example situations, with matrix binsize "b" = 5x feature binsize "a" and matrix windowsize = 2
        #b1    b2    b3    b4    b5       matrix bins  
        #aaaaa aaaaa aaaaa aaaaa aaaaa    => 25a >= genomesize >= 24a:  same number of samples corresponding to matrix samples (b1,b2), (b2,b3), (b3,b4) (b4,b5)
        #aaaaa aaaaa aaaaa aaaaa aaaa     => 24a > genomesize >= 23a: last sample incomplete, only 2 samples
        #aaaaa aaaaa aaaaa aaaaa aaa      => 23a > genomesize >= 22a: last sample incomplete, again only 2 samples
        #etc. - always one sample less
        elif nr_samples_list[0] > 0 and nr_samples_list[1] > 0 and nr_samples_list[0] == nr_samples_list[1] - 1:
            #for training or validation, it is sufficient to just reduce the number of samples by one 
            #and leave out the last sample
            if self.mode == "training" or self.mode == "validation":
                nr_samples_list[1] -= 1
                msg = "Info: leaving out one sample due to different binsizes matrix / chrom. features"
                print(msg)
            #for prediction, this is not good, as the predicted matrix will then be one binsize smaller than its actual size
            #it is in this case better to fill with zeros
            else:
                msg = "Info: filling last feature bins with zeros (diff. binsizes matrix / chrom. feats)"
                print(msg)
                target_size_bp = self.matrix_binsize * self.sparseHiCMatrix.shape[0]
                actual_feature_size_bp = self.FactorDataArray.shape[0] * self.feature_binsize
                diff_bins = int( np.ceil( (target_size_bp - actual_feature_size_bp) / self.feature_binsize) )
                assert diff_bins < (binsizeDiffFactor)
                zeros = np.zeros((diff_bins, self.FactorDataArray.shape[1]), dtype=self.FactorDataArray.dtype)
                self.FactorDataArray = np.concatenate([self.FactorDataArray, zeros], axis=0 )
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
        trainmatrix = np.array(np.nan_to_num(trainmatrix))[0,:]
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
        startInd = idx + flankingsize * self.matrix_binsize
        stopInd = startInd + windowsize * self.matrix_binsize
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
        resolutionFactor = 1
        if isinstance(self.matrix_binsize, int):
            resolutionFactor = self.matrix_binsize//self.feature_binsize
        startInd = idx * resolutionFactor
        stopInd = startInd + (2*flankingsize + windowsize) * resolutionFactor
        factorArray = self.FactorDataArray[startInd:stopInd,:]
        return factorArray

    def getSampleData(self, idx):
        if not self.data_loaded:
            return None
        factorArray = self.__getFactorData(idx)
        matrixArray = self.__getMatrixData(idx)
        sequenceArray = self.__getSequenceData(idx)
        return {"factorData": factorArray.astype("float32"), 
                "out_matrixData": matrixArray.astype("float32"), 
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
                                        pBinsize=self.feature_binsize,
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
            idx = int(np.floor(position / self.feature_binsize))
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