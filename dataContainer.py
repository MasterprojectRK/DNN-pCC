import utils
import records
import os
import numpy as np
from Bio import SeqIO
from tensorflow import dtypes as tfdtypes
from scipy.sparse import save_npz, csr_matrix

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

    def __loadFactorData(self, ignoreChromLengths=False, scaleFeatures=False, clampFeatures=False):
        #load chromatin factor data from bigwig files
        if self.chromatinFolder is None:
            return
        #ensure that binsizes for matrix (if given) and factors match
        if self.binsize is None:
            msg = "No binsize given; use a Hi-C matrix or explicitly specify binsize for the container"   
            raise ValueError(msg)
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
                raise ValueError(msg)
        #the chromosome lengths should be equal in all bigwig files
        if len(set(chromSizeList)) != 1 and not ignoreChromLengths:
            msg = "Invalid data. Chromosome lengths differ in bigwig files:"
            for i, filename in enumerate(bigwigFileList):
                msg += "{:s}: {:d}\n".format(filename, chromSizeList[i])
            raise ValueError(msg)
        elif len(set(chromSizeList)) != 1 and ignoreChromLengths:
            chromSize_factors = min(chromSizeList)
        else:
            chromSize_factors = chromSizeList[0]
        #the chromosome lengths of matrices and bigwig files must be equal
        if self.chromSize_matrix is not None \
                and self.chromSize_matrix != chromSize_factors:
            msg = "Chrom lengths not equal between matrix and bigwig files\n"
            msg += "Matrix: {:d} -- Factors: {:d}".format(self.chromSize_matrix, chromSize_factors)
            raise ValueError(msg)
        if self.chromSize_sequence is not None \
                and self.chromSize_sequence != chromSize_factors:
            msg = "Chrom lengths not equal between sequence and bigwig files\n"
            msg += "Sequence: {:d} -- Factors: {:d}".format(self.chromSize_sequence, chromSize_factors)
            raise ValueError(msg)
        #load the data into memory now
        self.factorNames = [os.path.splitext(os.path.basename(name))[0] for name in bigwigFileList]
        self.nr_factors = len(self.factorNames)
        self.prefixDict_factors = prefixDict_factors
        self.chromSize_factors = chromSize_factors
        nr_bins = int( np.ceil(self.chromSize_factors / self.binsize) )
        self.FactorDataArray = np.empty(shape=(len(bigwigFileList),nr_bins))
        msg = "Loaded {:d} chromatin features from folder {:s}\n"
        msg = msg.format(self.nr_factors, self.chromatinFolder)
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
            msg += "{:s} - min. {:.3f} - max {:.3f} - nonzero {:d} ({:.2f}%)\n"
            msg = msg.format(bigwigFile, tmpArray.min(), tmpArray.max(), nr_nonzero_abs, nr_nonzero_perc)
        self.FactorDataArray = np.transpose(self.FactorDataArray)
        print(msg)
            
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
            raise ValueError(msg)
        #scale to 0..1, if requested
        if scaleMatrix:
            sparseHiCMatrix = utils.scaleArray(sparseHiCMatrix)       
        #ensure that chrom sizes for matrix and factors are the same
        if self.chromSize_factors is not None and self.chromSize_factors != chromsize_matrix:
            msg = "Chromsize of matrix does not match bigwig files\n"
            msg += "Matrix: {:d} -- Bigwig files: {:d}"
            msg = msg.format(chromsize_matrix, self.chromSize_factors)
            raise ValueError(msg)
        self.chromSize_matrix = chromsize_matrix
        #ensure that binsizes for matrix and factors (if given) match
        if self.binsize is None or self.binsize == binsize:
            self.binsize = binsize
            self.sparseHiCMatrix = sparseHiCMatrix
        elif self.binsize is not None and self.binsize != binsize:
            msg = "Matrix has wrong binsize\n"
            msg += "Matrix: {:d} -- Binned chromatin factors {:d}"
            msg = msg.format(binsize, self.binsize)
            raise ValueError(msg)
        msg = "Loaded chromosome {:s} from cooler matrix {:s}\n"
        msg = msg.format(self.chromosome, self.matrixfilepath)
        msg += "Matrix shape {:d}x{:d} -- min {:d} -- max {:d}"
        msg = msg.format(self.sparseHiCMatrix.shape[0], self.sparseHiCMatrix.shape[1], int(self.sparseHiCMatrix.min()), int(self.sparseHiCMatrix.max()))
        print(msg)

    def __loadSequenceData(self):
        #load DNA sequence from Fasta file
        if self.sequencefilepath is None:
            return
        try:
            records = SeqIO.index(self.sequencefilepath, format="fasta")
        except Exception as e:
            print(e)
            msg = "Could not read sequence file. Wrong format?"
            raise SystemExit(msg)
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
            raise ValueError(msg)
        #length check
        chromLengthSequence = len(records[ chromname ])
        if self.chromSize_factors is not None and self.chromSize_factors != chromLengthSequence:
            msg = "Chromosome {:s} in sequence file {:s} has wrong length\n"
            msg += "Chrom. factors: {:d} - Sequence File {:d}"    
            msg = msg.format(chromname, self.sequencefilepath, self.chromSize_factors, chromLengthSequence)
            raise ValueError(msg)
        elif self.chromSize_matrix is not None and self.chromSize_matrix != chromLengthSequence:
            msg = "Chromosome {:s} in sequence file {:s} has wrong length\n"
            msg += "Matrix: {:d} - Sequence File {:d}"    
            msg = msg.format(chromname, self.sequencefilepath, self.chromSize_matrix, chromLengthSequence)
            raise ValueError(msg)    
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
    
    def loadData(self, scaleFeatures=False, clampFeatures=False, scaleTargets=False):
        self.__loadMatrixData(scaleMatrix=scaleTargets)
        self.__loadFactorData(scaleFeatures=scaleFeatures, clampFeatures=clampFeatures)
        self.__loadSequenceData()

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
        #check if the same kind of data is available for all containers
        factorsOK = type(self.FactorDataArray) == type(container.FactorDataArray)
        matrixOK = type(self.sparseHiCMatrix) == type(container.sparseHiCMatrix)
        sequenceOK = type(self.sequenceArray) == type(container.sequenceArray)
        #if chromatin factors are present, they need be loaded to check compatibility
        if self.chromatinFolder is not None and self.nr_factors is None:
            return False
        if container.chromatinFolder is not None and container.nr_factors is None:
            return False
        #if chromatin factors are present, the number of chromatin factors must match
        factorsOK = factorsOK and (self.nr_factors == container.nr_factors)
        #if DNA sequences are present, they need be loaded to decide compatibility
        if self.sequencefilepath is not None and self.sequenceSymbols is None:
            return False
        if container.sequencefilepath is not None and container.sequenceSymbols is None:
            return False
        #if DNA sequences are present, the number of symbols must match
        sequenceOK = sequenceOK and (self.sequenceSymbols == container.sequenceSymbols)
        return factorsOK and matrixOK and sequenceOK
        
    def writeTFRecord(self, pWindowsize, pOutfolder, pFlankingsize=None, pMaxdist=None, pRecordSize=None):
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

        windowsize = pWindowsize
        flankingsize = windowsize
        if pFlankingsize is not None and pFlankingsize > 0:
            flankingsize = pFlankingsize
        maxdist = pMaxdist
        if pMaxdist is not None:
            maxdist = min(pMaxdist, windowsize-1)
        #get the number of samples
        if self.binsize is None:
            self.loadData()
        nr_samples = self.getNumberSamples(flankingsize=flankingsize, windowsize=windowsize)
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
        for recordfile, firstIndex, lastIndex in zip(recordfiles, sample_indices, sample_indices[1:]):
            recordDict, storedFeaturesDict = self.__prepareWriteoutDict(pFirstIndex=firstIndex, 
                                                                        pLastIndex=lastIndex, 
                                                                        pWindowsize=windowsize, 
                                                                        pOutfolder=pOutfolder,
                                                                        pFlankingsize=flankingsize, 
                                                                        pMaxdist=maxdist)
            records.writeTFRecord(pFilename=recordfile, pRecordDict=recordDict)
        self.storedFiles = recordfiles
        self.storedFeatures = storedFeaturesDict
        return recordfiles

    def getNumberSamples(self, flankingsize, windowsize):
        if self.binsize is None:
            self.loadData()
        featureArrays = [self.FactorDataArray, self.sparseHiCMatrix, self.sequenceArray]
        cutouts = [windowsize+2*flankingsize, windowsize+2*flankingsize, (windowsize+2*flankingsize)*self.binsize]
        nr_samples_list = []
        for featureArray, cutout in zip(featureArrays, cutouts):
            if featureArray is not None:
                nr_samples_list.append(featureArray.shape[0] - cutout + 1)
            else:
                nr_samples_list.append(0)
        #check if all features have the same number of samples
        if len(set( [x for x in nr_samples_list if x>0] )) != 1:
            msg = "Error: sample binning / DNA sequence encoding went wrong"
            raise ValueError(msg)
        return max(nr_samples_list)

    def __getMatrixData(self, idx, flankingsize, windowsize, maxdist):
        if self.matrixfilepath is None:
            return None # this can't work
        if self.matrixfilepath is not None and self.sparseHiCMatrix is None:
            self.loadData()
        #the 0-th matrix starts flankingsize away from the boundary
        startInd = idx + flankingsize
        stopInd = startInd + windowsize
        trainmatrix = None
        if isinstance(maxdist, int) and maxdist < windowsize: #trapezoids, i.e. distance limited submatrices
            trainmatrix = self.sparseHiCMatrix[startInd:stopInd,startInd:stopInd].todense()[np.mask_indices(windowsize, utils.maskFunc, maxdist)]
        else: #triangles, i. e. full submatrices
            trainmatrix = self.sparseHiCMatrix[startInd:stopInd,startInd:stopInd].todense()[np.triu_indices(windowsize)]
        trainmatrix = np.nan_to_num(trainmatrix)
        return trainmatrix
    
    def __getSequenceData(self, idx, flankingsize, windowsize):
        if self.sequencefilepath is None:
            return None
        if self.sequencefilepath is not None and self.sequenceArray is None:
            self.loadData()
        startInd = idx + flankingsize * self.binsize
        stopInd = startInd + windowsize * self.binsize
        seqArray = self.sequenceArray[startInd:stopInd,:]
        return seqArray

    def __getFactorData(self, idx, flankingsize, windowsize):
        if self.chromatinFolder is None:
            return None
        if self.chromatinFolder is not None and self.FactorDataArray is None:
            self.loadData()
        #the 0-th feature matrix starts at position 0
        startInd = idx
        stopInd = startInd + 2*flankingsize + windowsize
        factorArray = self.FactorDataArray[startInd:stopInd,:]
        return factorArray

    def getSampleData(self, idx, flankingsize, windowsize, maxdist):
        factorArray = self.__getFactorData(idx, flankingsize, windowsize)
        matrixArray = self.__getMatrixData(idx, flankingsize, windowsize, maxdist)
        sequenceArray = self.__getSequenceData(idx, flankingsize, windowsize)
        return {"factorArray": factorArray, 
                "matrixArray": matrixArray, 
                "sequenceArray": sequenceArray}
        
    def plotFeatureAtIndex(self, idx, flankingsize, windowsize, maxdist, outpath, figuretype="png"):
        if self.binsize is None:
            self.loadData()
        if self.FactorDataArray is None or self.chromSize_factors is None:
            msg = "Error: cannot plot features when they are not present"
            raise ValueError(msg)
        if not isinstance(flankingsize, int) \
                or not isinstance(windowsize, int):
            msg = "Error: Flankingsize and Windowsize must be integers"
            raise ValueError(msg)
        if isinstance(idx, int) and (idx >= self.FactorDataArray.shape[0] or idx < 0):
            msg = "Error: Invalid index {:d}; must be None or integer in 0..{:d}".format(idx, self.FactorDataArray.shape[0]-1)
            raise ValueError(msg)
        if isinstance(idx, int):
            factorArray = self.__getFactorData(idx, flankingsize, windowsize)
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
    
    def plotFeaturesAtPosition(self, position, flankingsize, windowsize, maxdist, outpath, figuretype="png"):
        if self.binsize is None:
            self.loadData()
        if self.FactorDataArray is None or self.chromSize_factors is None:
            msg = "Error: cannot plot features when they are not present"
            raise ValueError(msg)
        if not isinstance(flankingsize, int) \
                or not isinstance(windowsize, int) \
                or not isinstance(maxdist, int):
            msg = "Error: Flankingsize, Windowsize and Maxdist must be integers"
            raise ValueError(msg)
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
                                        flankingsize=flankingsize, 
                                        windowsize=windowsize, 
                                        maxdist=maxdist,
                                        outpath=outpath,
                                        figuretype=figuretype)

    def saveMatrix(self, flankingsize, windowsize, maxdist, outputpath, index=None):
        sparseMatrix = None
        if isinstance(maxdist, int) and maxdist <= windowsize:
            maxdistInt = maxdist
        else:
            maxdistInt = windowsize
        if isinstance(index, int) and index < self.getNumberSamples(flankingsize, windowsize):
            tmpMat = np.zeros(shape=(windowsize, windowsize))
            indices = np.mask_indices(windowsize, utils.maskFunc, k=maxdistInt)
            tmpMat[indices] = self.__getMatrixData(idx=index, flankingsize=flankingsize, windowsize=windowsize, maxdist=maxdist)
            sparseMatrix = csr_matrix(tmpMat)
        else:
            sparseMatrix = self.sparseHiCMatrix
        folderName = self.chromatinFolder.rstrip("/").replace("/","-")
        filename = "matrix_{:s}_chr{:s}_{:s}".format(folderName, str(self.chromosome), str(index))
        filename = os.path.join(outputpath, filename)
        save_npz(file=filename, matrix=sparseMatrix)

    def __prepareWriteoutDict(self, pFirstIndex, pLastIndex, pWindowsize, pOutfolder, pFlankingsize=None, pMaxdist=None):
        factorData = []
        matrixData = []
        sequenceData = []
        for i in range(pFirstIndex,pLastIndex):
            data = self.getSampleData(idx=i, 
                                        flankingsize=pFlankingsize, 
                                        windowsize=pWindowsize, 
                                        maxdist=pMaxdist)
            factorData.append(data["factorArray"])
            matrixData.append(data["matrixArray"])
            sequenceData.append(data["sequenceArray"])
        recordDict = dict()
        storedFeaturesDict = dict()
        if not any(elem is None for elem in factorData):
            recordDict["factorData"] = np.array(factorData)
            storedFeaturesDict["factorData"] = {"shape": recordDict["factorData"].shape[1:], "dtype": tfdtypes.as_dtype(recordDict["factorData"].dtype)}
        if not any(elem is None for elem in sequenceData):
            recordDict["sequenceData"] = np.array(sequenceData)
            storedFeaturesDict["sequenceData"] = {"shape": recordDict["sequenceData"].shape[1:], "dtype": tfdtypes.as_dtype(recordDict["sequenceData"].dtype)}
        if not any(elem is None for elem in matrixData):
            recordDict["out_matrixData"] = np.array(matrixData)
            storedFeaturesDict["out_matrixData"] = {"shape": recordDict["out_matrixData"].shape[1:], "dtype": tfdtypes.as_dtype(recordDict["out_matrixData"].dtype)}
        return recordDict, storedFeaturesDict