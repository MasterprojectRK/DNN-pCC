import utils
import records
import os
import numpy as np
from Bio import SeqIO

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

    def __loadFactorData(self, ignoreChromLengths=False):
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
        self.factorNames = [os.path.splitext(name)[0] for name in bigwigFileList]
        self.nr_factors = len(self.factorNames)
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
        self.prefixDict_factors = prefixDict_factors
        #the chromosome lengths should be equal in all bigwig files
        if len(set(chromSizeList)) != 1 and not ignoreChromLengths:
            msg = "Invalid data. Chromosome lengths differ in bigwig files:"
            for i, filename in enumerate(bigwigFileList):
                msg += "{:s}: {:d}\n".format(filename, chromSizeList[i])
            raise ValueError(msg)
        elif len(set(chromSizeList)) != 1 and ignoreChromLengths:
            self.chromSize_factors = min(chromSizeList)
        else:
            self.chromSize_factors = chromSizeList[0]
        #the chromosome lengths of matrices and bigwig files must be equal
        if self.chromSize_matrix is not None \
                and self.chromSize_matrix != self.chromSize_factors:
            msg = "Chrom lengths not equal between matrix and bigwig files\n"
            msg += "Matrix: {:d} -- Factors: {:d}".format(self.chromSize_matrix, self.chromSize_factors)
            raise ValueError(msg)
        if self.chromSize_sequence is not None \
                and self.chromSize_sequence != self.chromSize_factors:
            msg = "Chrom lengths not equal between sequence and bigwig files\n"
            msg += "Sequence: {:d} -- Factors: {:d}".format(self.chromSize_sequence, self.chromSize_factors)
            raise ValueError(msg)
        #load the data into memory now
        nr_bins = int( np.ceil(self.chromSize_factors / self.binsize) )
        self.FactorDataArray = np.empty(shape=(len(bigwigFileList),nr_bins))
        for i, bigwigFile in enumerate(bigwigFileList):
            chromname = self.prefixDict_factors[bigwigFile] + self.chromosome
            self.FactorDataArray[i] = utils.binChromatinFactor(pBigwigFileName=bigwigFile,
                                                                    pBinSizeInt=self.binsize,
                                                                    pChromStr=chromname,
                                                                    pChromSize=self.chromSize_factors)
        self.FactorDataArray = np.transpose(self.FactorDataArray)
            
    def __loadMatrixData(self):
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
        #ensure that chrom sizes for matrix and factors are the same
        if self.chromSize_factors is not None and self.chromSize_factors != chromsize_matrix:
            msg = "Chromsize of matrix does not match bigwig files\n"
            msg += "Matrix: {:d} -- Bigwig files: {:d}"
            msg = msg.format(chromsize_matrix, self.chromSize_factors)
            raise ValueError(msg)
        #ensure that binsizes for matrix and factors (if given) match
        if self.binsize is None or self.binsize == binsize:
            self.binsize = binsize
            self.sparseHiCMatrix = sparseHiCMatrix
        elif self.binsize is not None and self.binsize != binsize:
            msg = "Matrix has wrong binsize\n"
            msg += "Matrix: {:d} -- Binned chromatin factors {:d}"
            msg = msg.format(binsize, self.binsize)
            raise ValueError(msg)

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
        self.sequenceArray = utils.fillEncodedSequence(utils.encodeSequence(seqStr, self.sequenceSymbols), self.binsize)
        del seqStr
        records.close()
    
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
    
    def loadData(self):
        self.__loadMatrixData()
        self.__loadFactorData()
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
        recordfiles = [os.path.join(pOutfolder, "{:03d}.tfrecord".format(i + 1)) for i in range(nr_files)]
        for recordfile, firstIndex, lastIndex in zip(recordfiles, sample_indices, sample_indices[1:]):
            factorData = []
            matrixData = []
            sequenceData = []
            for i in range(firstIndex,lastIndex):
                data = self.getSampleData(idx=i, 
                                          flankingsize=flankingsize, 
                                          windowsize=windowsize, 
                                          maxdist=maxdist)
                factorData.append(data["factorArray"])
                matrixData.append(data["matrixArray"])
                sequenceData.append(data["sequenceArray"])
            recordDict = dict()
            if not None in factorData:
                recordDict["factorData"] = np.array(factorData)
            if not None in sequenceData:
                recordDict["sequenceData"] = np.array(sequenceData)
            if not None in matrixData:
                recordDict["out_matrixData"] = np.array(matrixData)
            records.writeTFRecord(pFilename=recordfile, pRecordDict=recordDict)
        self.storedFiles = recordfiles
        self.storedFeatures = [key for key in recordDict]
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
        if maxdist >= windowsize: #triangles, i. e. full submatrices
            trainmatrix = self.sparseHiCMatrix[startInd:stopInd,startInd:stopInd].todense()[np.triu_indices(windowsize)]
        else: #trapezoids, i.e. distance limited submatrices
            trainmatrix = self.sparseHiCMatrix[startInd:stopInd,startInd:stopInd].todense()[np.mask_indices(windowsize, utils.maskFunc, maxdist)]
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
                or not isinstance(windowsize, int) \
                or not isinstance(maxdist, int):
            msg = "Error: Flankingsize, Windowsize and Maxdist must be integers"
            raise ValueError(msg)
        if isinstance(idx, int) and (idx >= self.FactorDataArray.shape[0] or idx < 0):
            msg = "Error: Invalid index {:d}; must be None or in 0..{:d}".format(idx, self.FactorDataArray.shape[0])
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