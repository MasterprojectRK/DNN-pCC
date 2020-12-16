import click
import utils
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy import sparse

@click.option("--matrix", "-m", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="cooler file to split")
@click.option("--pcaFile", "-pca", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="bedgraph file with PCA values (e.g. from hicPCA)" )
@click.option("--chromosome","-chrom", required=True, type=str)
@click.option("--outfolder", "-o", required=True,
             type=click.Path(exists=True, file_okay=False),
             help="path for output")
@click.command()
def matrixCompartmentSplit(matrix, pcafile, chromosome, outfolder):
    '''
    split a cooler matrix in two such that in one file 
    all A-compartment-bins and mixed A/B-compartment-bins are zeroed and in the other
    all B-compartment-bins and mixed A/B-compartment-bins are zeroed
    no sanity checks and nothing, also no computational efficiency, use at your own risk
    note high memory demands for high-res cooler matrices

    Parameters:
    matrix: str, path to matrix in cooler format
    pcafile: str, path to pca1 file in bedgraph format, same resolution as matrix
    chromosome: str, the chromosome to use
    outfolder: str, the path where results will be stored

    Returns:
    Nothing, but three matrices will be stored in outfolder,
    one with A-compartment-rows/columns zeroed, one with B-Compartment-rows/columns zeroed
    and one where only bins which are considered interactions between A/B are kept 
    '''
    sp1, binsize = utils.getMatrixFromCooler(matrix, chromosome)
    df1 = pd.read_csv(pcafile, sep="\t", header=None, names=["chrom", "start", "end", "value"])
    filter1 = df1["chrom"] == chromosome
    df1 = df1.loc[filter1]
    df1["start"] = np.uint32(np.ceil(df1["start"] / binsize))
    df1["end"] = np.uint32(np.ceil(df1["end"] / binsize))

    filter2 = df1["value"] >= 0
    pos_starts = list(df1.loc[filter2]["start"])
    pos_ends = list(df1.loc[filter2]["end"])
    neg_starts = list(df1.loc[~filter2]["start"])
    neg_ends = list(df1.loc[~filter2]["end"])

    sign = df1["value"].map(np.sign)
    diff1 = sign.diff(periods=1).fillna(0)
    diff2 = sign.diff(periods=-1).fillna(0)
    e1 = list(df1.loc[diff1[diff1 != 0].index]["start"])
    s1 = list(df1.loc[diff2[diff2 != 0].index]["end"])
    if s1[0] != 0:
        s1 = [0] + s1
    if e1[-1] != df1["end"].iloc[-1]:
        e1 = e1 + [df1["end"].iloc[-1]]
    
    posFilename = os.path.basename(matrix)[:-5] + "_posVals_chr{:s}.cool".format(chromosome)
    posFilename = os.path.join(outfolder, posFilename)
    negFilename = os.path.basename(matrix)[:-5] + "_negVals_chr{:s}.cool".format(chromosome)
    negFilename = os.path.join(outfolder, negFilename)
    mixedFilename = os.path.basename(matrix)[:-5] + "_mixedVals_chr{:s}.cool".format(chromosome)
    mixedFilename = os.path.join(outfolder, mixedFilename)

    sp2 = sp1.copy().tolil()
    for start, end in tqdm(zip(pos_starts, pos_ends), total=len(pos_starts), desc="zeroing pos. + mixed"):
        sp2[start:end,:] = 0
        sp2[:,start:end] = 0
    sp2 = sparse.csr_matrix(sp2)
    print("matshape", sp2.shape)
    utils.writeCooler([sp2],binsize, posFilename, [chromosome])
    del sp2
    sp2 = sp1.copy().tolil()
    for start, end in tqdm(zip(neg_starts, neg_ends), total=len(neg_starts), desc="zeroing neg. + mixed"):
        sp2[start:end,:] = 0
        sp2[:,start:end] = 0
    sp2 = sparse.csr_matrix(sp2)
    utils.writeCooler([sp2],binsize, negFilename, [chromosome])

    sp2 = sp1.copy().tolil()
    for start, end in tqdm(zip(s1, e1), total=len(s1), desc="zeroing non-mixed"):
        sp2[start:end,start:end] = 0
    sp2 = sparse.csr_matrix(sp2)
    utils.writeCooler([sp2],binsize, mixedFilename, [chromosome])


if __name__=="__main__":
    matrixCompartmentSplit() #pylint: disable=no-value-for-parameter