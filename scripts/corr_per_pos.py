import numpy as np
import pandas as pd
import cooler
import click
from tqdm import tqdm
import matplotlib.pyplot as plt

@click.option("--trueMatrix", "-tm", type=click.Path(exists=True, dir_okay=False))
@click.option("--predMatrix", "-pm", type=click.Path(exists=True, dir_okay=False), multiple=True)
@click.option("--chrom", type=str)
@click.option("--winsize", type=click.IntRange(min=1), help="windowsize in bins")
@click.option("--figuretype", "-ft", type=click.Choice(["pdf", "png", "svg"]), default="png")
@click.option("--outfilename", "-o", type=click.Path(dir_okay=False, writable=True))
@click.option("--legend", "-l", type=str, multiple=True)
@click.command()

def corrPerPos(truematrix, predmatrix, chrom, winsize, figuretype, outfilename, legend):
	if len(legend) != len(predmatrix):
		legend = [name for name in predmatrix]
	try:
		binsize_true, matrix_true = loadChromFromCooler(chrom, truematrix)
	except Exception as e:
		raise e
	binsizes_pred = set() 
	matrices_pred = []
	for matrix in predmatrix:
		try:
			binsize, mat = 	loadChromFromCooler(chrom, matrix)
			binsizes_pred.add(binsize)
			matrices_pred.append(mat)
		except Exception as e:
			raise e
	#sanity check binsizes	
	if len(binsizes_pred) != 1:
		msg = "binsizes of predicted matrices are not equal: {:s}".format(", ".join(list(binsizes_pred)))
		raise SystemExit(msg)
	binsize_pred = list(binsizes_pred)[0]
	if binsize_true != binsize_pred:
		msg = "Binsizes are not equal: true: {:d}, pred: {:d}".format(binsize_true, binsize_pred)
		raise SystemExit(msg)
		
	#compute pearson correlations for sliding window along diagonal
	pearson_list = []
	for matrix_pred in matrices_pred:
		try:
			pearson_list.append(computePearsonValues(matrix_true, matrix_pred, winsize))
		except Exception as e:
			raise e
	
	#plot correlation vs. positions
	fig1, ax1 = plt.subplots()
	for pearson_arr in pearson_list:
		x_vals = np.arange(pearson_arr.shape[0]) * binsize_true / 1e6
		ax1.plot(x_vals, pearson_arr)
	ax1.set_xlabel("genomic position / Mbp")
	ax1.set_ylabel("Pearson correlation")
	ax1.set_title("Pearson correlation vs. genomic position\nChromosome {:s}".format(chrom))
	if not outfilename.endswith("." + figuretype):
		outfilename = outfilename + "." + figuretype
	ax1.grid(True)
	ax1.legend(legend)
	fig1.savefig(outfilename)
	plt.close(fig1)
	del fig1, ax1





def loadChromFromCooler(chrom, coolermatrix):
	if chrom.startswith("chr"): 
		chrom = chrom[3:]
	#check if the matrix is loadable
	try:
		matrix = cooler.Cooler(coolermatrix)
		binsize = matrix.binsize
	except Exception as e:
		msg = "unable to load cooler matrix {:s}, maybe no cooler?".format(coolermatrix)
		raise ValueError(msg)
	#check if the chromosome is available:
	matrix_chr_1 = matrix_chr_2 = None
	try:
		matrix_chr_1 = matrix.matrix(sparse=True,balance=False).fetch(chrom).tocsr()
	except:
		pass
	try:
		matrix_chr_2 = matrix.matrix(sparse=True,balance=False).fetch("chr" + chrom).tocsr()
	except:
		pass
	if matrix_chr_1 is not None:
		return binsize, matrix_chr_1
	elif matrix_chr_2 is not None:
		return binsize, matrix_chr_2
	else:
		msg = "chrom {:s} not found in matrix {:s}".format(chrom, coolermatrix)
		raise ValueError(msg)	


def computePearsonValues(sparseMatrix1, sparseMatrix2, winsize):
	if sparseMatrix1.shape[0] != sparseMatrix2.shape[0]:
		msg = "matrix shapes not equal: {:d} - {:d}".format(sparseMatrix1.shape[0], sparseMatrix2.shape[0])
		raise ValueError(msg)
	#considering upper triangular part is sufficient
	triu_indices = np.triu_indices(winsize)
	pearson_list = []
	nr_windows = sparseMatrix1.shape[0] - winsize + 1
	for i in tqdm(range(nr_windows)):
		np1 = sparseMatrix1[i:i+winsize,i:i+winsize].todense()
		np1 = np1[triu_indices].flatten()
		np2 = sparseMatrix2[i:i+winsize,i:i+winsize].todense()
		np2 = np2[triu_indices].flatten()
		corr = np.corrcoef(np1, np2)[0,1]
		pearson_list.append(corr)
	pearson_arr = np.nan_to_num(np.array(pearson_list), nan=0.0)
	return pearson_arr	








if __name__ == "__main__":
	corrPerPos()
