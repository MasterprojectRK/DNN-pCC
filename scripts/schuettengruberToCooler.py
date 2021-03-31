import click
import pandas as pd
import numpy as np
import cooler

DM3_CHROM_SIZES = {"2L": 23011544, 
									 "2LHet": 368872,
									 "2R": 21146708,
									 "2RHet": 3288761,
									 "3L": 24543557,
									 "3LHet": 2555491,
									 "3R": 27905053,
									 "3RHet": 2517507,
									 "4": 1351857,
									 "U": 10049037,
									 "Uextra": 29004656,
									 "X": 22422827,
									 "XHet": 204112,
									 "YHet":	347038,
									 "M": 19517}

@click.option("--pixels", "-p", type=click.Path(exists=True, readable=True, dir_okay=False), required=True)
@click.option("--bins", "-b", type=click.Path(exists=True, readable=True, dir_okay=False), required=True)
@click.option("--outfile", "-o", type=click.Path(writable=True, dir_okay=False), required=True)
@click.command()
def toCooler(pixels, bins, outfile):
	#check if the inputs are as expected
	try:
		df_pixels = pd.read_csv(pixels, sep="\t", index_col=False)
		df_bins = pd.read_csv(bins, sep="\t", index_col=False)
	except Exception as e:
		msg = str(e) + "\nCould not read infiles, wrong files/format etc.?"
		raise SystemExit(msg)
	pixels_columns = {"cbin1", "cbin2", "expected_count", "observed_count"}
	bins_columns = {"cbin", "chr", "from.coord", "to.coord", "count"}
	if len( pixels_columns.intersection(set(df_pixels.columns)) ) != len(pixels_columns):
		msg = "pixels: not the expected column names"
		raise SystemExit(msg)
	if len( bins_columns.intersection(set(df_bins.columns)) ) != len(bins_columns):
		msg = "bins: not the expected column names"
		raise SystemExit(msg)

	#prepare the pixels for cooler
	df_pixels.rename(columns={"cbin1": "bin1_id", "cbin2": "bin2_id", "observed_count": "count"}, inplace=True)
	df_pixels.drop(columns=["expected_count"], inplace=True)
	gt = df_pixels["bin1_id"] > df_pixels["bin2_id"]
	df_pixels = df_pixels[~gt] #drop duplicate entries, keep upper triangular part of matrix

	#prepare the bins for cooler
	df_bins.rename(columns={"chr": "chrom", "from.coord": "start", "to.coord": "end"}, inplace=True)
	df_bins.drop(columns=["count"], inplace=True)
	binsize = df_bins.iloc[0,:]["end"] - df_bins.iloc[0,:]["start"]
	chromnames = list(df_bins["chrom"].unique())
	print("chromnames:", chromnames)
	print("detected binsize:", binsize)
	#sometimes the last bin is present, but "end" does not point to chromsize, but max. size given by bin
	#need to set the max. chrom size in this case, otherwise the last bin will be duplicated, once with end==chromsize and once with end==maxbin*binsize
	for chrom in chromnames:
		max_allowed_val = int(np.ceil(DM3_CHROM_SIZES[chrom] / binsize) * binsize)
		chromfltr = df_bins["chrom"] == chrom
		max_given_val = df_bins[chromfltr]["end"].values[-1]
		if max_given_val == max_allowed_val:
			idx = df_bins.loc[chromfltr, "end"].index[-1]
			df_bins.loc[idx, "end"] = DM3_CHROM_SIZES[chrom] 
			msg = "INFO: reset size of chromosome {:s} from {:d} to {:d}".format(chrom, max_given_val, DM3_CHROM_SIZES[chrom])
			print(msg)
		elif max_given_val < max_allowed_val:
			pass
		else:
			msg = "Chrom {:s} is larger than expected from ref. genome dm3.".format(chrom)
			raise SystemExit(msg)
	
	#The provided bins dataframe is sparse (missing bins at start, end, and in-between)
	#So create a new one, which contains all bins
	df_bins_cpl = pd.DataFrame()
	for chr in chromnames:
		chromsize = DM3_CHROM_SIZES[chr]
		start_list = [x for x in range(0, chromsize, binsize)]
		end_list = [x for x in range(binsize, chromsize, binsize)] + [chromsize]
		df1 = pd.DataFrame()
		df1["start"] = start_list
		df1["end"] = end_list
		df1["chrom"] = chr
		df_bins_cpl = df_bins_cpl.append(df1, ignore_index=True)
	print(df_bins_cpl[df_bins_cpl["chrom"] == "2R"].tail())
	df_bins_cpl.reset_index(inplace=True, drop=True)
	#get the old indices into the cpl bins dataframe, need them to update the pixels dataframe later
	df_bins_cpl = df_bins_cpl.merge(df_bins, on=["start","end","chrom"], how="outer")
	df_bins_cpl["cbin"].fillna(-1, inplace=True)
	df_bins_cpl["cbin"] = df_bins_cpl["cbin"].astype("int64")
	df_bins_cpl.sort_values(by=["chrom", "start","end"], inplace=True)
	df_bins_cpl.reset_index(inplace=True, drop=True)
	df_bins_cpl["new_index"] = df_bins_cpl.index
	#update the bin ids in pixels df	
	df_pixels = df_pixels.merge(df_bins_cpl, left_on="bin1_id", right_on="cbin", how="inner")
	df_pixels["bin1_id"] = df_pixels["new_index"]
	df_pixels.drop(columns=["new_index", "cbin", "start", "end", "chrom"], inplace=True)
	df_pixels = df_pixels.merge(df_bins_cpl, left_on="bin2_id", right_on="cbin", how="inner")
	df_pixels["bin2_id"] = df_pixels["new_index"]
	df_pixels.drop(columns=["new_index", "cbin", "start", "end", "chrom"], inplace=True)
	
	df_bins_cpl.drop(columns=["cbin", "new_index"], inplace=True)
	
	print("\nsome lines of bins df:")
	print(df_bins_cpl.head(10))
	print(df_bins_cpl[df_bins_cpl["chrom"] == "2L"].tail())
	print(df_bins_cpl[df_bins_cpl["chrom"] == "2R"].head())
	print(df_bins_cpl[df_bins_cpl["chrom"] == "2R"].tail())
	print(df_bins_cpl[df_bins_cpl["chrom"] == "3L"].head())
	print(df_bins_cpl[df_bins_cpl["chrom"] == "3L"].tail())
	print(df_bins_cpl[df_bins_cpl["chrom"] == "3R"].head())
	print(df_bins_cpl[df_bins_cpl["chrom"] == "3R"].tail())
	print(df_bins_cpl[df_bins_cpl["chrom"] == "4"].head())
	print(df_bins_cpl[df_bins_cpl["chrom"] == "4"].tail())
	print(df_bins_cpl[df_bins_cpl["chrom"] == "X"].head())
	print(df_bins_cpl.tail())
	
	print("\nsome lines of pixels df:")
	print(df_pixels.head())
	print(df_pixels.tail())
	
	#write the cooler file
	cooler.create_cooler(outfile, 
						bins=df_bins_cpl, 
						pixels=df_pixels,  
						ordered=True, 
						metadata={"fromFilenames": [pixels, bins]})





if __name__ == "__main__":
	toCooler()
