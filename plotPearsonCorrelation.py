import utils
import click
import os

@click.option("--predMatrix", "-pm",
                multiple=True, required=True,
                type=click.Path(exists=True, dir_okay=False, readable=True),
                help="predicted cooler matrices")
@click.option("--legend", "-l",
                multiple=True, required=False,
                type=str,
                help="text for plot legend")
@click.option("--targetMatrix", "-tm",
                required=True,
                type=click.Path(exists=True, dir_okay=False, readable=True),
                help="target cooler matrix")
@click.option("--targetChrom", "-tchrom",
                type=str, required=True,
                help="target chromosome; must be present in all matrices")
@click.option("--targetCellLine", "-tcl",
                type=str, required=True,
                help="target cell line; for the plot title; e.g. 'K562'")
@click.option("--modelCellLines", "-mcl",
                type=str, required=True,
                help="model cell lines; for the plot title; e.g. 'GM12878' or 'HUVEC, HMEC' or 'various'")
@click.option("--modelChromosomes", "-mchrom",
                type=str, required=True,
                help="model chromosomes; for the plot title; e.g. '17' or '10, 11, 12' or 'various'")
@click.option("--windowsize", "-ws",
                type=click.IntRange(min=1),
                required=True,
                help="Windowsize in basepairs (i.e., max. distance between interacting regions)")
@click.option("--keepCsv", "-k",
                type=bool, required=False, default=False,
                help="write csv files with statistical data")
@click.option("--csvPath", "-csv",
                type=click.Path(exists=True, writable=True, file_okay=False),
                required=False, default="./",
                help="Path where csv files will be written to if --keepCsv=True")
@click.option("--outfile", "-o",
                type=click.Path(writable=True,dir_okay=False),
                required=True,
                help="Filename of figure (should end in .png .pdf or .svg)")
@click.option("--corrMethod", "-cm",
                type=click.Choice(["pearson", "spearman"]),
                required=False, default="pearson",
                help="Correlation method to use")
@click.command()
def plotPearsonCorrelation(predmatrix, 
                            legend,
                            targetmatrix, 
                            targetchrom,
                            targetcellline,
                            modelcelllines,
                            modelchromosomes,
                            windowsize,
                            keepcsv,
                            csvpath,
                            outfile,
                            corrmethod):
    legendList = []
    if len(legend) == 0:
        legendList = [None for x in predmatrix]
    else:
        legendList = list(legend)
    if len(legendList) != len(predmatrix):
        msg = "If legends given, there must be as many as matrices\n"
        msg += "#matrices {:d}, #legends {:d}"
        msg = msg.format(len(predmatrix), len(legendList))
        raise SystemExit(msg) 
    
    statDfList = []
    for predictedMatrix in predmatrix:
        csvFileName = None
        if keepcsv:
            csvFileName = os.path.splitext(os.path.basename(predictedMatrix))[0] + ".csv"
            csvFileName = os.path.join(csvpath, csvFileName)
        statDfList.append(utils.computePearsonCorrelation(pCoolerFile1=predictedMatrix,
                                                pCoolerFile2=targetmatrix,
                                                pWindowsize_bp=windowsize,
                                                pModelChromList=[modelchromosomes],
                                                pTargetChromStr=targetchrom,
                                                pModelCellLineList=[modelcelllines],
                                                pTargetCellLineStr=targetcellline,
                                                pPlotOutputFile=None,
                                                pCsvOutputFile=csvFileName))
    utils.plotPearsonCorrelationDf(pResultsDfList=statDfList,
                                    pLegendList=legendList,
                                    pOutfile=outfile,
                                    pMethod=corrmethod)



if __name__=="__main__":
    plotPearsonCorrelation() #pylint: disable=no-value-for-parameter