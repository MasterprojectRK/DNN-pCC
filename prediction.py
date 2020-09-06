import click

@click.option("--validationmatrix","-vm", required=False,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Target matrix in cooler format for statistical result evaluation, if available")
@click.option("--chromatinPath","-cp", required=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where results will be stored")
@click.command()
def prediction(validationmatrix, chromatinpath, outputpath):
    #check inputs

    #feed inputs through neural network

    #compute statistics, if validation matrix provided

    #store results
    pass

if __name__ == "__main__":
    prediction() #pylint: disable=no-value-for-parameter