import click

@click.option("--trainmatrix","-tm",required=True,
                    type=click.Path(exists=True,dir_okay=False,readable=True),
                    help="Training matrix in cooler format")
@click.option("--chromatinPath","-cp", required=True,
                    type=click.Path(exists=True,readable=True,file_okay=False),
                    help="Path where chromatin factor data in bigwig format resides")
@click.option("--outputPath", "-o", required=True,
                    type=click.Path(exists=True,file_okay=False,writable=True),
                    help="Output path where trained network will be stored")
@click.command()
def training(trainmatrix, chromatinpath, outputpath):
    
    #check inputs

    #compose inputs into useful dataset

    #build neural network as described by Farre et al.

    #train the neural network

    #store the trained network
    pass


if __name__ == "__main__":
    training() #pylint: disable=no-value-for-parameter