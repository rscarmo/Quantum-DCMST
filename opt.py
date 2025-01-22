import argparse
import os.path as opath
from qopt import QOPT
import numpy as np


#  de alguma forma vc deve criar uma função que recebe parâmetro, faz avaliação no quantum algorithm e retorna uma loss como número real
# exemplo toy
def sphere(x):
    return np.sum(x**2)


def main(args):

    qopt = QOPT(args.config)
    # faz optimização com funcção de loss sphere
    results = qopt(sphere)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum optimization")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the configuration file",
        default=opath.join("experiments", "configs", "config.yaml"),
    )
    args = parser.parse_args()
    main(args)
