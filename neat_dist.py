import numpy as np

import jenkspy


def get_split(loss):
    return jenkspy.jenks_breaks(loss, nb_class=2)


if __name__ == "__main__":
    pass
