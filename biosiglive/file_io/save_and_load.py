import pickle
import numpy as np
from pathlib import Path
from ..streaming.utils import dic_merger


def save(data_dict, data_path):
    """This function adds data to a pickle file. It not open the file, but appends the data to the end, so it's fast.

    Parameters
    ----------
    data_dict : dict
        The data to be added to the file.
    data_path : str
        The path to the file. The file must exist.
    """
    if Path(data_path).suffix != ".bio":
        if Path(data_path).suffix == "":
            data_path += ".bio"
        else:
            raise ValueError("The file must be a .bio file.")
    with open(data_path, "ab") as outf:
        pickle.dump(data_dict, outf, pickle.HIGHEST_PROTOCOL)


# TODO add dict merger
def load(filename, number_of_line=None):
    """This function reads data from a pickle file to concatenate them into one dictionary.

    Parameters
    ----------
    filename : str
        The path to the file.
    number_of_line : int
        The number of lines to read. If None, all lines are read.

    Returns
    -------
    data : dict
        The data read from the file.

    """
    if Path(filename).suffix != ".bio":
        raise ValueError("The file must be a .bio file.")
    data = None
    limit = 2 if not number_of_line else number_of_line
    with open(filename, "rb") as file:
        count = 0
        while count < limit:
            try:
                data_tmp = pickle.load(file)
                if not data:
                    data = data_tmp
                else:
                    data = dic_merger(data, data_tmp)
                if number_of_line:
                    count += 1
                else:
                    count = 1
            except EOFError:
                break
    return data
