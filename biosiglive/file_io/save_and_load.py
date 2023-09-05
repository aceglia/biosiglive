import os.path

import pickle
from pathlib import Path
from ..streaming.utils import dic_merger
import pickletools
import gzip
from multiprocessing import Pool


def _load_and_compress(origin_file, target_file=None):
    data = load(origin_file)
    if not target_file:
        target_file = Path(origin_file).stem + ".bio.gzip"
    if Path(target_file).suffix != ".gzip":
        target_file = target_file + ".gzip"
    f = gzip.open(target_file, "wb")
    data_pick = pickletools.optimize(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))
    f.write(data_pick)
    f.close()


def compress(origin_file, target_file=None, multi_proc=False):
    if multi_proc:
        pool = Pool(processes=os.cpu_count())
        pool.map(_load_and_compress, origin_file)
    else:
        _load_and_compress(origin_file, target_file)


def is_int_file_end(data_path, last_n = None):
    """This function checks if the file extension is an integer.

    Parameters
    ----------
    filename : str
        The path to the file.

    Returns
    -------
    bool
        True if the file extension is an integer, False otherwise.
    """
    try:
        n = int(Path(data_path).stem[-1])
        if Path(data_path).stem[-2] == "_":
            data_path = Path(data_path).stem[:-2] + "_" + str(n + 1) + ".bio"
        else:
            data_path = is_int_file_end(Path(data_path).stem[-1] + ".bio", last_n=n)
    except ValueError:
        return Path(data_path).stem + "_1" + ".bio"
    return data_path


def save(data_dict, data_path, safe=False):
    """This function adds data to a pickle file. It not open the file, but appends the data to the end, so it's fast.

    Parameters
    ----------
    data_dict : dict
        The data to be added to the file.
    data_path : str
        The path to the file. The file must exist.
    safe : bool, optional
        If True, the data are saved in a new file. The default is False.
    """
    if Path(data_path).suffix != ".bio":
        if Path(data_path).suffix == "":
            data_path += ".bio"
        else:
            raise ValueError("The file must be a .bio file.")

    if safe and os.path.isfile(data_path):
        data_path = is_int_file_end(data_path)

    with open(data_path, "ab") as outf:
        pickle.dump(data_dict, outf, pickle.HIGHEST_PROTOCOL)


def load(filename, number_of_line=None, merge=True):
    """This function reads data from a pickle file to concatenate them into one dictionary.

    Parameters
    ----------
    filename : str
        The path to the file.
    number_of_line : int
        The number of lines to read. If None, all lines are read.
    merge : bool
        If True, the data are merged into one dictionary. If False, the data are returned as a list of dictionaries.

    Returns
    -------
    data : dict
        The data read from the file.

    """
    # if Path(filename).suffix == ".bio":
    #     with_gzip = False
    # elif Path(filename).suffix == ".gzip":
    #     with_gzip = True
    # else:
    # #     raise ValueError("The file must be a .bio or a .bio.gzip file.")
    with_gzip = False
    data = None if merge else []
    limit = 2 if not number_of_line else number_of_line
    if with_gzip:
        with gzip.open(filename, "rb") as file:
            count = 0
            while count < limit:
                try:
                    data_tmp = pickle.load(file)
                    if not merge:
                        data.append(data_tmp)
                    else:
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
    else:
        with open(filename, "rb") as file:
            count = 0
            while count < limit:
                try:
                    data_tmp = pickle.load(file)
                    if not merge:
                        data.append(data_tmp)
                    else:
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
