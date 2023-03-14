import numpy as np


def dic_merger(dic_to_merge: dict, new_dic: dict = None) -> dict:
    """Merge two dictionaries.

    Parameters
    ----------
    dic_to_merge : dict
        Existing dictionary to merge.
    new_dic : dict
        Temporary dictionary to merge with.

    Returns
    -------
    dict
        Merged dictionary.
    """

    if not new_dic:
        new_dic = dic_to_merge
    else:
        for key in dic_to_merge.keys():
            if not dic_to_merge[key]:
                dic_to_merge[key] = [dic_to_merge[key]]
            if not new_dic[key]:
                new_dic[key] = [new_dic[key]]
            if isinstance(new_dic[key], (int, float)):
                new_dic[key] = [new_dic[key]]
            if isinstance(dic_to_merge[key], (int, float)):
                dic_to_merge[key] = [dic_to_merge[key]]
            if isinstance(dic_to_merge[key], dict):
                new_dic[key] = dic_merger(dic_to_merge[key], new_dic[key])
            elif isinstance(dic_to_merge[key], list):
                new_dic[key] = dic_to_merge[key] + new_dic[key]
            elif isinstance(dic_to_merge[key], np.ndarray):
                if not isinstance(new_dic[key], np.ndarray):
                    new_dic[key] = np.array(new_dic[key])
                if len(new_dic[key].shape) == 1:
                    new_dic[key] = new_dic[key][:, np.newaxis]
                new_dic[key] = np.append(dic_to_merge[key], new_dic[key], axis=0)
            else:
                raise ValueError("Type not supported")
        for key in new_dic.keys():
            if key not in dic_to_merge.keys():
                new_dic[key] = new_dic[key]
    return new_dic

