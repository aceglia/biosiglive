import numpy as np


def dic_merger(dic_to_merge, new_dic=None):
    """Merge two dictionaries.

    Parameters
    ----------
    dic_to_merge : dict
        Dictionary to merge.
    new_dic : dict
        Dictionary to merge with.

    Returns
    -------
    dict
        Merged dictionary.
    """

    if not new_dic:
        new_dic = dic_to_merge
    else:
        for key in dic_to_merge.keys():
            if isinstance(dic_to_merge[key], dict):
                new_dic[key] = dic_merger(dic_to_merge[key], new_dic[key])
            elif isinstance(dic_to_merge[key], list):
                new_dic[key] = dic_to_merge[key] + new_dic[key]
            elif isinstance(dic_to_merge[key], np.ndarray):
                new_dic[key] = np.append(dic_to_merge[key], new_dic[key], axis=0)
            elif isinstance(dic_to_merge[key], (int, float)):
                if isinstance(new_dic[key], (int, float)):
                    new_dic[key] = [new_dic[key]]
                new_dic[key] = [new_dic[key]] + [dic_to_merge[key]]
            else:
                raise ValueError("Type not supported")
        for key in new_dic.keys():
            if key not in dic_to_merge.keys():
                new_dic[key] = new_dic[key]
    return new_dic
