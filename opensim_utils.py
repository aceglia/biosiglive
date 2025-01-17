import numpy as np
import csv
import glob
import os
from biosiglive import load


def read_sto_mot_file(filename):
    """
    Read sto or mot file from Opensim
    ----------
    filename: path
        Path of the file witch have to be read
    Returns
    -------
    Data Dictionary with file informations
    """
    data = {}
    data_row = []
    first_line = ()
    end_header = False
    with open(f"{filename}", "rt") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) == 0:
                pass
            elif row[0][:9] == "endheader":
                end_header = True
                first_line = idx + 1
            elif end_header is True and row[0][:9] != "endheader":
                row_list = row[0].split("\t")
                if idx == first_line:
                    names = row_list
                else:
                    data_row.append(row_list)
    data_mat = np.zeros((len(names), len(data_row)))
    for r in range(len(data_row)):
        data_mat[:, r] = np.array(data_row[r], dtype=float)
    return data_mat, names


# def write_sto_mot_file(all_paths, vicon_markers, depth_markers):
#     all_data = []
#     files = glob.glob(f"{all_paths['trial_dir']}Res*")
#     with open(files[0], 'r') as file:
#         csvreader = csv.reader(file, delimiter='\n')
#         for row in csvreader:
#             all_data.append(np.array(row[0].split("\t")))
#     all_data = np.array(all_data, dtype=float).T
#     data_index = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
#     all_data = all_data[data_index, :]
#     all_data = np.append(all_data, np.zeros((3, all_data.shape[1])), axis=0)
#
#     source = ["vicon", "depth"]
#     rate = [120, 60]
#     interp_size = [vicon_markers.shape[2], depth_markers.shape[2]]
#     for i in range(2):
#         x = np.linspace(0, 100, all_data.shape[1])
#         f = interp1d(x, all_data)
#         x_new = np.linspace(0, 100, interp_size[i])
#         all_data_int = f(x_new)
#         dic_data = {
#             "RFX": all_data_int[0, :],
#             "RFY": all_data_int[1, :],
#             "RFZ": all_data_int[2, :],
#             "RMX": all_data_int[3, :],
#             "RMY": all_data_int[4, :],
#             "RMZ": all_data_int[5, :],
#             "LFX": all_data_int[6, :],
#             "LFY": all_data_int[7, :],
#             "LFZ": all_data_int[8, :],
#             "LMX": all_data_int[9, :],
#             "LMY": all_data_int[10, :],
#             "LMZ": all_data_int[11, :],
#             "px": all_data_int[-1, :],
#             "py": all_data_int[-1, :],
#             "pz": all_data_int[-1, :]
#         }
#         # save(dic_data, f"{dir}/{participant}_{trial}_sensix_{source[i]}.bio")
#         headers = _prepare_mot(f"{all_paths['trial_dir']}{participant}_{trial}_sensix_{source[i]}.mot",
#                                all_data_int.shape[1], all_data_int.shape[0], list(dic_data.keys()))
#         duration = all_data_int.shape[1] / rate[i]
#         time = np.around(np.linspace(0, duration, all_data_int.shape[1]), decimals=3)
#         for frame in range(all_data_int.shape[1]):
#             row = [time[frame]]
#             for j in range(all_data_int.shape[0]):
#                 row.append(all_data_int[j, frame])
#             headers.append(row)
#         with open(f"{all_paths['trial_dir']}{participant}_{trial}_sensix_{source[i]}.mot", 'w', newline='') as file:
#             writer = csv.writer(file, delimiter='\t')
#             writer.writerows(headers)


def write_sto_mot_file(q, path):
    dof_names = [
        "thorax_tx",
        "thorax_ty",
        "thorax_tz",
        "thorax_tilt",
        "thorax_list",
        "thorax_rotation",
        "sternoclavicular_left_r1",
        "sternoclavicular_left_r2",
        "Acromioclavicular_left_r1",
        "Acromioclavicular_left_r2",
        "Acromioclavicular_left_r3",
        "shoulder_left_plane",
        "shoulder_left_ele",
        "shoulder_left_rotation",
        "elbow_left_flexion",
        "pro_sup_left",
    ]
    rate = 120
    headers = _prepare_mot(path, q.shape[1], q.shape[0], dof_names)
    duration = q.shape[1] / rate
    time = np.around(np.linspace(0, duration, q.shape[1]), decimals=3)
    for frame in range(q.shape[1]):
        row = [time[frame]]
        for j in range(q.shape[0]):
            row.append(q[j, frame])
        headers.append(row)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows(headers)


def _prepare_mot(output_file, n_rows, n_columns, columns_names):
    headers = [
        [output_file],
        ["version = 1"],
        [f"nRows = {n_rows}"],
        [f"nColumns = {n_columns + 1}"],
        ["inDegrees=yes"],
        ["endheader"],
    ]
    first_row = [
        "time",
    ]
    for i in range(len(columns_names)):
        first_row.append(columns_names[i])
    headers.append(first_row)
    return headers


def get_all_file(participants, data_dir, trial_names=None, to_include=(), to_exclude=()):
    all_path = []
    parts = []
    if trial_names and len(trial_names) != len(participants):
        trial_names = [trial_names for _ in participants]
    for p, part in enumerate(participants):
        try:
            all_files = os.listdir(f"{data_dir}{os.sep}{part}")
        except FileNotFoundError:
            print(f"Participant {part} not found in {data_dir}")
            continue
        if trial_names:
            to_include += trial_names[p] if isinstance(trial_names[p], list) else trial_names
        all_files = [
            file
            for file in all_files
            if any([ext in file for ext in to_include]) and not any([ext in file for ext in to_exclude])
        ]
        final_files = [f"{data_dir}{os.sep}{part}{os.sep}{file}" for file in all_files]
        parts.append([part for _ in final_files])
        all_path.append(final_files)
    return sum(all_path, []), sum(parts, [])


if __name__ == "__main__":
    source = ["dlc_technical_marker"]  # , "vicon_markerless"]#, "vicon", "minimal_vicon"]
    participants = [f"P{i}" for i in range(10, 11)]  # , "P15", "P16"]#, "P14", "P15", "P16"]
    participant = participants[0]
    s = source[0]
    # participants.pop(participants.index("P12"))
    prefix = "/mnt/shared/" if os.name == "posix" else r"Q:/"
    data_dir = f"{prefix}Projet_hand_bike_markerless/optim_params/reference_data"
    model_dir = f"{prefix}Projet_hand_bike_markerless/RGBD/"
    files, part = get_all_file(participants, data_dir, to_include=["reference_torque_gear_20_with_technical_marker"])
    data = load(files[0])
    end_idx = 1000
    q = data["q_ocp"][..., :end_idx]
    q[3:, ...] = np.degrees(q[3:, ...])
    q_dot = data["q_dot_ocp"][..., :end_idx]
    tau = data["tau_ocp"][..., :end_idx]
    data_path = "/mnt/shared/Projet_hand_bike_markerless/RGBD"
    write_sto_mot_file(q, f"{data_dir}/{participant}/torque_gear_20_with_technical_marker.mot")
