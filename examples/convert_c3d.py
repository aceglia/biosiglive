from pyomeca import Analogs, Markers
from  biosiglive import save
import os
import glob

participant = ["P4"]
markers_names = ['ster', 'xiph', 'c7', 't5', 'ribs', 'clavsc', 'clavac', 'scapts', 'scapia',
       'scapaa', 'delt', 'arml', 'epicm', 'epicl', 'elb', 'larml', 'stylr',
       'stylu', '*17', '*18', '*19', '*20']
trigger_name = ["Electric Current.1"]
muscle_names = ['pec.IM EMG1',
       'bic.IM EMG2',
       'tri.IM EMG3', 'lat.IM EMG4', 'trap.IM EMG5', 'delt_ant.IM EMG6',
       'delt_med.IM EMG7', 'delt_post.IM EMG8',]
for part in participant:
    file_dir = f"/run/user/1002/gvfs/smb-share:server=10.89.24.15,share=q/Projet_hand_bike_markerless/vicon/{part}/session_2"

    all_c3d = glob.glob(file_dir + "/*.c3d")
    all_c3d = [file_dir + "/gear_15.c3d"]
    for c3d in all_c3d:
        if os.path.isfile(f"{part}_{c3d.split('/')[-1][:-4]}_c3d.bio"):
            print(f"file : {part}_{c3d.split('/')[-1][:-4]}_c3d.bio already exists")
            continue
        trigger = []
        if 'calib' not in c3d:
            trigger = Analogs.from_c3d(filename=f"{c3d}", usecols=["Electric Current.1"])
        emg = []
        if not "anato" in c3d:
            emg = Analogs.from_c3d(filename=f"{c3d}", usecols=muscle_names)
        markers_vicon = []
        if "sprint" not in c3d:
            markers_vicon = Markers.from_c3d(filename=f"{c3d}") #, usecols=markers_names)
        save({"trigger":trigger, "markers":markers_vicon, "emg": emg}, f"{part}_{c3d.split('/')[-1][:-4]}_c3d.bio")
        print(f"file : {part}_{c3d.split('/')[-1][:-4]}_c3d.bio saved")
