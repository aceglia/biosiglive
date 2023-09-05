import numpy as np

from biosiglive import load, save
import matplotlib.pyplot as plt

data = load("stim_motomed_encodeur_compress.bio", merge=False)
angle = []
stim = None
data = data[0]
for i in range(len(data)):
    angle.append(data[i]["angle"])
    if stim is None:
        stim = data[i]["stim"]
    else:
        stim = np.append(stim, data[i]["stim"], axis=1)
x = np.linspace(0, stim.shape[1], stim.shape[1])
stim = stim[:, 1500*100:2000*100]
angle = angle[1500:2000]
t = np.linspace(0, stim.shape[1]/10000, len(angle))
t2 = np.linspace(0, stim.shape[1]/10000, stim.shape[1])
stim = np.clip(a=stim, a_min=0, a_max=0.09)/0.09
plt.figure()

plt.plot(t, angle, "g",)
plt.legend(["Crank angle"])
plt.ylabel("Crank angle (°)")
plt.xlabel("Time (s)")
ax2 = plt.twinx()
for i in range(1, 3):
    ax2.plot(t2, stim[i, :], alpha=0.5, label="stim"+str(i))

plt.legend(["Biceps and Anterior Deltoid stimulations", "Triceps and Posterior Deltoid stimulations"])
plt.ylabel("Normalized Stimulation (%)")

#add axis label
plt.xlabel("Time (s)")
plt.savefig("stim_motomed.svg")

data = load("walk_stim_bis.bio", merge=False)
angle = []
stim = None
f_z = None

for i in range(len(data)):
    if stim is None:
        stim = data[i]["stim"][0]
        f_z = data[i]["force_z_raw"]
    else:
        stim = np.append(stim, data[i]["stim"][0], axis=1)
        f_z = np.append(f_z, data[i]["force_z_raw"], axis=1)

stim = stim[0:1, 252360:267630] * 2
t = np.linspace(0, stim.shape[1]/3000, stim.shape[1])
f_z = f_z[0, 252360:267630]
stim = np.clip(a=stim, a_min=0, a_max=0.6) / 0.6
plt.figure()
plt.plot(t, f_z*1000, "g")
plt.legend(["Left foot ground reaction force"])
plt.ylabel("Force (N)")
plt.xlabel("Time (s)")
ax2 = plt.twinx()
ax2.plot(t, stim[0, :], alpha=0.5)
plt.legend(["Left Soleus stimulations"])
plt.ylabel("Normalized Stimulation (%)")
plt.xlabel("Time (s)")
plt.savefig("stim_fz.svg")
plt.show()

