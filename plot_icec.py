import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
#from netCDF4 import Dataset
import pickle
import cmocean
import string

plt.style.use('default')
#mpl.rcParams['lines.linewidth'] = 1

basePath = './'
dataDir = os.path.join(basePath,'data')

drifter_IDs = ['RockBLOCK_14432', 'RockBLOCK_14437', 'RockBLOCK_14435', 'RockBLOCK_14438']

fwidth = 140/25.4
golden = (1.0 + np.sqrt(5.0))/2.0
fheight = fwidth/golden

# goign to set up a subplot for each forcing
#plotDir = os.path.join(homeDir, 'plots')
plotDir = os.path.join(basePath, 'plots')
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

# transition time from ice to no ice
t0 = pd.to_datetime('2018-09-25')
# just read in data for each
force_all = ['eccc', 'topaz']
force_lab = {'eccc':'CAPS', 'topaz':'TOPAZ'}
dd_obs = {}
for force in force_all:
    with open(os.path.join(dataDir, 'drifter_{}.pickle'.format(force)), 'rb') as f:
        dd_obs.update({force:pickle.load(f, fix_imports=True, encoding="latin1")})

fig, axarr = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(fwidth, fheight))
fig.subplots_adjust(wspace=0.05)

# no point looping again, add stuff here
cols = plt.cm.Dark2
cmap = cols
alph = 0.9

drifter_cols = {'RockBLOCK_14438':cmap(3, alpha=alph), 'RockBLOCK_14435':cmap(0, alpha=alph),\
                'RockBLOCK_14437':cmap(1, alpha=alph), 'RockBLOCK_14432':cmap(2, alpha=alph)}
lines = {'eccc':'-', 'topaz':'--'}
lw = 1.0
fig2, ax2 = plt.subplots(1, figsize=(fwidth, fheight))

for iax, force in enumerate(force_all):
    for iid, ID in enumerate(drifter_IDs):
        t = dd_obs[force][ID].index
        A = dd_obs[force][ID].icec
        A[A<0.0] = 0.0
        axarr[iax].plot(t, A, '-', label=ID[-5:], color=drifter_cols[ID])
        # plot big one
        lab_txt = '{} [{}]'.format(ID[-5:],force_lab[force])
        ax2.plot(t, A, color=drifter_cols[ID], linestyle=lines[force], label=lab_txt, linewidth=lw, alpha=alph)

    axarr[iax].text(0, 1.025, '{})'.format(string.ascii_lowercase[iax]), transform=axarr[iax].transAxes,\
            size=8, weight='bold')
    axarr[iax].axvline(x=t0, color=(0.5,0.5,0.5), linewidth=2, alpha=0.8)
    axarr[iax].legend(loc='upper right', ncol=1, prop={'size':10})
    axarr[iax].set(title=force_lab[force])
    if iax > 0:
        axarr[iax].yaxis.tick_right()
        axarr[iax].yaxis.set_label_position('right')
    else:
        axarr[iax].set(ylabel=r'$A$')


fig.autofmt_xdate()
fig.savefig(os.path.join(plotDir, 'icec.pdf'), bbox_inches='tight')
plt.close(fig)

ax2.axvline(x=t0, color=(0.25,0.25,0.25), linewidth=2, alpha=0.5)
ax2.set(ylabel=r'$A$')
ax2.legend(loc='upper right', ncol=1, prop={'size':8})
fig2.autofmt_xdate()
fig2.savefig(os.path.join(plotDir, 'icec2.pdf'), bbox_inches='tight')
plt.close(fig2)

