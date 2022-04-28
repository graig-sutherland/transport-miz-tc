import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os, sys
import pickle
import string
from utils import calculate_drifter_speed_forward as drifter_speed


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def get_heading(w):
    ang = np.mod(90.0 - np.angle(w, deg=True), 360)
    ang[ang>180] = ang[ang>180]-360
    return ang

def plot_angle(x, y, ax, lab_txt, col, lsty='-', lw=1.1, marker=None, markeredgecolor=None, markerfacecolor=None, ms=None):
    '''plot angle on axes'''
    ang1 = get_heading(y)
    da = np.diff(ang1)
    iis = np.argwhere(np.abs(da)>180) + 1
    if len(iis) > 0:
        ip = 0
        for i in iis.flatten():
            ax.plot(x[ip:i], ang1[ip:i], '-', linewidth=lw, linestyle=lsty, \
                    label='_nolegend_', color=col, marker=marker, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, markersize=ms)
            ip=i
        ax.plot(x[ip:], ang1[ip:], '-', linewidth=lw, linestyle=lsty, label=lab_txt, color=col, marker=marker, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, markersize=ms)
    else:
        ax.plot(t, ang1, '-', linewidth=lw, linestyle=lsty, label=lab_txt, color=col, marker=marker, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, markersize=ms)



# define some directories for reading and plotting data
basePath = './'
dataDir = os.path.join(basePath, 'data')
plotDir = os.path.join(basePath, 'plots')

# drifters
drifter_IDs = ['RockBLOCK_14432', 'RockBLOCK_14437', 'RockBLOCK_14435', 'RockBLOCK_14438']
# forcing
force_all = ['eccc', 'topaz']
force_lab = {'eccc':'CAPS', 'topaz':'TOPAZ'}
# read in dataframe for each forcing
dd_obs = {}
for force in force_all:
    with open(os.path.join(dataDir, 'drifter_{}.pickle'.format(force)), 'rb') as fh:
        dd_obs.update({force:pickle.load(fh, fix_imports=True, encoding="latin1")})

# figure stuff
golden = 0.5*(1.0 + np.sqrt(5.0))
fwidth = 140/25.4
fheight = fwidth*golden
alph = 0.9
cols = plt.cm.tab10
# transition time from ice to no ice
t0 = pd.to_datetime('2018-09-25')
lw = 1.1
#resamp_freq = '3H'
resamp_freq = 'raw'
#markersize
ms = 2
mlcol = (0.15,0.15,0.15, alph)
vel_cols = {'drifter':mlcol, 'ocean':cols(0, alpha=alph), 'ice':cols(2, alpha=alph), 'wind':cols(1, alpha=alph), 'topaz':cols(2, alpha=alph), 'eccc':cols(0, alpha=alph)}
vel_cols.update({'eccc_ice':plt.cm.tab20(1, alpha=alph), 'topaz_ice':plt.cm.tab20(5, alpha=alph)})
markers = {'drifter':None, 'ocean':None, 'ice':None, 'wind':None, 'topaz':None, 'eccc':None, 'topaz_ice':None, 'eccc_ice':None}

# limits for vel and dir
dlim = [-180, 180]
ms2kmday = 86.4

# going to plot velocity for all drifters
fig, axarr = plt.subplots(ncols=2, nrows=len(drifter_IDs), sharex=True, sharey='col', figsize=(fwidth,fheight))
for ii, ID in enumerate(drifter_IDs):
    for ff, force in enumerate(force_all):
        # going to do some resampling to test
        if resamp_freq == 'raw':
            df_filt = dd_obs[force][ID].copy()
        else:
            idx = dd_obs[force][ID].asfreq(resamp_freq).index
            df_filt = dd_obs[force][ID].reindex(dd_obs[force][ID].index.union(idx)).interpolate('index').reindex(idx)
        velf = drifter_speed(df_filt)
        drf = ms2kmday*(velf[:,0] + 1j*velf[:,1])/3.6 # converts km/h -> m/s
        oce = ms2kmday*(df_filt.uuw + 1j*df_filt.vvw)
        ice = ms2kmday*(df_filt.uui + 1j*df_filt.vvi)
        wind = ms2kmday*(df_filt.uu + 1j*df_filt.vv)
        wind *= 0.02
        A = df_filt.icec.copy()
        A[A<0] = 0.0
        t = df_filt.index
        # plot forcing
        axarr[ii,0].plot(t, np.abs(oce), '-o', color=vel_cols[force], \
                label=r'$u_w$-{}'.format(force_lab[force]), linewidth=lw, markersize=ms)
        plot_angle(t, oce, ax=axarr[ii,1], lab_txt=r'$u_w$-{}'.format(force_lab[force]),\
                col=vel_cols[force], lw=lw, marker='o', ms=ms)
        axarr[ii,0].plot(t, np.abs(ice), '-o', color=vel_cols['{}_ice'.format(force)], \
                label=r'$u_i$-{}'.format(force_lab[force]), linewidth=lw, markersize=ms)
        plot_angle(t, oce, ax=axarr[ii,1], lab_txt=r'$u_i$-{}'.format(force_lab[force]),\
                col=vel_cols['{}_ice'.format(force)], lw=lw, marker='o', ms=ms)

    # plot wind
    axarr[ii,0].plot(t, np.abs(wind), '-o', color=vel_cols['wind'], label=r'2%$U_{10}$', linewidth=lw, markersize=ms)
    # plot angles
    plot_angle(t, wind, ax=axarr[ii,1], lab_txt=r'2%$U_{10}$', col=vel_cols['wind'], lw=lw, marker='o', ms=ms)
    # plot drifter
    axarr[ii,0].plot(t, np.abs(drf), '-o', color=vel_cols['drifter'], label=r'$u_o$', linewidth=lw, markersize=ms)
    # plot angles
    plot_angle(t, drf, ax=axarr[ii,1], lab_txt=r'$u_o$', col=vel_cols['drifter'], lw=lw, marker='o', ms=ms)
    for jj in range(2):
        axarr[ii,jj].axvline(x=t0, color=(0.25,0.25,0.25), linewidth=3, alpha=0.5)
        axarr[ii,jj].text(0, 1.025, '{})'.format(string.ascii_lowercase[2*ii+jj]), transform=axarr[ii,jj].transAxes, size=8, weight='bold')
        axarr[ii,0].text(0.95,0.9, ID[-5:], transform=axarr[ii,0].transAxes, size=8, weight='bold', horizontalalignment='right')
        # plot title
        #axarr[ii,0].set(title=ID[-5:])

for ax in axarr[:,-1]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
for ax in axarr[:,1]:
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set(ylabel=r'Dir [$^\circ$N]')
for ax in axarr[:,0]:
    ax.set(ylabel=r'[km/day]')
    ax2 = ax.twinx()
    ax2.spines["left"].set_position(("axes", -0.3))
    make_patch_spines_invisible(ax2)
    ax2.spines["left"].set_visible(True)
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')
    mn, mx = ax.get_ylim()
    ax2.set_ylim(mn/ms2kmday,mx/ms2kmday)
    ax2.set_ylabel('Speed [m/s]')

#axarr[0,0].set(ylabel=r'Speed [m/s]')
#axarr[1,0].set(ylabel=r'Dir [$^\circ$N]')
axarr[0,1].legend(ncol=1, loc='upper right', prop={'size':6})
fig.autofmt_xdate()
fig.savefig(os.path.join(plotDir, 'vel_forcing_{}_{}.pdf'.format('All2',resamp_freq)), bbox_inches='tight')

plt.close(fig)

