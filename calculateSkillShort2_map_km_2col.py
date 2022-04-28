import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import string
from utils import lonlat2xykm as ll2xy
import os
import sys
'''
A script to plot obs and simulated trajectories in km
'''

# input base directory
basePath = './'
plotDir = os.path.join(basePath, 'plots')
#plotDir = os.path.join('/home/gsu000/public_html/Nansen_icedrift_TC')
if not os.path.exists(plotDir):
    print('{} does not exist. Making it.'.format(plotDir))
    os.makedirs(plotDir)
dataDir = os.path.join(basePath, 'data')
expDir = os.path.join(basePath, 'data')

# make list of known drifters from obs
drifter_IDs = ['RockBLOCK_14432', 'RockBLOCK_14437', 'RockBLOCK_14435', 'RockBLOCK_14438']

# make experiment array

experiments = ['nostokes_ao3.0_aoang0_ai2.0aiang-30_interp_kiLINEAR_caps_orig', \
               'nostokes_ao3.0_aoang0_ai0.0aiang0_interp_kiSINTEF_caps_orig', \
               'nostokes_ao3.0_aoang0_ai0.0aiang0_interp_kiICE_caps_orig', \
               'nostokes_ao3.0_aoang0_ai0.0aiang0_interp_kiOCEAN_caps_orig', \
               'nostokes_ao3.0_aoang0_ai2.0aiang-30_interp_kiLINEAR_topaz', \
               'nostokes_ao3.0_aoang0_ai0.0aiang0_interp_kiSINTEF_topaz', \
               'nostokes_ao3.0_aoang0_ai0.0aiang0_interp_kiICE_topaz', \
               'nostokes_ao3.0_aoang0_ai0.0aiang0_interp_kiOCEAN_topaz']

fileName = '2018092000_000.csv'
start_times = pd.date_range('2018-09-20', '2018-09-23', freq='2D')
# resample frequency (same as MLDP output frequency)
resamp_freq = '60T'

markers = {'LINEAR':'^', 'SINTEF':'v', 'vector':'d', 'scalar':'o', '1D':'o', '2D':'d', 'ICE':'s', 'OCEAN':'o'}
lstys = {'LINEAR':'-', 'SINTEF':'-', 'vector':'-', 'scalar':'-', '1D':'-','2D':'-'}
lstys_time = {24:'-', 48:'-'}
mcols_cmap = plt.cm.Dark2
mcols = {'eccc':mcols_cmap(0), 'topaz':mcols_cmap(1), \
        'eccc-LINEAR':mcols_cmap(0), 'eccc-SINTEF':mcols_cmap(1),\
        'topaz-LINEAR':mcols_cmap(2), 'topaz-SINTEF':mcols_cmap(3),\
        'eccc-LINEAR-2D':mcols_cmap(1),'topaz-LINEAR-2D':mcols_cmap(3)}
mcols.update({'LINEAR':mcols_cmap(0), 'SINTEF':mcols_cmap(1), 'ICE':mcols_cmap(2), 'OCEAN':mcols_cmap(3)})
# define constant to convert m/s -> km/day
ms2kd = 3600.0*24*1e-3
force_libs = {'caps_orig':'eccc', 'topaz':'topaz'}
icemodel_libs = {'LINEAR':'linear', 'SINTEF':'80/30', 'ICE':'ice', 'OCEAN':'ocean'}
force_col = {0:'CAPS', 1:'TOPAZ'}

# read in obs time and position
df_obs = {}
for ID in drifter_IDs:
   for x in os.listdir(dataDir):
      if x.startswith(ID) and x.endswith('.csv'):
         driftFile = x
         print('Found file {}'.format(driftFile))
   # read in obs
   df_obs.update({ID:pd.read_csv(os.path.join(dataDir, driftFile))})
   df_obs[ID]['Timestamp'] = pd.to_datetime(df_obs[ID]['Timestamp'])
   df_obs[ID].set_index('Timestamp', inplace=True)
   # interpolate to same 60 minute grid
   df_obs[ID] = df_obs[ID].resample(resamp_freq).mean().interpolate(method='time')

# read mldp dataframes
df_exp = {}
for exp in experiments:
    df_exp.update({exp:pd.read_csv(os.path.join(expDir,exp,fileName))})
    df_exp[exp].set_index('drifter ID', inplace=True)
    df_exp[exp]['Timestamp'] = pd.to_datetime(df_exp[exp]['Date-Time (UTC)'])

# quickly get lat/lon dimensions for each time
df_latlon = {}
dl = 0.2 # buffer around min/max
for t0 in start_times:
    lat_min, lat_max = 90.0, 0.0
    lon_min, lon_max = 180.0, -180.0
    # get a string version to check with MLDP experiment (format is drifter_ID+'_'+st0)
    time_array = pd.date_range(t0+pd.to_timedelta('1H'),t0+pd.to_timedelta('48H'),freq='1H')
    for ID in drifter_IDs:
        dfo = df_obs[ID].loc[time_array]
        if dfo.Latitude.max()>lat_max:
            lat_max = dfo.Latitude.max()
        if dfo.Latitude.min()<lat_min:
            lat_min = dfo.Latitude.min()
        if dfo.Longitude.max()>lon_max:
            lon_max = dfo.Longitude.max()
        if dfo.Longitude.min()<lon_min:
            lon_min = dfo.Longitude.min()
    # add offset
    lon_min -= dl
    lon_max += dl
    lat_min -= dl
    lat_max += dl
    # write to dictionary
    df_latlon.update({t0:{'lon_min':lon_min, 'lon_max':lon_max, 'lat_min':lat_min, 'lat_max':lat_max}})

# cartopy and plot stuff
fwidth = 140.0/25.4
golden = 0.5*(1.0+np.sqrt(5.0))
fheight = 1.16*fwidth/golden

lw = 1.0
ms = 2
ind_sub = [0, 23, 47]
ntime = len(start_times)
ndrift = len(drifter_IDs)

cmap = plt.cm.Dark2

# plot a 2x2 subplot with each subplot a particular drifter with each experiment within each subplot
fig, ax = plt.subplots(nrows=ndrift, ncols=4, sharex=True, sharey=True, figsize=(fwidth,fheight))
for ii, t0 in enumerate(start_times):
    # going to make subplot for each 
    # get a string version to check with MLDP experiment (format is drifter_ID+'_'+st0)
    time_array = pd.date_range(t0+pd.to_timedelta('1H'),t0+pd.to_timedelta('48H'),freq='1H')
    st0 = t0.strftime('%Y%m%d%H%M')
    # create the figure object
    lat_min, lat_max = df_latlon[t0]['lat_min'], df_latlon[t0]['lat_max']
    lon_min, lon_max = df_latlon[t0]['lon_min'], df_latlon[t0]['lon_max']
    lon_mean = 0.5*(lon_min+lon_max)
    lat_mean = 0.5*(lat_min+lat_max)
    # add time stamp
    stlab = t0.strftime('%Y-%m-%d')
    ax[0,2*ii].text(0.5, 0.05, r'$t_0$ = {}'.format(stlab), transform=ax[0,2*ii].transAxes, size=7, horizontalalignment='center')
    ax[0,2*ii+1].text(0.5, 0.05, r'$t_0$ = {}'.format(stlab), transform=ax[0,2*ii+1].transAxes, size=7, horizontalalignment='center')
    #ax[ii,-1].text(1.075, 0.5, stlab, transform=ax[ii,-1].transAxes, size=8, horizontalalignment='center', verticalalignment='center', weight='bold', rotation=270)
    # loop through drifters
    for jj, ID in enumerate(drifter_IDs):
        # plot name on right hand side for first time
        if ii == 0:
            ax[jj,-1].text(1.075, 0.5, ID[-5:], transform=ax[jj,-1].transAxes, size=7, horizontalalignment='center', verticalalignment='center', weight='bold', rotation=270)
        #ax.update({ii:plt.subplot(2, 2, ii+1, projection=src_latlon)})
        #ax[ii].set_extent([lon_min,lon_max,lat_min,lat_max], src_latlon)
        dfo = df_obs[ID].loc[time_array]
        # change lon and lat mean to first point of trajectory
        #lon_mean, lat_mean = dfo.Longitude[0], dfo.Latitude[0]
        xo, yo = ll2xy(dfo.Longitude, dfo.Latitude, lon_mean, lat_mean)
        # plot for each column
        ioff = 2*ii
        for k in range(2):
            ax[jj,k+ioff].plot(xo,yo, 'k-', label='obs', linewidth=lw)
            ax[jj,k+ioff].plot(xo[ind_sub], yo[ind_sub], 'ko', markersize=ms)

        # loop through experiments and plot each
        for exp in experiments:
            dfe = df_exp[exp].loc['{}_{}'.format(ID,st0)]
            # change the index for this subset
            dfe.set_index('Timestamp', inplace=True)
            # plot the same dates as in time array
            #ff = force_libs[exp[37:]]
            #icemodel = icemod_libs[exp[30:36]]
            if 'caps_orig' in exp:
                ff = force_libs['caps_orig']
                icol = 2*ii
            elif 'topaz' in exp:
                ff = force_libs['topaz']
                icol=2*ii + 1
            else:
                print('{} is in wrong format'.format(exp))
                sys.exit()

            for mods in icemodel_libs.keys():
                if mods in exp:
                    icemodel = icemodel_libs[mods]
                    # define colour by transfer model
                    col = mcols[mods]
            
            dfe = dfe.loc[time_array]
            xe, ye = ll2xy(dfe.Longitude, dfe.Latitude, lon_mean, lat_mean)
            h, = ax[jj,icol].plot(xe, ye, '-', label='{}'.format(icemodel), color=col, alpha=0.5, linewidth=lw)
            ax[jj,icol].plot(xe[ind_sub], ye[ind_sub],'o', color=h.get_color(), markersize=ms)


# some labelling stuff
for a in ax[:,0]:
    a.set_ylabel(r'$y$ / km')
for a in ax[-1,:]:
    a.set_xlabel(r'$x$ / km')
for a in ax[0,[0,2]]:
    a.text(0.5,1.025, 'CAPS', transform=a.transAxes, size=7, weight='bold', horizontalalignment='center')
for a in ax[0,[1,3]]:
    a.text(0.5,1.025, 'TOPAZ', transform=a.transAxes, size=7, weight='bold', horizontalalignment='center')
# add a legend
ax[0,1].legend(prop={'size':7}, loc='lower center',ncol=5,  bbox_to_anchor=(1.05, 1.2))

# make axis labels smaller
for ii, a in enumerate(ax.flatten()):
    a.text(0, 1.025, '{})'.format(string.ascii_lowercase[ii]), transform=a.transAxes, size=7, weight='bold')
    a.tick_params(axis='both', which='major', labelsize=8)
    a.set_aspect('equal')
#    #a.xaxis.get_label().set_size(8)
#    #a.yaxis.get_label().set_size(8)


# save figure
fig.savefig(os.path.join(plotDir, 'Drifters_all_equal.pdf'), bbox_inches='tight')
plt.close(fig)


