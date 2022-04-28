import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cmocean
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import sys
import string

basePath = './'
dataDir = os.path.join(basePath,'data')
topazDir = os.path.join(dataDir, 'topaz')
capsDir = os.path.join(dataDir, 'caps')

alph = 0.85
cmap = plt.cm.Dark2
drifter_IDs = ['RockBLOCK_14432', 'RockBLOCK_14437', 'RockBLOCK_14435', 'RockBLOCK_14438']
drifter_cols = {'RockBLOCK_14438':cmap(3, alpha=alph), 'RockBLOCK_14435':cmap(0, alpha=alph),\
                'RockBLOCK_14437':cmap(1, alpha=alph), 'RockBLOCK_14432':cmap(2, alpha=alph)}

# read in drifter data
with open(os.path.join(dataDir,'drifter_topaz.pickle'), 'rb') as f:
   dd_obs_topaz = pickle.load(f, fix_imports=True, encoding="latin1")
with open(os.path.join(dataDir,'drifter_eccc.pickle'), 'rb') as f:
   dd_obs_eccc = pickle.load(f, fix_imports=True, encoding="latin1")

# some plot stuff
plotDir = os.path.join(basePath, 'plots')
if not os.path.exists(plotDir):
   os.makedirs(plotDir)

# define the times for plotting
time0 = pd.to_datetime('2018-09-20')
time1 = pd.to_datetime('2018-09-23')
time2 = pd.to_datetime('2018-09-26')
time_rng = np.array([time0,time1,time2])

# define the domain domain
#lat_min, lat_max = 80, 83.5
lat_min, lat_max = 80, 83
lon_min, lon_max = 8, 26
lat_mean = 0.5*(lat_min + lat_max)
lon_mean = 0.5*(lon_min + lon_max)
# xlocs and ylocs for ticks
dlon, dlat = 4.0, 0.5
xlocs = np.arange(np.floor(lon_min), np.ceil(lon_max)+dlon, dlon)
ylocs = np.arange(np.floor(lat_min), np.ceil(lat_max)+dlat, dlat)

# plot stuff
golden = 0.5*(1.0+np.sqrt(5.0))
fwidth = 140/25.4
fheight = fwidth*2/golden

# cartopy stuff
proj = ccrs.Mercator(central_longitude=lon_mean, min_latitude=lat_min, max_latitude=lat_max, latitude_true_scale=lat_mean)
src_latlon = ccrs.PlateCarree()
land_50 = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor=cfeature.COLORS['land'])

## create plot
fig, axarr = plt.subplots(nrows=len(time_rng), ncols=2, sharex=True, sharey=True, figsize=(fwidth,fheight), subplot_kw={'projection':proj})
fig.subplots_adjust(wspace=0.0, hspace=0.1, top=1.0, bottom=0.075)

# some dictionary to offset labels
lon_off = {'RockBLOCK_14432':0.5, 'RockBLOCK_14437':0.5, 'RockBLOCK_14435':0.5, 'RockBLOCK_14438':0.5}
lat_off = {'RockBLOCK_14432':0.0, 'RockBLOCK_14437':-0.04, 'RockBLOCK_14435':-0.07, 'RockBLOCK_14438':-0.1}


## lets loop through the times
for itime, time in enumerate(time_rng):
   year, month, day, hour = time.year, time.month, time.day, time.hour
   if hour < 12:
      sdate = '{:04d}{:02d}{:02d}00_{:03d}'.format(year,month,day,hour)
   else:
      sdate = '{:04d}{:02d}{:02d}12_{:03d}'.format(year,month,day,hour-12)

   print('Calculating for {}'.format(sdate))
   # read in data
   with open(os.path.join(capsDir, 'eccc_gl_{}.pickle'.format(time.strftime('%Y%m%d%H%M%S'))), 'rb') as fh:
       lon2, lat2, gl2 = pickle.load(fh)
   with open(os.path.join(topazDir, 'topaz_gl_{}.pickle'.format(time.strftime('%Y%m%d%H%M%S'))), 'rb') as fh:
       topaz_lon, topaz_lat, tgl = pickle.load(fh)

   # start plotting
   ax0 = axarr[itime, 0]
   ax1 = axarr[itime, 1]
   # plot CAPS ice
   ax0.coastlines()
   ax0.set_extent((lon_min, lon_max, lat_min, lat_max), src_latlon)
   ax1.coastlines()
   ax1.set_extent((lon_min, lon_max, lat_min, lat_max), src_latlon)
   gln0 = ax0.gridlines(draw_labels=True, xlocs=xlocs, ylocs=ylocs)
   gln0.xlabels_top = False
   gln0.ylabels_right = False
   gln0.xformatter = LONGITUDE_FORMATTER
   gln0.yformatter = LATITUDE_FORMATTER
   gln0.xlabel_style = {'size':8}
   gln0.ylabel_style = {'size':8}
   gln1 = ax1.gridlines(draw_labels=True, xlocs=xlocs, ylocs=ylocs)
   gln1.xlabels_top = False
   gln1.ylabels_left = False
   gln1.xformatter = LONGITUDE_FORMATTER
   gln1.yformatter = LATITUDE_FORMATTER
   gln1.xlabel_style = {'size':8}
   gln1.ylabel_style = {'size':8}
   # no xlabels except when the end
   if itime < len(time_rng)-1:
       gln0.xlabels_bottom = False
       gln1.xlabels_bottom = False

   # plot ice
   Qice = ax0.contourf(lon2, lat2, gl2, np.linspace(0, 1, 11), transform=src_latlon, cmap=cmocean.cm.ice)

   # plot ice TOPAZ
   Qice_tz = ax1.contourf(topaz_lon, topaz_lat, tgl, np.linspace(0, 1, 11), transform=src_latlon, cmap=cmocean.cm.ice)

   # plot trajectories
   for ID in drifter_IDs:
      ind0 = np.argmin(np.abs(dd_obs_eccc[ID].index - time))
      for a in (ax0, ax1):
         # first plot the names
         # plot trajectories
         a.plot(dd_obs_eccc[ID].Longitude, dd_obs_eccc[ID].Latitude, '-', \
              color=drifter_cols[ID], transform=src_latlon, label='_nolegend_', linewidth=1.2)
         if np.abs(dd_obs_eccc[ID].index[ind0]-time) < pd.to_timedelta('6H'):
            a.plot(dd_obs_eccc[ID].Longitude[ind0], dd_obs_eccc[ID].Latitude[ind0], 'o', markersize=5,\
               color=drifter_cols[ID], alpha=1.0, markeredgecolor='k', transform=src_latlon, label=ID[-5:])
   if itime == -1:
      ax0.set_title('CAPS', fontsize=10, weight='normal')
      ax1.set_title('TOPAZ', fontsize=10, weight='normal')
   # set labels
   ax0.text(0, 1.025, '{})'.format(string.ascii_lowercase[itime*2]), transform=ax0.transAxes, size=8, weight='bold')
   ax1.text(0, 1.025, '{})'.format(string.ascii_lowercase[itime*2+1]), transform=ax1.transAxes, size=8, weight='bold')
   #plt.suptitle(time, fontsize=10)
   
   ax0.text(0.5, 1.025, 'CAPS: {}'.format(time.strftime('%Y-%m-%d')), transform=ax0.transAxes, size=8, horizontalalignment='center')
   ax1.text(0.5, 1.025, 'TOPAZ: {}'.format(time.strftime('%Y-%m-%d')), transform=ax1.transAxes, size=8, horizontalalignment='center')

   for a in (ax0, ax1):
      #a.text(0.05, 0.9, time.strftime('%Y-%m-%d'), transform=a.transAxes, size=8, horizontalalignment='left')
      a.legend(loc='upper center', ncol=2, prop={'size':8}, bbox_to_anchor=(0.5,1.02))
   #ax1.legend(loc='upper right', ncol=2, prop={'size':8})

   # add features
   ax0.add_feature(land_50)
   ax1.add_feature(land_50)

# add colorbar at bottom
p0 = ax0.get_position().get_points().flatten()
p1 = ax1.get_position().get_points().flatten()
cbar_ax = fig.add_axes([p0[0], 0, p1[2]-p0[0], 0.03])
cbar = fig.colorbar(Qice, cax=cbar_ax, orientation='horizontal', ticks=np.linspace(0,1,11))
cbar.ax.set(xlabel=r'$A$')
# save figure
fig.savefig(os.path.join(plotDir, 'drift_tracks_cartoon.pdf'.format(itime)), bbox_inches='tight')
plt.close(fig)
