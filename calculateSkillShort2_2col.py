import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import pickle
import string
from skillFunctions import calculateSkillScores, calculate_skill_score

'''
A script to plot the separation distance at 48 hour lead time
'''

def interp_time(ts, target):
    if target in ts.index:
        return ts[target]
    ts1 = ts.sort_index()
    b = (ts1.index > target).argmax() # index of first entry after target
    s = ts1.iloc[b-1:b+1]
    # Insert empty value at target time.
    s = s.reindex(pd.to_datetime(list(s.index.values) + [pd.to_datetime(target)]))
    return s.interpolate(method='time').loc[target]

# input base directory
basePath = './'
#plotDir = os.path.join(basePath, 'plots')
plotDir = os.path.join(basePath,'plots')
if not os.path.exists(plotDir):
    print('{} does not exist. Making it.'.format(plotDir))
    os.makedirs(plotDir)
dataDir = os.path.join(basePath, 'data')
expDir = os.path.join(basePath, 'data')

# make list of known drifters from obs
drifter_IDs = ['RockBLOCK_14432', 'RockBLOCK_14437', 'RockBLOCK_14435','RockBLOCK_14438']

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
# resample frequency (same as MLDP output frequency)
resamp_freq = '60T'
hours_per_sample = 1.0

# make the meanSkill 
readSkill = True
skillFile = os.path.join(dataDir, 'MeanSkill_all_{}.pickle'.format(resamp_freq))
min_points = 12 # have at least min_points to calc skill

## loop through drifters
df_force = {}
force_all = ['eccc', 'topaz']

# pickle files are in python 2
for force in force_all:
    with open(os.path.join(dataDir, 'drifter_{}.pickle'.format(force)), 'rb') as fh:
        df_force.update({force:pickle.load(fh, fix_imports=True, encoding="latin1")})
# old way
df_obs = {}
for ID in drifter_IDs:
   for x in os.listdir(dataDir):
      if x.startswith(ID):
         driftFile = x
         print('Found file {}'.format(driftFile))
   # read in obs
   df_obs.update({ID:pd.read_csv(os.path.join(dataDir, driftFile))})
   df_obs[ID]['Timestamp'] = pd.to_datetime(df_obs[ID]['Timestamp'])
   df_obs[ID].set_index('Timestamp', inplace=True)
   # interpolate to same 15 minute grid
   df_obs[ID] = df_obs[ID].resample(resamp_freq).mean().interpolate(method='time')

if readSkill:
    # now loop through experiments to test
    meanSkill = {}
    for iexp, exp in enumerate(experiments):
    
       # read in simulated drift to data frame
       df_exp = pd.read_csv(os.path.join(expDir, exp, fileName))
       df_exp.set_index('drifter ID', inplace=True)
       df_exp['Timestamp'] = pd.to_datetime(df_exp['Date-Time (UTC)'])
       
       # get a list of unique ids
       unique_IDs = df_exp.index.unique()
       # get number of good points for comp
       ncomp = len(df_exp.loc[unique_IDs[0]].Latitude)
       nids = len(unique_IDs)
    
       # make a dictionary of mean skill values
       meanSkill.update({exp:{\
                    'drifter': [],\
                    'len': np.zeros([nids,]),\
                    'sep': np.zeros([ncomp, nids]),\
                    'disp': np.zeros([ncomp, nids]),\
                    'disp_obs': np.zeros([ncomp, nids]),\
                    'liu': np.zeros([ncomp, nids]),\
                    'molcard': np.zeros([ncomp, nids]),\
                    'toner': np.zeros([ncomp, nids]),\
                    'gjs': np.zeros([ncomp, nids]),\
                    'ratio': np.zeros([ncomp, nids])\
                    }})
    
       # loop through unique IDs and do a comparison with observations
       for nid, driftID in enumerate(unique_IDs):
          # find drifter to compare with observations
          for ID in drifter_IDs:
             if driftID.startswith(ID):
                meanSkill[exp]['drifter'].append(driftID)
                if nid < len(drifter_IDs):
                    time_start = df_exp.loc[driftID].Timestamp.min() #- pd.to_timedelta(resamp_freq)
                else:
                    time_start = df_exp.loc[driftID].Timestamp.min()
                time_stop = df_exp.loc[driftID].Timestamp.max()
                df_obs_exp = df_obs[ID][np.logical_and(df_obs[ID].index>=time_start, df_obs[ID].index<=time_stop)]
    
                if iexp == 0:
                    print('{} has {} records'.format(driftID, len(df_obs_exp)-1))
          
                # check that the length is sufficient
                nobs = len(df_obs_exp)
                nexp = len(df_exp.loc[driftID])
                if nobs < nexp:
                    npts = nobs
                else:
                    npts = nexp
                rng = np.arange(npts)
                if npts > min_points:
                   print('Calculating skill scores')
                   # calculate skill score
                   #liu, molcard, _, gjs, ratio, sep, disp_mod, disp_obs, dist_obs, dist_mod = calculateSkillScores(df_obs_exp, df_exp.loc[driftID])
                   d_obs = df_obs_exp.iloc[rng]
                   d_exp = df_exp.loc[driftID].iloc[rng]
                   liu = calculate_skill_score(d_obs, d_exp, method='liu')
                   gjs = calculate_skill_score(d_obs, d_exp, method='area')
                   molcard = calculate_skill_score(d_obs, d_exp, method='molcard')
                   toner = calculate_skill_score(d_obs, d_exp, method='toner')
                   diffus = calculate_skill_score(d_obs, d_exp, method='dispersion')
                   sep = calculate_skill_score(d_obs, d_exp, method='sep')
    
                   # add to dictionary with accumulated values
                   npts = len(liu)
                   rng = np.arange(npts)
                   meanSkill[exp]['len'][nid] = npts
                   meanSkill[exp]['sep'][rng, nid] = sep
                   meanSkill[exp]['liu'][rng, nid] = liu
                   meanSkill[exp]['gjs'][rng, nid] = gjs
                   meanSkill[exp]['molcard'][rng, nid] = toner
                   meanSkill[exp]['toner'][rng, nid] = molcard
                   meanSkill[exp]['ratio'][rng, nid] = diffus
                else:
                   print('len of df_obs is {} and not {}'.format(len(df_obs_exp),ncomp+1))
    # writ meanSkill dict to pickle
    with open(skillFile, 'wb') as f:
        pickle.dump(meanSkill, f)
else:
    with open(skillFile, 'rb') as f:
        meanSkill = pickle.load(f)
# make a plot
fwidth = 140.0/25.4
fheight = fwidth/1.618

for e in meanSkill.keys():
    ncomp, nids = meanSkill[e]['sep'].shape
    break
# forecast lead time
fc_lead = np.arange(ncomp)*hours_per_sample + 0.5

fc_max = 5*24

indexes = [48] # indices to compare (hourly output)
#markers = {'eccc':'o', 'topaz':'s'}
#mcols = {'LINEAR':plt.cm.tab10(0), 'SINTEF':plt.cm.tab10(1)}
falpha = 0.5
markers = {'LINEAR':'^', 'SINTEF':'v', 'vector':'d', 'scalar':'o', '1D':'o', '2D':'d', 'ICE':'s', 'OCEAN':'o'}
markers_time = {24:'o', 48:'s'}
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
icemod_libs = {'LINEAR':'linear', 'SINTEF':'80/30', 'ICE':'ice', 'OCEAN':'ocean'}

# one figure for each drifter
for ind in indexes:
    figd, axd = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=True, figsize=(fwidth,2*fheight))
    for dd, ID in enumerate(drifter_IDs):
        # loop through experiments
        for exp in experiments:
            ## these are for old format
            #ff = force_libs[exp[37:]]
            if 'caps_orig' in exp:
                ff = force_libs['caps_orig']
                icol = 0
            elif 'topaz' in exp:
                ff = force_libs['topaz']
                icol = 1
            else:
                print('{} is in wrong format'.format(exp))
                sys.exit()
            if 'kiLINEAR' in exp:
                icemodel = 'LINEAR'
            elif 'kiSINTEF' in exp:
                icemodel = 'SINTEF'
            elif 'kiICE' in exp:
                icemodel = 'ICE'
            elif 'kiOCEAN' in exp:
                icemodel = 'OCEAN'
            else:
                print('{} is in wrong format'.format(exp))
                sys.exit()
            
            mm = markers[icemodel]
            df = df_force[ff][ID]
            cc = mcols[icemodel]
            lsty = '-'
            plotLab = True
            nd = 1.0 
            dave = []
            for ii in range(nids):
                if meanSkill[exp]['drifter'][ii].startswith(ID):
                    tt = pd.to_datetime(meanSkill[exp]['drifter'][ii][16:])
                    time_obs = tt + pd.to_timedelta(ind, unit='H')
                    df_int = interp_time(df_force[ff][ID], time_obs)
                    icec = df_int.icec
                    if icec < 0:
                        icec = 0.0
                    t0 = datetime(tt.year,tt.month,tt.day,tt.hour,tt.minute)
                    d = meanSkill[exp]['sep'][ind,ii]
                    nd += 1
                    dave.append(d)
                    if plotLab:
                        labs = icemod_libs[icemodel]
                        axd[dd,icol].plot(t0, d, '.', marker=mm, color=cc, label=labs, alpha=falpha)
                        plotLab = False
                    else:
                        axd[dd,icol].plot(t0, d, '.', marker=mm, color=cc, alpha=falpha)

            # output mean and standard deviation
            print('{}-{}h {}'.format(ind,ID[-5:],exp))
            print('  mean +- std ={:.1f} +- {:.1f}'.format(np.mean(dave),np.std(dave)))

        #axd[dd].set_title(ID[-5:], fontdict={'fontsize': 8, 'fontweight': 'medium'})
        axd[dd,0].set_ylabel(r'$d$ / km')
        axd[dd,0].text(0, 1.025, '{})'.format(string.ascii_lowercase[2*dd]), transform=axd[dd,0].transAxes, size=8, weight='bold')
        axd[dd,1].text(0, 1.025, '{})'.format(string.ascii_lowercase[2*dd + 1]), transform=axd[dd,1].transAxes, size=8, weight='bold')
        axd[dd,0].text(0.5, 1.025, '{}'.format('CAPS'), transform=axd[dd,0].transAxes, size=10, weight='bold', horizontalalignment='center')
        axd[dd,1].text(0.5, 1.025, '{}'.format('TOPAZ'), transform=axd[dd,1].transAxes, size=10, weight='bold', horizontalalignment='center')
        for a in axd[dd,:]:
            a.text(0.99, 1.025, '{}'.format(ID[-5:]), transform=a.transAxes, size=8, horizontalalignment='right')

        
        axd[0,0].legend(prop={'size':6}, ncol=1, loc='upper left')
    
    figd.autofmt_xdate()
    figd.savefig(os.path.join(plotDir, 'AllDrifters_sep_{}h_2col.pdf'.format(ind)), bbox_inches='tight')
    plt.close(figd)
