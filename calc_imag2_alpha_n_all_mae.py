import pandas as pd
import numpy as np
import seawater as sw
import os, sys
from utils import meridional_curvature, prime_curvature
from utils import calculate_drifter_speed_forward as drifter_speed
import pickle
import cmocean
import string

def get_kice(A, method='power', n=1, tanh_mid=0.55, tanh_mult=2*np.pi, step1=0.3, step2=0.8):
    if method == 'power':
        kice = A**n
    elif method == 'linear':
        kice = A
    elif method == 'sintef':
        kice = (A-0.3)/0.5
        kice[A<0.3] = 0.0
        kice[A>=0.8] = 1.0
    elif method == 'step':
        kice = (A-step1)/(step2-step1)
        kice[A<step1] = 0.0
        kice[A>=step2] = 1.0
    elif method == 'tanh':
        kice = np.tanh(tanh_mult*(A-tanh_mid))
        kice -= kice.min()
        kice /= (kice.max() - kice.min())
    else:
        print('Bad input method {}'.format(method))
    return kice

def calculate_error(obs, mod, method='MAE'):
    d = obs-mod
    if method == 'MAE':
        err = np.mean(np.abs(d))
    elif method == 'RMS':
        err = np.sqrt(np.mean(d*np.conj(d)))
    elif method == 'MAPE':
        err = 100.0*np.mean(np.abs(d/obs))
    else:
        sys.exit('do not understand input {}'.format(method))
    return err

## you will need to comment this out if not running on science network

basePath = './'
dataDir = os.path.join(basePath,'data')

drifter_IDs = ['RockBLOCK_14432', 'RockBLOCK_14437', 'RockBLOCK_14435', 'RockBLOCK_14438']


# define fit method
rms_method = 'MAE'

# add some flags to normalize by drift velocity
norm_flag = False
if norm_flag:
    norm_ext = '_norm'
    norm_conv = 1.0
else:
    norm_ext = ''
    norm_conv = 24*3600.0*1e-3

# calculate for different alpha for ice and ocean
alpha_oce_mag, alpha_oce_ang = 0.03, 0.0
alpha_ice_mag, alpha_ice_ang = 0.02, -30.0
alpha_ocean_constant = alpha_oce_mag * np.exp(1j*alpha_oce_ang*np.pi/180)
alpha_ice_constant = alpha_ice_mag * np.exp(1j*alpha_ice_ang*np.pi/180)

alpha_real_values = np.linspace(-0.04, 0.10, 29)
alpha_imag_values = np.linspace(-0.07, 0.07, 29)
# save pickle objects
with open(os.path.join(dataDir,'Fit_values_real.pickle'), 'wb') as fh:
    pickle.dump(alpha_real_values, fh)
with open(os.path.join(dataDir,'Fit_values_imag.pickle'), 'wb') as fh:
    pickle.dump(alpha_imag_values, fh)

ai_ticks = np.arange(-0.4, 0.1, 0.04)
aw_ticks = ai_ticks
nmax = 1
n_rng = np.unique(np.exp(np.linspace(np.log(1.0/nmax), np.log(nmax), 2*nmax + 1)))

# kice list
kice_method_list = ['linear'] #,'sintef']

# make time limits
tmax = pd.to_datetime('27-Sep-2018 00:00:00')
tmax_string = '_2709'
# just read in data for each
force_all = ['eccc', 'topaz']
force_lab = {'eccc':'CAPS', 'topaz':'TOPAZ'}
dd_obs = {}
for force in force_all:
    ## some changes for eccc
    with open(os.path.join(dataDir, 'drifter_{}.pickle'.format(force)), 'rb') as f:
            dd_obs.update({force:pickle.load(f, fix_imports=True, encoding="latin1")})

for kice_method in kice_method_list:
    # one figure for all drifters
    for iid, ID in enumerate(drifter_IDs):
        print('Processing drifter {} for kice method {}'.format(ID, kice_method))
        # loop through forcing
        for iax, force in enumerate(force_all):  
            MAE_iceconst = np.zeros([len(alpha_imag_values), len(alpha_real_values)])
            MAE_oceconst = np.zeros_like(MAE_iceconst)
            w = dd_obs[force][ID].u + 1j*dd_obs[force][ID].v
            velf = drifter_speed(dd_obs[force][ID])
            wf = velf[:,0] + 1j*velf[:,1]
            wf /= 3.6 # convert km/h -> m/s
            dd_obs[force][ID] = dd_obs[force][ID].assign(wf = pd.Series(wf).values)

            # filter by time
            dd_obs[force][ID] = dd_obs[force][ID][dd_obs[force][ID].index <= tmax]
            
            # define some parameters
            obs = dd_obs[force][ID].wf
            A = dd_obs[force][ID].icec
            # make sure there are no negative ice concentrations
            A[A<0] = 0.0
            kice = get_kice(A, kice_method)
            oce_vel = dd_obs[force][ID].uuw + 1j*dd_obs[force][ID].vvw
            wind_vel = dd_obs[force][ID].uu + 1j*dd_obs[force][ID].vv
            ice_vel = dd_obs[force][ID].uui + 1j*dd_obs[force][ID].vvi
            if norm_flag:
                norm_vel = 1.0/obs
            else:
                norm_vel = 1.0 +1j*0.0
            # normalize
            obs *= norm_vel
            oce_vel *= norm_vel
            wind_vel *= norm_vel
            ice_vel *= norm_vel
            # loop through real and imaginary for wind
            for ii, alpha_imag in enumerate(alpha_imag_values):
                for jj, alpha_real in enumerate(alpha_real_values):
                    alpha_var = alpha_real + 1j*alpha_imag
                    mod_iceconst = kice*(ice_vel + alpha_ice_constant*wind_vel) + \
                            (1.0-kice)*(oce_vel + alpha_var*wind_vel)
                    MAE_iceconst[ii,jj] = norm_conv*np.mean(np.abs(mod_iceconst-obs))
                    mod_oceconst = kice*(ice_vel + alpha_var*wind_vel) + \
                            (1.0-kice)*(oce_vel + alpha_ocean_constant*wind_vel)
                    MAE_oceconst[ii,jj] = norm_conv*np.mean(np.abs(mod_oceconst-obs))

            ## save MAE and ice values
            with open(os.path.join(dataDir,'Fit_values_MAE_iceconst_ai{:.2f}aiang{:.0f}_{}_{}_{}{}{}.pickle'.format(alpha_ice_mag,alpha_ice_ang,ID,force,kice_method,tmax_string,norm_ext)), 'wb') as fh:
                pickle.dump(MAE_iceconst, fh)
            with open(os.path.join(os.path.join(dataDir,'Fit_values_MAE_oceconst_ao{:.2f}aoang{:.0f}_{}_{}_{}{}{}.pickle'.format(alpha_oce_mag,alpha_oce_ang,ID,force,kice_method, tmax_string, norm_ext))), 'wb') as fh:
                pickle.dump(MAE_oceconst, fh)
            
