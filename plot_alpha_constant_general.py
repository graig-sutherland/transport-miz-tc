import matplotlib.pyplot as plt
import numpy as np
import pickle
import cmocean
import os, sys
import string
from matplotlib.ticker import MultipleLocator

# define some import directories
basePath = './'
dataDir = os.path.join(basePath, 'data')

# define some plot parameters
plotDir = os.path.join(basePath, 'plots')
fwidth = 140/25.4
golden = 0.5*(1.0 + np.sqrt(5.0))
fheight = fwidth*golden

# define drifters
drifter_IDs = ['RockBLOCK_14432', 'RockBLOCK_14437', 'RockBLOCK_14435', 'RockBLOCK_14438']

#rms_method = 'MAPE'
rms_method = 'MAE'

#fst_flag = '_FST'
fst_flag = '_2709'

# define forcing
force_all = ['eccc', 'topaz']
force_lab = {'eccc':'CAPS', 'topaz':'TOPAZ'}
force_col = {'eccc':plt.cm.tab10(0), 'topaz':plt.cm.tab10(1)}
kice_method_list = ['linear']

# constant values for plotting
alpha_oce_mag, alpha_oce_ang = 0.03, 0.0
alpha_ice_mag, alpha_ice_ang = 0.02, -30.0

if rms_method == 'MAPE':
    rms_levs = np.arange(25, 251, 25)
    rms_sole = 10
    rms_units = '%'
    rms_extend = 'both'
else:
    rms_levs = np.arange(10, 51, 5)
    rms_sole = 1.0
    rms_units = 'km/day'
    rms_extend = 'both'

rms_cmap = cmocean.cm.speed
line_col = (0,0,0,0.5)
ai_ticks = np.arange(-0.4, 0.1, 0.04)
aw_ticks = ai_ticks

## Read in data ##
# first read in complex ocean and ice values for MAE matrix
with open(os.path.join(dataDir, 'Fit_values_real.pickle'), 'rb') as fh:
    alpha_real_range = pickle.load(fh)
with open(os.path.join(dataDir, 'Fit_values_imag.pickle'), 'rb') as fh:
    alpha_imag_range = pickle.load(fh)

# read in MAE matrix, which is different for each drifter, forcing and ice transfer function
# matrix is organized as MAE[ice_values, ocean_values]
# do a single plot for each drifter

for kice_method in kice_method_list:
    # one figure for all drifters
    fig_mse, ax_mse = plt.subplots(nrows=len(drifter_IDs), ncols=2, sharey=True, sharex=True, figsize=(fwidth, fheight))
    fig_mse.subplots_adjust(wspace=0.05, hspace=0.15, right=0.92)#, bottom=0.08, right=0.9)
    fig2_mse, ax2_mse = plt.subplots(nrows=len(drifter_IDs), ncols=2, sharey=True, sharex=True, figsize=(fwidth, fheight))
    fig2_mse.subplots_adjust(wspace=0.05, hspace=0.15, right=0.92)#, bottom=0.08, right=0.9)
    cnt = 0
    for iid, ID in enumerate(drifter_IDs):
        print('Processing drifter {} for kice method {}'.format(ID, kice_method))
        # loop through forcing
        for iax, force in enumerate(force_all):  
            # read in MAE data for ocean constant
            with open(os.path.join(dataDir,'Fit_values_{}_oceconst_ao{:.2f}aoang{:.0f}_{}_{}_{}{}.pickle'.format(rms_method,alpha_oce_mag,alpha_oce_ang,ID,force,kice_method,fst_flag)), 'rb') as fh:
                MAE_oceconst = pickle.load(fh)
            
            # now plot (constant ocean leeway first)
            cs = ax_mse[iid,iax].contourf(alpha_real_range, alpha_imag_range, MAE_oceconst, levels=rms_levs, cmap=rms_cmap, extend=rms_extend)
            imin = np.unravel_index(MAE_oceconst.argmin(), MAE_oceconst.shape)
            ai_min, aw_min, rms_min = alpha_imag_range[imin[0]], alpha_real_range[imin[1]], MAE_oceconst.min()
            # add a 1 km/day contour
            rms2_levs = rms_min + np.array([rms_sole])
            cs2 = ax_mse[iid,iax].contour(alpha_real_range, alpha_imag_range, MAE_oceconst, \
                    levels=rms2_levs, linestyles='-', colors='k', linewidths=1)
            vals = cs2.allsegs[0][0]
            vmax, vmin = np.argmax(vals,0), np.argmin(vals,0)
            # plot
            ax_mse[iid,iax].plot([vals[vmax[0],0], vals[vmax[0],0]], [vals[vmax[0],1], alpha_imag_range.min()], color=line_col, \
                    linewidth=1, linestyle='--', label='_nolegend_')
            ax_mse[iid,iax].plot([vals[vmin[0],0], vals[vmin[0],0]], [vals[vmin[0],1], alpha_imag_range.min()], color=line_col, \
                    linewidth=1, linestyle='--', label='_nolegend_')
            ax_mse[iid,iax].plot([vals[vmax[1],0], alpha_real_range.min()], [vals[vmax[1],1], vals[vmax[1],1]], color=line_col, \
                    linewidth=1, linestyle='--', label='_nolegend_')
            ax_mse[iid,iax].plot([vals[vmin[1],0], alpha_real_range.min()], [vals[vmin[1],1], vals[vmin[1],1]], color=line_col, \
                    linewidth=1, linestyle='--', label='_nolegend_')

            lab_txt = r"$\alpha_i$ = {:.3f}, {:.1f}$^\circ$".format(np.abs(aw_min+1j*ai_min),np.angle(aw_min+1j*ai_min,deg=True))+"\n"+r"{} = {:.1f} {}".format(rms_method,rms_min,rms_units)
            ax_mse[iid,iax].plot(alpha_real_range[imin[1]], alpha_imag_range[imin[0]], 'ko', \
                    label=lab_txt)
            
            # some figure stuff
            # first annotate axis
            ax_mse[iid,iax].text(0.01, 1.025, '{})'.format(string.ascii_lowercase[cnt]), transform=ax_mse[iid,iax].transAxes,\
                    size=10, weight='bold')
            ax_mse[iid,iax].set(xlim=[alpha_real_range.min(), alpha_real_range.max()-1e-3], \
                    ylim=[alpha_imag_range.min(), alpha_imag_range.max()-1e-3],\
                    yticks=ai_ticks, xticks=aw_ticks)
            ax_mse[iid,iax].xaxis.set_minor_locator(MultipleLocator(0.01))
            ax_mse[iid,iax].yaxis.set_minor_locator(MultipleLocator(0.01))
            ax_mse[iid,iax].legend(loc='upper right', prop={'size':8})
            # yaxis labels
            if np.mod(cnt,2) == 0:
                ax_mse[iid,iax].set(ylabel=r'Crosswind')
            else:
                ax_mse[iid,iax].set(ylabel=ID[-5:])
                #ax_mse[cnt].yaxis.tick_right()
                ax_mse[iid,iax].yaxis.set_label_position("right")
            if iid == 0:
                ax_mse[iid,iax].set(title=force_lab[force])
            elif iid == len(drifter_IDs)-1:
                ax_mse[iid,iax].set(xlabel=r'Downwind')

            # read in MAE for ice constant
            with open(os.path.join(dataDir,'Fit_values_{}_iceconst_ai{:.2f}aiang{:.0f}_{}_{}_{}{}.pickle'.format(rms_method,alpha_ice_mag,alpha_ice_ang,ID,force,kice_method,fst_flag)), 'rb') as fh:
                MAE_iceconst = pickle.load(fh)
            
            # now plot (constant ice leeway second)
            cs = ax2_mse[iid,iax].contourf(alpha_real_range, alpha_imag_range, MAE_iceconst, levels=rms_levs, cmap=rms_cmap, extend=rms_extend)
            imin = np.unravel_index(MAE_iceconst.argmin(), MAE_iceconst.shape)
            ai_min, aw_min, rms_min = alpha_imag_range[imin[0]], alpha_real_range[imin[1]], MAE_iceconst.min()
            # add a 1 km/day contour
            rms2_levs = rms_min + np.array([rms_sole])
            cs2 = ax2_mse[iid,iax].contour(alpha_real_range, alpha_imag_range, MAE_iceconst, \
                    levels=rms2_levs, linestyles='-', colors='k', linewidths=1)
            vals = cs2.allsegs[0][0]
            vmax, vmin = np.argmax(vals,0), np.argmin(vals,0)
            # plot
            ax2_mse[iid,iax].plot([vals[vmax[0],0], vals[vmax[0],0]], [vals[vmax[0],1], alpha_imag_range.min()], color=line_col, \
                    linewidth=1, linestyle='--', label='_nolegend_')
            ax2_mse[iid,iax].plot([vals[vmin[0],0], vals[vmin[0],0]], [vals[vmin[0],1], alpha_imag_range.min()], color=line_col, \
                    linewidth=1, linestyle='--', label='_nolegend_')
            ax2_mse[iid,iax].plot([vals[vmax[1],0], alpha_real_range.min()], [vals[vmax[1],1], vals[vmax[1],1]], color=line_col, \
                    linewidth=1, linestyle='--', label='_nolegend_')
            ax2_mse[iid,iax].plot([vals[vmin[1],0], alpha_real_range.min()], [vals[vmin[1],1], vals[vmin[1],1]], color=line_col, \
                    linewidth=1, linestyle='--', label='_nolegend_')

            lab_txt = r"$\alpha_w$ = {:.3f}, {:.1f}$^\circ$".format(np.abs(aw_min+1j*ai_min),np.angle(aw_min+1j*ai_min,deg=True))+"\n"+r"{} = {:.1f} {}".format(rms_method,rms_min,rms_units)
            ax2_mse[iid,iax].plot(alpha_real_range[imin[1]], alpha_imag_range[imin[0]], 'ko', \
                    label=lab_txt)
            
            # some figure stuff
            # first annotate axis
            ax2_mse[iid,iax].text(0.01, 1.025, '{})'.format(string.ascii_lowercase[cnt]), transform=ax2_mse[iid,iax].transAxes,\
                    size=10, weight='bold')
            ax2_mse[iid,iax].set(xlim=[alpha_real_range.min(), alpha_real_range.max()-1e-3], \
                    ylim=[alpha_imag_range.min(), alpha_imag_range.max()-1e-3],\
                    yticks=ai_ticks, xticks=aw_ticks)
            ax2_mse[iid,iax].xaxis.set_minor_locator(MultipleLocator(0.01))
            ax2_mse[iid,iax].yaxis.set_minor_locator(MultipleLocator(0.01))
            ax2_mse[iid,iax].legend(loc='upper right', prop={'size':8})
            # yaxis labels
            if np.mod(cnt,2) == 0:
                ax2_mse[iid,iax].set(ylabel=r'Crosswind')
            else:
                ax2_mse[iid,iax].set(ylabel=ID[-5:])
                #ax_mse[cnt].yaxis.tick_right()
                ax2_mse[iid,iax].yaxis.set_label_position("right")
            if iid == 0:
                ax2_mse[iid,iax].set(title=force_lab[force])
            elif iid == len(drifter_IDs)-1:
                ax2_mse[iid,iax].set(xlabel=r'Downwind')

            cnt += 1

    # personalized colourbar
    p0 = ax_mse[-1,1].get_position().get_points().flatten()
    p1 = ax_mse[0,1].get_position().get_points().flatten()
    cbar_ax = fig_mse.add_axes([1, p0[1], 0.03, p1[3]-p0[1]])
    cbar = fig_mse.colorbar(cs, cax=cbar_ax, ticks=rms_levs)
    cbar.ax.set(ylabel='{} [{}]'.format(rms_method,rms_units))#, rotation=1)
    #fig_mse.tight_layout()
    fig_mse.savefig(os.path.join(plotDir,'{}_{}_ki-{}_sub2_fwd_kmday_oceconst_ao{:.2f}aoang{:.0f}{}.pdf'.format('All',rms_method,kice_method,alpha_oce_mag,alpha_oce_ang,fst_flag)), bbox_inches='tight')
    plt.close(fig2_mse)
    # personalized colourbar
    p0 = ax2_mse[-1,1].get_position().get_points().flatten()
    p1 = ax2_mse[0,1].get_position().get_points().flatten()
    cbar_ax = fig2_mse.add_axes([1, p0[1], 0.03, p1[3]-p0[1]])
    cbar = fig2_mse.colorbar(cs, cax=cbar_ax, ticks=rms_levs)
    cbar.ax.set(ylabel='{} [{}]'.format(rms_method,rms_units))#, rotation=0)
    #fig_mse.tight_layout()
    fig2_mse.savefig(os.path.join(plotDir,'{}_{}_ki-{}_sub2_fwd_kmday_iceconst_ai{:.2f}aiang{:.0f}{}.pdf'.format('All',rms_method,kice_method,alpha_ice_mag,alpha_ice_ang,fst_flag)), bbox_inches='tight')
    plt.close(fig2_mse)

