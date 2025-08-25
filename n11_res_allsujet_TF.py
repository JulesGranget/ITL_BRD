
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import seaborn as sns
import gc

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

debug = False







#########################
######## PLOT TF ########
#########################



#chan = chan_list_eeg_short[0]
def save_tf_allsujet():

    data_diff_allchan = []

    stats_allcond_allchan = []
    stats_allcond_mne_allchan = []

    for chan_i, chan in enumerate(chan_list_eeg_short):

        print(f'#### COMPUTE TF {chan} ####', flush=True)

        #### load data
        print('#### LOAD BASELINE ####', flush=True)

        os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH'))

        tf_stretch_baseline_allsujet = np.zeros((len(sujet_list_FC), nfrex, stretch_point_ERP))

        #sujet_i, sujet = 46, sujet_list_FC[46]
        for sujet_i, sujet in enumerate(sujet_list_FC):

            tf_stretch_baseline_allsujet[sujet_i,:,:] = np.load(f'{sujet}_VS_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]

        print('#### LOAD COND ####', flush=True)

        tf_stretch_cond_allsujet = np.zeros((len(sujet_list_FC), nfrex, stretch_point_ERP))

        #sujet_i, sujet = 0, sujet_list_FC[47]
        for sujet_i, sujet in enumerate(sujet_list_FC):

            tf_stretch_cond_allsujet[sujet_i,:,:] = np.load(f'{sujet}_CHARGE_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]

        data_diff_allchan.append(np.median(tf_stretch_cond_allsujet - tf_stretch_baseline_allsujet, axis=0))
        
        #### load data thresh
        os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH_STATS'))
        
        stats_allcond_allchan.append(np.load(f'{chan}_allsujet_tf_STATS.npy'))
        stats_allcond_mne_allchan.append(np.load(f'{chan}_allsujet_tf_STATS_MNE.npy'))

    data_diff_allchan = np.array(data_diff_allchan)
    stats_allcond_allchan = np.array(stats_allcond_allchan)
    stats_allcond_mne_allchan = np.array(stats_allcond_mne_allchan)

    #### scale    
    vlim = np.abs(np.array([data_diff_allchan.min(), data_diff_allchan.max()])).max()

    vec_phase = [125, 375, 625, 875] 

    for chan_i, chan in enumerate(chan_list_eeg_short):

        print(f"plot:{chan}")

        for stat_type in ['HOMEMADE', 'MNE']:

            #### plot 
            fig, ax = plt.subplots()

            plt.suptitle(f'{chan} tf allsujet count:{len(sujet_list_FC)}')

            fig.set_figheight(5)
            fig.set_figwidth(8)

            #### generate time vec
            time_vec = np.arange(stretch_point_ERP)

            #### plot
            pcm = ax.pcolormesh(time_vec, frex, data_diff_allchan[chan_i], vmin=-vlim, vmax=vlim, shading='gouraud', cmap=plt.get_cmap('seismic'))
            # ax.pcolormesh(time_vec, frex, data_allcond[cond], shading='gouraud', cmap=plt.get_cmap('seismic'))
            ax.set_yscale('log')

            #### stats
            if stat_type == 'HOMEMADE':
                ax.contour(time_vec, frex, stats_allcond_allchan[chan_i], levels=0, colors='g')
            else:
                ax.contour(time_vec, frex, stats_allcond_mne_allchan[chan_i], levels=0, colors='g')

            # ax.vlines(stretch_point_ERP/2, ymin=frex[0], ymax=frex[-1], colors='g')
            ax.vlines(vec_phase, ymin=frex[0], ymax=frex[-1], colors='k')
            ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
            ax.set_ylim(frex[0], frex[-1])

            cbar = fig.colorbar(pcm, ax=ax, orientation='vertical')
            cbar.set_label('robust z-score')  # <-- Change label to fit your data

            #plt.show()

            #### save
            os.chdir(os.path.join(path_results, 'TF', 'allsujet'))
            fig.savefig(f'{chan}_{stat_type}.jpeg', dpi=150)

            fig.clf()
            plt.close('all')
            gc.collect()



   
#chan = chan_list_eeg_short[0]
def save_tf_subjectwise(chan):

    #### identify if already computed for all
    os.chdir(os.path.join(path_results, 'TF'))

    if os.path.exists(f'allsujet_{chan}.png'):
        print(f'{chan} ALREADY COMPUTED', flush=True)
        return

    #### load data
    print(f'#### LOAD BASELINE {chan} ####', flush=True)

    os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH'))

    tf_stretch_baseline_allsujet = np.zeros((len(sujet_list_FC), nfrex, stretch_point_ERP))

    #sujet_i, sujet = 46, sujet_list_FC[46]
    for sujet_i, sujet in enumerate(sujet_list_FC):

        tf_stretch_baseline_allsujet[sujet_i,:,:] = np.load(f'{sujet}_VS_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]

    print(f'#### LOAD COND {chan} ####', flush=True)

    tf_stretch_cond_allsujet = np.zeros((len(sujet_list_FC), nfrex, stretch_point_ERP))

    #sujet_i, sujet = 0, sujet_list_FC[47]
    for sujet_i, sujet in enumerate(sujet_list_FC):

        tf_stretch_cond_allsujet[sujet_i,:,:] = np.load(f'{sujet}_CHARGE_tf_stretch.npy')[np.where(chan_list_eeg_short == chan)[0][0],:,:]    

    data_allcond = {'VS' : tf_stretch_baseline_allsujet, 'CHARGE' : tf_stretch_cond_allsujet}

    print(f'#### PLOT {chan} ####', flush=True)

    #### plot 
    #sujet_i, sujet = 0, sujet_list_FC[0]
    for sujet_i, sujet in enumerate(sujet_list_FC):

        fig, axs = plt.subplots(ncols=len(cond_list)+1)

        plt.suptitle(f'{sujet} {chan} tf allsujet count:{len(sujet_list_FC)}')

        fig.set_figheight(5)
        fig.set_figwidth(18)

        #### for plotting l_gamma down
        #c, cond = 1, cond_to_plot[1]
        for c, cond in enumerate(cond_list):

            ax = axs[c]
            ax.set_title(cond, fontweight='bold', rotation=0)

            #### generate time vec
            time_vec = np.arange(stretch_point_ERP)

            #### plot
            ax.pcolormesh(time_vec, frex, data_allcond[cond][sujet_i,:,:], shading='gouraud', cmap=plt.get_cmap('seismic'))
            ax.set_yscale('log')

            ax.vlines(stretch_point_ERP/2, ymin=frex[0], ymax=frex[-1], colors='g')
            ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
            ax.set_ylim(frex[0], frex[-1])

        ax = axs[2]
        ax.set_title('diff', fontweight='bold', rotation=0)

        #### generate time vec
        time_vec = np.arange(stretch_point_ERP)

        #### plot
        ax.pcolormesh(time_vec, frex, data_allcond['CHARGE'][sujet_i,:,:] - data_allcond['VS'][sujet_i,:,:], shading='gouraud', cmap=plt.get_cmap('seismic'))
        ax.set_yscale('log')

        ax.vlines(stretch_point_ERP/2, ymin=frex[0], ymax=frex[-1], colors='g')
        ax.set_yticks([2,8,10,30,50,100,150], labels=[2,8,10,30,50,100,150])
        ax.set_ylim(frex[0], frex[-1])

        #plt.show()

        #### save
        os.chdir(os.path.join(path_results, 'TF', 'subjectwise', chan))
        fig.savefig(f'{sujet}_{chan}.jpeg', dpi=150)

        fig.clf()
        plt.close('all')
        gc.collect()


             

    

#################################
######## TOPOPLOT TF ########
#################################



def save_topoplot_allsujet():

    print(f'#### COMPUTE TOPOPLOT ####', flush=True)

    #### params
    phase_list = ['I', 'T_IE', 'E', 'T_EI']
    phase_shift = 125 
    # 0-125, 125-375, 375-625, 625-875, 875-1000, shift on origial TF
    phase_vec = {'I' : np.arange(250), 'T_IE' : np.arange(250)+250, 'E' : np.arange(250)+500, 'T_EI' : np.arange(250)+750} 

    point_thresh = 0.05

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg_short)
    info = mne.create_info(chan_list_eeg_short.tolist(), ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    #### load data
    print('#### LOAD DATA ####', flush=True)

    tf = np.zeros((len(cond_list), len(chan_list_eeg_short), nfrex, stretch_point_ERP))
    tf_stretch_allsujet = np.zeros((2, len(sujet_list_FC), len(chan_list_eeg_short), nfrex, stretch_point_ERP))

    os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH'))

    for cond_i, cond in enumerate(cond_list):

        print(cond)

        #sujet_i, sujet = 46, sujet_list_FC[46]
        for sujet_i, sujet in enumerate(sujet_list_FC):

            tf_stretch_allsujet[cond_i, sujet_i,:,:,:] = np.load(f'{sujet}_{cond}_tf_stretch.npy')

    tf_dict = {'cond' : cond_list, 'sujet' : sujet_list_FC, 'chan_list' : chan_list_eeg_short, 'frex' : frex, 'time' : np.arange(stretch_point_ERP)}
    xr_tf = xr.DataArray(data=tf_stretch_allsujet, dims=tf_dict.keys(), coords=tf_dict.values())

    shifted_xr_tf = xr_tf.roll(time=-phase_shift, roll_coords=False)

    #### chunk data perm 2g
    print('CHUNK')
    topoplot_data_2g = np.zeros((len(phase_list), len(freq_band_fc_list), len(chan_list_eeg_short)))
    topoplot_signi_2g = np.zeros((len(phase_list), len(freq_band_fc_list), len(chan_list_eeg_short)), dtype='bool')

    #phase_i, phase = 0, phase_list[0]
    for phase_i, phase in enumerate(phase_list):

        #band_i, band = 0, freq_band_fc_list[0]
        for band_i, band in enumerate(freq_band_fc_list):

            frex_mask = (frex >= freq_band_fc[band][0]) & (frex < freq_band_fc[band][1]) 
            tf_chunk = shifted_xr_tf.loc[:,:,:,:,phase_vec[phase]][:, :, :, frex_mask]

            for chan_i, chan in enumerate(chan_list_eeg_short):

                data_baseline, data_cond = tf_chunk.loc['VS',:,chan].median(['frex', 'time']).values, tf_chunk.loc['CHARGE',:,chan].median(['frex', 'time']).values

                mask = get_permutation_2groups(data_baseline, data_cond, n_surr_fc, stat_design=stat_design, mode_grouped=mode_grouped, 
                                                                    mode_generate_surr=mode_generate_surr_2g, percentile_thresh=percentile_thresh)

                if mask:

                    topoplot_data_2g[phase_i, band_i, chan_i] = np.median(data_cond - data_baseline)
                    topoplot_signi_2g[phase_i, band_i, chan_i] = True

    #### chunk data perm TF
    topoplot_data_TF = np.zeros((len(phase_list), len(freq_band_fc_list), len(chan_list_eeg_short)))
    topoplot_signi_TF = np.zeros((len(phase_list), len(freq_band_fc_list), len(chan_list_eeg_short)), dtype='bool')

    os.chdir(os.path.join(path_precompute, 'TF', 'STRETCH_STATS'))

    #chan_i, chan = 4, chan_list_eeg_short[4]
    for chan_i, chan in enumerate(chan_list_eeg_short):

        stats_allcond_chan = np.load(f'{chan}_allsujet_tf_STATS.npy')
        dict_xr = {'frex' : frex, 'time' : np.arange(stretch_point_ERP)}
        xr_stat_chan = xr.DataArray(data=stats_allcond_chan, dims=dict_xr.keys(), coords=dict_xr.values())
        xr_stat_chan = xr_stat_chan.roll(time=-phase_shift, roll_coords=False)

        #phase_i, phase = 1, phase_list[1]
        for phase_i, phase in enumerate(phase_list):

            #band_i, band = 0, freq_band_fc_list[0]
            for band_i, band in enumerate(freq_band_fc_list):

                frex_mask = (frex >= freq_band_fc[band][0]) & (frex < freq_band_fc[band][1]) 
                tf_chunk = shifted_xr_tf.loc[:,:,:,:,phase_vec[phase]][:, :, :, frex_mask]

                data_baseline, data_cond = tf_chunk.loc['VS',:,chan].median(['frex', 'time']).values, tf_chunk.loc['CHARGE',:,chan].median(['frex', 'time']).values

                mask_stats = xr_stat_chan.loc[:,phase_vec[phase]][frex_mask].values
                if np.sum(mask_stats) / mask_stats.size >= 0.05:

                    topoplot_data_TF[phase_i, band_i, chan_i] = np.median(data_cond - data_baseline)
                    topoplot_signi_TF[phase_i, band_i, chan_i] = True
 
    #### plot
    os.chdir(os.path.join(path_results, 'TF', 'allsujet'))

    for stat_type in ['2g', 'TFperm']:

        if stat_type == '2g':
            topoplot_data = topoplot_data_2g
            topoplot_signi = topoplot_signi_2g

        if stat_type == 'TFperm':
            topoplot_data = topoplot_data_TF
            topoplot_signi = topoplot_signi_TF

        #### vlim
        vlim = np.zeros((len(freq_band_fc_list)))
        #band_i, band = 0, freq_band_fc_list[0]
        for band_i, band in enumerate(freq_band_fc_list):

            vlim[band_i] = np.abs(np.array([topoplot_data[:, band_i, :].reshape(-1).min(), topoplot_data[:, band_i, :].reshape(-1).max()])).max()

        #band_i, band = 0, freq_band_fc_list[0]
        for band_i, band in enumerate(freq_band_fc_list):

            fig, axs = plt.subplots(nrows=1, ncols=len(phase_list), figsize=(15,5))

            #phase_i, phase = 0, phase_list[0]
            for phase_i, phase in enumerate(phase_list):

                ax = axs[phase_i]

                im = mne.viz.plot_topomap(data=topoplot_data[phase_i, band_i, :], axes=ax, show=False, names=chan_list_eeg_short, pos=info,
                                mask=topoplot_signi[phase_i, band_i, :], mask_params=mask_params, vlim=(-vlim[band_i], vlim[band_i]), cmap='seismic', extrapolate='local')

                ax.set_title(f'{phase}')

            cbar = fig.colorbar(im[0], ax=axs, orientation='vertical', shrink=0.7)
            cbar.set_label('robust z-score (a.u.)')  # Label for colorbar

            plt.suptitle(f'{band} lim:{np.round(vlim[band_i],2)} stat:{stat_type}')

            # plt.show()

            fig.savefig(f"topoplot_{band}_allsujet_{stat_type}.jpeg")

            plt.close('all')
        
        






########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    #chan = chan_list_eeg_short[0]
    for chan in chan_list_eeg_short:
                
        save_tf_subjectwise(chan)

    save_tf_allsujet()
    save_topoplot_allsujet()



        