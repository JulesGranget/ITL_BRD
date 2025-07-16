
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import seaborn as sns
import gc
import statsmodels.api as sm

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

debug = False




             

    

#################################
######## TOPOPLOT TF ########
#################################



def REG_TF():

    print(f'#### COMPUTE TOPOPLOT ####', flush=True)

    #### params
    phase_list = ['I', 'T_IE', 'E', 'T_EI']
    phase_shift = 125 
    # 0-125, 125-375, 375-625, 625-875, 875-1000, shift on origial TF
    phase_vec = {'I' : np.arange(250), 'T_IE' : np.arange(250)+250, 'E' : np.arange(250)+500, 'T_EI' : np.arange(250)+750} 

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

    #### chunk data
    print('CHUNK')
    sujet_list_sel = [sujet for sujet in shifted_xr_tf['sujet'].values if sujet[2:4] == 'DL']
    topoplot_data = np.zeros((len(phase_list), len(freq_band_fc_list), len(chan_list_eeg_short), len(sujet_list_sel)))
    dict_coords_topoplot = {'phase' : phase_list, 'band' : freq_band_fc_list, 'chan' : chan_list_eeg_short, 'sujet' : sujet_list_sel}

    for sujet_i, sujet in enumerate(sujet_list_sel):

        print(sujet)

        #phase_i, phase = 0, phase_list[0]
        for phase_i, phase in enumerate(phase_list):

            #band_i, band = 0, freq_band_fc_list[0]
            for band_i, band in enumerate(freq_band_fc_list):

                frex_mask = (frex >= freq_band_fc[band][0]) & (frex < freq_band_fc[band][1]) 
                tf_chunk = shifted_xr_tf.loc[:,sujet,:,:,phase_vec[phase]][:, :, frex_mask]

                for chan_i, chan in enumerate(chan_list_eeg_short):

                    data_baseline, data_cond = tf_chunk.loc['VS',chan].median(['frex', 'time']).values, tf_chunk.loc['CHARGE',chan].median(['frex', 'time']).values

                    topoplot_data[phase_i, band_i, chan_i, sujet_i] = data_cond - data_baseline

    xr_topoplot_diff = xr.DataArray(data=topoplot_data, dims=dict_coords_topoplot.keys(), coords=dict_coords_topoplot)

    #### get mdp
    os.chdir(os.path.join(path_data, 'DYSLEARN'))
    df_mdp = pd.read_excel('DYSLEARN mdp short.xlsx')

    dict_mdp = {'sujet' : [], 'A2' :  [], 'A1' :  [], 'A1+A2' :  []}

    for sujet in sujet_list_sel:

        for ID_i, ID in enumerate(df_mdp['ID'].values):

            if len(ID.astype(str)) == 1:
                ID_search = f"0{ID}"
            else:
                ID_search = f"{ID}"

            if ID_search.find(sujet[-2:]) != -1:
                ID_corres = ID_i
                break

        dict_mdp['sujet'].append(sujet)
        dict_mdp['A2'].append(df_mdp.iloc[ID_corres]['A2 TOTAL'])
        dict_mdp['A1'].append(df_mdp.iloc[ID_corres]['MDP A1'])
        dict_mdp['A1+A2'].append(df_mdp.iloc[ID_corres]['A1+A2'])
        
    df_mdp_sujet = pd.DataFrame(dict_mdp)

    #### reg
    percentile_thresh_list = [2.5, 97.5]
    n_perm_spearman = 1000
    mdp_metric_list = ['A1', 'A2', 'A1+A2']

    dict_reg = {'metric' : [], 'band' : [], 'chan' : [], 'phase' : [], 'rho' : [], 'OLS_a' : [], 'rho_signi' : [], 'OLS_a_signi' : []}

    for metric in mdp_metric_list:

        x = df_mdp_sujet[metric].values

        for band in freq_band_fc_list:

            print(metric, band)

            for chan in chan_list_eeg_short:

                for phase in phase_list:

                    y = xr_topoplot_diff.loc[phase,band,chan,:].values

                    rho_obs, _ = scipy.stats.spearmanr(x, y)

                    mdl = sm.OLS(y, x).fit()
                    # intercept = np.round(mdl.params[0], 4)
                    OLS_a_obs = np.round(mdl.params[-1], 4)
                    # pval = np.round(mdl.pvalues[-1], 4)

                    surr_distrib_rho = []
                    surr_distrib_OLS_a = []

                    for i in range(n_perm_spearman):
                    
                        y_perm = np.random.permutation(y)   

                        _rho, _ = scipy.stats.spearmanr(x, y_perm)
                        surr_distrib_rho.append(_rho)

                        _mdl = sm.OLS(y_perm, x).fit()
                        _OLS_a = np.round(_mdl.params[-1], 4)
                        surr_distrib_OLS_a.append(_OLS_a)

                    if debug:

                        count, _, _ = plt.hist(surr_distrib_rho, bins=50)
                        plt.vlines(np.percentile(surr_distrib_rho, [2.5, 97.5]), ymin=0, ymax=count.max(), color='r')
                        plt.vlines(rho_obs, ymin=0, ymax=count.max(), color='g')
                        plt.show()

                        count, _, _ = plt.hist(surr_distrib_OLS_a, bins=50)
                        plt.vlines(np.percentile(surr_distrib_OLS_a, [2.5, 97.5]), ymin=0, ymax=count.max(), color='r')
                        plt.vlines(OLS_a_obs, ymin=0, ymax=count.max(), color='g')
                        plt.show()

                    if rho_obs < np.percentile(surr_distrib_rho, percentile_thresh_list[0]) or rho_obs > np.percentile(surr_distrib_rho, percentile_thresh_list[1]):
                        rho_signi = True
                    else:
                        rho_signi = False

                    if OLS_a_obs < np.percentile(surr_distrib_OLS_a, percentile_thresh_list[0]) or OLS_a_obs > np.percentile(surr_distrib_OLS_a, percentile_thresh_list[1]):
                        OLS_a_signi = True
                    else:
                        OLS_a_signi = False

                    if debug:

                        sns.lmplot(data=pd.DataFrame({'sujet' : sujet_list_sel, 'A1' : df_mdp_sujet['A1'].values, 'Pxx' : y}), x='A1', y='Pxx')
                        plt.show()

                    dict_reg['metric'].append(metric)
                    dict_reg['band'].append(band)
                    dict_reg['chan'].append(chan)
                    dict_reg['phase'].append(phase)
                    dict_reg['rho'].append(rho_obs)
                    dict_reg['OLS_a'].append(OLS_a_obs)
                    dict_reg['rho_signi'].append(rho_signi)
                    dict_reg['OLS_a_signi'].append(OLS_a_signi)

    df_reg = pd.DataFrame(dict_reg)

    #### plot lmplot
    os.chdir(os.path.join(path_results, 'REG'))
    for metric in mdp_metric_list:

        x = df_mdp_sujet[metric].values

        for band in freq_band_fc_list:

            print(metric, band)

            df_chan_regplot = pd.DataFrame()

            for phase in phase_list:

                for chan in chan_list_eeg_short:
                
                    y = xr_topoplot_diff.loc[phase,band,chan,:].values
                    _df_chan = pd.DataFrame({'sujet' : sujet_list_sel, 'A1' : df_mdp_sujet['A1'].values, 'Pxx' : y, 'chan' : [chan]*len(sujet_list_sel), 'phase' : [phase]*len(sujet_list_sel)})
                    df_chan_regplot = pd.concat([df_chan_regplot, _df_chan])

            sns.lmplot(data=df_chan_regplot, x='A1', y='Pxx', hue='chan', ci=None, col='phase')
            plt.suptitle(f"{metric} {band}")
            # plt.show()
            plt.savefig(f"Pxx_lmplot_{metric}_{band}.png")
            plt.close('all')

    #### topoplot
    os.chdir(os.path.join(path_results, 'REG'))
    
    for metric_mdp in mdp_metric_list:

        #### vlim
        vlim = np.zeros((2, len(freq_band_fc_list)))
        for metric_reg_i, metric_reg in enumerate(['rho', 'OLS_a']):
            #band_i, band = 0, freq_band_fc_list[0]
            for band_i, band in enumerate(freq_band_fc_list):

                if metric_reg == 'rho':

                    vlim[metric_reg_i, band_i] = 1

                else:

                    vlim[metric_reg_i, band_i] = np.abs(np.array([df_reg.query(f"band == '{band}' and metric == '{metric_mdp}'")[metric_reg].values.min(), 
                                                    df_reg.query(f"band == '{band}' and metric == '{metric_mdp}'")[metric_reg].values.max()])).max()
    
        #### plot
        #band_i, band = 0, freq_band_fc_list[0]
        for band_i, band in enumerate(freq_band_fc_list):

            im_list = []

            fig, axs = plt.subplots(nrows=2, ncols=len(phase_list), figsize=(15,5))

            for row_i, metric_reg in enumerate(['rho', 'OLS_a']):

                #phase_i, phase = 0, phase_list[0]
                for col_i, phase in enumerate(phase_list):

                    ax = axs[row_i, col_i]
                    _data_plot = df_reg.query(f"band == '{band}' and metric == '{metric_mdp}' and phase == '{phase}'")[f'{metric_reg}'].values
                    _data_mask = df_reg.query(f"band == '{band}' and metric == '{metric_mdp}' and phase == '{phase}'")[f'{metric_reg}_signi'].values

                    im, _ = mne.viz.plot_topomap(data=_data_plot, axes=ax, show=False, names=chan_list_eeg_short, pos=info,
                                    mask=_data_mask, mask_params=mask_params, vlim=(-vlim[row_i, band_i], vlim[row_i, band_i]), cmap='seismic', extrapolate='local')

                    if col_i == 0:
                        im_list.append(im)  # Save image to add colorbar later
                    
                    ax.set_title(f'{phase}')
                    ax.set_ylabel(f"{metric_reg}")

            for row_i, im in enumerate(im_list):
                # Position: [left, bottom, width, height] in figure coordinates
                cbar_ax = fig.add_axes([0.92, 0.56 - row_i * 0.48, 0.015, 0.35])
                plt.colorbar(im, cax=cbar_ax)

            plt.suptitle(f'{metric_mdp} {band}, lim_rho:{np.round(vlim[0, band_i],2)}, lim_OLS_a:{np.round(vlim[1, band_i],2)}')

            # plt.show()

            fig.savefig(f"Pxx_topoplot_{metric_mdp}_{band}_allsujet.jpeg")

            plt.close('all')
            
            




#################################
######## FC ########
#################################



def REG_FC():

    print(f'#### COMPUTE TOPOPLOT ####', flush=True)

    #### load fc data
    fc_metric = 'WPLI'

    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')     

    fc_allsujet = np.zeros((len(sujet_list_FC), len(pairs_to_compute), len(freq_band_fc_list), stretch_point_FC))

    os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

    for sujet_i, sujet in enumerate(sujet_list_FC):

        print_advancement(sujet_i, len(sujet_list_FC))
        
        _fc_sujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch_rscore.nc')
        _fc_sujet = _fc_sujet.loc[:, 'CHARGE'] - _fc_sujet.loc[:, 'VS']
        _fc_sujet = _fc_sujet.median('cycle')

        fc_allsujet[sujet_i] = _fc_sujet

    fc_allsujet_dict = {'sujet' : sujet_list_FC, 'pair' : pairs_to_compute, 'band' : freq_band_fc_list, 'time' : np.arange(stretch_point_FC)}

    fc_allsujet = xr.DataArray(data=fc_allsujet, dims=fc_allsujet_dict.keys(), coords=fc_allsujet_dict.values())

    mask_params = dict(markersize=15, markerfacecolor='y')

    ch_types = ['eeg'] * len(chan_list_eeg_short)
    info = mne.create_info(chan_list_eeg_short.tolist(), ch_types=ch_types, sfreq=srate)
    info.set_montage('standard_1020')

    time_vec = fc_allsujet['time'].values
    phase_list = ['I', 'T_IE', 'E', 'T_EI']
    phase_shift = int(stretch_point_FC/4) 
    phase_vec = {'I' : np.arange(phase_shift), 'T_IE' : np.arange(phase_shift)+phase_shift, 
                'E' : np.arange(phase_shift)+phase_shift*2, 'T_EI' : np.arange(phase_shift)+phase_shift*3} 
    
    shifted_fc_allsujet = fc_allsujet.roll(time=-phase_shift, roll_coords=False)

    data_type = 'rscore'

    os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

    clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_state_stretch.nc')

    fc_mat_phase = np.zeros((len(sujet_list_FC), len(freq_band_fc_list), len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
    fc_mat_only_signi_phase = np.zeros((len(sujet_list_FC), len(freq_band_fc_list), len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

    #band_i, band = 0, freq_band_fc_list[0]
    for band_i, band in enumerate(freq_band_fc_list):

        #### phase
        #phase_i, phase = 0, 'I'
        for phase_i, phase in enumerate(phase_list):

            #pair_i, pair = 2, pairs_to_compute[2]
            for pair_i, pair in enumerate(pairs_to_compute):

                A, B = pair.split('-')
                A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                data_chunk_diff = shifted_fc_allsujet.loc[:, pair, band, phase_vec[phase]].values

                fc_val = np.median(data_chunk_diff, axis=1)

                fc_mat_phase[:, band_i, phase_i, A_i, B_i], fc_mat_phase[:, band_i, phase_i, B_i, A_i] = fc_val, fc_val

                if clusters.loc[data_type, phase, band, pair].values.astype('bool'):
                    fc_mat_only_signi_phase[:, band_i, phase_i, A_i, B_i], fc_mat_only_signi_phase[:, band_i, phase_i, B_i, A_i] = fc_val, fc_val

    coords_fc_mat = {'sujet' : sujet_list_FC, 'band' : freq_band_fc_list, 'phase' : phase_list, 'chanA' : chan_list_eeg_short, 'chanB' : chan_list_eeg_short}
    xr_fc_mat = xr.DataArray(data=fc_mat_phase, dims=coords_fc_mat.keys(), coords=coords_fc_mat)
    xr_fc_mat_onlysigni = xr.DataArray(data=fc_mat_only_signi_phase, dims=coords_fc_mat.keys(), coords=coords_fc_mat)

    #### get mdp
    os.chdir(os.path.join(path_data, 'DYSLEARN'))
    df_mdp = pd.read_excel('DYSLEARN mdp short.xlsx')
    sujet_list_sel = [sujet for sujet in xr_fc_mat['sujet'].values if sujet[2:4] == 'DL']

    dict_mdp = {'sujet' : [], 'A2' :  [], 'A1' :  [], 'A1+A2' :  []}

    for sujet in sujet_list_sel:

        for ID_i, ID in enumerate(df_mdp['ID'].values):

            if len(ID.astype(str)) == 1:
                ID_search = f"0{ID}"
            else:
                ID_search = f"{ID}"

            if ID_search.find(sujet[-2:]) != -1:
                ID_corres = ID_i
                break

        dict_mdp['sujet'].append(sujet)
        dict_mdp['A2'].append(df_mdp.iloc[ID_corres]['A2 TOTAL'])
        dict_mdp['A1'].append(df_mdp.iloc[ID_corres]['MDP A1'])
        dict_mdp['A1+A2'].append(df_mdp.iloc[ID_corres]['A1+A2'])
        
    df_mdp_sujet = pd.DataFrame(dict_mdp)

    #### reg
    percentile_thresh_list = [2.5, 97.5]
    n_perm_spearman = 1000
    mdp_metric_list = ['A1', 'A2', 'A1+A2']

    xr_sel = xr_fc_mat.loc[sujet_list_sel]
    # xr_sel = xr_fc_mat_onlysigni.loc[sujet_list_sel]

    dict_reg = {'metric' : [], 'band' : [], 'chan' : [], 'phase' : [], 'rho' : [], 'OLS_a' : [], 'rho_signi' : [], 'OLS_a_signi' : []}

    for metric in mdp_metric_list:

        x = df_mdp_sujet[metric].values

        for band in freq_band_fc_list:

            print(metric, band)

            for chan in chan_list_eeg_short:

                for phase in phase_list:

                    y_raw = xr_sel.loc[:,band,phase,chan,:].values
                    mask_zeros = y_raw[0,:] != 0
                    y = np.median(y_raw[:,mask_zeros], axis=1)

                    rho_obs, _ = scipy.stats.spearmanr(x, y)

                    mdl = sm.OLS(y, x).fit()
                    # intercept = np.round(mdl.params[0], 4)
                    OLS_a_obs = np.round(mdl.params[-1], 4)
                    # pval = np.round(mdl.pvalues[-1], 4)

                    surr_distrib_rho = []
                    surr_distrib_OLS_a = []

                    for i in range(n_perm_spearman):
                    
                        y_perm = np.random.permutation(y)   
                        _rho, _ = scipy.stats.spearmanr(x, y_perm)
                        surr_distrib_rho.append(_rho)

                        _mdl = sm.OLS(y_perm, x).fit()
                        _OLS_a = np.round(_mdl.params[-1], 4)
                        surr_distrib_OLS_a.append(_rho)

                    if debug:

                        count, _, _ = plt.hist(surr_distrib_rho, bins=50)
                        plt.vlines(np.percentile(surr_distrib_rho, [2.5, 97.5]), ymin=0, ymax=count.max(), color='r')
                        plt.vlines(rho_obs, ymin=0, ymax=count.max(), color='g')
                        plt.show()

                        count, _, _ = plt.hist(surr_distrib_OLS_a, bins=50)
                        plt.vlines(np.percentile(surr_distrib_OLS_a, [2.5, 97.5]), ymin=0, ymax=count.max(), color='r')
                        plt.vlines(OLS_a_obs, ymin=0, ymax=count.max(), color='g')
                        plt.show()

                    if rho_obs < np.percentile(surr_distrib_rho, percentile_thresh_list[0]) or rho_obs > np.percentile(surr_distrib_rho, percentile_thresh_list[1]):
                        rho_signi = True
                    else:
                        rho_signi = False

                    if OLS_a_obs < np.percentile(surr_distrib_OLS_a, percentile_thresh_list[0]) or OLS_a_obs > np.percentile(surr_distrib_OLS_a, percentile_thresh_list[1]):
                        OLS_a_signi = True
                    else:
                        OLS_a_signi = False

                    if debug:

                        sns.lmplot(data=pd.DataFrame({'sujet' : sujet_list_sel, 'A1' : df_mdp_sujet['A1'].values, 'Pxx' : y}), x='A1', y='Pxx')
                        plt.show()
        
                    dict_reg['metric'].append(metric)
                    dict_reg['band'].append(band)
                    dict_reg['chan'].append(chan)
                    dict_reg['phase'].append(phase)
                    dict_reg['rho'].append(rho_obs)
                    dict_reg['OLS_a'].append(OLS_a_obs)
                    dict_reg['rho_signi'].append(rho_signi)
                    dict_reg['OLS_a_signi'].append(OLS_a_signi)

    df_reg = pd.DataFrame(dict_reg)

    #### plot lmplot
    os.chdir(os.path.join(path_results, 'REG'))
    for metric in mdp_metric_list:

        x = df_mdp_sujet[metric].values

        for band in freq_band_fc_list:

            print(metric, band)

            df_chan_regplot = pd.DataFrame()

            for phase in phase_list:

                for chan in chan_list_eeg_short:
                
                    y_raw = xr_sel.loc[:,band,phase,chan,:].values
                    mask_zeros = y_raw[0,:] != 0
                    y = np.median(y_raw[:,mask_zeros], axis=1)

                    _df_chan = pd.DataFrame({'sujet' : sujet_list_sel, 'A1' : df_mdp_sujet['A1'].values, 'FC' : y, 'chan' : [chan]*len(sujet_list_sel), 'phase' : [phase]*len(sujet_list_sel)})
                    df_chan_regplot = pd.concat([df_chan_regplot, _df_chan])

            sns.lmplot(data=df_chan_regplot, x='A1', y='FC', hue='chan', ci=None, col='phase')
            plt.suptitle(f"{metric} {band}")
            # plt.show()
            plt.savefig(f"FC_lmplot_{metric}_{band}.png")
            plt.close('all')

    #### topoplot
    os.chdir(os.path.join(path_results, 'REG'))
    
    for metric_mdp in mdp_metric_list:

        #### vlim
        vlim = np.zeros((2, len(freq_band_fc_list)))
        for metric_reg_i, metric_reg in enumerate(['rho', 'OLS_a']):
            #band_i, band = 0, freq_band_fc_list[0]
            for band_i, band in enumerate(freq_band_fc_list):

                if metric_reg == 'rho':

                    vlim[metric_reg_i, band_i] = 1

                else:    

                    vlim[metric_reg_i, band_i] = np.abs(np.array([df_reg.query(f"band == '{band}' and metric == '{metric_mdp}'")[metric_reg].values.min(), 
                                                    df_reg.query(f"band == '{band}' and metric == '{metric_mdp}'")[metric_reg].values.max()])).max()
    
        #### plot
        #band_i, band = 0, freq_band_fc_list[0]
        for band_i, band in enumerate(freq_band_fc_list):

            im_list = []

            fig, axs = plt.subplots(nrows=2, ncols=len(phase_list), figsize=(15,5))

            for row_i, metric_reg in enumerate(['rho', 'OLS_a']):

                #phase_i, phase = 0, phase_list[0]
                for col_i, phase in enumerate(phase_list):

                    ax = axs[row_i, col_i]
                    _data_plot = df_reg.query(f"band == '{band}' and metric == '{metric_mdp}' and phase == '{phase}'")[f'{metric_reg}'].values
                    _data_mask = df_reg.query(f"band == '{band}' and metric == '{metric_mdp}' and phase == '{phase}'")[f'{metric_reg}_signi'].values

                    im, _ = mne.viz.plot_topomap(data=_data_plot, axes=ax, show=False, names=chan_list_eeg_short, pos=info,
                                    mask=_data_mask, mask_params=mask_params, vlim=(-vlim[row_i, band_i], vlim[row_i, band_i]), cmap='seismic', extrapolate='local')
                    
                    if col_i == 0:
                        im_list.append(im)

                    ax.set_title(f'{phase}')
                    ax.set_ylabel(f"{metric_reg}")

            plt.suptitle(f'{metric_mdp} {band}, lim_rho:{np.round(vlim[0, band_i],2)}, lim_OLS_a:{np.round(vlim[1, band_i],2)}')

            for row_i, im in enumerate(im_list):
                
                cbar_ax = fig.add_axes([0.92, 0.56 - row_i * 0.48, 0.015, 0.35])
                plt.colorbar(im, cax=cbar_ax)

            # plt.show()

            fig.savefig(f"FC_topoplot_{metric_mdp}_{band}_allsujet.jpeg")

            plt.close('all')
            
            





########################################
######## EXECUTE AND SAVE ########
########################################


if __name__ == '__main__':

    REG_TF()
    REG_FC()


        


