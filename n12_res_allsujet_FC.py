
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import joblib
import xarray as xr
import seaborn as sns
import gc
from matplotlib.animation import FuncAnimation
import networkx as nx
import matplotlib.patches as patches

from n00_config_params import *
from n00bis_config_analysis_functions import *
from n00ter_stats import *

debug = False






########################################
######## SUPPORT FUNCTIONS ########
########################################

#data, pairs = clusters.loc['whole', :].values, clusters['pair'].values
def from_pairs_2mat(data, pairs):

    unique_channels = sorted(set([ch for pair in pairs for ch in pair.split("-")]))
    channel_to_index = {ch: i for i, ch in enumerate(unique_channels)}  # Mapping channel -> index

    mat = np.zeros((len(unique_channels), len(unique_channels)))

    for pair, value in zip(pairs, data):
        ch1, ch2 = pair.split("-") 
        i, j = channel_to_index[ch1], channel_to_index[ch2] 
        mat[i, j] = value
        mat[j, i] = value 

    if debug:

        plt.imshow(mat)
        plt.show()

    return mat







################################
######## PLOT FC ########
################################


def plot_allsujet_FC_time_stretch():

    #fc_metric = 'ISPC'
    for fc_metric in ['MI', 'ISPC', 'WPLI']:

        print(f'{fc_metric} PLOT stretch', flush=True)

        #### load
        pairs_to_compute = []

        for pair_A in chan_list_eeg_short:

            for pair_B in chan_list_eeg_short:

                if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                    continue

                pairs_to_compute.append(f'{pair_A}-{pair_B}')     

        #data_type = 'raw'
        for data_type in ['raw', 'rscore']:

            if fc_metric == 'MI':
                fc_allsujet = np.zeros((len(sujet_list_FC), len(pairs_to_compute), stretch_point_FC))
            else:
                fc_allsujet = np.zeros((len(sujet_list_FC), len(pairs_to_compute), len(freq_band_fc_list), stretch_point_FC))

            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

            for sujet_i, sujet in enumerate(sujet_list_FC):

                print_advancement(sujet_i, len(sujet_list_FC))

                if data_type == 'raw':

                    if fc_metric == 'MI':
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_stretch_{sujet}.nc')
                        _fc_sujet = _fc_sujet.loc['CHARGE'] - _fc_sujet.loc['VS']
                        _fc_sujet = _fc_sujet.median('ntrials')
                    
                    else:
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch.nc')
                        _fc_sujet = _fc_sujet.loc[:, 'CHARGE'] - _fc_sujet.loc[:, 'VS']
                        _fc_sujet = _fc_sujet.median('cycle')

                else:

                    if fc_metric == 'MI':
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_stretch_{sujet}_rscore.nc')
                        _fc_sujet = _fc_sujet.loc['CHARGE'] - _fc_sujet.loc['VS']
                        _fc_sujet = _fc_sujet.median('ntrials')
                    
                    else:
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch_rscore.nc')
                        _fc_sujet = _fc_sujet.loc[:, 'CHARGE'] - _fc_sujet.loc[:, 'VS']
                        _fc_sujet = _fc_sujet.median('cycle')

                fc_allsujet[sujet_i] = _fc_sujet

            if fc_metric == 'MI':
                fc_allsujet_dict = {'sujet' : sujet_list_FC, 'pair' : pairs_to_compute, 'time' : np.arange(stretch_point_FC)}
            else:
                fc_allsujet_dict = {'sujet' : sujet_list_FC, 'pair' : pairs_to_compute, 'band' : freq_band_fc_list, 'time' : np.arange(stretch_point_FC)}

            fc_allsujet = xr.DataArray(data=fc_allsujet, dims=fc_allsujet_dict.keys(), coords=fc_allsujet_dict.values())

            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))
            if data_type == 'raw':
                clusters_homemade = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_HOMEMADE_time_stretch.nc')
                clusters_mne = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_MNE_time_stretch.nc')
            else:
                clusters_homemade = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_HOMEMADE_time_stretch_rscore.nc')
                clusters_mne = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_MNE_time_stretch_rscore.nc')

            pairs_to_compute = fc_allsujet['pair'].values
            time_vec = fc_allsujet['time'].values

            if debug:

                pair = pairs_to_compute[0]
                cond = 'CHARGE'
                for sujet_i, sujet in enumerate(sujet_list_FC):
                    plt.plot(fc_allsujet.loc[sujet,cond,pair,:], alpha=0.2)
                plt.plot(fc_allsujet.loc[:,cond,pair,:].median('sujet'), color='r')
                plt.show()

            ######## IDENTIFY MIN MAX ########

            #### identify min max for allsujet

            vlim_band = {}
            vlim_band_whole = {}

            for band in freq_band_fc_list:

                vlim_band[band] = {}
                vlim_band_whole[band] = {}
            
                #pair_i, pair = 0, pairs_to_compute[0]
                for pair_i, pair in enumerate(pairs_to_compute):

                    if fc_metric == 'MI':
                        vlim_band[band][pair] = {'min' : fc_allsujet.loc[:, pair, :].values.min(), 'max' : fc_allsujet.loc[:, pair, :].values.max()}
                        vlim_band_whole[band][pair] = {'min' : fc_allsujet.loc[:, pair, :].values.min(), 'max' : fc_allsujet.loc[:, pair, :].values.max()}
                    else:
                        vlim_band[band][pair] = {'min' : fc_allsujet.loc[:, pair, band, :].values.min(), 'max' : fc_allsujet.loc[:, pair, band, :].values.max()}                        
                        vlim_band_whole[band][pair] = {'min' : fc_allsujet.loc[:, pair, band, :].values.min(), 'max' : fc_allsujet.loc[:, pair, band, :].values.max()}                        

            if debug:

                for pair in pairs_to_compute:

                    plt.plot(fc_allsujet.loc[:, pair, 'CHARGE', :].mean('sujet'))
                
                plt.show()

            n_sujet = fc_allsujet['sujet'].shape[0]

            for band in freq_band_fc_list:

                #pair_i, pair = 1, pairs_to_compute[1]
                for pair_i, pair in enumerate(pairs_to_compute):

                    fig, ax = plt.subplots()

                    fig.set_figheight(5)
                    fig.set_figwidth(8)

                    plt.suptitle(f'stretch {pair} nsujet:{n_sujet}')

                    ax.set_ylim(vlim_band[band][pair]['min'], vlim_band[band][pair]['max'])

                    for sujet_i, sujet in enumerate(sujet_list_FC):

                        if fc_metric == 'MI':
                            ax.plot(time_vec, fc_allsujet.loc[sujet, pair], alpha=0.2)
                        else:
                            ax.plot(time_vec, fc_allsujet.loc[sujet, pair, band], alpha=0.2)

                    if fc_metric == 'MI':
                        ax.plot(time_vec, fc_allsujet.loc[:, pair].median('sujet'), color='r')
                    else:
                        ax.plot(time_vec, fc_allsujet.loc[:, pair, band].median('sujet'), color='r')

                    if fc_metric == 'MI':
                        _clusters = clusters_homemade.loc[pair, :].values
                    else:
                        _clusters = clusters_homemade.loc[band, pair, :].values

                    ax.fill_between(time_vec, vlim_band[band][pair]['min'], vlim_band[band][pair]['max'], where=_clusters.astype('int'), alpha=0.3, color='r')

                    if fc_metric == 'MI':
                        _clusters = clusters_mne.loc[pair, :].values
                    else:
                        _clusters = clusters_mne.loc[band, pair, :].values
                        
                    ax.fill_between(time_vec, vlim_band[band][pair]['min'], vlim_band[band][pair]['max'], where=_clusters.astype('int'), alpha=0.3, color='g')

                    ax.vlines(stretch_point_FC/2, ymin=vlim_band[band][pair]['min'], ymax=vlim_band[band][pair]['max'], colors='g')  

                    fig.tight_layout()
                    plt.legend()

                    # plt.show()

                    #### save
                    os.chdir(os.path.join(path_results, 'FC', fc_metric, 'allpairs'))

                    if data_type == 'raw':

                        if fc_metric == 'MI':
                            fig.savefig(f'RAW_stretch_{pair}.jpeg', dpi=150)

                        else:
                            fig.savefig(f'RAW_{band}_stretch_{pair}.jpeg', dpi=150)

                    else:

                        if fc_metric == 'MI':
                            fig.savefig(f'RSCORE_stretch_{pair}.jpeg', dpi=150)

                        else:
                            fig.savefig(f'RSCORE_{band}_stretch_{pair}.jpeg', dpi=150)

                    fig.clf()
                    plt.close('all')
                    gc.collect()

                        





def plot_allsujet_FC_mat_stretch():

    # EEG 10-20 system positions (approximate, normalized for plotting)
    eeg_positions = {
        'C3': (-0.5, 0.3), 'C4': (0.5, 0.3),
        'CP1': (-0.3, 0.0), 'CP2': (0.3, 0.0),
        'Cz': (0.0, 0.3), 'F3': (-0.5, 0.7),
        'F4': (0.5, 0.7), 'FC1': (-0.3, 0.5),
        'FC2': (0.3, 0.5), 'Fz': (0.0, 0.7)
    }

    #### load
    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')        

    #fc_metric = 'MI'
    for fc_metric in ['MI', 'ISPC', 'WPLI']:

        for data_type in ['raw', 'rscore']:

            print(f'{fc_metric} {data_type} PLOT stretch', flush=True)

            if fc_metric == 'MI':
                fc_allsujet = np.zeros((len(sujet_list_FC), len(pairs_to_compute), stretch_point_FC))
            else:
                fc_allsujet = np.zeros((len(sujet_list_FC), len(pairs_to_compute), len(freq_band_fc_list), stretch_point_FC))

            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

            for sujet_i, sujet in enumerate(sujet_list_FC):

                print_advancement(sujet_i, len(sujet_list_FC))

                if data_type == 'raw':
                    if fc_metric == 'MI':
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_stretch_{sujet}.nc')
                        _fc_sujet = _fc_sujet.loc['CHARGE'] - _fc_sujet.loc['VS']
                        _fc_sujet = _fc_sujet.median('ntrials')                
                    else:
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch.nc')
                        _fc_sujet = _fc_sujet.loc[:, 'CHARGE'] - _fc_sujet.loc[:, 'VS']
                        _fc_sujet = _fc_sujet.median('cycle')
                else:
                    if fc_metric == 'MI':
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_stretch_{sujet}_rscore.nc')
                        _fc_sujet = _fc_sujet.loc['CHARGE'] - _fc_sujet.loc['VS']
                        _fc_sujet = _fc_sujet.median('ntrials')                
                    else:
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch_rscore.nc')
                        _fc_sujet = _fc_sujet.loc[:, 'CHARGE'] - _fc_sujet.loc[:, 'VS']
                        _fc_sujet = _fc_sujet.median('cycle')

                fc_allsujet[sujet_i] = _fc_sujet

            if fc_metric == 'MI':
                fc_allsujet_dict = {'sujet' : sujet_list_FC, 'pair' : pairs_to_compute, 'time' : np.arange(stretch_point_FC)}
            else:
                fc_allsujet_dict = {'sujet' : sujet_list_FC, 'pair' : pairs_to_compute, 'band' : freq_band_fc_list, 'time' : np.arange(stretch_point_FC)}

            fc_allsujet = xr.DataArray(data=fc_allsujet, dims=fc_allsujet_dict.keys(), coords=fc_allsujet_dict.values())

            time_vec = fc_allsujet['time'].values
            phase_list = ['I', 'T_IE', 'E', 'T_EI']
            phase_shift = int(stretch_point_FC/4) 
            phase_vec = {'I' : np.arange(phase_shift), 'T_IE' : np.arange(phase_shift)+phase_shift, 
                        'E' : np.arange(phase_shift)+phase_shift*2, 'T_EI' : np.arange(phase_shift)+phase_shift*3} 
            
            shifted_fc_allsujet = fc_allsujet.roll(time=-phase_shift, roll_coords=False)

            #band_i, band = 0, freq_band_fc_list[0]
            for band_i, band in enumerate(freq_band_fc_list):

                for stat_type in ['HOMEMADE', 'MNE']:

                    os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

                    if stat_type == 'HOMEMADE':
                        clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_state_stretch.nc')
                    else:
                        clusters = xr.open_dataarray(f'{fc_metric}_mne_allsujet_STATS_state_stretch.nc')

                    fc_mat_whole = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    fc_mat_phase = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    fc_mat_mask_signi_phase = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    fc_mat_only_signi_whole = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    fc_mat_only_signi_phase = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

                    #### whole
                    if fc_metric == 'MI':
                        fc_mat_mask_signi_whole = from_pairs_2mat(clusters.loc[data_type, 'whole',:], pairs_to_compute)
                    else:
                        fc_mat_mask_signi_whole = from_pairs_2mat(clusters.loc[data_type, 'whole',band,:], pairs_to_compute)

                    #pair_i, pair = 2, pairs_to_compute[2]
                    for pair_i, pair in enumerate(pairs_to_compute):

                        A, B = pair.split('-')
                        A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                        if fc_metric == 'MI':
                            data_chunk_diff = shifted_fc_allsujet.loc[:, pair].median('sujet').values
                            
                        else:
                            data_chunk_diff = shifted_fc_allsujet.loc[:, pair, band].median('sujet').values

                        fc_val = np.median(data_chunk_diff)

                        fc_mat_whole[A_i, B_i], fc_mat_whole[B_i, A_i] = fc_val, fc_val

                        if fc_metric == 'MI' and clusters.loc[data_type, 'whole',pair].values.astype('bool'):
                            fc_mat_only_signi_whole[A_i, B_i], fc_mat_only_signi_whole[B_i, A_i] = fc_val, fc_val
                        elif fc_metric != 'MI' and clusters.loc[data_type, 'whole',band,pair].values.astype('bool'):
                            fc_mat_only_signi_whole[A_i, B_i], fc_mat_only_signi_whole[B_i, A_i] = fc_val, fc_val

                    #### phase

                    #phase_i, phase = 0, 'I'
                    for phase_i, phase in enumerate(phase_list):

                        if fc_metric == 'MI':
                            fc_mat_mask_signi_phase[phase_i,:,:] = from_pairs_2mat(clusters.loc[data_type, phase,:], pairs_to_compute)
                        else:
                            fc_mat_mask_signi_phase[phase_i,:,:] = from_pairs_2mat(clusters.loc[data_type, phase,band,:], pairs_to_compute)

                        #pair_i, pair = 2, pairs_to_compute[2]
                        for pair_i, pair in enumerate(pairs_to_compute):

                            A, B = pair.split('-')
                            A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                            if fc_metric == 'MI':
                                data_chunk_diff = shifted_fc_allsujet.loc[:, pair, phase_vec[phase]].median('sujet').values
                                
                            else:
                                data_chunk_diff = shifted_fc_allsujet.loc[:, pair, band, phase_vec[phase]].median('sujet').values

                            fc_val = np.median(data_chunk_diff)

                            fc_mat_phase[phase_i, A_i, B_i], fc_mat_phase[phase_i, B_i, A_i] = fc_val, fc_val

                            if fc_metric == 'MI' and clusters.loc[data_type, phase,pair].values.astype('bool'):
                                fc_mat_only_signi_phase[phase_i, A_i, B_i], fc_mat_only_signi_phase[phase_i, B_i, A_i] = fc_val, fc_val
                            elif fc_metric != 'MI' and clusters.loc[data_type, phase,band,pair].values.astype('bool'):
                                fc_mat_only_signi_phase[phase_i, A_i, B_i], fc_mat_only_signi_phase[phase_i, B_i, A_i] = fc_val, fc_val

                    #### plot
                    vlim_whole = np.abs((fc_mat_whole.min(), fc_mat_whole.max())).max()
                    vlim_phase = np.abs((fc_mat_phase.min(), fc_mat_phase.max())).max()

                    #fc_type = 'signimat'
                    for fc_type in ['fullmat', 'signimat']:

                        #### whole
                        fig, ax = plt.subplots(figsize=(8,8)) 

                        if fc_type == 'fullmat':
                            im = ax.imshow(fc_mat_whole, cmap='seismic', vmin=-vlim_whole, vmax=vlim_whole)
                        elif fc_type == 'signimat':
                            im = ax.imshow(fc_mat_only_signi_whole, cmap='seismic', vmin=-vlim_whole, vmax=vlim_whole)
                        ax.set_xticks(ticks=np.arange(fc_mat_whole.shape[1]), labels=chan_list_eeg_short, rotation=90)
                        ax.set_xlabel("Electrodes")

                        ax.set_yticks(ticks=np.arange(fc_mat_whole.shape[1]), labels=chan_list_eeg_short)
                        ax.set_ylabel("Electrodes")

                        if fc_type == 'fullmat':
                            for i in range(fc_mat_mask_signi_whole.shape[0]):
                                for j in range(fc_mat_mask_signi_whole.shape[1]):
                                    if fc_mat_only_signi_whole[i, j]:
                                        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=4, edgecolor='g', facecolor='none')
                                        ax.add_patch(rect)

                        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, label="Connectivity Strength")

                        if fc_metric == 'MI':
                            plt.suptitle(f"WHOLE MI {stat_type} {fc_type} _{data_type}")
                        else:
                            plt.suptitle(f'WHOLE {fc_metric} {band} {stat_type} {fc_type} _{data_type}')

                        # plt.show()

                        os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))

                        if fc_metric == 'MI':
                            plt.savefig(f'whole_stretch_MI_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)

                        else:
                            plt.savefig(f'whole_stretch_{fc_metric}_{band}_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)
                            
                        if fc_type == "signimat":
                            os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot', 'summary'))
                            if fc_metric == 'MI':
                                plt.savefig(f'whole_stretch_MI_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)
                            else:
                                plt.savefig(f'whole_stretch_{fc_metric}_{band}_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)

                        plt.close('all')
                        gc.collect()

                        #### phase

                        fig, axs = plt.subplots(ncols=len(phase_list), figsize=(12,5)) 

                        for phase_i, phase in enumerate(phase_list):

                            ax = axs[phase_i]

                            if fc_type == 'fullmat':
                                im = ax.imshow(fc_mat_phase[phase_i, :, :], cmap='seismic', vmin=-vlim_phase, vmax=vlim_phase)
                            elif fc_type == 'signimat':
                                im = ax.imshow(fc_mat_only_signi_phase[phase_i, :, :], cmap='seismic', vmin=-vlim_phase, vmax=vlim_phase)
                            ax.set_xticks(ticks=np.arange(fc_mat_phase.shape[1]), labels=chan_list_eeg_short, rotation=90)
                            ax.set_xlabel("Electrodes")

                            if phase_i == 0:
                                ax.set_yticks(ticks=np.arange(fc_mat_phase.shape[1]), labels=chan_list_eeg_short)
                                ax.set_ylabel("Electrodes")

                            if fc_type == 'fullmat':
                                _fc_mat_mask_signi = fc_mat_mask_signi_phase[phase_i,:,:]

                                for i in range(_fc_mat_mask_signi.shape[0]):
                                    for j in range(_fc_mat_mask_signi.shape[1]):
                                        if _fc_mat_mask_signi[i, j]:
                                            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=4, edgecolor='g', facecolor='none')
                                            ax.add_patch(rect)

                            ax.set_title(phase)

                        fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label="Connectivity Strength")

                        if fc_metric == 'MI':
                            plt.suptitle(f"MI FC {stat_type} {fc_type} _{data_type}")
                        else:
                            plt.suptitle(f'{fc_metric} {band} {stat_type} {fc_type} _{data_type}')

                        # plt.show()

                        os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))

                        if fc_metric == 'MI':
                            plt.savefig(f'phase_stretch_MI_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)

                        else:
                            plt.savefig(f'phase_stretch_{fc_metric}_{band}_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)
                            
                        if fc_type == "signimat":
                            os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot', 'summary'))
                            if fc_metric == 'MI':
                                plt.savefig(f'phase_stretch_MI_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)
                            else:
                                plt.savefig(f'phase_stretch_{fc_metric}_{band}_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)

                        plt.close('all')
                        gc.collect()

                        if fc_type == 'signimat':

                            #### whole

                            fig, ax = plt.subplots(figsize=(8,8)) 

                            G = nx.Graph()
                            G.add_nodes_from(eeg_positions.keys())
                                
                            mat_connectivity = fc_mat_only_signi_whole
                            
                            for i, ch1 in enumerate(eeg_positions.keys()):
                                for j, ch2 in enumerate(eeg_positions.keys()):
                                    if i < j:  # Avoid duplicate edges
                                        weight = mat_connectivity[i, j]
                                        color = 'red' if weight > 0 else 'blue'
                                        G.add_edge(ch1, ch2, weight=abs(weight) * 5, color=color)
                            
                            # Draw the graph
                            edges = G.edges()
                            edge_colors = [G[u][v]['color'] for u, v in edges]
                            edge_widths = [G[u][v]['weight'] for u, v in edges]
                                
                            nx.draw(G, eeg_positions, with_labels=True, node_size=500, font_size=8, edge_color=edge_colors, width=edge_widths, ax=ax)
                            ax.set_title(phase)
                                    
                            # Add colorbar
                            sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-vlim_whole, vmax=vlim_whole))
                            sm.set_array([])
                            plt.colorbar(sm, ax=ax, label='Connectivity Strength')
                                
                            if fc_metric == 'MI':
                                plt.suptitle(f"MI FC {stat_type} {fc_type} _{data_type}")
                            else:
                                plt.suptitle(f'{fc_metric} {band} {stat_type} {fc_type} _{data_type}')

                            # plt.show()

                            os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))

                            if fc_metric == 'MI':
                                plt.savefig(f'whole_stretch_GRAPH_MI_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)

                            else:
                                plt.savefig(f'whole_stretch_GRAPH_{fc_metric}_{band}_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)
                                
                            if fc_type == "signimat":
                                os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot', 'summary'))
                                if fc_metric == 'MI':
                                    plt.savefig(f'whole_stretch_GRAPH_MI_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)
                                else:
                                    plt.savefig(f'whole_stretch_GRAPH_{fc_metric}_{band}_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)

                            plt.close('all')
                            gc.collect()

                            #### phase

                            fig, axs = plt.subplots(ncols=len(phase_list), figsize=(12,5)) 

                            for phase_i, phase in enumerate(phase_list):

                                ax = axs[phase_i]

                                G = nx.Graph()
                                G.add_nodes_from(eeg_positions.keys())
                                    
                                mat_connectivity = fc_mat_only_signi_phase[phase_i, :, :]
                                
                                for i, ch1 in enumerate(eeg_positions.keys()):
                                    for j, ch2 in enumerate(eeg_positions.keys()):
                                        if i < j:  # Avoid duplicate edges
                                            weight = mat_connectivity[i, j]
                                            color = 'red' if weight > 0 else 'blue'
                                            G.add_edge(ch1, ch2, weight=abs(weight) * 5, color=color)
                                
                                # Draw the graph
                                edges = G.edges()
                                edge_colors = [G[u][v]['color'] for u, v in edges]
                                edge_widths = [G[u][v]['weight'] for u, v in edges]
                                    
                                nx.draw(G, eeg_positions, with_labels=True, node_size=500, font_size=8, edge_color=edge_colors, width=edge_widths, ax=ax)
                                ax.set_title(phase)
                                    
                            # Add colorbar
                            sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-vlim_phase, vmax=vlim_phase))
                            sm.set_array([])
                            plt.colorbar(sm, ax=ax, label='Connectivity Strength')
                                
                            if fc_metric == 'MI':
                                plt.suptitle(f"MI FC {stat_type} {fc_type} _{data_type}")
                            else:
                                plt.suptitle(f'{fc_metric} {band} {stat_type} {fc_type} _{data_type}')

                            # plt.show()

                            os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))

                            if fc_metric == 'MI':
                                plt.savefig(f'phase_stretch_GRAPH_MI_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)

                            else:
                                plt.savefig(f'phase_stretch_GRAPH_{fc_metric}_{band}_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)
                                
                            if fc_type == "signimat":
                                os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot', 'summary'))
                                if fc_metric == 'MI':
                                    plt.savefig(f'phase_stretch_GRAPH_MI_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)
                                else:
                                    plt.savefig(f'phase_stretch_GRAPH_{fc_metric}_{band}_FC_{fc_type}_{stat_type}_{data_type}.jpeg', dpi=150)

                            plt.close('all')
                            gc.collect()

                    continue


                    #### ANIMATION

                    print('animation')

                    def get_visu_data_fullmat(time_window):

                        start_window = np.arange(0, time_vec.size, time_window)
                        n_times = np.arange(start_window.size)
                        
                        cluster_mask_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))
                        data_chunk_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))

                        #win_i, win_start = 15, start_window[15]
                        for win_i, win_start in enumerate(start_window):

                            if win_start == start_window[-1]:
                                continue
                            
                            win_stop = start_window[win_i+1]

                            _fc_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
                            _cluster_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

                            #pair_i, pair = 2, pairs_to_compute[2]
                            for pair_i, pair in enumerate(pairs_to_compute):

                                A, B = pair.split('-')
                                A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                                if fc_metric == 'MI':
                                    data_chunk_diff = shifted_fc_allsujet_rscore[:, pair_i, win_start:win_stop].median('sujet').values
                                    _clusters = clusters_time[pair_i, win_start:win_stop].values
                                else:
                                    data_chunk_diff = shifted_fc_allsujet_rscore[:, pair_i, band_i, win_start:win_stop] .median('sujet').values
                                    _clusters = clusters_time[band_i, pair_i, win_start:win_stop].values

                                fc_val = np.median(data_chunk_diff)
                                _fc_mat[A_i, B_i], _fc_mat[B_i, A_i] = fc_val, fc_val
                                
                                cluster_val = _clusters.sum() > 0
                                _cluster_mat[A_i, B_i], _cluster_mat[B_i, A_i] = cluster_val, cluster_val

                            data_chunk_wins[win_i, :,:] = _fc_mat
                            cluster_mask_wins[win_i, :,:] = _cluster_mat

                        return n_times, start_window, data_chunk_wins, cluster_mask_wins
                    
                    def get_visu_data_matsigni(time_window):

                        start_window = np.arange(0, time_vec.size, time_window)
                        n_times = np.arange(start_window.size)
                        
                        cluster_mask_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))
                        data_chunk_wins = np.zeros((start_window.size, len(chan_list_eeg_short), len(chan_list_eeg_short)))

                        #win_i, win_start = 23, start_window[23]
                        for win_i, win_start in enumerate(start_window):

                            if win_start == start_window[-1]:
                                continue
                            
                            win_stop = start_window[win_i+1]

                            _fc_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
                            _cluster_mat = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))

                            #pair_i, pair = 2, pairs_to_compute[2]
                            for pair_i, pair in enumerate(pairs_to_compute):
                                A, B = pair.split('-')
                                A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                                if fc_metric == 'MI':
                                    data_chunk_diff = shifted_fc_allsujet_rscore[:, pair_i, win_start:win_stop].median('sujet').values
                                    _clusters = clusters_time[pair_i, win_start:win_stop].values
                                else:
                                    data_chunk_diff = shifted_fc_allsujet_rscore[:, pair_i, band_i, win_start:win_stop].median('sujet').values
                                    _clusters = clusters_time[band_i, pair_i, win_start:win_stop].values

                                fc_val = np.median(data_chunk_diff)
                                if _clusters.sum() > 0:
                                    _fc_mat[A_i, B_i], _fc_mat[B_i, A_i] = fc_val, fc_val
                                
                                cluster_val = _clusters.sum() > 0
                                _cluster_mat[A_i, B_i], _cluster_mat[B_i, A_i] = cluster_val, cluster_val

                            data_chunk_wins[win_i, :,:] = _fc_mat
                            cluster_mask_wins[win_i, :,:] = _cluster_mat

                        return n_times, start_window, data_chunk_wins, cluster_mask_wins
                    
                    def update_fullmat(frame):
                            
                        ax.clear()

                        ax.set_title(f'{np.round(time_vec[start_window[frame]], 5)}')
                        
                        ax.imshow(data_chunk_wins[frame,:,:], cmap='seismic', vmin=-vlim, vmax=vlim)

                        for i in range(chan_list_eeg_short.shape[0]):
                            for j in range(chan_list_eeg_short.shape[0]):
                                if cluster_mask_wins[frame, i, j]:
                                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                                    edgecolor='green', facecolor='none', lw=4)
                                    ax.add_patch(rect)

                        phase_title = [phase for phase in phase_vec if time_vec[start_window[frame]] in phase_vec[phase]][0]
                        ax.set_xlabel("Electrodes")
                        ax.set_ylabel("Electrodes")
                        ax.set_title(f"{fc_metric} {phase_title} : {np.round(time_vec[start_window[frame]], 2)}")

                        return [ax]
                    
                    def update_matsigni(frame):
                            
                        ax.clear()
                        
                        ax.set_title(f'{np.round(time_vec[start_window[frame]], 5)}')
                        
                        ax.imshow(data_chunk_wins[frame,:,:], cmap='seismic', vmin=-vlim, vmax=vlim)

                        ax.set_xticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short, rotation=90)
                        ax.set_yticks(ticks=np.arange(chan_list_eeg_short.shape[0]), labels=chan_list_eeg_short)

                        phase_title = [phase for phase in phase_vec if time_vec[start_window[frame]] in phase_vec[phase]][0]
                        ax.set_xlabel("Electrodes")
                        ax.set_ylabel("Electrodes")
                        ax.set_title(f"{fc_metric} {phase_title} : {np.round(time_vec[start_window[frame]], 2)}")

                        return [ax]
                    
                    #fc_type = 'fullmat'
                    for fc_type in ['fullmat', 'matsigni']:

                        time_window = 10
                        if fc_type == 'fullmat':
                            n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data_fullmat(time_window)
                        elif fc_type == 'matsigni':
                            n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data_matsigni(time_window)
                        vlim = np.abs((data_chunk_wins.min(), data_chunk_wins.max())).max()

                        if debug:

                            win_i = 10

                            plt.matshow(data_chunk_wins[win_i,:,:], cmap='seismic')

                            for i in range(chan_list_eeg_short.shape[0]):
                                for j in range(chan_list_eeg_short.shape[0]):
                                    if cluster_mask_wins[win_i, i, j]:
                                        plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='green', facecolor='none', lw=2))

                            plt.colorbar(label='Connectivity Strength')
                            plt.clim(-vlim, vlim)
                            plt.xticks(ticks=np.arange(fc_mat_whole.shape[0]), labels=chan_list_eeg_short, rotation=90)
                            plt.yticks(ticks=np.arange(fc_mat_whole.shape[0]), labels=chan_list_eeg_short)
                            plt.xlabel("Electrodes")
                            plt.ylabel("Electrodes")
                            if fc_metric == 'MI':
                                plt.title(f"MI FC start{np.round(time_vec[start_window[win_i]], 2)}")
                            else:
                                plt.title(f"{fc_metric} {band} FC start{np.round(time_vec[start_window[win_i]], 2)}")
                            plt.show()

                        # Create topoplot frames
                        fig, ax = plt.subplots()
                        cax = ax.matshow(np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short))), cmap='seismic', vmin=-vlim_whole, vmax=vlim_whole)
                        colorbar = plt.colorbar(cax, ax=ax)

                        # Animation
                        if fc_type == 'fullmat':
                            ani = FuncAnimation(fig, update_fullmat, frames=n_times, interval=1000)
                        elif fc_type == 'matsigni':
                            ani = FuncAnimation(fig, update_matsigni, frames=n_times, interval=1000)
                        # plt.show()

                        os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))
                        
                        if fc_metric == 'MI':
                            ani.save(f"stretch_{fc_metric}_FC_mat_animation_allsujet_{fc_type}.gif", writer="pillow")
                        else:
                            ani.save(f"stretch_{fc_metric}_{band}_FC_mat_animation_allsujet_{fc_type}.gif", writer="pillow")  
                        
                        if fc_type == 'matsigni':

                            time_window = 10
                            n_times, start_window, data_chunk_wins, cluster_mask_wins = get_visu_data_matsigni(time_window)
                            vlim = np.abs((data_chunk_wins.min(), data_chunk_wins.max())).max()

                            # Create graph
                            G = nx.Graph()
                            G.add_nodes_from(eeg_positions.keys())

                            # Define the update function for animation
                            def update_graph(frame):

                                plt.clf()
                                ax = plt.gca()

                                phase_title = [phase for phase in phase_vec if time_vec[start_window[frame]] in phase_vec[phase]][0]
                                ax.set_title(f"{fc_metric} {phase_title} : {np.round(time_vec[start_window[frame]], 2)}")
                                
                                # Clear and add new edges
                                G.clear()
                                G.add_nodes_from(eeg_positions.keys())
                                
                                mat_connectivity = data_chunk_wins[frame,:,:]
                                
                                for i, ch1 in enumerate(eeg_positions.keys()):
                                    for j, ch2 in enumerate(eeg_positions.keys()):
                                        if i < j:  # Avoid duplicate edges
                                            weight = mat_connectivity[i, j]
                                            color = 'red' if weight > 0 else 'blue'
                                            G.add_edge(ch1, ch2, weight=abs(weight) * 5, color=color)
                                
                                # Draw the graph
                                edges = G.edges()
                                edge_colors = [G[u][v]['color'] for u, v in edges]
                                edge_widths = [G[u][v]['weight'] for u, v in edges]
                                
                                nx.draw(G, eeg_positions, with_labels=True, node_size=500, font_size=8, edge_color=edge_colors, width=edge_widths, ax=ax)
                                
                                # Add colorbar
                                sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-vlim, vmax=vlim))
                                sm.set_array([])
                                plt.colorbar(sm, ax=ax, label='Connectivity Strength')

                            # Create animation
                            fig = plt.figure()
                            ani = FuncAnimation(fig, update_graph, frames=n_times, interval=1000)
                            # plt.show()

                            os.chdir(os.path.join(path_results, 'FC', fc_metric, 'matplot'))
                        
                            if fc_metric == 'MI':
                                ani.save(f"stretch_{fc_metric}_FC_graph_animation_allsujet_{fc_type}.gif", writer="pillow")
                            else:
                                ani.save(f"stretch_{fc_metric}_{band}_FC_graph_animation_allsujet_{fc_type}.gif", writer="pillow")  






def plot_allsujet_FC_graph_stretch():

    #### load
    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')        

    #fc_metric = 'WPLI'
    for fc_metric in ['MI', 'ISPC', 'WPLI']:

        #data_type = 'rscore'
        for data_type in ['raw', 'rscore']:

            print(f'{fc_metric} {data_type} PLOT stretch', flush=True)

            if fc_metric == 'MI':
                fc_allsujet = np.zeros((len(sujet_list_FC), len(pairs_to_compute), stretch_point_FC))
            else:
                fc_allsujet = np.zeros((len(sujet_list_FC), len(pairs_to_compute), len(freq_band_fc_list), stretch_point_FC))

            os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

            for sujet_i, sujet in enumerate(sujet_list_FC):

                print_advancement(sujet_i, len(sujet_list_FC))

                if data_type == 'raw':
                    if fc_metric == 'MI':
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_stretch_{sujet}.nc')
                        _fc_sujet = _fc_sujet.loc['CHARGE'] - _fc_sujet.loc['VS']
                        _fc_sujet = _fc_sujet.median('ntrials')                
                    else:
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch.nc')
                        _fc_sujet = _fc_sujet.loc[:, 'CHARGE'] - _fc_sujet.loc[:, 'VS']
                        _fc_sujet = _fc_sujet.median('cycle')
                else:
                    if fc_metric == 'MI':
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_stretch_{sujet}_rscore.nc')
                        _fc_sujet = _fc_sujet.loc['CHARGE'] - _fc_sujet.loc['VS']
                        _fc_sujet = _fc_sujet.median('ntrials')                
                    else:
                        _fc_sujet = xr.open_dataarray(f'{fc_metric}_{sujet}_stretch_rscore.nc')
                        _fc_sujet = _fc_sujet.loc[:, 'CHARGE'] - _fc_sujet.loc[:, 'VS']
                        _fc_sujet = _fc_sujet.median('cycle')

                fc_allsujet[sujet_i] = _fc_sujet

            if fc_metric == 'MI':
                fc_allsujet_dict = {'sujet' : sujet_list_FC, 'pair' : pairs_to_compute, 'time' : np.arange(stretch_point_FC)}
            else:
                fc_allsujet_dict = {'sujet' : sujet_list_FC, 'pair' : pairs_to_compute, 'band' : freq_band_fc_list, 'time' : np.arange(stretch_point_FC)}

            fc_allsujet = xr.DataArray(data=fc_allsujet, dims=fc_allsujet_dict.keys(), coords=fc_allsujet_dict.values())

            time_vec = fc_allsujet['time'].values
            phase_list = ['I', 'T_IE', 'E', 'T_EI']
            phase_shift = int(stretch_point_FC/4) 
            phase_vec = {'I' : np.arange(phase_shift), 'T_IE' : np.arange(phase_shift)+phase_shift, 
                        'E' : np.arange(phase_shift)+phase_shift*2, 'T_EI' : np.arange(phase_shift)+phase_shift*3} 
            
            shifted_fc_allsujet = fc_allsujet.roll(time=-phase_shift, roll_coords=False)

            #band_i, band = 0, freq_band_fc_list[0]
            for band_i, band in enumerate(freq_band_fc_list):

                #stat_type = 'HOMEMADE'
                for stat_type in ['HOMEMADE', 'MNE']:

                    os.chdir(os.path.join(path_precompute, 'FC', fc_metric))

                    if stat_type == 'HOMEMADE':
                        clusters = xr.open_dataarray(f'{fc_metric}_allsujet_STATS_state_stretch.nc')
                    else:
                        clusters = xr.open_dataarray(f'{fc_metric}_mne_allsujet_STATS_state_stretch.nc')

                    fc_mat_whole = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    fc_mat_phase = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    fc_mat_mask_signi_phase = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    fc_mat_only_signi_whole = np.zeros((len(chan_list_eeg_short), len(chan_list_eeg_short)))
                    fc_mat_only_signi_phase = np.zeros((len(phase_list), len(chan_list_eeg_short), len(chan_list_eeg_short)))

                    #### whole
                    if fc_metric == 'MI':
                        fc_mat_mask_signi_whole = from_pairs_2mat(clusters.loc[data_type, 'whole',:], pairs_to_compute)
                    else:
                        fc_mat_mask_signi_whole = from_pairs_2mat(clusters.loc[data_type, 'whole',band,:], pairs_to_compute)

                    #pair_i, pair = 2, pairs_to_compute[2]
                    for pair_i, pair in enumerate(pairs_to_compute):

                        A, B = pair.split('-')
                        A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                        if fc_metric == 'MI':
                            data_chunk_diff = shifted_fc_allsujet.loc[:, pair].median('sujet').values
                            
                        else:
                            data_chunk_diff = shifted_fc_allsujet.loc[:, pair, band].median('sujet').values

                        fc_val = np.median(data_chunk_diff)

                        fc_mat_whole[A_i, B_i], fc_mat_whole[B_i, A_i] = fc_val, fc_val

                        if fc_metric == 'MI' and clusters.loc[data_type, 'whole',pair].values.astype('bool'):
                            fc_mat_only_signi_whole[A_i, B_i], fc_mat_only_signi_whole[B_i, A_i] = fc_val, fc_val
                        elif fc_metric != 'MI' and clusters.loc[data_type, 'whole',band,pair].values.astype('bool'):
                            fc_mat_only_signi_whole[A_i, B_i], fc_mat_only_signi_whole[B_i, A_i] = fc_val, fc_val

                    #### phase

                    #phase_i, phase = 0, 'I'
                    for phase_i, phase in enumerate(phase_list):

                        if fc_metric == 'MI':
                            fc_mat_mask_signi_phase[phase_i,:,:] = from_pairs_2mat(clusters.loc[data_type, phase,:], pairs_to_compute)
                        else:
                            fc_mat_mask_signi_phase[phase_i,:,:] = from_pairs_2mat(clusters.loc[data_type, phase,band,:], pairs_to_compute)

                        #pair_i, pair = 2, pairs_to_compute[2]
                        for pair_i, pair in enumerate(pairs_to_compute):

                            A, B = pair.split('-')
                            A_i, B_i = np.where(chan_list_eeg_short == A)[0][0], np.where(chan_list_eeg_short == B)[0][0]

                            if fc_metric == 'MI':
                                data_chunk_diff = shifted_fc_allsujet.loc[:, pair, phase_vec[phase]].median('sujet').values
                                
                            else:
                                data_chunk_diff = shifted_fc_allsujet.loc[:, pair, band, phase_vec[phase]].median('sujet').values

                            fc_val = np.median(data_chunk_diff)

                            fc_mat_phase[phase_i, A_i, B_i], fc_mat_phase[phase_i, B_i, A_i] = fc_val, fc_val

                            if fc_metric == 'MI' and clusters.loc[data_type, phase,pair].values.astype('bool'):
                                fc_mat_only_signi_phase[phase_i, A_i, B_i], fc_mat_only_signi_phase[phase_i, B_i, A_i] = fc_val, fc_val
                            elif fc_metric != 'MI' and clusters.loc[data_type, phase,band,pair].values.astype('bool'):
                                fc_mat_only_signi_phase[phase_i, A_i, B_i], fc_mat_only_signi_phase[phase_i, B_i, A_i] = fc_val, fc_val

                    #mat, mask_graph_metric = fc_mat[0, 0, :, :], fc_mat_mask_signi[0,0,:,:]
                    def thresh_fc_mat(mat, mode='mask', percentile_graph_metric=50, mask_graph_metric=None):

                        if mode == 'percentile':

                            mat_thresh = mat.copy()

                            mat_values = mat[np.triu_indices(mat.shape[0], k=1)]
                            
                            if debug:
                                np.sum(mat_values > np.percentile(mat_values, 90))

                                count, bin, fig = plt.hist(mat_values)
                                plt.vlines(np.percentile(mat_values, 99), ymin=count.min(), ymax=count.max(), color='r')
                                plt.vlines(np.percentile(mat_values, 95), ymin=count.min(), ymax=count.max(), color='r')
                                plt.vlines(np.percentile(mat_values, 90), ymin=count.min(), ymax=count.max(), color='r')
                                plt.vlines(np.percentile(mat_values, 75), ymin=count.min(), ymax=count.max(), color='r')
                                plt.show()

                            #### apply thresh
                            for chan_i in range(mat.shape[0]):
                                mat_thresh[chan_i,:][np.where(mat[chan_i,:] < np.percentile(mat_values, percentile_graph_metric))[0]] = 0

                        if mode == 'mask':

                            mat_thresh = mat * mask_graph_metric
                            
                        if mat_thresh.sum() == 0:

                            return mat_thresh

                        #### verify that the graph is fully connected
                        chan_i_to_remove = []
                        for chan_i in range(mat_thresh.shape[0]):
                            if np.sum(mat_thresh[chan_i,:]) == 0:
                                chan_i_to_remove.append(chan_i)

                        mat_thresh_i_mask = [i for i in range(mat_thresh.shape[0]) if i not in chan_i_to_remove]

                        if len(chan_i_to_remove) != 0:
                            for row in range(2):
                                if row == 0:
                                    mat_thresh = mat_thresh[mat_thresh_i_mask,:]
                                elif row == 1:
                                    mat_thresh = mat_thresh[:,mat_thresh_i_mask]

                        if debug:
                            plt.imshow(mat_thresh)
                            plt.show()

                        return mat_thresh
                    
                    #### node wise
                    df_graph_metrics_node_wise = pd.DataFrame()

                    for mode in ['percentile', 'mask']:

                        for sujet_i, sujet in enumerate(sujet_list_FC):

                            for phase_i, phase in enumerate(phase_list):

                                if fc_mat_mask_signi_phase[phase_i,:,:].sum() == 0:

                                    _df = pd.DataFrame({'sujet' : [sujet], 'phase' : [phase], 'mode' : [mode], 'chan' : [chan], 'degree' : [0], 'clustering_coeff' : [0], 
                                                'betweenness' : [0], 'eigenvector' : [0]})

                                    df_graph_metrics_node_wise = pd.concat((df_graph_metrics_node_wise, _df))

                                else:

                                    # Create a graph from the adjacency matrix
                                    mat = thresh_fc_mat(fc_mat_phase[phase_i,:,:], mode=mode, percentile_graph_metric=50 ,mask_graph_metric=fc_mat_mask_signi_phase[phase_i,:,:])
                                    graph = nx.from_numpy_array(mat)
                                    nx.relabel_nodes(graph, mapping=dict(enumerate(chan_list_eeg_short)), copy=False)

                                    if debug:
                                        # Plot the graph
                                        plt.figure(figsize=(10, 8))
                                        pos = nx.spring_layout(graph)  # Layout for visualization
                                        nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=10)
                                        plt.title("Graph Visualization")
                                        plt.show()

                                    #### metrics
                                    degree = nx.degree_centrality(graph)
                                    # Formula: degree_centrality(v) = degree(v) / (n - 1), where n is the number of nodes

                                    clustering_coeff = nx.clustering(graph)

                                    betweenness = nx.betweenness_centrality(graph)
                                    # Formula: betweenness_centrality(v) = sum of (shortest paths through v / total shortest paths)

                                    eigenvector = nx.eigenvector_centrality(graph)

                                    for chan in chan_list_eeg_short:

                                        try:
                                            _df = pd.DataFrame({'sujet' : [sujet], 'phase' : [phase], 'mode' : [mode], 'chan' : [chan], 'degree' : [degree[chan]], 'clustering_coeff' : [clustering_coeff[chan]],
                                                                'betweenness' : [betweenness[chan]], 'eigenvector' : [eigenvector[chan]]})

                                            df_graph_metrics_node_wise = pd.concat((df_graph_metrics_node_wise, _df))

                                        except:

                                            _df = pd.DataFrame({'sujet' : [sujet], 'phase' : [phase], 'mode' : [mode], 'chan' : [chan], 'degree' : [0], 'clustering_coeff' : [0], 
                                                    'betweenness' : [0], 'eigenvector' : [0]})

                                            df_graph_metrics_node_wise = pd.concat((df_graph_metrics_node_wise, _df))

                    #### graph wise
                    df_graph_metrics_graph_wise = pd.DataFrame()

                    for mode in ['percentile', 'mask']:

                        for sujet_i, sujet in enumerate(sujet_list_FC):

                            for phase_i, phase in enumerate(phase_list):

                                if fc_mat_mask_signi_phase[phase_i,:,:].sum() == 0:

                                    _df = pd.DataFrame({'sujet' : [sujet], 'phase' : [phase], 'mode' : [mode], 'global_efficiency' : [0], 'path_length' : [0],
                                                'small_worldness' : [0], 'modularity' : [0]})

                                    df_graph_metrics_graph_wise = pd.concat((df_graph_metrics_graph_wise, _df))

                                else:

                                    # Create a graph from the adjacency matrix
                                    mat = thresh_fc_mat(fc_mat_phase[phase_i,:,:], mode=mode, percentile_graph_metric=50 ,mask_graph_metric=fc_mat_mask_signi_phase[phase_i,:,:])
                                    graph = nx.from_numpy_array(mat)
                                    nx.relabel_nodes(graph, mapping=dict(enumerate(chan_list_eeg_short)), copy=False)

                                    if debug:
                                        # Plot the graph
                                        plt.figure(figsize=(10, 8))
                                        pos = nx.spring_layout(graph)  # Layout for visualization
                                        nx.draw(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=10)
                                        plt.title("Graph Visualization")
                                        plt.show()

                                    #### metrics
                                    global_efficiency = nx.global_efficiency(graph)

                                    # Handle disconnected graph for path length calculation
                                    if nx.is_connected(graph):
                                        path_length = nx.average_shortest_path_length(graph)
                                    else:
                                        # Compute average shortest path length per connected component
                                        components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
                                        path_lengths = [nx.average_shortest_path_length(comp) for comp in components]
                                        path_length = np.mean(path_lengths)  # Take the mean over components

                                    # Compute approximate small-worldness
                                    if path_length > 0:
                                        small_worldness = nx.transitivity(graph) / path_length
                                    else:
                                        small_worldness = 0  # Avoid division by zero

                                    # Compute modularity (handling single communities)
                                    communities = list(nx.community.greedy_modularity_communities(graph))
                                    if len(communities) > 1:
                                        modularity = nx.community.modularity(graph, communities)
                                    else:
                                        modularity = 0  # If there's only one community, modularity is undefined

                                    try:
                                        _df = pd.DataFrame({'sujet' : [sujet], 'phase' : [phase], 'mode' : [mode], 'global_efficiency' : [global_efficiency], 'path_length' : [path_length],
                                                            'small_worldness' : [small_worldness], 'modularity' : [modularity]})

                                        df_graph_metrics_graph_wise = pd.concat((df_graph_metrics_graph_wise, _df))

                                    except:

                                        _df = pd.DataFrame({'sujet' : [sujet], 'phase' : [phase], 'mode' : [mode], 'global_efficiency' : [0], 'path_length' : [0], 
                                                'small_worldness' : [0], 'modularity' : [0]})

                                        df_graph_metrics_graph_wise = pd.concat((df_graph_metrics_graph_wise, _df))

                    #### plot node wise
                    os.chdir(os.path.join(path_results, 'FC', fc_metric, 'graph'))

                    #metric = 'degree'
                    for metric in ['degree', 'clustering_coeff', 'betweenness', 'eigenvector']:

                        for mode in ['mask', 'percentile']: 

                            fig, ax = plt.subplots(figsize=(15, 5), sharey=True)

                            df_plot = df_graph_metrics_node_wise.query(f"mode == '{mode}'")
                            sns.barplot(data=df_plot, x="chan", y=metric, hue="phase", alpha=0.6, ax=ax)
                            ax.set_xlabel("Channel")
                            if phase_i == 0:
                                ax.set_ylabel(metric)
                            else:
                                ax.set_ylabel("")

                            if debug:

                                df_plot = df_graph_metrics_node_wise.query(f"mode == '{mode}' and cond =='VS'")
                                fig, ax = plt.subplots()
                                sns.barplot(data=df_plot, x="chan", y=metric, hue='phase', alpha=0.6, ax=ax)
                                plt.show()

                            if fc_metric == 'MI':
                                plt.suptitle(f"{metric} mode:{mode} {data_type} {stat_type}")
                            else :
                                plt.suptitle(f"{metric} {band} mode:{mode} {data_type} {stat_type}")

                            # plt.show()

                            if fc_metric == 'MI':
                                plt.savefig(f"NODES_stretch_{fc_metric}_{mode}_graph_{metric}_{data_type}_{stat_type}.png")
                            else:
                                plt.savefig(f"NODES_stretch_{fc_metric}_{band}_{mode}_graph_{metric}_{data_type}_{stat_type}.png")

                            plt.close("all")

                    #### plot graph wise
                    os.chdir(os.path.join(path_results, 'FC', fc_metric, 'graph'))

                    #mode = 'mask'
                    for mode in ['mask', 'percentile']: 

                        fig, axs = plt.subplots(ncols=4, figsize=(15, 5), sharey=True)

                        for metric_i, metric in enumerate(['global_efficiency', 'path_length', 'small_worldness', 'modularity']):

                            ax = axs[metric_i]

                            df_plot = df_graph_metrics_graph_wise.query(f"mode == '{mode}'")
                            sns.barplot(data=df_plot, hue="phase", y=metric, alpha=0.6, ax=ax)
                            ax.set_title(metric) 

                        if debug:

                            df_plot = df_graph_metrics_graph_wise.query(f"mode == '{mode}' and cond =='VS'")
                            fig, ax = plt.subplots()
                            sns.barplot(data=df_plot, x="chan", y=metric, hue='phase', alpha=0.6, ax=ax)
                            plt.show()

                        if fc_metric == 'MI':
                            plt.suptitle(f"mode:{mode} {data_type} {stat_type}")
                        else :
                            plt.suptitle(f"{band} mode:{mode} {data_type} {stat_type}")

                        # plt.show()

                        if fc_metric == 'MI':
                            plt.savefig(f"GRAPHS_stretch_{fc_metric}_{mode}_graph_{data_type}_{stat_type}.png")
                        else:
                            plt.savefig(f"GRAPHS_stretch_{fc_metric}_{band}_{mode}_graph_{data_type}_{stat_type}.png")

                        plt.close("all")









################################
######## EXECUTE ########
################################


if __name__ == '__main__':

    plot_allsujet_FC_time_stretch()
    plot_allsujet_FC_mat_stretch()
    plot_allsujet_FC_graph_stretch()




