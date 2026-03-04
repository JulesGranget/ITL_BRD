

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import joblib
import xarray as xr

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False







########################################
######## MUTUAL INFORMATION ########
########################################

#sujet = sujet_list[0]
def get_MI_sujet_stretch(sujet):

    #### verify computation
    if os.path.exists(os.path.join(path_precompute, 'FC', 'MI', f'MI_stretch_{sujet}_beta.nc')):
        print(f'{sujet} ALREADY DONE MI STRETCH')
        return
    
    print(sujet)

    #### params
    os.chdir(path_prep)
    # xr_norm_params = xr.open_dataarray('norm_params.nc')
    respfeatures = load_respfeatures(sujet)

    cond_sel = ['VS', 'CHARGE']

    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')        

    #### container
    MI_sujet = np.zeros((len(cond_sel), len(pairs_to_compute), nrespcycle_FC, stretch_point_FC))

    #cond_i, cond = 0, 'VS'
    for cond_i, cond in enumerate(cond_sel):

        print(cond)

        data = load_data_sujet_CSD(sujet, cond)

        data_rscore = data.copy()
        for chan_i, chan in enumerate(chan_list_eeg):

            data_rscore[chan_i] = (data[chan_i] - np.median(data[chan_i])) * 0.6745 / scipy.stats.median_abs_deviation(data[chan_i])
        
            # data_rscore = (data - xr_norm_params.loc[sujet, 'CSD', 'median', chan].values.reshape(-1,1)) * 0.6745 / xr_norm_params.loc[sujet, 'CSD', 'mad', :].values.reshape(-1,1)
        
        if debug:

            plt.plot(scipy.stats.zscore(data[0,:]), label='raw')
            plt.plot(scipy.stats.zscore(data_rscore[0,:]), label='rscore')
            plt.legend()
            plt.show()

        respfeatures_i = respfeatures[cond]

        #pair_i, pair = 0, pairs_to_compute[0]
        for pair_i, pair in enumerate(pairs_to_compute):

            print_advancement(pair_i, len(pairs_to_compute), [25,50,75])

            A, B = pair.split('-')[0], pair.split('-')[1]

            x = data[chan_list_eeg.tolist().index(A),:]
            y = data[chan_list_eeg.tolist().index(B),:]

            #### pad sig for the conv
            x_pad = np.pad(x, int(MI_window_size*fc_win_overlap/2), mode='reflect')
            y_pad = np.pad(y, int(MI_window_size*fc_win_overlap/2), mode='reflect')

            #### slide
            win_vec = np.arange(0, x_pad.size-int(MI_window_size*fc_win_overlap/2), int(MI_window_size*fc_win_overlap/2)).astype('int')
            MI_pair = []
            for i in win_vec:
                MI_pair.append(get_MI_2sig(x_pad[i:i+MI_window_size], y_pad[i:i+MI_window_size]))
            MI_pair = np.array(MI_pair)

            #### interpol
            f = scipy.interpolate.interp1d(np.linspace(0, x.size, MI_pair.size), MI_pair)
            MI_pair_interp = f(np.arange(x.size))

            # MI_conv_pair = np.random.rand(x.size)

            if debug:

                plt.plot(np.linspace(0, x.size, MI_pair.size), MI_pair)
                plt.plot(np.arange(x.size), MI_pair_interp)
                plt.show()

            #### stretch
            MI_stretch_pair, mean_inspi_ratio = stretch_data(respfeatures_i, stretch_point_FC, MI_pair_interp, srate)

            #### cap epochs
            MI_stretch_pair = MI_stretch_pair[:nrespcycle_FC,:]

            if debug:

                for cycle_i in range(MI_stretch_pair.shape[0]):

                    plt.plot(MI_stretch_pair[cycle_i,:], alpha=0.2)

                plt.plot(np.median(MI_stretch_pair, axis=0), color='r')
                plt.show()

            MI_sujet[cond_i, pair_i, :, :] = MI_stretch_pair

    #### export
    MI_dict = {'cond' : cond_sel, 'pair' : pairs_to_compute, 'ntrials' : np.arange(nrespcycle_FC), 'time' : np.arange(stretch_point_FC)}
    xr_MI_sujet = xr.DataArray(data=MI_sujet, dims=MI_dict.keys(), coords=MI_dict.values())

    MI_sujet_rscore = MI_sujet.copy()
    for cond_i, cond in enumerate(cond_list):
        for pair_i, pair in enumerate(pairs_to_compute):
            for cycle_i in range(nrespcycle_FC):
                MI_sujet_rscore[cond_i, pair_i, cycle_i] = (MI_sujet[cond_i, pair_i, cycle_i, :] - np.median(MI_sujet[cond_i, pair_i, cycle_i, :])) * 0.6745 / scipy.stats.median_abs_deviation(MI_sujet[cond_i, pair_i, cycle_i, :])
        
    xr_MI_sujet_rscore = xr.DataArray(data=MI_sujet_rscore, dims=MI_dict.keys(), coords=MI_dict.values())

    if debug:
        
        pair_i = 0

        fc_diff = xr_MI_sujet.values[1, pair_i] - xr_MI_sujet.values[0, pair_i]

        for cycle_i in range(nrespcycle_FC):

            plt.plot(fc_diff[cycle_i], alpha=0.2)
            
        plt.plot(np.median(fc_diff, axis=0), color='r')
        plt.show()

        fc_diff = xr_MI_sujet_rscore.values[1, pair_i] - xr_MI_sujet_rscore.values[0, pair_i]

        for cycle_i in range(nrespcycle_FC):

            plt.plot(fc_diff[cycle_i], alpha=0.2)
            
        plt.plot(np.median(fc_diff, axis=0), color='r')
        plt.show()

        plt.plot(np.median(xr_MI_sujet_rscore.values[1, pair_i] - xr_MI_sujet_rscore.values[0, pair_i], axis=0), color='r')
        plt.plot(np.median(xr_MI_sujet.values[1, pair_i] - xr_MI_sujet.values[0, pair_i], axis=0), color='r')
        plt.show()
    
    os.chdir(os.path.join(path_precompute, 'FC', 'MI'))
    xr_MI_sujet.to_netcdf(f'MI_stretch_{sujet}_beta.nc')
    xr_MI_sujet_rscore.to_netcdf(f'MI_stretch_{sujet}_rscore_beta.nc')

    print('done')












################################
######## WPLI ISPC ######## 
################################


#sujet = sujet_list_FC[38]
def get_ISPC_WPLI_stretch(sujet):
    
    # Check if results already exist
    ispc_path = os.path.join(path_precompute, 'FC', 'ISPC', f'ISPC_{sujet}_stretch_beta.nc')
    wpli_path = os.path.join(path_precompute, 'FC', 'WPLI', f'WPLI_{sujet}_stretch_beta.nc')
    
    if os.path.exists(ispc_path) and os.path.exists(wpli_path):
        print(f'{sujet} ALREADY DONE')
        return
    
    print(sujet)

    cond_sel = ['VS', 'CHARGE']
    
    pairs_to_compute = []

    for pair_A in chan_list_eeg_short:

        for pair_B in chan_list_eeg_short:

            if pair_A == pair_B or f'{pair_A}-{pair_B}' in pairs_to_compute or f'{pair_B}-{pair_A}' in pairs_to_compute:
                continue

            pairs_to_compute.append(f'{pair_A}-{pair_B}')        
    
    #pair_i = 0
    def compute_pair(pair_i):

        os.chdir(path_prep)
        # xr_norm_params = xr.open_dataarray('norm_params.nc')
        respfeatures_sujet = load_respfeatures(sujet)
    
        pair = pairs_to_compute[pair_i]
        pair_A, pair_B = pair.split('-')
        idx_A, idx_B = np.where(chan_list_eeg_short == pair_A)[0][0], np.where(chan_list_eeg_short == pair_B)[0][0]

        print(f"{pair} {int(pair_i*100/len(pairs_to_compute))}%", flush=True)
        
        ISPC_all_bands = np.zeros((len(cond_sel), len(freq_band_fc_list), nrespcycle_FC, stretch_point_FC))
        WPLI_all_bands = np.zeros((len(cond_sel), len(freq_band_fc_list), nrespcycle_FC, stretch_point_FC))
        
        for band_i, band in enumerate(freq_band_fc_list):

            wavelets = get_wavelets_fc(freq_band_fc[band])
            win_slide = ISPC_window_size[band]
            
            for cond_i, cond in enumerate(cond_sel):

                respfeatures_sujet_chunk = respfeatures_sujet[cond][:nrespcycle_FC+20] 
                len_sig_to_analyze = respfeatures_sujet_chunk['next_inspi_index'].values[-1]+1*srate

                data = load_data_sujet_CSD(sujet, cond)[:,:len_sig_to_analyze]
                data = data[[idx_A, idx_B],:]

                data_rscore = data.copy()

                for i in range(2):

                    data_rscore[i] = (data[i] - np.median(data[i])) * 0.6745 / scipy.stats.median_abs_deviation(data[i])

                    # data_rscore = (data - xr_norm_params.loc[sujet, 'CSD', 'median', [pair_A, pair_B]].values.reshape(-1,1))
                    #               * 0.6745 / xr_norm_params.loc[sujet, 'CSD', 'mad', [pair_A, pair_B]].values.reshape(-1,1)
                                
                convolutions = np.array([
                    np.array([scipy.signal.fftconvolve(data_rscore[ch, :], wavelet, 'same') for wavelet in wavelets])
                    for ch in [0, 1]
                ])
                
                x_conv, y_conv = convolutions[0], convolutions[1]

                #### slide
                x_pad = np.pad(x_conv, ((0, 0), (int(win_slide*fc_win_overlap/2), int(win_slide*fc_win_overlap/2))), mode='reflect')
                y_pad = np.pad(y_conv, ((0, 0), (int(win_slide*fc_win_overlap/2), int(win_slide*fc_win_overlap/2))), mode='reflect')

                win_vec = np.arange(0, x_conv.shape[-1]-int(win_slide*fc_win_overlap/2), int(win_slide*fc_win_overlap/2)).astype('int')

                ISPC_slide = np.zeros((x_conv.shape[0], win_vec.size))
                WPLI_slide = np.zeros((x_conv.shape[0], win_vec.size))
                for wavelet_i in range(x_conv.shape[0]):
                    for win_i in range(win_vec.size):
                        ISPC_slide[wavelet_i,win_i] = get_ISPC_2sig(x_pad[wavelet_i,win_vec[win_i]:win_vec[win_i]+win_slide], y_pad[wavelet_i,win_vec[win_i]:win_vec[win_i]+win_slide])
                        WPLI_slide[wavelet_i,win_i] = get_WPLI_2sig(x_pad[wavelet_i,win_vec[win_i]:win_vec[win_i]+win_slide], y_pad[wavelet_i,win_vec[win_i]:win_vec[win_i]+win_slide])

                ISPC_slide = np.median(ISPC_slide, axis=0)
                WPLI_slide = np.median(WPLI_slide, axis=0)

                #### interpol
                f = scipy.interpolate.interp1d(np.linspace(0, x_conv.shape[-1], ISPC_slide.size), ISPC_slide)
                ISPC_slide_interp = f(np.arange(x_conv.shape[-1]))

                f = scipy.interpolate.interp1d(np.linspace(0, x_conv.shape[-1], WPLI_slide.size), ISPC_slide)
                WPLI_slide_interp = f(np.arange(x_conv.shape[-1]))
                
                ISPC_stretch = stretch_data(respfeatures_sujet_chunk, stretch_point_FC, ISPC_slide_interp, srate)[0]
                WPLI_stretch = stretch_data(respfeatures_sujet_chunk, stretch_point_FC, WPLI_slide_interp, srate)[0]

                if debug:

                    for cycle_i in range(ISPC_stretch.shape[0]):

                        plt.plot(ISPC_stretch[cycle_i,:], alpha=0.2)

                    plt.plot(np.median(ISPC_stretch, axis=0), color='r')
                    plt.show()
                
                ISPC_all_bands[cond_i, band_i] = ISPC_stretch[:nrespcycle_FC,:]
                WPLI_all_bands[cond_i, band_i] = WPLI_stretch[:nrespcycle_FC,:]

        if debug:

            band_i = 0
            plt.plot(np.median(WPLI_all_bands[0,band_i], axis=0))
            plt.plot(np.median(WPLI_all_bands[1,band_i], axis=0))
            plt.show()
                
        return ISPC_all_bands, WPLI_all_bands
    
    results = joblib.Parallel(n_jobs=n_core, prefer='processes', batch_size=1)(
        joblib.delayed(compute_pair)(pair_i) for pair_i in range(len(pairs_to_compute))
    )

    # Preallocate results
    ISPC_sujet = np.zeros((len(pairs_to_compute), len(cond_sel), len(freq_band_fc_list), nrespcycle_FC, stretch_point_FC))
    WPLI_sujet = np.zeros((len(pairs_to_compute), len(cond_sel), len(freq_band_fc_list), nrespcycle_FC, stretch_point_FC))
    
    for pair_i in range(len(pairs_to_compute)):
        ISPC_sujet[pair_i] = results[pair_i][0]
        WPLI_sujet[pair_i] = results[pair_i][1]
    
    xr_dict = {'pair': pairs_to_compute, 'cond': cond_sel, 'band': freq_band_fc_list, 'cycle': np.arange(nrespcycle_FC), 'time': np.arange(stretch_point_FC)}
    xr_ispc = xr.DataArray(data=ISPC_sujet, dims=xr_dict.keys(), coords=xr_dict.values())
    xr_wpli = xr.DataArray(data=WPLI_sujet, dims=xr_dict.keys(), coords=xr_dict.values())

    ISPC_sujet_rscore = ISPC_sujet.copy()
    WPLI_sujet_rscore = WPLI_sujet.copy()

    for cond_i, cond in enumerate(cond_list):
        for pair_i, pair in enumerate(pairs_to_compute):
            for band_i, band in enumerate(freq_band_fc_list):
                for cycle_i in range(nrespcycle_FC):
                    ISPC_sujet_rscore[pair_i, cond_i, band_i, cycle_i] = (ISPC_sujet[pair_i, cond_i, band_i, cycle_i, :] - np.median(ISPC_sujet[pair_i, cond_i, band_i, cycle_i, :])) * 0.6745 / scipy.stats.median_abs_deviation(ISPC_sujet[pair_i, cond_i, band_i, cycle_i, :])
                    WPLI_sujet_rscore[pair_i, cond_i, band_i, cycle_i] = (WPLI_sujet[pair_i, cond_i, band_i, cycle_i, :] - np.median(WPLI_sujet[pair_i, cond_i, band_i, cycle_i, :])) * 0.6745 / scipy.stats.median_abs_deviation(WPLI_sujet[pair_i, cond_i, band_i, cycle_i, :])
    
    xr_ispc_rscore = xr.DataArray(data=ISPC_sujet_rscore, dims=xr_dict.keys(), coords=xr_dict.values())
    xr_wpli_rscore = xr.DataArray(data=WPLI_sujet_rscore, dims=xr_dict.keys(), coords=xr_dict.values())

    if debug:
        
        pair_i = 0
        band_i = 0

        fc_diff = ISPC_sujet.values[pair_i, 1, band_i] - ISPC_sujet.values[pair_i, 0, band_i]
        fc_diff = WPLI_sujet.values[pair_i, 1, band_i] - WPLI_sujet.values[pair_i, 0, band_i]

        for cycle_i in range(nrespcycle_FC):

            plt.plot(fc_diff[cycle_i], alpha=0.2)
            
        plt.plot(np.median(fc_diff, axis=0), color='r')
        plt.show()

        fc_diff = ISPC_sujet_rscore.values[pair_i, 1, band_i] - ISPC_sujet_rscore.values[pair_i, 0, band_i]
        fc_diff = WPLI_sujet_rscore.values[pair_i, 1, band_i] - WPLI_sujet_rscore.values[pair_i, 0, band_i]

        for cycle_i in range(nrespcycle_FC):

            plt.plot(fc_diff[cycle_i], alpha=0.2)
            
        plt.plot(np.median(fc_diff, axis=0), color='r')
        plt.show()

        plt.plot(np.median(ISPC_sujet.values[pair_i, 1, band_i] - ISPC_sujet.values[pair_i, 0, band_i], axis=0), color='r')
        plt.plot(np.median(ISPC_sujet_rscore.values[pair_i, 1, band_i] - ISPC_sujet_rscore.values[pair_i, 0, band_i], axis=0), color='r')
        plt.show()

        plt.plot(np.median(WPLI_sujet.values[pair_i, 1, band_i] - WPLI_sujet.values[pair_i, 0, band_i], axis=0), color='r')
        plt.plot(np.median(WPLI_sujet_rscore.values[pair_i, 1, band_i] - WPLI_sujet_rscore.values[pair_i, 0, band_i], axis=0), color='r')
        plt.show()
        
    os.chdir(os.path.join(path_precompute, 'FC', 'ISPC'))
    xr_ispc.to_netcdf(f'ISPC_{sujet}_stretch_beta.nc')
    xr_ispc_rscore.to_netcdf(f'ISPC_{sujet}_stretch_rscore_beta.nc')
    os.chdir(os.path.join(path_precompute, 'FC', 'WPLI'))
    xr_wpli.to_netcdf(f'WPLI_{sujet}_stretch_beta.nc')
    xr_wpli_rscore.to_netcdf(f'WPLI_{sujet}_stretch_rscore_beta.nc')
    
    print('done')

    

    





################################
######## EXECUTE ########
################################



if __name__ == '__main__':

    ######## COMPUTE FC ALLSUJET CLUSTER ########

    #get_MI_sujet_stretch()
    # execute_function_in_slurm_bash('n07_precompute_FC', 'get_MI_sujet_stretch', [[sujet] for sujet in sujet_list_FC], n_core=15, mem='20G')
    #sync_folders__push_to_crnldata()

    #get_ISPC_WPLI_stretch()
    # execute_function_in_slurm_bash('n07_precompute_FC', 'get_ISPC_WPLI_stretch', [[sujet] for sujet in sujet_list_FC], n_core=20, mem='30G')
    #sync_folders__push_to_crnldata()
    

    ######## COMPUTE FC ALLSUJET NYC ########

    for sujet in sujet_list_FC:
        get_ISPC_WPLI_stretch(sujet)



    
