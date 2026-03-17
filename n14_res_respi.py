

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False




############################
######## LOAD DATA ########
############################


def load_respfeatures_plot(sujet):

    #### load data
    os.chdir(path_prep)

    respfeatures_allcond = {}
    respi_allcond = {}

    #cond = 'CHARGE'
    for cond in cond_list:

        respi = load_data_sujet(sujet,cond)[np.where(chan_list == 'pression')[0][0],:]
        respi = scipy.signal.detrend(respi, type='linear')
        respi_allcond[cond] = respi

        params = physio.get_respiration_parameters('human_airflow')
        params['cycle_clean']['low_limit_log_ratio'] = 6
        # params['cycle_detection']['inspiration_adjust_on_derivative'] = True

        diff1 = np.diff(respi)*-1
        peaks, _ = scipy.signal.find_peaks(diff1, prominence=(diff1).std()*3)

        baseline_val = []

        for peak_i, peak in enumerate(peaks):

            if peak_i == peaks.size-1:
                continue

            backward_signal = diff1[peaks[peak_i]:peaks[peak_i+1]][::-1]
            backward_val = np.where(np.diff(backward_signal) > 0)[0][0]
            baseline_val.append(respi[peak-backward_val])

        baseline = np.median(baseline_val)

        if debug:

            plt.plot(respi)
            plt.hlines(np.median(baseline_val), xmin=0, xmax=respi.size, color='r')
            plt.show()

        params['baseline']['baseline_mode'] = 'manual'
        params['baseline']['baseline'] = baseline

        respi_clean, resp_features_i = physio.compute_respiration(raw_resp=respi, srate=srate, parameters=params)

        respi -= baseline

        time_vec = np.arange(respi.size)/srate

        fig_final, ax = plt.subplots(figsize=(18, 10))
        ax.plot(time_vec, respi)

        ax.scatter(resp_features_i['inspi_index'].values/srate, respi[resp_features_i['inspi_index'].values], color='g', label='inspi')
        ax.scatter(resp_features_i['expi_index'].values/srate, respi[resp_features_i['expi_index'].values], color='r', label='expi')
        ax.scatter(resp_features_i['next_inspi_index'].values/srate, respi[resp_features_i['next_inspi_index'].values], color='g', label='inspi')

        plt.legend()
        # plt.show()

        # cycles = physio.detect_respiration_cycles(respi, srate, baseline_mode='median',
        #                                           baseline=None, epsilon_factor1=10, epsilon_factor2=5, inspiration_adjust_on_derivative=False)
        
        # cycles = debugged_detect_respiration_cycles(respi, srate, baseline_mode='median',
        #                                             baseline=None, epsilon_factor1=10, epsilon_factor2=5, inspiration_adjust_on_derivative=False)
        
        # if debug:

        #     fig, ax = plt.subplots()
        #     ax.plot(respi)
        #     ax.scatter(cycles[:,0], respi[cycles[:,0]], color='g')
        #     plt.show()

        # cycles, fig_respi_exclusion, fig_final = exclude_bad_cycles(respi, cycles, srate, exclusion_coeff=1)
            
        # if debug:

        #     fig, ax = plt.subplots()
        #     ax.plot(respi)
        #     ax.scatter(cycles[:,0], respi[cycles[:,0]], color='r')
        #     plt.show()

        # #### get resp_features
        # resp_features_i = physio.compute_respiration_cycle_features(respi, srate, cycles, baseline=None)

        # select_vec = np.ones((resp_features_i.index.shape[0]), dtype='int')
        # resp_features_i.insert(resp_features_i.columns.shape[0], 'select', select_vec)
        
        respfeatures_allcond[cond] = [resp_features_i, fig_final]

    return respi_allcond, respfeatures_allcond




####################################
######## RESPI BEHAVIOR ########
####################################

def plot_mean_respi():
    
    stretch_resp_allsujet = np.zeros((len(sujet_list_FC), len(cond_list), stretch_point_ERP))

    for sujet_i, sujet in enumerate(sujet_list_FC):

        respi_allcond, respfeatures = load_respfeatures_plot(sujet)

        #### load
        #cond = 'VS'
        for cond_i, cond in enumerate(cond_list):

            respi_stretch = stretch_data(respfeatures[cond][0], stretch_point_ERP, respi_allcond[cond], srate)[0]
            stretch_resp_allsujet[sujet_i, cond_i] = np.median(respi_stretch[:nrespcycle_TF], axis=0)

    #### extract r score
    rzscore_data = []
    ymin_rzscore, ymax_rzscore = [], []

    for sujet_i, sujet in enumerate(sujet_list_FC):  

        _both_cond_sig = np.concat([stretch_resp_allsujet[sujet_i,0], stretch_resp_allsujet[sujet_i,1]], axis=0)

        _median = np.median(_both_cond_sig)
        _mad = np.median(np.abs(_both_cond_sig - _median), axis=0)

        _rzscore_sujet = []
        _ymin_rzscore_sujet, _ymax_rzscore_sujet = [], []

        for cond_i, cond in enumerate(cond_list):

            _rzscore_x = (stretch_resp_allsujet[sujet_i,cond_i] - _median) * 0.6745 / _mad
            _rzscore_sujet.append(_rzscore_x)
            _ymin_rzscore_sujet.append(_rzscore_x.min())
            _ymax_rzscore_sujet.append(_rzscore_x.max())

        rzscore_data.append(_rzscore_sujet)
        ymin_rzscore.append(_ymin_rzscore_sujet)
        ymax_rzscore.append(_ymax_rzscore_sujet)

    rzscore_data = np.array(rzscore_data)
    ymin_rzscore, ymax_rzscore = np.array(ymin_rzscore).min(), np.array(ymax_rzscore).max()
            

    #### plot
    time_vec = np.arange(stretch_point_ERP)
    colors_respi = {'VS' : 'b', 'CHARGE' : 'r'}
    colors_respi_sem = {'VS' : 'c', 'CHARGE' : 'm'}

    stretch_resp_cond = np.median(rzscore_data, axis=0)
    sem_resp_cond = []
    lmin_cond, lmax_cond = [], []

    for cond_i, cond in enumerate(cond_list):

        _sem = stretch_resp_allsujet[:,cond_i].std(axis=0)/np.sqrt(stretch_resp_allsujet[:,cond_i].shape[0])
        sem_resp_cond.append(_sem)
        lmin_cond.append(stretch_resp_cond[cond_i]-_sem)
        lmax_cond.append(stretch_resp_cond[cond_i]+_sem) 

    sem_resp_cond = np.array(sem_resp_cond)
    lmin, lmax = np.array(lmin_cond).min(), np.array(lmax_cond).max()

    #### median
    fig_median, ax = plt.subplots()

    #cond = 'VS'
    for cond_i, cond in enumerate(cond_list):

        ax.plot(time_vec, stretch_resp_cond[cond_i], color=colors_respi[cond], label=cond)
        ax.fill_between(time_vec, stretch_resp_cond[cond_i]+sem_resp_cond[cond_i], stretch_resp_cond[cond_i]-sem_resp_cond[cond_i], alpha=0.25, color=colors_respi_sem[cond])

    ax.vlines(stretch_point_ERP/2, ymin=lmin, ymax=lmax, color='r')
    plt.ylim(lmin, lmax)
    plt.title('median')
    plt.legend()

    # fig_median.show()

    os.chdir(os.path.join(path_results, 'RESPI', 'plot'))
    fig_median.savefig(f"respi_median.png")

    #### allsujet
    fig_cond_allsujet, axs = plt.subplots(ncols=len(cond_list), figsize=(11,5))

    #cond = 'VS'
    for cond_i, cond in enumerate(cond_list):

        ax = axs[cond_i]

        for sujet_i, sujet in enumerate(sujet_list_FC):

            ax.plot(time_vec, rzscore_data[sujet_i,cond_i])

        ax.vlines(stretch_point_ERP/2, ymin=ymin_rzscore, ymax=ymax_rzscore, color='r')
        ax.set_title(f'{cond}')

    plt.suptitle('allsujet')

    # fig_cond_allsujet.show()

    os.chdir(os.path.join(path_results, 'RESPI', 'plot'))
    fig_cond_allsujet.savefig(f"allsujet_respi_median.png")

    plt.close('all')


def plot_MDP():

    sujet_list_DL = [int(_sujet[-2:]) for _sujet in sujet_list_FC if _sujet.find('DL') != -1]

    path_MDP_raw = os.path.join(path_data, 'DYSLEARN', 'DYSLEARN MDP raw.xlsx')
    mdp_raw = pd.read_excel(path_MDP_raw)
    mdp_raw_filt = mdp_raw.iloc[:35].query(f"ID in {sujet_list_DL}")

    print(f"median A1 {np.median(mdp_raw_filt['MDP A1'])}")
    print(f"median A2 {np.median(mdp_raw_filt['A2 TOTAL'])}")
    print(f"mode QS {scipy.stats.mode(mdp_raw_filt['QS  Best 4'].values.astype('int'))}")

    path_export_plot = os.path.join(path_results, 'MDP')

    fig_A1, ax = plt.subplots(figsize=(6, 5))
    bins = np.arange(0.5, 11.5, 1)
    sns.histplot(mdp_raw_filt, x='MDP A1', bins=bins, ax=ax)
    ax.set_xlim(0.5, 11.5)
    ax.set_xticks(range(11))
    ax.set_title('A1')
    # fig_A1.show()
    fig_A1.savefig(path_export_plot + "/MDP_A1.png")

    fig_A2, ax = plt.subplots(figsize=(6, 5))
    bins = np.arange(0, 55, 5)
    sns.histplot(mdp_raw_filt, x='A2 TOTAL', bins=bins)
    ax.set_xlim(0, 50)
    ax.set_xticks(np.arange(0, 55, 5))
    ax.set_title('A2 TOTAL')
    # fig_A2.show()
    fig_A2.savefig(path_export_plot + "/MDP_A2TOT.png")

    fig_QS, ax = plt.subplots(figsize=(5, 5))
    bins = np.arange(0.5, 6.5, 1)
    sns.histplot(mdp_raw_filt, x='QS  Best 4', bins=bins)
    ax.set_xlim(0.5, 5.5)
    ticks = [1, 2, 3, 4, 5]
    labels = ['muscle work', 'hunger for air', 'constricted', 'mental effort', 'breathing a lot']
    ax.set_xticks(ticks, labels, rotation=45)
    ax.set_title('BEST QS')
    plt.tight_layout()
    # fig_QS.show()
    fig_QS.savefig(path_export_plot + "/MDP_QS.png")

    plt.close('all')







############################
######## EXECUTE ########
############################



if __name__ == '__main__':


    plot_mean_respi()



