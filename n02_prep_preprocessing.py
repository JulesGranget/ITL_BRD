
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import mne
import pandas as pd

import physio

import seaborn as sns

from n00_config_params import *
from n00bis_config_analysis_functions import *

debug = False






################################
######## OPEN DATA ########
################################




#sujet, cond = sujet_list[0], 'CHARGE'
def open_raw_data(sujet, cond):

    ######## identify project and sujet ########        
    sujet_project = sujet_project_nomenclature[sujet[2:4]]
    sujet_init_name = list(sujet_list_correspondance.keys())[list(sujet_list_correspondance.values()).index(sujet)][3:]

    ######## OPEN DATA ########
    if sujet_project == 'NORMATIVE':

        os.chdir(os.path.join(path_data, sujet_project, 'first', sujet_init_name))

        print(f"OPEN {sujet_project} : {sujet}")

        _data = mne.io.read_raw_brainvision(f"{sujet_init_name}_{cond}_ValidICM.vhdr")
        _chan_list_eeg = _data.info['ch_names'][:-5]
        _data_eeg = _data.get_data()[:-5,:]
        pression_chan_i = _data.info['ch_names'].index('Pression')
        _respi = _data.get_data()[pression_chan_i,:]
        _srate_init = _data.info['sfreq']

        _trig = _data.annotations.onset

        if debug:
            mne.viz.plot_raw(_data, n_channels=1)

            plt.plot(respi)
            plt.show()

        #### sel chan 
        chan_sel_list_i = [chan_list_project_wise[sujet_project].index(chan) for chan in chan_list_eeg if chan in chan_list_project_wise[sujet_project]]
        _data_eeg = _data_eeg[chan_sel_list_i,:]

    elif sujet_project == 'PHYSIOLOGY':

        os.chdir(os.path.join(path_data, sujet_project, sujet_init_name))

        print(f"OPEN {sujet_project} : {sujet}")

        _data = mne.io.read_raw_brainvision(f"{sujet_init_name}_CONTINU_64Ch_A2Ref.vhdr")
        if sujet == '21PH_SB':
            _chan_list_eeg = _data.info['ch_names'][:-4]
            _data_eeg = _data.get_data()[:-4,:]
        else:
            _chan_list_eeg = _data.info['ch_names'][1:-4]
            _data_eeg = _data.get_data()[1:-4,:]
        pression_chan_i = _data.info['ch_names'].index('Pression')
        _respi = _data.get_data()[pression_chan_i,:]
        _srate_init = _data.info['sfreq']

        _trig = _data.annotations.onset

        if debug:
            mne.viz.plot_raw(_data, n_channels=1)

        #### sel chan 
        chan_sel_list_i = [chan_list_project_wise[sujet_project].index(chan) for chan in chan_list_eeg if chan in chan_list_project_wise[sujet_project]]
        _data_eeg = _data_eeg[chan_sel_list_i,:]

        #### chunk cond
        start, stop = int(section_timming_PHYSIOLOGY[sujet][cond][0]*_srate_init), int(section_timming_PHYSIOLOGY[sujet][cond][1]*_srate_init)
        
        _data_eeg = _data_eeg[:,start:stop]
        _respi = _respi[start:stop]
        _trig = _trig[(_trig>=section_timming_PHYSIOLOGY[sujet][cond][0]) & (_trig<=section_timming_PHYSIOLOGY[sujet][cond][1])]
        _trig -= start

        if debug:

            plt.plot(_respi)
            plt.vlines(_trig*_srate_init, ymin=_respi.min(), ymax=_respi.max(), color='r')
            plt.show()

    elif sujet_project == 'ITL_LEO':

        os.chdir(os.path.join(path_data, 'ITL_LEO'))

        print(f"OPEN {sujet_project} : {sujet}")

        if cond == 'VS':
            cond_to_search = 'VS'
        elif cond == 'CHARGE':
            cond_to_search = 'ITL'

        file_name = [file for file in os.listdir() if file.find(sujet_init_name) != -1 and file.find(f'{cond_to_search}.edf') != -1][0]
        file_name_marker = [file for file in os.listdir() if file.find(sujet_init_name) != -1 and file.find(f'{cond_to_search}.Markers') != -1][0]

        _data = mne.io.read_raw_edf(file_name)
        _chan_list_eeg = _data.info['ch_names'][:-3]
        _data_eeg = _data.get_data()[:-3,:]
        pression_chan_i = _data.info['ch_names'].index('PRESSION')
        _respi = _data.get_data()[pression_chan_i,:]
        _srate_init = _data.info['sfreq']

        f = open(file_name_marker, "r")
        _trig = [int(line.split(',')[2][1:]) for line_i, line in enumerate(f.read().split('\n')) if len(line.split(',')) == 5 and line.split(',')[0] == 'Response']
        _trig = np.array(_trig) / _srate_init

        #### sel chan 
        chan_sel_list_i = [chan_list_project_wise[sujet_project].index(chan) for chan in chan_list_eeg if chan in chan_list_project_wise[sujet_project]]
        _data_eeg = _data_eeg[chan_sel_list_i,:]

    elif sujet_project == 'DYSLEARN':

        print(f"OPEN {sujet_project} : {sujet}")

        if cond == 'CHARGE':
            cond_corrected = 'ITL'
        else:
            cond_corrected = cond

        os.chdir(os.path.join(path_data, sujet_project, cond_corrected))

        if sujet_init_name in ['08']:
            file_open = [file for file in os.listdir() if file.find('vhdr') != -1 and file.find(f"DYSLEARN_00{sujet_init_name}") != -1][-1]
        else:
            file_open = [file for file in os.listdir() if file.find('vhdr') != -1 and file.find(f"DYSLEARN_00{sujet_init_name}") != -1][0]    

        _data = mne.io.read_raw_brainvision(file_open)
        _chan_list_eeg = _data.info['ch_names'][:-3]
        _data_eeg = _data.get_data()[:-3,:]
        pression_chan_i = _data.info['ch_names'].index('PRESS')
        _respi = _data.get_data()[pression_chan_i,:]
        _srate_init = _data.info['sfreq']

        _trig_onset = _data.annotations.onset
        _trig_name = _data.annotations.description

        if sujet == '42DL_11':
            start, stop = int(_trig_onset[np.where(_trig_name == f'Comment/{cond_corrected} DEBUT')[0][0]]*_srate_init), int(_trig_onset[np.where(_trig_name == f'Comment/VS FIN')[0][0]]*_srate_init)
        else:
            start, stop = int(_trig_onset[np.where(_trig_name == f'Comment/{cond_corrected} DEBUT')[0][0]]*_srate_init), int(_trig_onset[np.where(_trig_name == f'Comment/{cond_corrected} FIN')[0][0]]*_srate_init)
        
        _data_eeg = _data_eeg[:,start:stop]
        _respi = _respi[start:stop]

        _trig = _trig_onset[(_trig_onset>=start/_srate_init) & (_trig_onset<=stop/_srate_init)]
        _trig -= start/_srate_init

        #### sel chan 
        chan_sel_list_i = [chan_list_project_wise[sujet_project].index(chan) for chan in chan_list_eeg if chan in chan_list_project_wise[sujet_project]]
        _data_eeg = _data_eeg[chan_sel_list_i,:]

    ######## ADJUST RESPI ########
    if sujet_respi_adjust[sujet] == 'inverse':
        _respi *= -1

    ######## RESAMPLE ########
    if _srate_init != 500:

        #### EEG
        _time_vec_origin = np.arange(0, _data_eeg.shape[-1]/_srate_init, 1/_srate_init)
        _time_vec_dwsampled = np.arange(0, _data_eeg.shape[-1]/_srate_init, 1/srate)

        _data_dwsampled = np.zeros((len(chan_list_eeg), _time_vec_dwsampled.shape[0]))

        for chan_i in range(_data_eeg.shape[0]):
            x = _data_eeg[chan_i,:]
            x_dwsampled = np.interp(_time_vec_dwsampled, _time_vec_origin, x)
            _data_dwsampled[chan_i,:] = x_dwsampled

        _data_eeg = _data_dwsampled

        #### AUX
        _respi = np.interp(_time_vec_dwsampled, _time_vec_origin, _respi) 

    ######## EXTRACT TRIG ########

    if debug:

        time_vec = np.arange(0, _respi.shape[0]/srate, 1/srate)
            
        plt.plot(time_vec, _respi)
        plt.title(f"{sujet}, {cond}")
        plt.vlines(_trig, ymin=_respi.min(), ymax=_respi.max(), color='r')
        plt.show()

    return _data_eeg, _respi, _trig











################################
######## VIEWER ########
################################

#data, data_aux = raw_eeg.get_data(), raw_aux.get_data()
def view_data(data_eeg, respi, return_fig=False):

    #### downsample
    print('resample')
    srate_downsample = 250

    time_vec = np.linspace(0,data_eeg.shape[-1],data_eeg.shape[-1])/srate
    time_vec_resample = np.linspace(0,data_eeg.shape[-1],int(data_eeg.shape[-1] * (srate_downsample / srate)))/srate

    data_resampled = np.zeros((data_eeg.shape[0], time_vec_resample.shape[0]))
    respi_resampled = np.zeros((time_vec_resample.shape[0]))

    for chan_i in range(data_eeg.shape[0]):
        f = scipy.interpolate.interp1d(time_vec, data_eeg[chan_i,:], kind='linear', fill_value="extrapolate")
        data_resampled[chan_i,:] = f(time_vec_resample)

    f = scipy.interpolate.interp1d(time_vec, respi, kind='quadratic', fill_value="extrapolate")
    respi_resampled = f(time_vec_resample)

    print('plot')

    fig, ax = plt.subplots()

    ax.plot(time_vec_resample, zscore(respi_resampled), label='respi')

    for chan_i, chan in enumerate(chan_list_eeg):
    
        x = data_resampled[chan_i,:]
        ax.plot(time_vec_resample, zscore(x)+3*(chan_i)+1, label=chan)
    
    plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='upper left')  # reverse to keep order consistent

    if return_fig:
        return fig
    else:
        plt.show()







################################
######## AUX PREPROC ########
################################



def respi_preproc(respi):

    #### inspect Pxx
    if debug:
        plt.plot(np.arange(respi.shape[0])/srate, respi)
        plt.show()

        nwind = int(10*srate)
        nfft = nwind
        noverlap = np.round(nwind/2)
        hannw = scipy.signal.windows.hann(nwind)
        hzPxx, Pxx = scipy.signal.welch(respi,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        plt.semilogy(hzPxx, Pxx, label='respi')
        plt.legend()
        plt.xlim(0,60)
        plt.show()

    #### filter respi physio
    respi_filt = physio.preprocess(respi, srate, band=25., btype='lowpass', ftype='bessel', order=5, normalize=False)
    respi_filt = physio.smooth_signal(respi_filt, srate, win_shape='gaussian', sigma_ms=40.0)

    if debug:
        plt.plot(respi, label='respi')
        plt.plot(respi_filt, label='respi_filtered')
        plt.legend()
        plt.show()

        hzPxx, Pxx_pre = scipy.signal.welch(respi,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        hzPxx, Pxx_post = scipy.signal.welch(respi_filt,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
        plt.semilogy(hzPxx, Pxx_pre, label='pre')
        plt.semilogy(hzPxx, Pxx_post, label='post')
        plt.legend()
        plt.xlim(0,60)
        plt.show()

    return respi_filt











################################
######## COMPARISON ########
################################


# to compare during preprocessing
#chan_name = 'C3'
def compare_pre_post(data_pre, data_post, srate, chan_name):

    # compare before after
    nchan_i = np.where(chan_list_eeg == chan_name)[0][0]
    x_pre = data_pre[nchan_i,:]
    x_post = data_post[nchan_i,:]
    time = np.arange(x_pre.shape[0]) / srate

    nwind = int(10*srate)
    nfft = nwind
    noverlap = np.round(nwind/2)
    hannw = scipy.signal.windows.hann(nwind)

    hzPxx, Pxx_pre = scipy.signal.welch(x_pre,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)
    hzPxx, Pxx_post = scipy.signal.welch(x_post,fs=srate,window=hannw,nperseg=nwind,noverlap=noverlap,nfft=nfft)

    plt.plot(time, x_pre, label='x_pre')
    plt.plot(time, x_post, label='x_post')
    plt.title(chan_name)
    plt.legend()
    plt.show()

    plt.semilogy(hzPxx, Pxx_pre, label='Pxx_pre')
    plt.semilogy(hzPxx, Pxx_post, label='Pxx_post')
    plt.title(chan_name)
    plt.legend()
    # plt.xlim(60,360)
    plt.show()










################################
######## PREPROCESSING ########
################################

#new_ref = prep_step['reref']['params']
def reref_eeg(raw_data, info_eeg, new_ref):

    raw_eeg_reref = mne.io.RawArray(raw_data, info_eeg)
    raw_eeg_reref, refdata = mne.set_eeg_reference(raw_eeg_reref, ref_channels=new_ref)

    if debug == True :
        duration = 3.
        n_chan = 20
        raw_eeg_reref.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

    raw_data_post = raw_eeg_reref.get_data() # if reref ok

    return raw_data_post




def detrend_mean_centered(raw_data):
        
    # mean centered
    data_mc = np.zeros(raw_data.shape)
    for chan_i in range(np.size(raw_data,0)):
        data_mc[chan_i,:] = scipy.signal.detrend(raw_data[chan_i,:], type='linear') 
        data_mc[chan_i,:] = data_mc[chan_i,:] - np.mean(data_mc[chan_i,:])

    return data_mc




def line_noise_removing(data_eeg, info_eeg):

    linenoise_freq = [50, 100, 150]

    raw_eeg_line_noise_removing = mne.io.RawArray(data_eeg, info_eeg)

    raw_eeg_line_noise_removing.notch_filter(linenoise_freq, verbose='critical')
    
    raw_eeg_line_noise_removing = raw_eeg_line_noise_removing.get_data()

    return raw_eeg_line_noise_removing





def filter(raw_data, info_eeg, h_freq, l_freq):

    #filter_length = int(srate*10) # give sec
    filter_length = 'auto'

    if debug == True :
        h = mne.filter.create_filter(raw_data, srate, l_freq=l_freq, h_freq=h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2')
        flim = (0.1, srate / 2.)
        mne.viz.plot_filter(h, srate, freq=None, gain=None, title=None, flim=flim, fscale='log')

    raw_eeg_filter = mne.io.RawArray(raw_data, info_eeg)

    raw_eeg_filter = raw_eeg_filter.filter(l_freq, h_freq, filter_length=filter_length, method='fir', phase='zero-double', fir_window='hamming', fir_design='firwin2', verbose='critical')
    data_filtered = raw_eeg_filter.get_data()

    if debug == True :
        duration = 60.
        n_chan = 20
        raw_eeg_filter.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

    return data_filtered





def ICA_computation(data_eeg, info_eeg):

    # n_components = np.size(raw.get_data(),0) # if int, use only the first n_components PCA components to compute the ICA decomposition
    n_components = 20
    random_state = 27
    method = 'fastica'

    raw_eeg_ica = mne.io.RawArray(data_eeg, info_eeg)

    ica = mne.preprocessing.ICA(n_components=n_components, random_state=random_state, method=method)

    reject = None
    decim = None
    # picks = mne.pick_types(raw.info, eeg=True, eog=True)
    picks = mne.pick_types(raw_eeg_ica.info)
    ica.fit(raw_eeg_ica)

    # for eeg signal
    ica.plot_sources(raw_eeg_ica)
    ica.plot_components()
        
    # apply ICA
    raw_ICA = raw_eeg_ica.copy()
    ica.apply(raw_ICA) # exclude component

    raw_ICA_export = raw_ICA.get_data()

    # verify
    if debug == True :

        # compare before after
        compare_pre_post(data_pre=data_eeg, data_post=raw_ICA_export, srate=srate, chan_name='Fp2')

        duration = .5
        n_chan = 10
        raw_ICA.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify

    return raw_ICA_export





def average_reref(raw):

    if debug:
        raw_post = raw.copy()
    else:
        raw_post = raw

    raw_post.set_eeg_reference('average')

    if debug == True :
        duration = .5
        n_chan = 10
        raw_post.plot(scalings='auto',duration=duration,n_channels=n_chan) # verify


    return raw_post




def csd_computation(raw):

    raw_csd = surface_laplacian(raw=raw, m=4, leg_order=50, smoothing=1e-5) # MXC way

    # compare before after
    # compare_pre_post(raw, raw_post, 4)

    return raw_csd




def preprocessing_eeg(data_eeg, info_eeg, prep_step):


    ######## PREPROC ########
    print('#### PREPROCESSING ####', flush=True)

    # data_init = data_eeg.copy()
    # data_eeg = data_init.copy()

    #### Execute preprocessing

    if prep_step['reref']['execute']:
        print('reref', flush=True)
        data_post = reref_eeg(data_eeg, info_eeg, prep_step['reref']['params'])
        #compare_pre_post(data_pre=data_init, data_post=data_post, srate=srate, chan_name='FC5')
        data_eeg = data_post


    if prep_step['detrend_mean_centered']['execute']:
        print('detrend_mean_centered', flush=True)
        data_post = detrend_mean_centered(data_eeg)
        #compare_pre_post(data_pre=data_init, data_post=data_post, srate=srate, chan_name='Fz')
        data_eeg = data_post


    if prep_step['line_noise_removing']['execute']:
        print('line_noise_removing', flush=True)
        data_post = line_noise_removing(data_eeg, info_eeg)
        #compare_pre_post(data_pre=data_init, data_post=data_post, srate=srate, chan_name='C3')
        data_eeg = data_post


    if prep_step['high_pass']['execute']:
        print('high_pass', flush=True)
        h_freq = prep_step['high_pass']['params']['h_freq']
        l_freq = prep_step['high_pass']['params']['l_freq']
        data_post = filter(data_eeg, info_eeg, h_freq, l_freq)
        #compare_pre_post(data_pre=data_init, data_post=data_post, srate=srate, chan_name='C3')
        data_eeg = data_post


    if prep_step['low_pass']['execute']:
        print('low_pass', flush=True)
        h_freq = prep_step['high_pass']['params']['h_freq']
        l_freq = prep_step['high_pass']['params']['l_freq']
        data_post = filter(data_eeg, info_eeg, h_freq, l_freq)
        #compare_pre_post(data_pre=data_init, data_post=data_post, srate=srate, chan_name='C3')
        data_eeg = data_post

    if prep_step['csd_computation']['execute']:
        print('csd_computation', flush=True)
        data_post = csd_computation(data_eeg)
        #compare_pre_post(data_pre=data_init, data_post=data_post, srate=srate, chan_name='C3')
        data_eeg = data_post

    if prep_step['ICA_computation']['execute']:
        print('ICA_computation', flush=True)
        data_post = ICA_computation(data_eeg, info_eeg)
        #compare_pre_post(data_pre=data_init, data_post=data_post, srate=srate, chan_name='C3')
        data_eeg = data_post


    if prep_step['average_reref']['execute']:
        print('average_reref', flush=True)
        data_post = average_reref(data_eeg)
        #compare_pre_post(data_pre=data_init, data_post=data_post, srate=srate, chan_name='C3')
        data_eeg = data_post

    #compare_pre_post(data_pre=data_init, data_post=data_eeg, srate=srate, chan_name='C3')

    return data_eeg









########################################
######## DETECT ARTIFACT ########
########################################



def detect_cross(sig, threshold):

    """
    Detect crossings
    ------
    inputs =
    - sig : numpy 1D array
    - show : plot figure showing rising zerox in red and decaying zerox in green (default = False)
    output =
    - pandas dataframe with index of rises and decays
    """

    rises, = np.where((sig[:-1] <=threshold) & (sig[1:] >threshold)) # detect where sign inversion from - to +
    decays, = np.where((sig[:-1] >=threshold) & (sig[1:] <threshold)) # detect where sign inversion from + to -

    if rises.size != 0:

        if rises[0] > decays[0]: # first point detected has to be a rise
            decays = decays[1:] # so remove the first decay if is before first rise
        if rises[-1] > decays[-1]: # last point detected has to be a decay
            rises = rises[:-1] # so remove the last rise if is after last decay

        return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T
    
    else:

        return None
    



def compute_rms(x):

    """Fast root mean square."""
    n = x.size
    ms = 0
    for i in range(n):
        ms += x[i] ** 2
    ms /= n

    return np.sqrt(ms)




def sliding_rms(x, sf, window=0.5, step=0.2, interp=True):

    halfdur = window / 2
    n = x.size
    total_dur = n / sf
    last = n - 1
    idx = np.arange(0, total_dur, step)
    out = np.zeros(idx.size)

    # Define beginning, end and time (centered) vector
    beg = ((idx - halfdur) * sf).astype(int)
    end = ((idx + halfdur) * sf).astype(int)
    beg[beg < 0] = 0
    end[end > last] = last
    # Alternatively, to cut off incomplete windows (comment the 2 lines above)
    # mask = ~((beg < 0) | (end > last))
    # beg, end = beg[mask], end[mask]
    t = np.column_stack((beg, end)).mean(1) / sf

    # Now loop over successive epochs
    for i in range(idx.size):
        out[i] = compute_rms(x[beg[i] : end[i]])

    # Finally interpolate
    if interp and step != 1 / sf:
        f = scipy.interpolate.interp1d(t, out, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True)
        t = np.arange(n) / sf
        out = f(t)

    return t, out

#sig = data
def iirfilt(sig, srate, lowcut=None, highcut=None, order=4, ftype='butter', verbose=False, show=False, axis=0):

    if len(sig.shape) == 1:

        axis = 0

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = scipy.signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    filtered_sig = scipy.signal.sosfiltfilt(sos, sig, axis=axis)

    return filtered_sig



def med_mad(data, constant = 1.4826):

    median = np.median(data)
    mad = np.median(np.abs(data - median)) * constant

    return median , mad



def compute_artifact_features(inds, srate):

    artifacts = pd.DataFrame()
    artifacts['start_ind'] = inds['rises'].astype(int)
    artifacts['stop_ind'] = inds['decays'].astype(int)
    artifacts['start_t'] = artifacts['start_ind'] / srate
    artifacts['stop_t'] = artifacts['stop_ind'] / srate
    artifacts['duration'] = artifacts['stop_t'] - artifacts['start_t']

    return artifacts



#data = data[16,:]
def detect_movement_artifacts(data, srate, n_chan_artifacted=5, n_deviations=5, low_freq=40, high_freq=150, wsize=1, step=0.2):
    
    eeg_filt = iirfilt(data, srate, low_freq, high_freq, ftype='bessel', order=2, axis=1)

    if len(eeg_filt.shape) != 1:

        masks = np.zeros((eeg_filt.shape), dtype='bool')

        for i in range(eeg_filt.shape[0]):

            print_advancement(i, eeg_filt.shape[0], steps=[25, 50, 75])

            sig_chan_filtered = eeg_filt[i,:]
            t, rms_chan = sliding_rms(sig_chan_filtered, sf=srate, window = wsize, step = step) 
            pos, dev = med_mad(rms_chan)
            detect_threshold = pos + n_deviations * dev
            masks[i,:] = rms_chan > detect_threshold

        compress_chans = masks.sum(axis = 0)
        inds = detect_cross(compress_chans, n_chan_artifacted+0.5)

        if type(inds) == type(None):
            print('none')
            return None

        artifacts = compute_artifact_features(inds, srate)

    else:

        sig_chan_filtered = eeg_filt
        t, rms_chan = sliding_rms(sig_chan_filtered, sf=srate, window = wsize, step = step) 
        pos, dev = med_mad(rms_chan)
        detect_threshold = pos + n_deviations * dev
        masks = rms_chan > detect_threshold

        compress_chans = masks*1
        inds = detect_cross(masks, 0.5)

        if inds == None:
            return None
        
        artifacts = compute_artifact_features(inds, srate)

    return artifacts


# chan_artifacts = artifacts
def insert_noise(sig, srate, chan_artifacts, freq_min=30., margin_s=0.2, seed=None):

    sig_corrected = sig.copy()

    margin = int(srate * margin_s)
    up = np.linspace(0, 1, margin)
    down = np.linspace(1, 0, margin)
    
    noise_size = np.sum(chan_artifacts['stop_ind'].values - chan_artifacts['start_ind'].values) + 2 * margin * chan_artifacts.shape[0]
    
    # estimate psd sig
    freqs, spectrum = scipy.signal.welch(sig, nperseg=noise_size, nfft=noise_size, noverlap=0, scaling='spectrum', window='box', return_onesided=False, average='median')
    
    spectrum = np.sqrt(spectrum)
    
    # pregenerate long noise piece
    rng = np.random.RandomState(seed=seed)
    
    long_noise = rng.randn(noise_size)
    noise_F = np.fft.fft(long_noise)
    #long_noise = np.fft.ifft(np.abs(noise_F) * spectrum * np.exp(1j * np.angle(noise_F)))
    long_noise = np.fft.ifft(spectrum * np.exp(1j * np.angle(noise_F)))
    long_noise = long_noise.astype(sig.dtype)
    sos = scipy.signal.iirfilter(2, freq_min / (srate / 2), analog=False, btype='highpass', ftype='bessel', output='sos')
    long_noise = scipy.signal.sosfiltfilt(sos, long_noise, axis=0)
    
    filtered_sig = scipy.signal.sosfiltfilt(sos, sig, axis=0)
    rms_sig = np.median(filtered_sig**2)
    rms_noise = np.median(long_noise**2)
    factor = np.sqrt(rms_sig) / np.sqrt(rms_noise)
    long_noise *= factor
    
    noise_ind = 0

    for _, artifact in chan_artifacts.iterrows():

        ind0, ind1 = int(artifact['start_ind']), int(artifact['stop_ind'])
        
        n_samples = ind1 - ind0 + 2 * margin
        
        sig_corrected[ind0:ind1] = 0
        sig_corrected[ind0-margin:ind0] *= down
        sig_corrected[ind1:ind1+margin] *= up
        
        noise = long_noise[noise_ind: noise_ind + n_samples]
        noise_ind += n_samples
        
        noise += np.linspace(sig[ind0-1-margin], sig[ind1+1+margin], n_samples)
        noise[:margin] *= up
        noise[-margin:] *= down
        
        sig_corrected[ind0-margin:ind1+margin] += noise
        
    return sig_corrected





#data = data_preproc 
def remove_artifacts(data, srate):

    #### detect on all chan
    print('#### ARTIFACT DETECTION ALLCHAN ####', flush=True)
    artifacts = detect_movement_artifacts(data, srate, n_chan_artifacted=5, n_deviations=5, low_freq=40 , high_freq=150, wsize=1, step=0.2)
    
    if type(artifacts) == type(None):
        print("NO ARTIFACT FOUND")
        data_corrected = data.copy()
        return data_corrected
    
    #### correct on all chan
    print('#### ARTIFACT CORRECTION ALLCHAN ####', flush=True)
    data_corrected = data.copy()

    for chan_i, chan_name in enumerate(chan_list_eeg):

        data_corrected[chan_i,:] = insert_noise(data[chan_i,:], srate, artifacts, freq_min=30., margin_s=0.2, seed=None)

    if debug:

        chan_i = 0

        plt.plot(data[chan_i,:], label='raw')
        plt.plot(data_corrected[chan_i,:], label='corrected')
        plt.vlines(artifacts['start_ind'].values, ymin=data[chan_i,:].min(), ymax=data[chan_i,:].max(), color='r', label='start')
        plt.vlines(artifacts['stop_ind'].values, ymin=data[chan_i,:].min(), ymax=data[chan_i,:].max(), color='g', label='stop')
        plt.legend()
        plt.show()

        n_chan_plot = 5

        fig, ax = plt.subplots()

        for chan_i, chan_name in enumerate(chan_list_eeg[:n_chan_plot]):
        
            ax.plot(zscore(data[chan_i,:])+3*chan_i, label=f"raw : {chan_name}")
            ax.plot(zscore(data_corrected[chan_i,:])+3*chan_i, label=f"correct raw : {chan_name}")

        plt.vlines(artifacts['start_ind'].values, ymin=0, ymax=3*chan_i, color='r', label='start')
        plt.vlines(artifacts['stop_ind'].values, ymin=0, ymax=3*chan_i, color='r', label='stop')
        
        plt.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper left')  # reverse to keep order consistent

        plt.show()

    return data_corrected
















################################
######## EXECUTE ########
################################


if __name__== '__main__':

    ########################################
    ######## GENERATE PREPROC FILES ########
    ########################################

    #sujet = sujet_list[2]
    for sujet in sujet_list:

        #cond = cond_list[1]
        for cond in cond_list:

            ########################################
            ######## CONSTRUCT ARBORESCENCE ########
            ########################################

            # construct_token = generate_folder_structure(sujet)

            # if construct_token != 0 :
                
            #     raise ValueError("""Folder structure has been generated 
            #     Lauch the script again for preproc""")

            ########################
            ######## PARAMS ########
            ########################

            # sujet_list = ['01NM_MW', '02NM_OL', '03NM_MC', '04NM_LS', '05NM_JS', '06NM_HC', '07NM_YB', '08NM_CM', '09NM_CV', '10NM_VA', '11NM_LC', '12NM_PS', '13NM_JP', '14NM_LD',
            #   '15PH_JS',  '16PH_LP',  '17PH_SB',  '18PH_TH',  '19PH_VA',  '20PH_VS',
            #   '21IL_NM', '22IL_DG', '23IL_DM', '24IL_DJ', '25IL_DC', '26IL_AP', '27IL_SL', '28IL_LL', '29IL_VR', '30IL_LC', '31IL_MA', '32IL_LY', '33IL_BA', '34IL_CM', '35IL_EA', '36IL_LT',
            #   '37DL_05', '38DL_06', '39DL_07', '40DL_08', '41DL_11', '42DL_12', '43DL_13', '44DL_14', '45DL_15', '46DL_16', '47DL_17', '48DL_18', '49DL_19', '50DL_20', '51DL_21', '52DL_22',
            #   '53DL_23', '54DL_24', '55DL_25', '56DL_26', '57DL_27', '58DL_28', '59DL_29', '60DL_30', '61DL_31', '62DL_32', '63DL_34',
            #   ]

            # sujet = '49DL_19'

            # cond_list = ['VS', 'CHARGE']

            # cond = 'VS'
            # cond = 'CHARGE'

            if os.path.exists(os.path.join(path_prep, f'{sujet}_{cond}.fif')):

                print(f"{sujet} ALREADY COMPTUED", flush=True)
                continue

            else:

                print(f'#### COMPUTE {sujet} ####', flush=True)

            ################################
            ######## EXTRACT DATA ########
            ################################

            #sujet, cond = sujet_list[0], 'VS'
            data_eeg, respi, trig = open_raw_data(sujet, cond)

            info_eeg = mne.create_info(ch_names=chan_list_eeg.tolist(), ch_types=['eeg']*data_eeg.shape[0], sfreq=srate)
            info_eeg.set_montage("standard_1020")

            #### verif power
            if debug:
                raw_eeg = mne.io.RawArray(data_eeg,info_eeg)

                mne.viz.plot_raw_psd(raw_eeg)

                view_data(data_eeg, respi)

            ################################
            ######## AUX PROCESSING ########
            ################################

            #### verif ecg and respi orientation
            if debug:
                plt.plot(respi)
                plt.show()

            respi = respi_preproc(respi)
                

            ########################################################
            ######## PREPROCESSING & ARTIFACT CORRECTION ########
            ########################################################

            data_preproc = preprocessing_eeg(data_eeg, info_eeg, prep_step)

            if debug:

                view_data(data_preproc, respi)
                compare_pre_post(data_pre=data_eeg, data_post=data_preproc, srate=srate, chan_name='C3')

            # data_preproc = ICA_computation(data_preproc, info_eeg)
            # data_preproc = mne.io.RawArray(data_preproc, info_eeg.info)

            if debug:

                view_data(data_preproc, respi)

            data_preproc_clean = remove_artifacts(data_preproc, srate)

            ########################################
            ######## FINAL VIZUALISATION ########
            ########################################

            #### pre
            fig_raw = view_data(data_eeg, respi, return_fig=True)
            ####post
            fig_post = view_data(data_preproc_clean, respi, return_fig=True)
            #### for one chan
            # compare_pre_post(data_pre=data_eeg, data_post=data_preproc_clean, srate=srate, chan_name='FC5')

            fig_raw.suptitle(f'{sujet}_{cond}_raw')
            fig_post.suptitle(f'{sujet}_{cond}_preproc')

            plt.show(block=True) 

            ################################
            ######## CHOP AND SAVE ########
            ################################

            print('#### SAVE ####', flush=True)
    
            #### save alldata + stim chan
            data_export = np.vstack((data_preproc_clean, respi))

            info_eeg_export = mne.create_info(ch_names=chan_list.tolist(), ch_types=['eeg']*data_eeg.shape[0] + ['misc'], sfreq=srate)
            info_eeg_export.set_montage("standard_1020")

            raw_export = mne.io.RawArray(data_export, info_eeg_export)

            df_trig = pd.DataFrame({'trig' : ['inspi']*trig.shape[0], 'time' : trig})

            os.chdir(path_prep)

            #### save all cond
            raw_export.save(f'{sujet}_{cond}.fif')
                
            df_trig.to_excel(f'{sujet}_{cond}_trig.xlsx')



    ################################
    ######## APPLY CSD ########
    ################################

    #sujet = sujet_list[0]
    for sujet in sujet_list:

        #cond = cond_list[0]
        for cond in cond_list:

            os.chdir(path_prep)
            raw = mne.io.read_raw_fif(f'{sujet}_{cond}.fif', preload=True, verbose='critical')
            raw_csd = mne.preprocessing.compute_current_source_density(raw)

            raw_csd.save(f'{sujet}_{cond}_CSD.fif')

            if debug:

                chan_i = 2

                plt.plot(scipy.stats.zscore(raw.get_data()[chan_i,:]), label='raw')
                plt.plot(scipy.stats.zscore(raw_csd.get_data()[chan_i,:]), label='csd')
                plt.legend()
                plt.show()


    ########################################
    ######## COMPUTE NORM PARAMS ########
    ########################################

    xr_data = np.zeros((len(sujet_list), 2, 4, len(chan_list)))
    params_list = ['mean', 'sd', 'median', 'mad']

    #sujet = sujet_list[2]
    for sujet_i, sujet in enumerate(sujet_list):

        #cond = cond_list[0]
        for cond_i, cond in enumerate(cond_list):

            if cond_i == 0:
                data = load_data_sujet(sujet, cond)
                data_CSD = load_data_sujet_CSD(sujet, cond)
            else:
                data = np.concatenate((data, load_data_sujet(sujet, cond)), axis=1)
                data_CSD = np.concatenate((data, load_data_sujet_CSD(sujet, cond)), axis=1)

        if debug:

            plt.plot(data_CSD[0,:])
            plt.show()

        xr_data[sujet_i, 0, 0, :], xr_data[sujet_i, 0, 1, :] = data.mean(axis=1), data.std(axis=1) 
        xr_data[sujet_i, 0, 2, :], xr_data[sujet_i, 0, 3, :] = np.median(data, axis=1), scipy.stats.median_abs_deviation(data, axis=1)

        xr_data[sujet_i, 1, 0, :], xr_data[sujet_i, 1, 1, :] = data_CSD.mean(axis=1), data_CSD.std(axis=1) 
        xr_data[sujet_i, 1, 2, :], xr_data[sujet_i, 1, 3, :] = np.median(data_CSD, axis=1), scipy.stats.median_abs_deviation(data_CSD, axis=1)

    xr_dict = {'sujet' : sujet_list, 'data_type' : ['standard', 'CSD'], 'norm_param' : params_list, 'chan' : chan_list}
    xr_param_norm = xr.DataArray(data=xr_data, dims=xr_dict.keys(), coords=xr_dict)

    os.chdir(path_prep)
    xr_param_norm.to_netcdf('norm_params.nc')

    ########################################
    ######## AGGREGATES PREPROC ########
    ########################################

    time_vec = np.arange(0, section_time_general, 1/srate)

    xr_dict_preproc = {'sujet' : sujet_list, 'cond' : cond_list, 'chan' : chan_list, 'time' : time_vec}
    xr_data_preproc = np.zeros(( len(sujet_list), len(cond_list), chan_list.shape[0], time_vec.shape[0] ))

    os.chdir(path_prep)

    for sujet_i, sujet in enumerate(sujet_list):

        print(sujet)

        #cond = cond_list[0]
        for cond_i, cond in enumerate(cond_list):

            raw = mne.io.read_raw_fif(f"{sujet}_{cond}.fif")
            xr_data_preproc[sujet_i, cond_i, :, :] = raw.get_data()

    xr_preproc = xr.DataArray(data=xr_data_preproc, dims=xr_dict_preproc.keys(), coords=xr_dict_preproc.values())
    xr_preproc.to_netcdf('alldata_preproc.nc')










