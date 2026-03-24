


# from n00_config_params import *
# from n00bis_config_analysis_functions import *

import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

debug = False
path_results = ""




################################
######## WAVELETS ########
################################




def get_wavelets(srate=500):

    #### compute wavelets
    wavetime = np.arange(-3,3,1/srate)
    wavelets = np.zeros((nfrex, len(wavetime)), dtype=complex)

    # create Morlet wavelet family
    for fi in range(nfrex):
        
        s = cycles[fi] / (2*np.pi*frex[fi])
        gw = np.exp(-wavetime**2/ (2*s**2)) 
        sw = np.exp(1j*(2*np.pi*frex[fi]*wavetime))
        mw =  gw * sw

        wavelets[fi,:] = mw

    if debug:

        plt.plot(np.sum(np.abs(wavelets),axis=1))
        plt.show()

        plt.pcolormesh(np.real(wavelets))
        plt.show()

        plt.plot(np.real(wavelets)[0,:])
        plt.show()

    return wavelets





################################################################
######## FUNCTIONAL CONNECTIVITY METRIC EXTRACTION ########
################################################################




def get_MI_2sig(x, y):

    #### Freedman and Diaconis rule
    nbins_x = int(np.ceil((x.max() - x.min()) / (2 * scipy.stats.iqr(x)*(x.size**(-1/3)))))
    nbins_y = int(np.ceil((y.max() - y.min()) / (2 * scipy.stats.iqr(y)*(y.size**(-1/3)))))

    #### compute proba
    hist_x = np.histogram(x,bins = nbins_x)[0]
    hist_x = hist_x/np.sum(hist_x)
    hist_y = np.histogram(y,bins = nbins_y)[0]
    hist_y = hist_y/np.sum(hist_y)

    hist_2d = np.histogram2d(x, y, bins=[nbins_x, nbins_y])[0]
    hist_2d = hist_2d / np.sum(hist_2d)

    #### compute MI
    E_x = 0
    E_y = 0
    E_x_y = 0

    for p in hist_x:
        if p!=0 :
            E_x += -p*np.log2(p)

    for p in hist_y:
        if p!=0 :
            E_y += -p*np.log2(p)

    for p0 in hist_2d:
        for p in p0 :
            if p!=0 :
                E_x_y += -p*np.log2(p)

    MI = E_x+E_y-E_x_y

    return MI



def get_ISPC_2sig(x, y):

    ##### collect "eulerized" phase angle differences
    phase_angle_diff = np.exp(1j*(np.angle(x)-np.angle(y)))

    ##### compute ISPC
    ISPC = np.abs( np.mean(phase_angle_diff) )

    return ISPC



def get_WPLI_2sig(x, y):

    sxy = x * np.conj(y)

    # Extract imaginary part (which is sin(phase difference))
    im_part = np.imag(sxy)
    
    # Compute the weighted phase lag index (wPLI)
    numerator = np.abs(np.mean(im_part))  # Mean of the sign of the imaginary part
    denominator = np.mean(np.abs(im_part))  # Mean of the absolute imaginary part
    
    WPLI = numerator / denominator

    return WPLI



def get_Cxy_2sig(x, y):

    xy = x * np.conj(y)
    xx = x * np.conj(x)
    yy = y * np.conj(y)

    num = np.mean(np.abs(xy))**2
    denom = np.mean(np.abs(xx)) * np.mean(np.abs(yy))

    # Prevent division by zero
    if denom <= 0:
        return 0.0  

    Cxy = num / denom

    return Cxy


########################################
######## SIGNAL SIMULATION ########
########################################


def generate_synchronized_signals_CHIRP(duration_tot=1200, num_windows=50, window_dur=5, noise_coeff=1, freq_sync=15, 
                                    srate=500, amp_coeff=2, phase_diff=np.pi/2, freq_var=10):
        
    total_samples = duration_tot * srate  
    t = np.arange(total_samples) / srate  

    # Ensure non-overlapping windows with at least one window in between
    available_indices = np.arange(window_dur * srate, total_samples - window_dur * srate, window_dur * srate)  # Possible start points
    available_indices = available_indices[np.arange(0,available_indices.size,2)]

    if available_indices.size < num_windows:
        raise ValueError('sig too small for num_windows')
    else:
        sel_rand_win = np.random.choice(np.arange(available_indices.size), size=num_windows, replace=False)
        sync_windows = np.sort(available_indices[sel_rand_win])

    win_size = window_dur * srate

    # Apply synchronization in selected windows
    def get_freq_vec(sync_windows):

        for start_i, start in enumerate(sync_windows):

            if start_i == 0:

                freq_vec_rand = [np.random.randint(low=freq_sync-freq_var, high=freq_sync+freq_var) for i in range(10) if np.random.randint(low=freq_sync-freq_var, high=freq_sync+freq_var) != freq_sync][0]
                
                freq_vec = np.linspace(freq_vec_rand, freq_sync, start)
                freq_vec = np.concatenate([freq_vec, np.linspace(freq_sync, freq_var, win_size)])

            elif start_i == sync_windows.size-1:

                freq_vec_rand = [np.random.randint(low=freq_sync-freq_var, high=freq_sync+freq_var) for i in range(10) if np.random.randint(low=freq_sync-freq_var, high=freq_sync+freq_var) != freq_sync][0]
                
                start_for_vec = freq_vec.size
                stop_for_vec = start
                mid_point = int((start_for_vec + stop_for_vec) / 2)

                freq_vec = np.concatenate([freq_vec, np.linspace(freq_sync, freq_vec_rand, mid_point-start_for_vec)])  
                freq_vec = np.concatenate([freq_vec, np.linspace(freq_vec_rand, freq_sync, mid_point-start_for_vec)])  
                freq_vec = np.concatenate([freq_vec, np.linspace(freq_sync, freq_sync, win_size)])

                freq_vec = np.concatenate([freq_vec, np.linspace(freq_sync, freq_vec_rand, total_samples-start+win_size)])

            else:

                freq_vec_rand = [np.random.randint(low=freq_sync-freq_var, high=freq_sync+freq_var) for i in range(10) if np.random.randint(low=freq_sync-freq_var, high=freq_sync+freq_var) != freq_sync][0]
                
                start_for_vec = freq_vec.size
                stop_for_vec = start
                mid_point = int((start_for_vec + stop_for_vec) / 2)

                freq_vec = np.concatenate([freq_vec, np.linspace(freq_sync, freq_vec_rand, mid_point-start_for_vec)])  
                freq_vec = np.concatenate([freq_vec, np.linspace(freq_vec_rand, freq_sync, mid_point-start_for_vec)])  
                freq_vec = np.concatenate([freq_vec, np.linspace(freq_sync, freq_sync, win_size)])

        return freq_vec
        
    freq_vec_1 = get_freq_vec(sync_windows)
    freq_vec_2 = get_freq_vec(sync_windows)

    if debug:

        plt.plot(freq_vec_1)
        plt.plot(freq_vec_2)
        plt.vlines(np.array(sync_windows), ymin=freq_sync-freq_var, ymax=freq_sync+freq_var, color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=freq_sync-freq_var, ymax=freq_sync+freq_var, color='r', linestyles='--')
        plt.show()

    phase_1 = 2 * np.pi * np.cumsum(freq_vec_1) / srate  # Integrate frequency to get phase
    chirp_signal_1 = np.sin(phase_1)

    phase_2 = 2 * np.pi * np.cumsum(freq_vec_2) / srate  # Integrate frequency to get phase
    chirp_signal_2 = np.sin(phase_2 + phase_diff)

    if debug:

        plt.plot(chirp_signal_1)
        plt.plot(chirp_signal_2)
        plt.vlines(np.array(sync_windows), ymin=chirp_signal_1.min(), ymax=chirp_signal_1.max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=chirp_signal_1.min(), ymax=chirp_signal_1.max(), color='r', linestyles='--')
        plt.show()

    signal1 = chirp_signal_1[:total_samples] * amp_coeff + np.random.randn(total_samples) * noise_coeff
    signal2 = chirp_signal_2[:total_samples] * amp_coeff + np.random.randn(total_samples) * noise_coeff

    if debug:

        plt.plot(signal1)
        plt.vlines(np.array(sync_windows), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='r', linestyles='--')
        plt.plot(signal2)
        plt.show()

    return signal1, signal2, sync_windows




def generate_synchronized_signals_RAWOSC(duration_tot=1200, num_windows=50, window_dur=5, noise_coeff=1, freq_sync=10, 
                                    srate=500, nshift=1000, amp_coeff=2, phase_diff=np.pi/2):
        
    total_samples = duration_tot * srate  
    t = np.arange(total_samples) / srate  

    # Initialize signals with random noise
    signal1 = np.random.randn(total_samples) * noise_coeff
    signal2 = np.random.randn(total_samples) * noise_coeff

    # Create oscillatory activity at the common frequency
    osc_total = np.sin(2 * np.pi * freq_sync * t)

    # Add oscillatory component 
    signal1 += osc_total

    shift_indices = np.random.randint(low=0, high=signal1.size, size=nshift)
    shift_indices = sorted(shift_indices)  # Get random non-overlapping windows

    for i in shift_indices:
        signal2[i:] *= -1

    if debug:
        plt.plot(signal1)
        plt.plot(signal2)
        plt.show()

    # Ensure non-overlapping windows with at least one window in between
    available_indices = np.arange(window_dur * srate, total_samples - window_dur * srate, window_dur * srate)  # Possible start points
    available_indices = available_indices[np.arange(0,available_indices.size,2)]

    if available_indices.size < num_windows:
        raise ValueError('sig too small for num_windows')
    else:
        sel_rand_win = np.random.choice(np.arange(available_indices.size), size=num_windows, replace=False)
        sync_windows = available_indices[sel_rand_win]

    # Apply synchronization in selected windows
    for start in sync_windows:
        end = start + window_dur * srate  # sec
        osc1 = np.sin(2 * np.pi * freq_sync * np.arange(0, window_dur, 1/srate))
        osc2 = np.sin(2 * np.pi * freq_sync * np.arange(0, window_dur, 1/srate) + phase_diff)

        # Overwrite the signal in the synchronization window
        signal1[start:end] = osc1 * amp_coeff + np.random.randn(osc1.size) * noise_coeff
        signal2[start:end] = osc2 * amp_coeff + np.random.randn(osc1.size) * noise_coeff

    if debug:

        plt.plot(signal1)
        plt.vlines(np.array(sync_windows), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='r', linestyles='--')
        plt.plot(signal2)
        plt.show()

    return signal1, signal2, sync_windows




def morlet_wavelet_transform(signal_data, srate=500, freq_sync=6, num_cycles=7):
    
    # Define time window for the wavelet
    t = np.arange(-5, 5, 1/srate)  # Time vector
    sigma_t = num_cycles / (2 * np.pi * freq_sync)  # Standard deviation in time

    # Manually create the Morlet wavelet
    wavelet = np.exp(2j * np.pi * freq_sync * t) * np.exp(-t**2 / (2 * sigma_t**2))

    # Convolve wavelet with signal using FFT for speed
    analytic_signal = scipy.signal.fftconvolve(signal_data, wavelet, mode='same')

    return analytic_signal



################################
######## EXECUTE ########
################################


# Generate signals
duration_tot=800 # sec
num_windows=50
window_dur=5 # sec
srate=100
nshift=1000
freq_sync=15
noise_coeff=2
amp_coeff=2
phase_diff=np.pi/2
freq_var=10


# wavelets params
nfrex = 150
ncycle_list = [7, 41]
freq_list = [2, 150]
wavetime = np.arange(-3,3,1/srate)
frex = np.logspace(np.log10(freq_list[0]), np.log10(freq_list[1]), nfrex) 
cycles = np.logspace(np.log10(ncycle_list[0]), np.log10(ncycle_list[1]), nfrex).astype('int')

# generate sig

time_vec = np.arange(0, duration_tot, 1/srate)

signal1, signal2, sync_windows = generate_synchronized_signals_RAWOSC(duration_tot=duration_tot, num_windows=num_windows, window_dur=window_dur, 
                                                                srate=srate, nshift=nshift, freq_sync=freq_sync, amp_coeff=amp_coeff, noise_coeff=noise_coeff,
                                                                phase_diff=phase_diff)

signal1, signal2, sync_windows = generate_synchronized_signals_CHIRP(duration_tot=duration_tot, num_windows=num_windows, window_dur=window_dur, 
                                                                     noise_coeff=noise_coeff, freq_sync=freq_sync, 
                                    srate=srate, amp_coeff=amp_coeff, phase_diff=phase_diff, freq_var=freq_var)

if debug:

    plt.plot(signal1)
    plt.vlines(np.array(sync_windows), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='g')
    plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='r', linestyles='--')
    plt.plot(signal2)
    plt.show()

wavelets = get_wavelets(srate=srate)

if debug:

    fig, ax = plt.subplots()
    ax.pcolormesh(np.real(wavelets))
    plt.show()

frex_chunk = frex[frex < 30]
wavelets = wavelets[frex < 30,:]

if debug:

    fig, ax = plt.subplots()
    ax.pcolormesh(np.real(wavelets))
    plt.show()

tf1 = np.zeros((frex_chunk.shape[0], signal1.shape[0]))
tf2 = np.zeros((frex_chunk.shape[0], signal2.shape[0]))

for fi in range(frex_chunk.shape[0]):
    
    tf1[fi,:] = abs(scipy.signal.fftconvolve(signal1, wavelets[fi,:], 'same'))**2 
    tf2[fi,:] = abs(scipy.signal.fftconvolve(signal2, wavelets[fi,:], 'same'))**2 

time = np.arange(0, signal1.size)
fig, ax = plt.subplots()
ax.pcolormesh(time, frex_chunk, tf1)
plt.show()

time = np.arange(0, signal1.size)
fig, ax = plt.subplots()
ax.pcolormesh(time, frex_chunk, tf2)
plt.show()

# Extract analytic signal using Morlet wavelet
x = morlet_wavelet_transform(signal1, srate=srate, freq_sync=freq_sync, num_cycles=7)
y = morlet_wavelet_transform(signal2, srate=srate, freq_sync=freq_sync, num_cycles=7)

if debug:

    x_plot = np.abs(x)**2
    y_plot = np.abs(y)**2
    plt.plot(x_plot, label='x')
    plt.plot(y_plot, label='y')
    sig_tot = np.concatenate((x_plot, y_plot))
    plt.vlines(np.array(sync_windows), ymin=sig_tot.min(), ymax=sig_tot.max(), color='g')
    plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=sig_tot.min(), ymax=sig_tot.max(), color='r', linestyles='--')
    plt.legend()
    plt.show()

# conv metrics
# win_conv = 1*srate
ncycle_FC = 10
win_slide = int(ncycle_FC/freq_sync*srate)
win_overlap = 0.5 #percentage

time_vec_win = np.arange(0, duration_tot, win_slide*win_overlap/srate)[:-1]

pre_win, post_win = 4, 4
res_win_size_fc = np.zeros((4, np.arange(2,30,4).size, (window_dur+pre_win+post_win)*srate))

for ncycle_FC_i, ncycle_FC in enumerate(np.arange(2,30,4)):

    win_conv = int(ncycle_FC/freq_sync*srate)

    # x_pad = np.pad(x, int(win_conv/2), mode='reflect')
    # y_pad = np.pad(y, int(win_conv/2), mode='reflect')

    # conv
    win_vec = np.arange(0, signal1.size-win_slide*win_overlap, win_conv*win_overlap).astype('int')
    MI_conv = []
    for i in win_vec:
        MI_conv.append(get_MI_2sig(signal1[i:i+win_slide], signal2[i:i+win_slide]))
    MI_conv = np.array(MI_conv)

    if debug:

        x_plot = MI_conv
        plt.plot(win_vec, x_plot)
        plt.vlines(np.array(sync_windows), ymin=x_plot.min(), ymax=x_plot.max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=x_plot.min(), ymax=x_plot.max(), color='r', linestyles='--')
        plt.show()

    win_vec = np.arange(0, signal1.size-win_slide*win_overlap, win_conv*win_overlap).astype('int')
    ISPC_conv = []
    for i in win_vec:
        ISPC_conv.append(get_ISPC_2sig(x[i:i+win_conv], y[i:i+win_conv]))
    ISPC_conv = np.array(ISPC_conv)

    ISPC_conv_full = np.array([get_ISPC_2sig(x[i:i+win_conv], y[i:i+win_conv]) for i in range(int(x.size))])

    if debug:

        plt.plot(win_vec, ISPC_conv)
        plt.plot(np.arange(x.size), ISPC_conv_full)
        plt.vlines(np.array(sync_windows), ymin=x_plot.min(), ymax=x_plot.max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=x_plot.min(), ymax=x_plot.max(), color='r', linestyles='--')
        plt.show()

        x_plot = ISPC_conv
        plt.plot(win_vec, x_plot)
        plt.vlines(np.array(sync_windows), ymin=x_plot.min(), ymax=x_plot.max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=x_plot.min(), ymax=x_plot.max(), color='r', linestyles='--')
        plt.show()

    win_vec = np.arange(0, signal1.size-win_slide*win_overlap, win_conv*win_overlap).astype('int')
    WPLI_conv = []
    for i in win_vec:
        WPLI_conv.append(get_WPLI_2sig(x[i:i+win_conv], y[i:i+win_conv]))
    WPLI_conv = np.array(WPLI_conv)

    if debug:

        x_plot = WPLI_conv
        plt.plot(win_vec, x_plot)
        plt.vlines(np.array(sync_windows), ymin=x_plot.min(), ymax=x_plot.max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=x_plot.min(), ymax=x_plot.max(), color='r', linestyles='--')
        plt.show()

    win_vec = np.arange(0, signal1.size-win_slide*win_overlap, win_conv*win_overlap).astype('int')
    Cxy_conv = []
    for i in win_vec:
        Cxy_conv.append(get_Cxy_2sig(x[i:i+win_conv], y[i:i+win_conv]))
    Cxy_conv = np.array(Cxy_conv)

    if debug:

        x_plot = Cxy_conv
        plt.plot(win_vec, x_plot)
        plt.vlines(np.array(sync_windows), ymin=x_plot.min(), ymax=x_plot.max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=x_plot.min(), ymax=x_plot.max(), color='r', linestyles='--')
        plt.show()

    win_time = window_dur + pre_win + post_win
    srate_slide = 1/(win_slide*win_overlap/srate)
    n_obs_chunk = int(win_time*srate_slide)
    epochs_MI = np.zeros((sync_windows.shape[0], n_obs_chunk))
    epochs_ISPC = np.zeros((sync_windows.shape[0], n_obs_chunk))
    epochs_WPLI = np.zeros((sync_windows.shape[0], n_obs_chunk))
    epochs_Cxy = np.zeros((sync_windows.shape[0], n_obs_chunk))
    
    for win_i, win_time_chunk in enumerate(sync_windows/srate):

        time_sel = (time_vec_win >= (win_time_chunk-pre_win)) & (time_vec_win <= (win_time_chunk+window_dur+post_win))
        epochs_ISPC[win_i,:] = ISPC_conv[time_sel][:n_obs_chunk]
        epochs_MI[win_i,:] = MI_conv[time_sel][:n_obs_chunk]
        epochs_WPLI[win_i,:] = WPLI_conv[time_sel][:n_obs_chunk]
        epochs_Cxy[win_i,:] = Cxy_conv[time_sel][:n_obs_chunk]

    if debug:

        x_plot = epochs_ISPC
        x_plot = epochs_MI
        x_plot = epochs_WPLI

        fig, axs = plt.subplots(4, 1)
        im0 = axs[0].pcolormesh(epochs_ISPC)
        im1 = axs[1].pcolormesh(epochs_MI)
        im2 = axs[2].pcolormesh(epochs_WPLI)
        im3 = axs[3].pcolormesh(epochs_Cxy)
        for im in [im0, im1, im2, im3]:
            im.set_clim(0, 1)
        plt.show()

        fig, axs = plt.subplots(4, 1)
        im0 = axs[0].plot(np.median(epochs_ISPC, axis=0))
        im1 = axs[1].plot(np.median(epochs_MI, axis=0))
        im2 = axs[2].plot(np.median(epochs_WPLI, axis=0))
        im3 = axs[3].plot(np.median(epochs_Cxy, axis=0))
        for im in [im0, im1, im2, im3]:
            im.set_ylim(0, 1)
        plt.show()

        plt.figure()
        plt.pcolormesh(epochs_ISPC)
        
        plt.figure()
        plt.pcolormesh(epochs_MI)
        
        plt.figure()
        plt.pcolormesh(epochs_WPLI)
        plt.show()

        plt.plot(np.median(x_plot, axis=0))
        plt.show()

    res_win_size_fc[0,ncycle_FC_i,:] = np.median(epochs_ISPC, axis=0)
    res_win_size_fc[1,ncycle_FC_i,:] = np.median(epochs_MI, axis=0)
    res_win_size_fc[2,ncycle_FC_i,:] = np.median(epochs_WPLI, axis=0)
    res_win_size_fc[3,ncycle_FC_i,:] = np.median(epochs_Cxy, axis=0)

    #### plot for one ncycle
    os.chdir(os.path.join(path_results, 'FC'))

    plt.pcolormesh(epochs_ISPC)
    plt.title(f'ISPC_ncycle:{ncycle_FC}')
    plt.savefig(f'ISPC_raster_ncycle{ncycle_FC}.jpg')
    # plt.show()
    plt.close('all')

    plt.pcolormesh(epochs_MI)
    plt.title(f'MI_ncycle:{ncycle_FC}')
    plt.savefig(f'MI_raster_ncycle{ncycle_FC}.jpg')
    # plt.show()
    plt.close('all')

    plt.pcolormesh(epochs_WPLI)
    plt.title(f'WPLI_ncycle:{ncycle_FC}')
    plt.savefig(f'WPLI_raster_ncycle{ncycle_FC}.jpg')
    # plt.show()
    plt.close('all')

    plt.pcolormesh(epochs_Cxy)
    plt.title(f'Cxy_ncycle:{ncycle_FC}')
    plt.savefig(f'Cxy_raster_ncycle{ncycle_FC}.jpg')
    # plt.show()
    plt.close('all')

    plt.plot(scipy.stats.zscore(np.median(epochs_ISPC, axis=0)), label='ISPC')
    plt.plot(scipy.stats.zscore(np.median(epochs_MI, axis=0)), label='MI')
    plt.plot(scipy.stats.zscore(np.median(epochs_WPLI, axis=0)), label='WPLI')
    plt.plot(scipy.stats.zscore(np.median(epochs_Cxy, axis=0)), label='Cxy')
    plt.legend()
    plt.savefig(f'median_FC_comparison_ncycle{ncycle_FC}.jpg')
    # plt.show()
    plt.close('all')

#### plot for all ncycle
os.chdir(os.path.join(path_results, 'FC'))

for ncycle_FC_i, ncycle_FC in enumerate(np.arange(2,30,2)):
    plt.plot(res_win_size_fc[0,ncycle_FC_i,:], label=ncycle_FC)
plt.legend()
plt.title('ISPC')
plt.savefig(f'ALLMETRIC_median_ISPC_comparison_ncycle.jpg')
# plt.show()
plt.close('all')

for ncycle_FC_i, ncycle_FC in enumerate(np.arange(2,30,2)):
    plt.plot(res_win_size_fc[1,ncycle_FC_i,:], label=ncycle_FC)
plt.legend()
plt.title('MI')
plt.savefig(f'ALLMETRIC_median_MI_comparison_ncycle.jpg')
# plt.show()
plt.close('all')

for ncycle_FC_i, ncycle_FC in enumerate(np.arange(2,30,2)):
    plt.plot(res_win_size_fc[2,ncycle_FC_i,:], label=ncycle_FC)
plt.legend()
plt.title('WPLI')
plt.savefig(f'ALLMETRIC_median_WPLI_comparison_ncycle.jpg')
# plt.show()
plt.close('all')

for ncycle_FC_i, ncycle_FC in enumerate(np.arange(2,30,2)):
    plt.plot(res_win_size_fc[3,ncycle_FC_i,:], label=ncycle_FC)
plt.legend()
plt.title('Cxy')
plt.savefig(f'ALLMETRIC_median_Cxy_comparison_ncycle.jpg')
# plt.show()
plt.close('all')



#### noise evaluation
noise_vec = np.arange(1,5,0.5)
ncycle_FC = 10
win_conv = int(ncycle_FC/freq_sync*srate)
pre_win, post_win = 4, 4
res_noise_coeff_fc = np.zeros((4, noise_vec.size, (window_dur+pre_win+post_win)*srate))

for noice_coeff_i, noise_coeff in enumerate(noise_vec):

    print(noise_coeff)
    
    signal1, signal2, sync_windows = generate_synchronized_signals_RAWOSC(duration_tot=duration_tot, num_windows=num_windows, window_dur=window_dur, 
                                                                srate=srate, nshift=nshift, freq_sync=freq_sync, amp_coeff=amp_coeff, noise_coeff=noise_coeff,
                                                                phase_diff=phase_diff)

    signal1, signal2, sync_windows = generate_synchronized_signals_CHIRP(duration_tot=duration_tot, num_windows=num_windows, window_dur=window_dur, 
                                                                        noise_coeff=noise_coeff, freq_sync=freq_sync, 
                                        srate=srate, amp_coeff=amp_coeff, phase_diff=phase_diff, freq_var=freq_var)
    
    if debug:

        plt.plot(signal1)
        plt.vlines(np.array(sync_windows), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='g')
        plt.vlines((np.array(sync_windows)+window_dur*srate), ymin=np.concatenate((signal1, signal2)).min(), ymax=np.concatenate((signal1, signal2)).max(), color='r', linestyles='--')
        plt.plot(signal2)
        plt.show()
    
    signal1_pad = np.pad(signal1, int(win_conv/2), mode='reflect')
    signal2_pad = np.pad(signal2, int(win_conv/2), mode='reflect')

    x_pad = np.pad(x, int(win_conv/2), mode='reflect')
    y_pad = np.pad(y, int(win_conv/2), mode='reflect')

    # conv
    MI_conv = np.array([get_MI_2sig(signal1_pad[i:i+win_conv], signal2_pad[i:i+win_conv]) for i in range(int(signal1_pad.size-win_conv))])
    ISPC_conv = np.array([get_ISPC_2sig(x_pad[i:i+win_conv], y_pad[i:i+win_conv]) for i in range(int(x_pad.size-win_conv))])
    WPLI_conv = np.array([get_WPLI_2sig(x_pad[i:i+win_conv], y_pad[i:i+win_conv]) for i in range(int(x_pad.size-win_conv))])
    Cxy_conv = np.array([get_Cxy_2sig(x_pad[i:i+win_conv], y_pad[i:i+win_conv]) for i in range(int(x_pad.size-win_conv))])

    epochs_MI = []
    epochs_ISPC = []
    epochs_WPLI = []
    epochs_Cxy = []
    
    for win_i, win_time in enumerate(sync_windows):

        start, stop = win_time-pre_win*srate, win_time+window_dur*srate+post_win*srate
        if start < 0 or stop > duration_tot*srate:
            continue
        epochs_ISPC.append(ISPC_conv[start:stop])
        epochs_MI.append(MI_conv[start:stop])
        epochs_WPLI.append(WPLI_conv[start:stop])
        epochs_Cxy.append(Cxy_conv[start:stop])

    epochs_ISPC = np.array(epochs_ISPC)
    epochs_MI = np.array(epochs_MI)
    epochs_WPLI = np.array(epochs_WPLI)
    epochs_Cxy = np.array(epochs_Cxy)

    res_noise_coeff_fc[0,noice_coeff_i,:] = np.median(epochs_ISPC, axis=0)
    res_noise_coeff_fc[1,noice_coeff_i,:] = np.median(epochs_MI, axis=0)
    res_noise_coeff_fc[2,noice_coeff_i,:] = np.median(epochs_WPLI, axis=0)
    res_noise_coeff_fc[3,noice_coeff_i,:] = np.median(epochs_Cxy, axis=0)

#### plot for all ncycle
os.chdir(os.path.join(path_results, 'FC'))

for noice_coeff_i, noise_coeff in enumerate(noise_vec):
    plt.plot(res_noise_coeff_fc[0,noice_coeff_i,:], label=noise_coeff)
plt.legend()
plt.title('ISPC')
plt.savefig(f'ALLMETRIC_median_ISPC_comparison_noise_coeff.jpg')
# plt.show()
plt.close('all')

for noice_coeff_i, noise_coeff in enumerate(noise_vec):
    plt.plot(res_noise_coeff_fc[1,noice_coeff_i,:], label=noise_coeff)
plt.legend()
plt.title('MI')
plt.savefig(f'ALLMETRIC_median_MI_comparison_noise_coeff.jpg')
# plt.show()
plt.close('all')

for noice_coeff_i, noise_coeff in enumerate(noise_vec):
    plt.plot(res_noise_coeff_fc[2,noice_coeff_i,:], label=noise_coeff)
plt.legend()
plt.title('WPLI')
plt.savefig(f'ALLMETRIC_median_WPLI_comparison_noise_coeff.jpg')
# plt.show()
plt.close('all')

for noice_coeff_i, noise_coeff in enumerate(noise_vec):
    plt.plot(res_noise_coeff_fc[3,noice_coeff_i,:], label=noise_coeff)
plt.legend()
plt.title('Cxy')
plt.savefig(f'ALLMETRIC_median_Cxy_comparison_noise_coeff.jpg')
# plt.show()
plt.close('all')

