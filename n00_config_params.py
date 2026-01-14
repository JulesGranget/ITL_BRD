
import numpy as np


################################
######## GENERAL PARAMS ######## 
################################

srate = 500

project_name_list_raw = ['COVEM_ITL', 'NORMATIVE', 'PHYSIOLOGY', 'SLP', 'ITL_LEO','DYSLEARN']
project_name_list = ['NORMATIVE', 'PHYSIOLOGY', 'ITL_LEO', 'DYSLEARN']

sujet_list_project_wise = {'COVEM_ITL': ['01NM', '02HM', '03DG', '04DM', '05DR', '06DJ', '07DC', '08AP', '09SL', '10LL', '11VR', '12LC', '13NN', '14MA', '15LY', '16BA', '17CM', '18EA', '19LT'],
                      'NORMATIVE' : ['MW02', 'OL04', 'MC05', 'LS07', 'JS08', 'HC09', 'YB10', 'CM12', 'CV13', 'VA14', 'LC15', 'PS16', 'JP19', 'LD20'], 
                      'PHYSIOLOGY' : ['JS08',  'LP26',  'MN23',  'SB27',  'TH24',  'VA14',  'VS06'], 
                      'SLP' : ['AB33', 'BK35', 'CD28', 'ES32', 'JC30', 'MM34', 'SG29', 'ZM31'],
                      'ITL_LEO' : ['01NM', '03DG', '04DM', '06DJ', '07DC', '08AP', '09SL', '10LL', '11VR', '12LC', '14MA', '15LY', '16BA', '17CM', '18EA', '19LT'],
                      'DYSLEARN' : ['05','06','07','08','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','34']
                      }

sujet_list = ['01NM_MW', '02NM_OL', '03NM_MC', '04NM_LS', '05NM_JS', '06NM_HC', '07NM_YB', '08NM_CM', '09NM_CV', '10NM_VA', '11NM_LC', '12NM_PS', '13NM_JP', '14NM_LD',
              '15PH_JS',  '16PH_LP',  '17PH_SB',  '18PH_TH',  '19PH_VA',  '20PH_VS',
              '21IL_NM', '22IL_DG', '23IL_DM', '24IL_DJ', '25IL_DC', '26IL_AP', '27IL_SL', '28IL_LL', '29IL_VR', '30IL_LC', '31IL_MA', '32IL_LY', '33IL_BA', '34IL_CM', '35IL_EA', '36IL_LT',
              '37DL_05', '38DL_06', '39DL_07', '40DL_08', '41DL_11', '42DL_12', '43DL_13', '44DL_14', '45DL_15', '46DL_16', '47DL_17', '48DL_18', '49DL_19', '50DL_20', '51DL_21', '52DL_22',
              '53DL_23', '54DL_24', '55DL_25', '56DL_26', '57DL_27', '58DL_28', '59DL_29', '60DL_30', '61DL_31', '62DL_32', '63DL_34',
              ]

sujet_list_FC = ['01NM_MW', '02NM_OL', '04NM_LS', '05NM_JS', '06NM_HC', '07NM_YB', '08NM_CM', '10NM_VA', '12NM_PS', '13NM_JP', '14NM_LD', '15PH_JS', '19PH_VA', 
                 '24IL_DJ', '26IL_AP', '27IL_SL', '28IL_LL', '31IL_MA', '33IL_BA', '34IL_CM', '35IL_EA', '37DL_05', '38DL_06', '39DL_07', '40DL_08', 
                 '43DL_13', '44DL_14', '45DL_15', '47DL_17', '49DL_19', '50DL_20', '51DL_21', '52DL_22', '54DL_24', '55DL_25', '56DL_26', 
                 '57DL_27', '58DL_28', '59DL_29', '61DL_31', '62DL_32', '63DL_34']

cond_list = ['VS', 'CHARGE']

sujet_project_nomenclature = {'NM' : 'NORMATIVE', 'PH' : 'PHYSIOLOGY', 'IL' : 'ITL_LEO', 'DL' : 'DYSLEARN'}

sujet_list_correspondance = {'NM_MW02' : '01NM_MW', 'NM_OL04' : '02NM_OL', 'NM_MC05' : '03NM_MC', 'NM_LS07' : '04NM_LS', 'NM_JS08' : '05NM_JS', 'NM_HC09' : '06NM_HC', 
                             'NM_YB10' : '07NM_YB', 'NM_CM12' : '08NM_CM', 'NM_CV13' : '09NM_CV', 'NM_VA14' : '10NM_VA', 'NM_LC15' : '11NM_LC', 'NM_PS16' : '12NM_PS', 
                             'NM_JP19' : '13NM_JP', 'NM_LD20' : '14NM_LD', 'PH_JS08' : '15PH_JS', 'PH_LP26' : '16PH_LP', 'PH_SB27' : '18=7PH_SB',
                             'PH_TH24' : '18PH_TH', 'PH_VA14' : '19PH_VA', 'PH_VS06' : '20PH_VS', 'IL_01NM' : '21IL_NM', 'IL_03DG' : '22IL_DG', 'IL_04DM' : '23IL_DM', 
                             'IL_06DJ' : '24IL_DJ', 'IL_07DC' : '25IL_DC', 'IL_08AP' : '26IL_AP', 'IL_09SL' : '27IL_SL', 'IL_10LL' : '28IL_LL', 'IL_11VR' : '29IL_VR', 
                             'IL_12LC' : '30IL_LC', 'IL_14MA' : '31IL_MA', 'IL_15LY' : '32IL_LY', 'IL_16BA' : '33IL_BA', 'IL_17CM' : '34IL_CM', 'IL_18EA' : '35IL_EA', 
                             'IL_19LT' : '36IL_LT',
                             'DL_05' : '37DL_05', 'DL_06' : '38DL_06', 'DL_07' : '39DL_07', 'DL_08' : '40DL_08', 'DL_11' : '41DL_11', 'DL_12' : '42DL_12', 
                             'DL_13' : '43DL_13', 'DL_14' : '44DL_14', 'DL_15' : '45DL_15', 'DL_16' : '46DL_16', 'DL_17' : '47DL_17', 'DL_18' : '48DL_18',
                             'DL_19' : '49DL_19', 'DL_20' : '50DL_20', 'DL_21' : '51DL_21', 'DL_22' : '52DL_22', 'DL_23' : '53DL_23', 'DL_24' : '54DL_24', 
                             'DL_25' : '55DL_25', 'DL_26' : '56DL_26', 'DL_27' : '57DL_27', 'DL_28' : '58DL_28', 'DL_29' : '59DL_29', 'DL_30' : '60DL_30', 
                             'DL_31' : '61DL_31', 'DL_32' : '62DL_32', 'DL_34' : '63DL_34',
                             }

chan_list_project_wise = {'COVEM_ITL': ['FC1', 'FC2', 'Cz', 'C2', 'CP1', 'CP2', 'EMG'],
                        'NORMATIVE' : ['Fp1', 'F7', 'F3', 'Fz', 'FC5', 'FC1', 'A1', 'T7', 'C3', 'Cz', 'TP9', 'CP5', 'CP1', 'P7', 'P3', 'Pz', 'Fp2', 'F4', 'F8', 'FC2', 'FC6', 'C4', 'T8', 'A2', 'CP2', 'CP6', 'TP10', 'P4', 'P8', 'O1', 'Oz', 'O2', 'Debit', 'Pression', 'EMG PS', 'ECG', 'FCz'], 
                        'PHYSIOLOGY' : ['EOG', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'Debit', 'Pression', 'PS', 'ECG'], 
                        'SLP' : ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'FCz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'ECG', 'ScalEMG'],
                        'ITL_LEO' : ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'A1', 'CP5', 'CP1', 'CP2', 'CP6', 'A2', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'EOG', 'EMG', 'PRESSION'],
                        'DYSLEARN' : ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'PRESS', 'ECG', 'TRIG']}

chan_list = np.array(['C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7',
       'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Fp2', 'Fz', 'O1', 'O2', 'Oz',
       'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8', 'pression'])

chan_list_eeg = np.array(['C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6', 'Cz', 'F3', 'F4', 'F7',
       'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Fp2', 'Fz', 'O1', 'O2', 'Oz',
       'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8'])

chan_list_eeg_short = np.array(['C3', 'C4', 'CP1', 'CP2', 'Cz', 'F3', 'F4', 'FC1', 'FC2', 'Fz'])

#### NOTES ####
# In PHYSIOLOGY sujet ['MC05', 'OL04'] have been excluded because bad load
# In ITL sujet ['NN', ] have been excluded due to bad signals in ITL
# In NORMATIVE ['VS06', 'ML11', 'SL18', 'DR05'] removed no signal in CHARGE

# sujets : [23, 41, 42, 46, 48, 53, 60] removed because of mvt and respiratory artifacts

condition_list_project_wise = {'COVEM_ITL': ['CHARGE'],
                      'NORMATIVE' : ['CHARGE', 'PETITE CHARGE', 'ARTIFACT', 'SNIFS', 'VS', 'VS2'], 
                      'PHYSIOLOGY' : ['VS', 'CHARGE'], 
                      'SLP' : ['CHARGE'],
                       'ITL_LEO' : ['VS', 'CO2', 'ITL'],
                       'DYSLEARN' : ['VS', 'ITL']}

params_extraction_data = {'COVEM_ITL' : {'time_cutoff' : 13},
                      'NORMATIVE' : { 'time_cutoff' : {'CHARGE' : 0, 'SNIFS' : 0, 'VS' : 0}},
                      'PHYSIOLOGY' : {'time_cutoff' : 0},
                      'SLP' : {'time_cutoff' : 0},
                       'ITL_LEO' : {'time_cutoff' : 0}}



########################################
######## PATH DEFINITION ########
########################################

import socket
import os
import platform
 
PC_OS = platform.system()
PC_ID = socket.gethostname()
init_workdir = os.getcwd()

if PC_ID == 'jules_pc':

    try: 
        os.chdir('N:\\')
        teleworking = False
    except:
        teleworking = True 

    if teleworking:

        PC_working = 'Jules_VPN'
        path_main_workdir = 'Z:\\cmo\\Projets\\PPI_Jules\\Scripts'
        path_general = 'Z:\\cmo\\Projets\\PPI_Jules'
        path_memmap = 'Z:\\cmo\\Projets\\PPI_Jules\\memmap'
        n_core = 4

    else:

        PC_working = 'Jules_VPN'
        path_main_workdir = 'N:\\cmo\\Projets\\PPI_Jules\\Scripts'
        path_general = 'N:\\cmo\\Projets\\PPI_Jules'
        path_memmap = 'N:\\cmo\\Projets\\PPI_Jules\\memmap'
        n_core = 4


elif PC_ID == 'jules-precisiont1700':

    PC_working = 'Jules_Labo_Linux'
    path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/PPI_Jules/Scripts'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/PPI_Jules'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/PPI_Jules/memmap'
    n_core = 5

elif PC_ID == 'DESKTOP-3IJUK7R':

    PC_working = 'Jules_Labo_Win'
    path_main_workdir = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_general = 'D:\\LPPR_CMO_PROJECT\\Lyon'
    path_memmap = 'D:\\LPPR_CMO_PROJECT\\Lyon\\memmap'
    n_core = 2

elif PC_ID == 'pc-jules' or PC_ID == 'LAPTOP-EI7OSP7K':

    PC_working = 'Jules_Labo_Linux'
    path_main_workdir = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/PPI_Jules/Scripts'
    path_general = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/PPI_Jules'
    path_memmap = '/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/PPI_Jules/memmap'
    n_core = 4

elif PC_ID == 'pc-valentin':

    PC_working = 'Valentin_Labo_Linux'
    path_main_workdir = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J'
    path_memmap = '/home/valentin/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/memmap'
    n_core = 6

elif PC_ID == 'nodeGPU':

    PC_working = 'nodeGPU'
    path_main_workdir = '/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Script_Python_EEG_Paris_git'
    path_general = '/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J'
    path_memmap = '/mnt/data/julesgranget/EEG_Paris_J/memmap'
    n_core = 15

#### interactif node from cluster
elif PC_ID == 'node14':

    PC_working = 'node14'
    path_main_workdir = '/crnldata/cmo/Projets/PPI_Jules/Scripts'
    path_general = '/crnldata/cmo/Projets/PPI_Jules'
    path_memmap = '/crnldata/cmo/Projets/PPI_Jules/memmap'
    n_core = 15

#### non interactif node from cluster
elif PC_ID == 'node13':

    PC_working = 'node13'
    path_main_workdir = '/mnt/data/julesgranget/PIPBox/Scripts'
    path_general = '/mnt/data/julesgranget/PIPBox'
    path_memmap = '/mnt/data/julesgranget/PIPBox/memmap'
    n_core = 15

elif PC_ID == 'jules-ubuntu1':

    PC_working = 'jules-ubuntu1'
    path_main_workdir = '/mnt/data/julesgranget/PIPBox/Scripts'
    path_general = '/mnt/data/julesgranget/PIPBox'
    path_memmap = '/mnt/data/julesgranget/PIPBox/memmap'

else:

    PC_working = 'node13'
    path_main_workdir = '/home/jules/Documents/PPI_Jules/Scripts'
    path_general = '/home/jules/Documents/PPI_Jules'
    path_memmap = '/home/jules/Documents/PPI_Jules/memmap'
    n_core = 15
    
path_mntdata = '/mnt/data/julesgranget/PIPBox'
path_data = os.path.join(path_general, 'Data')
path_prep = os.path.join(path_general, 'Analyses', 'preprocessing')
path_precompute = os.path.join(path_general, 'Analyses', 'precompute') 
path_results = os.path.join(path_general, 'Analyses', 'results') 
path_slurm = os.path.join(path_general, 'Scripts_slurm')

os.chdir(init_workdir)

#### slurm params
mem_crnl_cluster = '10G'
n_core_slurms = 10







################################
######## RESPI PARAMS ########
################################ 


sujet_respi_adjust = {
'01NM_MW':'normal',   '02NM_OL':'normal',   '03NM_MC':'normal',   '04NM_LS':'normal',
'05NM_JS':'normal',   '06NM_HC':'normal',   '07NM_YB':'normal',   '08NM_CM':'normal',
'09NM_CV':'normal',   '10NM_VA':'normal',   '11NM_LC':'normal',   '12NM_PS':'normal',  
'13NM_JP':'normal',   '14NM_LD':'normal',   '15PH_JS':'inverse',   '16PH_LP':'normal',   
'17PH_SB':'normal',   '18PH_TH':'inverse',   '19PH_VA':'inverse',   
'20PH_VS':'inverse',   '21IL_NM':'inverse', '22IL_DG':'inverse',   '23IL_DM':'inverse',   
'24IL_DJ':'inverse',   '25IL_DC':'inverse',   '26IL_AP':'inverse',   '27IL_SL':'inverse',   
'28IL_LL':'inverse',   '29IL_VR':'inverse',   '30IL_LC':'inverse',   '31IL_MA':'inverse',   
'32IL_LY':'inverse',   '33IL_BA':'inverse',   '34IL_CM':'inverse',   '35IL_EA':'inverse',   
'36IL_LT':'inverse',   '37DL_05':'inverse',   '38DL_06':'inverse',   '39DL_07':'inverse',   
'40DL_08':'inverse',   '41DL_11':'inverse',   '42DL_12':'inverse',   '43DL_13':'inverse',   
'44DL_14':'inverse',   '45DL_15':'inverse',   '46DL_16':'inverse',   '47DL_17':'inverse',   
'48DL_18':'inverse',   '49DL_19':'normal',   '50DL_20':'inverse',   '51DL_21':'inverse',   
'52DL_22':'inverse',   '53DL_23':'inverse',   '54DL_24':'inverse',   '55DL_25':'inverse',   
'56DL_26':'inverse',   '57DL_27':'inverse',   '58DL_28':'inverse',   '59DL_29':'inverse',   
'60DL_30':'inverse',   '61DL_31':'inverse',   '62DL_32':'inverse',   '63DL_34':'inverse',
}



cycle_detection_params = {
'exclusion_metrics' : 'med',
'sum_coeff_exclusion' : 3,
'time_coeff_exclusion' : 2,
'respi_scale' : [0.1, 0.35], #Hz
}





################################
######## PREP PARAMS ########
################################ 

section_time_general = 300 #sec

section_timming_PHYSIOLOGY = {
'15PH_JS': {'VS' : [0, 613], 'CHARGE' : [644, 1280]},   '16PH_LP': {'VS' : [0, 610], 'CHARGE' : [700, 1340]}, 
'17PH_SB': {'VS' : [0, 610], 'CHARGE' : [680, 1311]},   '18PH_TH': {'VS' : [0, 620], 'CHARGE' : [650, 1280]},   '19PH_VA': {'VS' : [0, 610], 'CHARGE' : [630, 1250]},   
'20PH_VS': {'VS' : [0, 635], 'CHARGE' : [720, 1340]}
}


prep_step_debug = {
'reref' : {'execute': True, 'params' : ['TP9']}, #chan = chan to reref
'mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': True, 'params' : {'l_freq' : 0, 'h_freq': 45}},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': True},
'csd_computation' : {'execute': True},
}

prep_step = {
'reref' : {'execute': False, 'params' : ['TP9', 'TP10']}, #chan = chan to reref
'detrend_mean_centered' : {'execute': True},
'line_noise_removing' : {'execute': True},
'high_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'low_pass' : {'execute': False, 'params' : {'l_freq' : None, 'h_freq': None}},
'csd_computation' : {'execute': False},
'ICA_computation' : {'execute': True},
'average_reref' : {'execute': False},
}






################################
######## ERP PARAMS ########
################################

stretch_point_ERP = 1000
stretch_TF_auto = False
ratio_stretch_TF = 0.5

PPI_time_vec = [-3, 1] #seconds
ERP_time_vec = [-3, 1]
mean_respi_ERP_time_vec = [-3,3]
PPI_lm_time = [-2.5, 0]

allplot_erp_ylim = (-0.3, 0.3)

ERP_n_surrogate = 1000



################################
######## WAVELETS ########
################################

nfrex = 50
nfrex_fc = 50



################################
######## STATS PERM ########
################################


tf_stats_percentile_cluster_manual_perm = 80
erp_time_cluster_thresh = 50 #ms

stat_design='within'
mode_grouped='median'
mode_generate_surr_2g='percentile'
percentile_thresh=[0.5, 99.5]

mode_generate_surr_1d='percentile_time'
mode_select_thresh_1d='percentile_time'
size_thresh_alpha=0.05



################################
######## TF & ITPC ########
################################

nfrex = 150
ncycle_list = [7, 41]
freq_list = [2, 150]
srate_dw = 10
wavetime = np.arange(-3,3,1/srate)
frex = np.logspace(np.log10(freq_list[0]), np.log10(freq_list[1]), nfrex) 
cycles = np.logspace(np.log10(ncycle_list[0]), np.log10(ncycle_list[1]), nfrex).astype('int')

nrespcycle_TF = 100
ratio_stretch_TF = 0.5
n_surrogates_tf = 1000
tf_stats_percentile_cluster = 95
tf_stats_percentile_cluster_size_thresh = 75



########################
######## FC ########
########################

nrespcycle_FC = 100
stretch_point_FC = 240
ncycle_FC = 10
fc_win_overlap = 0.5
MI_window_size = int(3*srate)
freq_band_fc_list = ['theta', 'alpha', 'gamma']
freq_band_fc = {'theta' : [4,8], 'alpha' : [8,12], 'gamma' : [80,150]}
n_surr_fc = 1000
ISPC_ncycles = {'theta' : 10, 'alpha' : 10, 'gamma' : 30}
ISPC_window_size = {'theta' : int(ISPC_ncycles['theta']/freq_band_fc['theta'][0]*srate), 'alpha' : int(ISPC_ncycles['alpha']/freq_band_fc['alpha'][0]*srate), 'gamma' : int(ISPC_ncycles['gamma']/freq_band_fc['gamma'][0]*srate)}







