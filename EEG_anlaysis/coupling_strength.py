#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:22:48 2025

@author: asan
"""


import glob
import os
import time
import pandas as pd
import mne
from scipy.signal import butter, filtfilt 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import numpy as np
import scipy.signal as signal
import matplotlib.pylab as plt
from scipy.signal import butter, filtfilt
import numpy as np
from fastdtw import fastdtw
import random
from scipy.stats import ranksums

# =============================================================================
# this parts read the names of all csv, edf, txt files in the folder and 
# check if the number of txt and edf files is equal to make sure every edf 
# files (recording) were staged. 
# =============================================================================

# path = '/Volumes/stanley_trx_cv/Ahmet/SCHEMA/Cacna1g_3months_final/disk_fixed/cacna1g_3month_light_SOs_fixed_15/results/spindles'

path = '/Users/asan/Desktop/SCHEMA/schema_wtih_WT_thre/3month/cacna1g/3_month_cacna1g_WT/results/spindles'
cp_files = glob.glob(path+'/*.spindles')
cp_files = sorted(cp_files) 

so_files = glob.glob(path+'/*so.so')
so_files = sorted(so_files) 


path_stages = '/Users/asan/Desktop/SCHEMA/schema_wtih_WT_thre/3month/cacna1g/3_month_cacna1g_WT/results/stages'
stg_files = glob.glob(path_stages+'/*.txt')
stg_files = sorted(stg_files) 
  
IDs = [cp_files[i].split('/')[-1].split('_')[0] for i in range(len(cp_files))]
# strength_final = pd.DataFrame(columns=['IDs', 'geno', 'phase', 'CH', 'strength', 'ratio_ss', 'ratio_so', 'count',
       # 'density'])
strength_final = pd.DataFrame([])

chs_all = ['EEG1','EEG2']

for chs in chs_all:
    
    strength_all = pd.DataFrame(columns=[])

    for i in range(len(cp_files)):
        if os.path.getsize(cp_files[i]) == 0:
            continue
        coupl = pd.read_csv(cp_files[i], delimiter = '\t')
        so_no = pd.read_csv(so_files[i], delimiter = '\t')
        number_of_so = len(so_no[so_no['CH'] == chs]) 
        
        df_stages = pd.read_csv(stg_files[i])
        nrem_length = len(df_stages[df_stages['Stage']==2])*10/60
        
        number_of_spindle = (len(coupl[(coupl['CH'] == chs) & (coupl['F'] == 11)]))    
        coupl = coupl[['ID','CH','F','SO_PHASE_PEAK']].dropna()
        total_coupling_no = (len(coupl[(coupl['CH'] == chs) & (coupl['F'] == 11)])) 
        angles = coupl[(coupl['CH'] == chs) & (coupl['F'] == 11)]['SO_PHASE_PEAK'].reset_index(drop=True)
        vector_strength = np.abs(np.mean(np.exp(1j * np.radians(angles))))
       
        coupled_density = (total_coupling_no/nrem_length)
    
        if 'ight' in coupl['ID'].iloc[0]:
            phase_name = 'LightCycle'
        else:
            phase_name = 'DarkCycle'
            
        st_temp = {'IDs' : coupl['ID'].iloc[0].split('_')[0],
                   'phase': phase_name,
                   'CH': chs,
                   'strength' : vector_strength,
                   'ratio_ss': (total_coupling_no/number_of_spindle),
                   'ratio_so': (total_coupling_no/number_of_so),
                   'coupled_no': total_coupling_no,
                   'uncoupled_spindles_no': (number_of_spindle - total_coupling_no),
                   'uncoupled_sos_no': (number_of_so - total_coupling_no),
                   'coupled_uncoupled_ratio_ss': (total_coupling_no / (number_of_spindle - total_coupling_no)),
                   'coupled_uncoupled_ratio_so': (total_coupling_no / (number_of_so - total_coupling_no)),
                   'density': coupled_density}
                
        strength_all = pd.concat([strength_all, pd.DataFrame(st_temp, index = [i])])
    
        
    id_info = pd.read_csv('/Volumes/stanley_trx_cv/Ahmet/SCHEMA/Cacna1g_3months_final/disk_fixed/cacna1g_3month_light_SOs_fixed_15/grouping.csv')
    id_info.rename(columns={'subject_id': 'IDs'}, inplace = True)
    id_info['IDs'] = id_info['IDs'].astype(str)
    strength_all = pd.merge(id_info,strength_all, how='inner')
    
    strength_final = pd.concat([strength_final,strength_all]).reset_index(drop=True)





plt.figure(figsize=[15,6])
# =============================================================================
# # strength comparison
# =============================================================================
# doses = strength_final['dose'].unique()
genos = ['WT','Het','KO']
select_phase = 'DarkCycle'
select_ch = ['EEG1','EEG2']
# select_ch = ['EEG1']
clrs = ['black','green','orange']

features = ['strength', 'ratio_ss', 'ratio_so','coupled_no', 'uncoupled_spindles_no', 'uncoupled_sos_no', 'coupled_uncoupled_ratio_ss', 'coupled_uncoupled_ratio_so', 'density']
# features = ['density']

import scipy.stats as stats

for chnl in range(len(select_ch)):
    for ff in range(len(features)):
        
        if chnl == 1:
            ff_sp = ff+9
        else:
            ff_sp =ff
            
        plt.subplot(2,9,ff_sp+1)
        for i in range(len(genos)):
            wt_st = strength_final[(strength_final['geno'] == genos[i]) & (strength_final['phase'] == select_phase) & (strength_final['CH'] == select_ch[chnl])][features[ff]]
            jitter = np.random.random(len(wt_st))*0.2
            plt.scatter(np.ones(len(wt_st))*i+jitter, wt_st, color = clrs[i], s=9)
        
        wt_ct = strength_final[(strength_final['geno'] == 'WT') & (strength_final['phase'] == select_phase) & (strength_final['CH'] == select_ch[chnl])][features[ff]] 
        ht_ct = strength_final[(strength_final['geno'] == 'Het') & (strength_final['phase'] == select_phase) & (strength_final['CH'] == select_ch[chnl])][features[ff]] 
        ko_ct = strength_final[(strength_final['geno'] == 'KO') & (strength_final['phase'] == select_phase) & (strength_final['CH'] == select_ch[chnl])][features[ff]]
        
        from scipy.stats import f_oneway
        p_value = f_oneway(wt_ct,ht_ct,ko_ct)[1]
        
        print(select_ch[chnl] + '_' + features[ff] + '_' + str(p_value))
            
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_position(('outward', 5))
        plt.gca().spines['bottom'].set_position(('outward', 5))
        
        plt.xlim(-.5,2.5)
        if p_value<0.05:
            plt.title(f"{features[ff]}\n{p_value:.3e}", fontsize=7, color='red')
            # plt.title(features[ff], fontsize = 7, color='red')
        else:
            plt.title(features[ff], fontsize = 7, color='black')
        plt.xticks(range(len(genos)),labels = genos, rotation=60)

plt.subplot(2,9,1)
plt.ylabel('Parietal')

plt.subplot(2,9,10)
plt.ylabel('Frontal')

plt.tight_layout()



from statsmodels.stats.multicomp import pairwise_tukeyhsd

combined_data = pd.concat([wt_ct, ht_ct, ko_ct], axis=0)
labels = ['WT'] * len(wt_ct) + ['Het'] * len(ht_ct) + ['KO'] * len(ko_ct)

tukey_result = pairwise_tukeyhsd(combined_data, labels)
print(tukey_result.summary())





