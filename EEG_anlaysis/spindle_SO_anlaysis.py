
import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 10  # Default font size for X axis tick labels
plt.rcParams['ytick.labelsize'] = 10  # Default font size for Y axis tick labels
plt.rcParams['font.size'] = 12  # Default font size for titles, labels, and tick labels
plt.rcParams['axes.labelsize'] = 10  # Default font size for axis labels
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'


import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import ranksums
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


### =============================================================================
### Spindle All
### =============================================================================

all_features = ['spindle_density', 'AMP', 'ISA', 'DUR']
# plt.subplots(4,2,figsize = [5.72,9.5]) # one freq
plt.subplots(4,2,figsize = [12,12]) # all freqs

count = 0
for spindle_feature in range(len(all_features)):
    if spindle_feature==0:
        df_all = pd.read_csv('spindles_mean.csv')
    else:
        df_all = pd.read_csv('spindles_details.csv')
        
### =============================================================================
### SOs All
### =============================================================================
        
# all_features = ['SO_Rate', 'DUR']
# plt.subplots(2,2) # one freq
# count = 0
# for spindle_feature in range(len(all_features)):
#     if spindle_feature==0:
#         df_all = pd.read_csv('so_rate_mean.csv')
#     else:
#         df_all = pd.read_csv('so_details_mean.csv')



    # this is cacna1g
    df = df_all[(df_all['subject_id'] != 106547) & (df_all['subject_id'] != 97561) & (df_all['subject_id'] != 96554) & (df_all['subject_id'] != 93533)]

    # freq = [9,11,13,15];
    freq = [11];

    chs = ['EEG1','EEG2'];
    
    phases = sorted(df['phase'].unique())
    if len(phases)>1:
        cycle = [phases[1],phases[1]] # when n is 1, it only uses light, 0 is for dark
    else:
        cycle = [phases[0],phases[0]] # if the len(phase) is less than 1, it means we are only looking at one cycle (ligth or dark)
    # cycle=['light','light']
    cycle=['LightCycle','LightCycle']
    genos = ['WT','Het','KO']
    
    clrs = ['black','green','orange'];
    # clrs = ['black','green','red'];
    spindle_all = [];
        
    jitter = 0.15  # Adjust the amount of jitter as needed
    err_jit = .5

    for sp in range(2):
        lcs = np.array([1,2.5,4])
        count = count+1
        for i in freq:  # Spindle
        # for i in range(1):  #SOs

            
            #### this part is for SOs       

            # KO_freq = df[(df['phase'] == cycle[sp]) & (df['CH'] == chs[sp]) & (df['geno'] == genos[0])][all_features[spindle_feature]];
            # HET_freq = df[(df['phase'] == cycle[sp]) & (df['CH'] == chs[sp]) & (df['geno'] == genos[1])][all_features[spindle_feature]];
            # WT_freq = df[(df['phase'] == cycle[sp]) & (df['CH'] == chs[sp]) & (df['geno'] == genos[2])][all_features[spindle_feature]];
            
            # ### this part is for spindles       
            
            KO_freq= df[(df['phase'] == cycle[sp]) & (df['CH'] == chs[sp]) & (df['geno'] == genos[0]) & (df['F'] == i)][all_features[spindle_feature]];
            HET_freq = df[(df['phase'] == cycle[sp]) & (df['CH'] == chs[sp]) & (df['geno'] == genos[1]) & (df['F'] == i)][all_features[spindle_feature]];
            WT_freq = df[(df['phase'] == cycle[sp]) & (df['CH'] == chs[sp]) & (df['geno'] == genos[2]) & (df['F'] == i)][all_features[spindle_feature]];
            
            
            ###### outlier rejection using the quartiles
        
            
            ## WT
            Q1 = WT_freq.quantile(0.25)
            Q3 = WT_freq.quantile(0.75)
            IQR = Q3 - Q1
            th_rate = 2
            lower_bound = Q1 - (th_rate * IQR)
            upper_bound = Q3 + (th_rate * IQR)
            WT_freq = WT_freq[(WT_freq>= lower_bound) & (WT_freq<= upper_bound)]
            # WT_freq = WT_freq[(WT_freq['spindle_density']>= lower_bound) & (WT_freq['spindle_density'].values<= upper_bound)]
            
            #### KO
            Q1 = KO_freq.quantile(0.25)
            Q3 = KO_freq.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (th_rate * IQR)
            upper_bound = Q3 + (th_rate * IQR)
            KO_freq = KO_freq[(KO_freq>= lower_bound) & (KO_freq<= upper_bound)]
            # KO_freq = KO_freq[(KO_freq['spindle_density']>= lower_bound) & (KO_freq['spindle_density'].values<= upper_bound)]
            
            #### HET
            Q1 = HET_freq.quantile(0.25)
            Q3 = HET_freq.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (th_rate * IQR)
            upper_bound = Q3 + (th_rate * IQR)
            HET_freq = HET_freq[(HET_freq>= lower_bound) & (HET_freq<= upper_bound)]
            # KO_freq = KO_freq[(KO_freq['spindle_density']>= lower_bound) & (KO_freq['spindle_density'].values<= upper_bound)]
            
            # ########################
            
            
            
            spindle_all.append(max(KO_freq))
            spindle_all.append(min(KO_freq))
            spindle_all.append(max(WT_freq))
            spindle_all.append(min(WT_freq))
            spindle_all.append(max(HET_freq))
            spindle_all.append(min(HET_freq))
            
            fig = plt.subplot(4,2,count)    #this is for spindle
            # fig = plt.subplot(2,2,count)      #this is for SOs
            
            
            for pl,jt in enumerate([WT_freq, HET_freq, KO_freq]):
                jittered_jt = np.ones(len(jt))*lcs[pl]+ np.random.uniform(-jitter, jitter, size=len(jt))
                plt.scatter(jittered_jt, jt, s = 35, color = clrs[pl], alpha = .6, marker = 'o', edgecolors = 'None')
           
            x = lcs;
            y = [np.mean(WT_freq),np.mean(HET_freq),np.mean(KO_freq)];
            yerr= [np.std(WT_freq),np.std(HET_freq),np.std(KO_freq)];
            [plt.errorbar(x[i]+err_jit, y[i], yerr = yerr[i], fmt='s', markersize = 3, capsize=5, color = clrs[i]) for i in range(3)]
                
                    
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_position(('outward', 5))    
            plt.gca().spines['bottom'].set_position(('outward', 5))    
            plt.gca().set_facecolor('white')
            
            
            l_d_all = np.concatenate((WT_freq, HET_freq, KO_freq))
            
            f_statistic, p_value = stats.f_oneway(WT_freq, HET_freq, KO_freq)
            print(p_value)

            if p_value<0.05:
                data = pd.concat([WT_freq, HET_freq, KO_freq], ignore_index=True)
                groups = (['WT'] * len(WT_freq) +
                          ['HET'] * len(HET_freq) +
                          ['KO'] * len(KO_freq))
                
                # Perform Tukey's HSD
                tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
                print(tukey)
                H1_light, p_value1 = ranksums(WT_freq,KO_freq)
                t_stat1, p_value1 = stats.ttest_ind(WT_freq,KO_freq)
                rg_sp = max(l_d_all)-min(l_d_all);
                if p_value1<0.05:
                    plt.plot([lcs[0],lcs[2]],[max(l_d_all)+(rg_sp*0.5),max(l_d_all)+(rg_sp*0.5)],color = clrs[2])
                    plt.plot([lcs[2],lcs[2]],[max(l_d_all)+(rg_sp*0.5),max(KO_freq)+.0],linestyle='--',linewidth = 1,alpha = .5, color = clrs[2])
                    plt.text(lcs[1]-.45, max(l_d_all)+(rg_sp*0.55), f"{p_value1:.2e}", fontsize=10, color=clrs[2])

                # H2_light, p_value2 = ranksums(WT_freq,HET_freq)
                t_stat2, p_value2 = stats.ttest_ind(WT_freq, HET_freq)
                rg_sp = max(l_d_all)-min(l_d_all);
                if p_value2<0.05:
                    plt.plot([lcs[0], lcs[1]],[max(l_d_all)+(rg_sp*0.2),max(l_d_all)+(rg_sp*0.2)],color = clrs[1])
                    plt.plot([lcs[1],lcs[1]],[max(l_d_all)+(rg_sp*0.2),max(HET_freq)+.0],linestyle='--', linewidth = 1,alpha = .5,color = clrs[1])
                    plt.text(np.mean(lcs[0:2])-.5, max(l_d_all)+(rg_sp*0.25), f"{p_value2:.2e}", fontsize=10, color=clrs[1])
            
            rg_sp = max(l_d_all)-min(l_d_all);
            # plt.ylim([min(l_d_all)-(rg_sp*0.2), max(l_d_all)+(rg_sp*0.8)])
            plt.xticks([],color = 'none'); 


            lcs = lcs+6

        
        if sp == 0:
            plt.ylabel(all_features[spindle_feature])
    
#     #### this part is for spindle
    plt.subplot(4,2,count) 
    plt.ylim([min(spindle_all)-(rg_sp*0.2), max(spindle_all)+(rg_sp*0.8)])
    plt.subplot(4,2,count-1) 
    plt.ylim([min(spindle_all)-(rg_sp*0.2), max(spindle_all)+(rg_sp*0.8)])

    #### this part is for spindle      
lcs_all = np.array([2.5,8.5,14.5,20.5])
plt.subplot(4,2,1);
plt.title('Parietal', fontsize=12)
plt.subplot(4,2,2);
plt.title('Frontal', fontsize=12)
plt.subplot(4,2,7);
plt.xticks([2.5], labels=['11Hz'], color='black' ,weight = 'bold') # one frequency
# plt.xticks(ticks=lcs_all, labels=['9Hz','11Hz','13Hz','15Hz'], color='black')  # all frequency
plt.subplot(4,2,8);
plt.xticks([2.5], labels=['11Hz'], color='black',weight = 'bold') # one frequency
# plt.xticks(ticks=lcs_all, labels=['9Hz','11Hz','13Hz','15Hz'], color='black')  # all frequency

plt.tight_layout()
    
#     ### this part is for SOs
# lcs = np.array([1,2.5,4])
           
# plt.subplot(2,2,1);
# plt.title('Parietal', fontsize=12)
# plt.subplot(2,2,2);
# plt.title('Frontal', fontsize=12)
# plt.subplot(2,2,3);
# plt.xticks(lcs,['WT','Het','KO'],fontsize = 12, fontweight = 'bold'); 
# plt.ylabel('SO-DUR')
# plt.subplot(2,2,4);
# plt.xticks(lcs,['WT','Het','KO'],fontsize = 12, fontweight = 'bold'); 
    
# plt.tight_layout()

    
