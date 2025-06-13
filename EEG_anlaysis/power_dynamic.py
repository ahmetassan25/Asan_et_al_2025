

# =============================================================================
# # dynamic power 
# =============================================================================


WT_light = np.array(WT_power_amplitude_light)
Het_light = np.array(Het_power_amplitude_light)
KO_light = np.array(KO_power_amplitude_light)

WT_dark = np.array(WT_power_amplitude_dark)
Het_dark = np.array(Het_power_amplitude_dark)
KO_dark = np.array(KO_power_amplitude_dark)

WT_light = np.concatenate((WT_light, WT_dark), axis=2) 
Het_light = np.concatenate((Het_light, Het_dark), axis=2)
KO_light = np.concatenate((KO_light, KO_dark), axis=2)


def drop_inf_rows(arr):
    # Check if each "row" along axis 0 has only finite values across axes 1 and 2
    finite_mask = np.all(np.isfinite(arr), axis=(1, 2))
    # Apply mask to keep rows without any inf values in any of their elements
    return arr[finite_mask]

# Apply the function to each array
WT_light = drop_inf_rows(WT_light)
Het_light = drop_inf_rows(Het_light)
KO_light = drop_inf_rows(KO_light)

from scipy.stats import f_oneway
import time
wt_all = []
het_all = []
ko_all = []
cc = 0
# plt.subplots(7,1,figsize=[18,18])
plt.figure(figsize=[5,4])
dur_len =24
tticks = []

for cy in range(0,24,1): #range(dur_len): 

    start = int(360*cy)
    stop = int(360*(cy+1))
    time_end = np.arange(start*10/60/60,(stop*10/60/60)+1) #this converts duration to the hours
    
    
    WT_mean_all = []
    Het_mean_all = []
    KO_mean_all = []
    # duration = 720 # just check the first 1 hour (360 episodes) 6 is 1 min
            
    for i in range(7):
        
        WT_mean = []
        for ii in range(WT_light.shape[0]):
            # power_mean = np.nanmean(WT_light[ii,eeg_ch,start:stop,i],0) # just check the first 1 hour (360 episodes)
            power_mean = np.mean(WT_light[ii,eeg_ch,start:stop,i],0) # just check the first 1 hour (360 episodes)
            WT_mean.append(power_mean)
        WT_mean_all.append(WT_mean)
        
        Het_mean = []
        for ii in range(Het_light.shape[0]):
            # power_mean = np.nanmean(Het_light[ii,eeg_ch,start:stop,i],0) # just check the first 1 hour (360 episodes)
            power_mean = np.mean(Het_light[ii,eeg_ch,start:stop,i],0) # just check the first 1 hour (360 episodes)
            Het_mean.append(power_mean)
        Het_mean_all.append(Het_mean)
            
        KO_mean = []    
        for ii in range(KO_light.shape[0]):
            # power_mean = np.nanmean(KO_light[ii,eeg_ch,start:stop,i],0) # just check the first 1 hour (360 episodes)
            power_mean = np.mean(KO_light[ii,eeg_ch,start:stop,i],0) # just check the first 1 hour (360 episodes)
            KO_mean.append(power_mean)
        KO_mean_all.append(KO_mean)

    
    from scipy import stats   
    # plt.subplots(1,7,figsize=[18,3])
    # cc = 0
    jitter=0.1
    # lcs = [0,1,2]
    for i in [2]: #range(7): 
        # plt.subplot(7,1,i+1)
        jittered_WT = np.ones(len(WT_mean_all[0]))*cc+ np.random.uniform(-jitter, jitter, size=len(WT_mean_all[0]))
        plt.scatter(jittered_WT,WT_mean_all[i],color = clrs[0],s=4)
        jittered_Het = np.ones(len(Het_mean_all[0]))*(cc+.5)+ np.random.uniform(-jitter, jitter, size=len(Het_mean_all[0]))
        plt.scatter(jittered_Het,Het_mean_all[i],color = clrs[1],s=4)
        jittered_KO = np.ones(len(KO_mean_all[0]))*(cc+1)+ np.random.uniform(-jitter, jitter, size=len(KO_mean_all[0]))
        plt.scatter(jittered_KO,KO_mean_all[i],color = clrs[2],s=4)
        
        lcs = [cc,cc+.5,cc+1]

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_position(('outward', 5))    
        plt.gca().spines['bottom'].set_position(('outward', 5))    
        plt.gca().set_facecolor('white')
        
        l_d_all = np.concatenate([WT_mean_all[i],Het_mean_all[i],KO_mean_all[i]])
        rg_sp = max(l_d_all)-min(l_d_all);
    
    wt_all.append(WT_mean_all)
    het_all.append(Het_mean_all)
    ko_all.append(KO_mean_all)


    cc = cc+3
    # plt.ylabel(power_label[i])
    tticks.append(lcs[1])
    if cy == dur_len-1:
        ticks_str = [str(tick+1) for tick in range(len(tticks))]
        plt.xticks(tticks[0::2],labels=ticks_str[0::2],fontsize=12)
        plt.yticks(fontsize=12)


for kl in [2]: #range(7): #[4]:
    # plt.subplot(7,1,kl+1)

    for il in range(17):
            
        test = np.array(wt_all)[:,kl,:]
        # test1 = np.mean(test[:,1,:],1)
        plt.plot(np.array(tticks)-0.5,test[:,il],color='black',alpha = 0.2)
    plt.plot(np.array(tticks),np.nanmean(test,1),color='black',alpha = 1,linewidth=4)
    for il in range(9):
        test = np.array(ko_all)[:,kl,:]
        # test1 = np.mean(test[:,1,:],1)
        plt.plot(np.array(tticks)+.5,test[:,il],color='orange',alpha = 0.2)
    plt.plot(np.array(tticks),np.nanmean(test,1),color='orange',alpha = 1,linewidth=4)
  
    for il in range(14):
        test = np.array(het_all)[:,kl,:]
        # test1 = np.mean(test[:,1,:],1)
        plt.plot(np.array(tticks),test[:,il],color='green',alpha = 0.2)
    plt.plot(np.array(tticks),np.nanmean(test,1),color='green',alpha = 1,linewidth=4)



test = np.array(het_all)
test1 = np.mean(test[:,1,:],1)
plt.plot(tticks,test1,color='green')

test = np.array(ko_all)
test1 = np.mean(test[:,1,:],1)
plt.plot(tticks,test1,color='orange')




for i in range(7):
    plt.subplot(7,1,i+1)
    plt.ylabel(power_label[i])
    if i<6:
        plt.xticks([])
plt.xlabel('Time (h)')    
plt.tight_layout()



wt_test_final = np.array([])
wt_test = np.array(wt_all)
band_all = []
cycle_all = []
time_all = []
IDs = []
band_names = ['SOs','Delta','Theta','Alpha','Sigma','Beta','Gamma']
for i in range(7):
    for ii in range(17):
        wt_test_final = np.concatenate((wt_test_final, wt_test[:,i,ii]))
        band_all.extend([band_names[i] for _ in range(24)])
        cycle_all.extend(['lightcycle' for _ in range(12)])
        cycle_all.extend(['darkcycle' for _ in range(12)])
        IDs.extend([str(ii) for _ in range(24)])
        time_all.extend([ttm for ttm in range(24)])

df = pd.DataFrame({'IDs': IDs,
                   'PSD': wt_test_final,
                   'Band': band_all,
                   'Cycle': cycle_all,
                   'Time': time_all})



###cosinor finction


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import f
import math

# # Define the cosine function
# def cosine_function(x, amplitude, frequency, phase, offset):
#     return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

def cosinor_function(x, mesor, amplitude, acrophase, period=24):
    # return mesor + amplitude * np.cos(2 * np.pi * x / period - (np.pi/2) - acrophase)
    # return mesor + abs(amplitude) * np.cos(2 * np.pi * x / period - (np.pi/2) - acrophase)
    # return mesor + amplitude * np.cos(2 * np.pi * x / period - (np.pi/2) - acrophase)
    return mesor + amplitude * np.cos(2 * np.pi * x / period - acrophase)

test_r2 = []
all_groups = [wt_all,het_all,ko_all]
max_final = []
phase_final = []
# select_bands = [0,1,2,3,4,5,6]
select_bands = [2]
jj = [-.1,0,.1,-.1,0,.1]
clrs = ['black','green','orange']
bands_to_look = 1
plt.subplots(bands_to_look,1,figsize=[3.5,4])

amp_final = []
freq_final = []
phase_final = []
offset_final = []
peak_time_final = []
r2_all = []
p_value_r2_all = []
mesor_final =[]
acrophase_final = []
for grp in range(3):

    lj3 = np.array(all_groups[grp])
    
    amp_band = []
    freq_band = []
    phase_band = []
    offset_band = []
    peak_time_band = []
    acrophase_band = []
    mesor_band = []
    
    for i in range(len(select_bands)):
        er = lj3[:,select_bands[i],:]
        plt.subplot(bands_to_look,1,i+1)
        # plt.plot(np.arange(0,24)+jj[grp], np.nanmean(er,1),color=clrs[grp], alpha = 1, linewidth = 2)
        max_all = []
        phase_ = []
        amp_each = []
        freq_each = []
        phase_each = []
        offset_each = []
        peak_time_each = []        
        mesor_ = []
        acrophase_each = []
        
        # er[np.isinf(er)] = np.nan
        y_data = np.nanmean(er,1)
        x_data = np.arange(len(y_data))   

        #####cosiner function
        initial_guess = [np.mean(y_data), np.std(y_data), 0]  # Period is assumed to be 24, so no need to guess it
        params, _ = curve_fit(cosinor_function, x_data, y_data, p0=initial_guess, maxfev=5000)
        mesor_fit, amplitude_fit, acrophase_fit = params
        y_fitted_mean = cosinor_function(x_data, mesor_fit, amplitude_fit, acrophase_fit)
        plt.plot(x_data, y_fitted_mean, label='Fitted Cosine', color=clrs[grp],linewidth=3)
        r2 = r2_score(y_data, y_fitted_mean)
        print(r2)


        ###cosine function for each gorup (use the mean)
        # # initial_guess = [np.std(y_data), 1/len(x_data), 0, np.mean(y_data)]
        # initial_guess = [np.std(y_data), 1/24, -2*np.pi, np.mean(y_data)]
        # # bounds = ([0, 1/30, -2*np.pi, min(y_data)], [2*np.std(y_data), 1/18, 2*np.pi, max(y_data)])
        # # bounds = ([0, 1/26, -2*np.pi, min(y_data)], [2*np.std(y_data), 1/22, 2*np.pi, max(y_data)])
        # # params, _ = curve_fit(cosine_function, x_data, y_data, p0=initial_guess, bounds=bounds, maxfev=5000)
        # params, _ = curve_fit(cosine_function, x_data, y_data, p0=initial_guess, maxfev=5000)
        # amplitude_fit, frequency_fit, phase_fit, offset_fit = params
        # y_fitted_mean = cosine_function(x_data, amplitude_fit, frequency_fit, phase_fit, offset_fit)
        # plt.plot(x_data, y_fitted_mean, label='Fitted Cosine', color=clrs[grp],linewidth=3)
        # r2 = r2_score(y_data, y_fitted_mean)
        # print(r2)
        y_fitted_all = []
        for kl in range(er.shape[1]):
            # plt.figure()
            # plt.plot(np.arange(0,24)+jj[grp],er[:,kl],color=clrs[grp],alpha = .2)
            y_data = er[:,kl]
            y_data[np.isnan(y_data)]=np.nanmean(y_data) # replace Nan with the mean
            x_data = np.arange(len(y_data)) 
            
            ####cosinor function for each animal
            initial_guess = [np.mean(y_data), np.std(y_data), 0]  # Period is assumed to be 24, so no need to guess it
            params, _ = curve_fit(cosinor_function, x_data, y_data, p0=initial_guess, maxfev=5000)
            mesor_fit, amplitude_fit, acrophase_fit = params
            y_fitted = cosinor_function(x_data, mesor_fit, amplitude_fit, acrophase_fit)
            # y_fitted_all = np.concatenate((y_fitted_all, y_fitted))
            y_fitted_all.append(y_fitted)
            r2 = r2_score(y_data, y_fitted)
            test_r2.append(r2)
            print(r2)
            # plt.plot(x_data, y_fitted, label='Fitted Cosine', color=clrs[grp],linewidth=1)
            # print(acrophase_fit)
            # print(np.degrees(acrophase_fit))
            
            mesor_.append(mesor_fit)
            
            if amplitude_fit < 0:
                acrophase_fit += np.pi 
                
            amp_each.append(abs(amplitude_fit)*2)
            # amp_each.append(amplitude_fit)

            acrophase_each.append(np.degrees(acrophase_fit))
                        
            
            # normalized_angle = (math.degrees(acrophase_fit) + 180) % 360
            # hours = (normalized_angle / 360) * 24
            # acrophase_each.append(acrophase_fit*24/6.28)
            # acrophase_each.append(np.degrees(acrophase_fit))

            
            
            # ####cosine function
            # # initial_guess = [np.std(y_data), 1/len(x_data), 0, np.mean(y_data)]
            # initial_guess = [np.std(y_data), 1/24, -2*np.pi, np.mean(y_data)]
            # bounds = ([0, 1/30, -2*np.pi, min(y_data)], [2*np.std(y_data), 1/18, 2*np.pi, max(y_data)])
            # # bounds = ([0, 1/26, -2*np.pi, min(y_data)], [2*np.std(y_data), 1/22, 2*np.pi, max(y_data)])
            # params, _ = curve_fit(cosine_function, x_data, y_data, p0=initial_guess, bounds=bounds, maxfev=5000)
            # # params, _ = curve_fit(cosine_function, x_data, y_data, p0=initial_guess, maxfev=5000)
            # amplitude_fit, frequency_fit, phase_fit, offset_fit = params
            # y_fitted = cosine_function(x_data, amplitude_fit, frequency_fit, phase_fit, offset_fit)
            # phase_.append(phase_fit)
            # # plt.plot(x_data, y_fitted, label='Fitted Cosine', color=clrs[grp],linewidth=1)
            # amp_each.append(amplitude_fit)
            # freq_each.append(frequency_fit)
            # phase_each.append(phase_fit)
            # offset_each.append(offset_fit)
            
            r2 = r2_score(y_data, y_fitted)
            r2_all.append(r2)
            n = len(y_data)
            k = 1
            f_statistic = (r2 / k) / ((1 - r2) / (n - k - 1))
            p_value_r2 = 1 - f.cdf(f_statistic, k, n - k - 1)
            p_value_r2_all.append(p_value_r2)            
            # print(f'R-squared: {r2:.3f}')

            
            # # Calculate and store peak time
            # peak_time = -phase_fit / (2 * np.pi * frequency_fit)
            # peak_time_each.append(peak_time)
        
        # plt.plot(x_data, y_fitted_mean, label='Fitted Cosine', color=clrs[grp],linewidth=2)

        ###cosinor
        mesor_band.append(mesor_)
        amp_band.append(amp_each)
        acrophase_band.append(acrophase_each)

        # ###cosine
        # amp_band.append(amp_each)
        # freq_band.append(freq_each)
        # phase_band.append(phase_each)
        # offset_band.append(offset_each)
        # peak_time_band.append(peak_time_each)

    ###cosinor
    mesor_band1 = np.array(mesor_band[0])
    Q1 = np.percentile(mesor_band1, 25)
    Q3 = np.percentile(mesor_band1, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    mesor_band1 = [[i for i in mesor_band1 if lower_bound<i<upper_bound]]  
    mesor_final.append(mesor_band1)
    
    amp_band1 = np.array(amp_band[0])
    Q1 = np.percentile(amp_band1, 25)
    Q3 = np.percentile(amp_band1, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    amp_band1 = [[i for i in amp_band1 if lower_bound<i<upper_bound]]  
    amp_final.append(amp_band1)
    
    acrophase_band1 = np.array(acrophase_band[0])
    Q1 = np.percentile(acrophase_band1, 25)
    Q3 = np.percentile(acrophase_band1, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    acrophase_band1 = [[i for i in acrophase_band1 if lower_bound<i<upper_bound]]
    acrophase_final.append(acrophase_band1)

    # ###cosine    
    # amp_final.append(amp_band)
    # freq_final.append(freq_band)
    # phase_final.append(phase_band)
    # offset_final.append(offset_band)
    # peak_time_final.append(peak_time_band) 
    
    spect_mean = np.mean(np.array(y_fitted_all),0);
    spect_ste = np.std(np.array(y_fitted_all),0)/np.sqrt(len(y_fitted_all))
    plt.fill_between(np.arange(0,24), spect_mean-spect_ste,spect_mean+spect_ste,color=clrs[grp],edgecolor='none', alpha=0.4, label='Standard Error')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_position(('outward', 5))
    plt.gca().spines['bottom'].set_position(('outward', 5))
    plt.gca().set_facecolor('white')
    plt.xticks(np.arange(1,25,3),np.arange(1,25,3),fontsize=12)
    plt.yticks(fontsize=12)

plt.tight_layout()

import numpy as np
import matplotlib.pyplot as plt
import pycircstat
from pycircstat import watson_williams
from scipy.stats import f_oneway
from scipy.stats import ttest_ind

# Example data for illustration purposes
# Replace with your actual data
fits = [mesor_final, amp_final, acrophase_final]  # cosinor
ttls = ['mesor', 'amplitude', 'phase']
# select_bands = [4]  # Assuming you have defined this elsewhere
clrs = ['black', 'green', 'orange']  # Define your colors here

plt.subplots(1, 3, figsize=[9, 4])
count = 1
for i in range(len(select_bands)):
    for ii in range(len(fits)):
        plt.subplot(1, 3, count)
        count += 1
        
        if ttls[ii] == 'phase':
            # Circular plot for phases
            colors = ['black', 'green', 'orange']
            labels = ['WT', 'Het', 'KO']

            # Flatten and concatenate phase data
            for j, (phase_data, color, label) in enumerate(zip(fits[ii], colors, labels)):
                # Ensure phase_data is in radians
                phase_data_rad = np.deg2rad(phase_data[i])
                # mean_direction = pycircstat.mean(phase_data_rad)
                mean_direction = np.angle(np.mean(np.exp(1j * phase_data_rad)))

                # Plot each group on a polar plot
                ax = plt.subplot(1, 3, count - 1, polar=True)
                ax.hist(phase_data_rad, bins=9, density=True, color=color, alpha=0.4, label=label, edgecolor='none')
                ax.plot([mean_direction, mean_direction], [0, 4], color=color, linestyle='--', linewidth=2)  # Circular mean line
                ax.set_yticklabels([])
                ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]), ['0', '45', '90', '135', '180', '225', '270', '315'],fontsize=12)
                # ax.set_xticks([])
                # ax.set_title(ttls[ii])
                ax.set_facecolor('white')  # Corrected line
                ax.grid(axis='x', color='black', alpha=.2)
                ax.grid(axis='y', color='black', alpha=0)
                ax.spines['polar'].set_visible(False)

            # plt.legend(loc='upper right')

        else:
            # Scatter plot for mesor and amplitude
            jitter = np.random.rand(len(fits[ii][0][i]))/5
            plt.scatter(np.ones(len(fits[ii][0][i])) + jitter, fits[ii][0][i], color='black')
            jitter = np.random.rand(len(fits[ii][1][i]))/5
            plt.scatter(np.ones(len(fits[ii][1][i])) * 2 + +jitter, fits[ii][1][i], color='green')
            jitter = np.random.rand(len(fits[ii][2][i]))/5
            plt.scatter(np.ones(len(fits[ii][2][i])) * 3 + +jitter, fits[ii][2][i], color='orange')
            
            
            plt.xticks([1, 2, 3], ['WT', 'Het', 'KO'],fontsize=12)
            plt.yticks(fontsize=12)
            # plt.title(ttls[ii])
        
        if ttls[ii] == 'phase':
            pass
        else:
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_position(('outward', 5))
            plt.gca().spines['bottom'].set_position(('outward', 5))
            plt.gca().set_facecolor('white')

        # Perform statistical tests for phase (circular) and amplitude/mesor (linear)
        if ttls[ii] == 'phase':
            # Convert phase data to radians
            rad_data = [np.deg2rad(fits[ii][0][i]), np.deg2rad(fits[ii][1][i]), np.deg2rad(fits[ii][2][i])]
            # Perform Watson-Williams test
            p_value1 = pycircstat.tests.watson_williams(rad_data[0], rad_data[1])
            p_value2 = pycircstat.tests.watson_williams(rad_data[0], rad_data[2])

            print(f"Watson-Williams p-value for phase comparison: {p_value}")

            if p_value2[0] < 0.05:
                all_fits = []
                all_fits.extend(rad_data[0])
                all_fits.extend(rad_data[2])
                mean_angle = np.angle(np.mean(np.exp(1j * rad_data[2])))
                all_range = max(all_fits) - min(all_fits)
                # ax.text(mean_angle, 1, f"{p_value2[0]:.2e}", fontsize=10, color=clrs[2], ha='center')
                ax.text(mean_angle, 1.2, '*', fontsize=15, color=clrs[2], ha='center')


            if p_value1[0] < 0.05:
                all_fits = []
                all_fits.extend(rad_data[0])
                all_fits.extend(rad_data[1])
                mean_angle = np.angle(np.mean(np.exp(1j * rad_data[1])))
                all_range = max(all_fits) - min(all_fits)
                # ax.text(mean_angle, 1, f"{p_value1[0]:.2e}", fontsize=10, color=clrs[1], ha='center')
                ax.text(mean_angle, 1.2, '*', fontsize=15, color=clrs[1], ha='center')

        else:
            # Use regular tests for amplitude and mesor (linear data)
            
            
            
            # group_1 = fits[ii][0][i]
            # group_2 = fits[ii][1][i]
            # group_3 = fits[ii][2][i]
            
            # data = pd.DataFrame({
            #     'value': np.concatenate([group_1, group_2, group_3]),
            #     'group': (['Group 1'] * len(group_1) + 
            #               ['Group 2'] * len(group_2) + 
            #               ['Group 3'] * len(group_3)),
            #     'subject': np.arange(len(group_1) + len(group_2) + len(group_3))  # Replace with actual subjects if available
            # })
            
            # # Fit the linear mixed-effects model
            # full_model = mixedlm("value ~ group", data, groups=data["subject"])
            # result_full = full_model.fit(reml=False)  # Set REML to False for ML estimation
            # print(result_full.summary())
            # null_model = mixedlm("value ~ 1", data, groups=data["subject"])
            # result_null = null_model.fit(reml=False)  # Set REML to False for ML estimation
                        
            # # Log-likelihood ratio test
            # lr_test_stat = 2 * (result_full.llf - result_null.llf)  # LLF: Log-Likelihood Function
            # df_diff = result_full.df_model - result_null.df_model   # Degrees of freedom difference
            # p_value = chi2.sf(lr_test_stat, df_diff)
            
            # lr_test_stat = 2 * (result_full.llf - result_null.llf)  # LLF: Log-Likelihood Function
            # df_diff = result_full.df_modelwc - result_null.df_modelwc  # Degrees of Freedom difference
            # p_value = chi2.sf(lr_test_stat, df_diff)  # P-value calculation


            
            # print(f"Likelihood Ratio Test Statistic: {lr_test_stat}")
            # print(f"Degrees of Freedom: {df_diff}")
            # print(f"p-value: {p_value}")
            
            
            
            f_statistic, p_value = f_oneway(fits[ii][0][i], fits[ii][1][i], fits[ii][2][i])
            print(p_value)
            if p_value < 0.5:
                H1_light, p_value1 = ttest_ind(fits[ii][0][i], fits[ii][2][i])
                if p_value1 < 0.05:
                    all_fits = []
                    all_fits.extend(fits[ii][0][i])
                    all_fits.extend(fits[ii][2][i])
                    all_range = max(all_fits) - min(all_fits)
                    plt.plot([1, 3], [max(all_fits) + (all_range * 0.4), max(all_fits) + (all_range * 0.4)], color=clrs[2])
                    plt.plot([3, 3], [max(all_fits) + (all_range * 0.4), max(fits[ii][2][i]) + .0], linestyle='--', linewidth=1, alpha=.5, color=clrs[2])
                    plt.text(2 - .45, max(all_fits) + (all_range * 0.45), f"{p_value1:.2e}", fontsize=12, color=clrs[2])
                
                H2_light, p_value2 = ttest_ind(fits[ii][0][i], fits[ii][1][i])
                if p_value2 < 0.05:
                    all_fits = []
                    all_fits.extend(fits[ii][0][i])
                    all_fits.extend(fits[ii][1][i])
                    all_range = max(all_fits) - min(all_fits)
                    plt.plot([1, 2], [max(all_fits) + (all_range * 0.2), max(all_fits) + (all_range * 0.2)], color=clrs[1])
                    plt.plot([2, 2], [max(all_fits) + (all_range * 0.2), max(fits[ii][2][i]) + .0], linestyle='--', linewidth=1, alpha=.5, color=clrs[1])
                    plt.text(1.5 - .45, max(all_fits) + (all_range * 0.25), f"{p_value2:.2e}", fontsize=12, color=clrs[1])

plt.tight_layout()


