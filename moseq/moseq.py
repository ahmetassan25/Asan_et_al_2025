#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:19:39 2025

@author: asan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# moseq dataframes
moseq_df=pd.read_csv('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/moseq_df.csv')
stats_df=pd.read_csv('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/stats_df.csv')

from tqdm.auto import trange
def permutation_test(box, rest, sample_size = 99999):
    x = np.hstack([box, rest])
    n_box = len(box)
    perm_mean = np.zeros(sample_size)
    for i in trange(sample_size):
        np.random.seed(i) 
        # compute shuffle box mean - rest mean
        perm_x = np.random.permutation(x)
        perm_mean[i] = perm_x[:n_box].mean() - perm_x[n_box:].mean()
    return (np.sum((box.mean()-rest.mean())<perm_mean) +1)/ (sample_size+1)
from scipy.stats import ttest_ind

def bootstrap(box, rest, sample_size = 1000, sample_iter=100000):
    b_mean_box = np.zeros(sample_iter)
    b_mean_rest= np.zeros(sample_iter)
    for i in trange(sample_iter):
        np.random.seed(i)
        b_mean_box[i]=np.random.choice(box, sample_size).mean()
        b_mean_rest[i]=np.random.choice(rest, sample_size).mean()
    return np.percentile(b_mean_box, [2.5,97.5]), np.percentile(b_mean_rest, [2.5, 97.5])


def relative_usage(genotype_kinematics, ref, sort_by = 'velocity_2d_mm_mean'):
    relative_u = pd.pivot_table(genotype_kinematics, values = 'usage', index='syllable', columns = 'group').reset_index()
    relative_u['K'] = relative_u['K']/relative_u['J']
    relative_u['G'] = relative_u['G']/relative_u['J']
    relative_u['J'] = relative_u['J']/relative_u['J']

    relative_u = pd.merge(relative_u, ref, left_on='syllable', right_on='syllable')
    # only use the top99% ones
    relative_u = relative_u[relative_u.syllable < 49].copy()
    relative_u = relative_u.sort_values(by=[sort_by], ascending=False).reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    sns.lineplot(data = relative_u[['J', 'K', 'G']], ax = ax, dashes=False)
    ax.set_title(f'Syllale in order of {sort_by} high to low', fontsize = 30)
    
def rolling_usage(genotype_kinematics, ref, roll_num, sort_by = 'velocity_2d_mm_mean', filename='velocity_2d_mm_mean'):
    relative_u = pd.pivot_table(genotype_kinematics, values = 'usage', index='syllable', columns = 'group').reset_index()
    relative_u = pd.merge(relative_u, ref, left_on='syllable', right_on='syllable')
    relative_u = relative_u[relative_u.syllable < 49].copy()
    relative_u = relative_u.sort_values(by=[sort_by], ascending=False).reset_index()
    rolling = pd.DataFrame({
                            'J': relative_u['J'].rolling(roll_num).mean(),
                           'K': relative_u['K'].rolling(roll_num).mean(),
                           'G': relative_u['G'].rolling(roll_num).mean()}).dropna()
    rolling['K'] =rolling['K']/rolling['J']
    rolling['G'] =rolling['G']/rolling['J']
    rolling['J'] =rolling['J']/rolling['J']

    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    sns.lineplot(data = rolling, ax = ax, dashes=False, palette=['b','r','g'])
    ax.set_title(f'Syllale in order of {sort_by} high to low', fontsize = 30)
    # ax.set_xticks([])
    fig.savefig(f'{filename}.pdf')
    return relative_u, rolling    

def plot_boxvsrest(stats_df, box, boxname, pair, metric, filename):

    stats_subset = stats_df.loc[(stats_df.group ==pair[0]) | (stats_df.group==pair[1])].copy()
    stats_subset[metric] *= 30
    print(stats_subset.group.unique())
    stats_subset = stats_subset.groupby(['syllable']).mean().reset_index()
    stats_subset['syllable_group'] = 'rest'
    stats_subset.loc[stats_subset['syllable'].isin(box), 'syllable_group'] = 'box'
    print(box)
    box_data = stats_subset[stats_subset['syllable_group'] == 'box'][metric]
    rest_data = stats_subset[stats_subset['syllable_group'] == 'rest'][metric]

    fig, axs = plt.subplots(1,1, figsize=(5,5))
    sns.set_theme(style="whitegrid")
    sns.boxplot(ax= axs, x = 'syllable_group', y = metric, data=stats_subset)
    sns.stripplot(ax= axs, x = 'syllable_group', y = metric, data=stats_subset, color='black')
    axs.set_xlabel('Syllable group')
    axs.set_ylabel(metric)
    axs.set_title(f'{boxname} syllables VS rest')
    plt.show()
    fig.savefig(f'{filename}-{boxname}.pdf')
    print('file name', f'{filename}-{boxname}.pdf')
    stats_subset[['syllable_group','syllable', metric]].to_csv(f'{filename}-{boxname}.csv',index=False)
    return box_data, rest_data    
    
import seaborn as sns
import matplotlib.pyplot as plt

genotype_kinematics = (
    stats_df[['group', 'uuid', 'syllable', 'usage', 'velocity_2d_mm_mean', 'dist_to_center_px_mean']]
    .groupby(['group', 'syllable'])
    .mean(numeric_only=True)  # Fix here
    .reset_index()
)

syll_v = (
    genotype_kinematics.groupby('syllable')
    .mean(numeric_only=True)['velocity_2d_mm_mean']  # Fix here
    .reset_index()
)

syll_d = (
    genotype_kinematics.groupby('syllable')
    .mean(numeric_only=True)['dist_to_center_px_mean']  # Fix here
    .reset_index()
)



import seaborn as sns
import matplotlib.pyplot as plt

def rolling_usage(genotype_kinematics, ref, roll_num, sort_by='velocity_2d_mm_mean', filename='velocity_2d_mm_mean'):
    # Merge and sort
    relative_u = pd.pivot_table(genotype_kinematics, values='usage', index='syllable', columns='group').reset_index()
    relative_u = pd.merge(relative_u, ref, on='syllable')
    relative_u = relative_u[relative_u.syllable < 49].sort_values(by=[sort_by], ascending=False).reset_index(drop=True)
    
    # Compute rolling mean and SEM
    rolling_sem = pd.DataFrame({
    'J': relative_u['J'].rolling(roll_num).std() / np.sqrt(roll_num),
    'K': relative_u['K'].rolling(roll_num).std() / np.sqrt(roll_num),
    'G': relative_u['G'].rolling(roll_num).std() / np.sqrt(roll_num)
}).dropna()
    
    rolling_mean = pd.DataFrame({
    'J': relative_u['J'].rolling(roll_num).mean(),
    'K': relative_u['K'].rolling(roll_num).mean(),
    'G': relative_u['G'].rolling(roll_num).mean()
}).dropna()

    # Normalize usage
    rolling_mean['K'] /= rolling_mean['J']
    rolling_mean['G'] /= rolling_mean['J']
    rolling_mean['J'] /= rolling_mean['J']
    
    rolling_sem['K'] /= rolling_mean['J']
    rolling_sem['G'] /= rolling_mean['J']
    rolling_sem['J'] /= rolling_mean['J']

    # Plot with SEM as shaded regions
    fig, ax = plt.subplots(figsize=(20, 10))
    for group, color in zip(['J', 'K', 'G'], ['blue', 'red', 'green']):
        ax.plot(rolling_mean.index, np.log10(rolling_mean[group]), label=f'{group} Mean', color=color)
        ax.fill_between(rolling_mean.index, 
                        rolling_mean[group] - rolling_sem[group], 
                        rolling_mean[group] + rolling_sem[group], 
                        color=color, alpha=0.2, label=f'{group} ± SEM')
    
    ax.set_title(f'Syllable Order by {sort_by} (High to Low)', fontsize=30)
    ax.set_xlabel('Rolling Window')
    ax.set_ylabel('Relative Usage')
    ax.legend(fontsize=15)
    sns.despine()
    
    fig.savefig(f'{filename}.pdf')
    plt.show()
    
    return relative_u, rolling_mean


relative_u, rolling = rolling_usage(genotype_kinematics, syll_v, 10, sort_by = 'velocity_2d_mm_mean', filename='velocityhightolow')



# =============================================================================
# # heatmap
# =============================================================================
moseq_df_subset = moseq_df.copy()
sessions = np.unique(moseq_df.uuid)

position_densities = []
groups = []
from scipy.signal import medfilt

for idx in sessions:
    moseq_df_session = moseq_df_subset[moseq_df_subset.uuid == idx]
    
    # Filter out rows with NaN values in either 'centroid_x_px' or 'centroid_y_px'
    moseq_df_session = moseq_df_session.dropna(subset=['centroid_x_px', 'centroid_y_px'])
    
    # Append the unique groups for each session
    groups.append(moseq_df_session.group.unique())
    
    # Create 2D histogram of positions
    position_density = plt.hist2d(moseq_df_session['centroid_x_px'], moseq_df_session['centroid_y_px'], bins=100, density=False)
    
    # Normalize the density by dividing by the sum
    position_densities.append(position_density[0] / position_density[0].sum())

position_densities = np.array(position_densities)
groups = np.concatenate(groups)


pd_cm =position_densities[np.where(groups =='K')[0], :, :].mean(axis=0)
pd_cf = position_densities[np.where(groups =='G')[0], :, :].mean(axis=0)
pd_kdm = position_densities[np.where(groups =='J')[0], :, :].mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
axes = axes.flatten()

temp_density = position_densities[np.where(groups =='K')[0], :, :].mean(axis=0)
temp_density = medfilt(temp_density, 11)
temp_density = (temp_density-temp_density.min())/(temp_density.max()-temp_density.min())

pos = axes[0].imshow(temp_density, cmap='hot')
axes[0].set_title('K')
fig.colorbar(pos,ax=axes[0])

temp_density = position_densities[np.where(groups =='G')[0], :, :].mean(axis=0)
temp_density = medfilt(temp_density, 11)
temp_density = (temp_density-temp_density.min())/(temp_density.max()-temp_density.min())

pos = axes[1].imshow(temp_density, cmap='hot')
axes[1].set_title('G')
fig.colorbar(pos,ax=axes[1])

temp_density = position_densities[np.where(groups =='J')[0], :, :].mean(axis=0)
temp_density = medfilt(temp_density, 11)
temp_density = (temp_density-temp_density.min())/(temp_density.max()-temp_density.min())
pos = axes[2].imshow(temp_density, cmap='hot')
axes[2].set_title('J')
fig.colorbar(pos,ax=axes[2])



from scipy.signal import medfilt
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharex=True, sharey=True)
vmin = 0.00001
cmap = 'seismic'
axes = axes.flatten()

# Assume pd_kdm, pd_cm, pd_kdf, pd_cf are predefined arrays with data
# Scale the data to go from -1 to 1
data = (medfilt(pd_cm, 11) - medfilt(pd_kdm, 11)) * 10000
norm = Normalize(vmin=data.min(), vmax=data.max())  # Normalization
print(data.min(), data.max())

# Plot the first data
pos = axes[0].imshow(data, cmap=cmap, vmin=-1, vmax=1)
# pos = axes[0].imshow(data, cmap=cmap, norm=norm)
axes[0].set_title('K - J')
fig.colorbar(pos, ax=axes[0])

# Scale the data to go from -1 to 1
data = (medfilt(pd_cf, 11) - medfilt(pd_kdm, 11)) * 10000
norm = Normalize(vmin=data.min(), vmax=data.max())  # Normalization
pos = axes[1].imshow(data, cmap=cmap, vmin=-1, vmax=1)
# pos = axes[1].imshow(data, cmap=cmap, norm=norm)
axes[1].set_title('G - J')
fig.colorbar(pos, ax=axes[1])

# Output the min and max values of the data for inspection
print(data.min(), data.max())

plt.show()



# =============================================================================
# # web figure
# =============================================================================


from math import ceil
import networkx as nx


def plot_transition_graph_group_diff(
    groups,
    trans_mats,
    usages,
    syll_include,
    save_dir=None,
    layout="circular",
    node_scaling=100000,
    show_syllable_names=False,
):
    syll_names = [f"{ix}" for ix in syll_include]

    # subsetting transmat
    trans_mats =[tm[syll_include, :][:, syll_include] for tm in trans_mats]
    usages = [[usage[i] for i in syll_include] for usage in usages]

    # take absolute value of usage
    abs_usages = [np.abs(usage) for usage in usages]
    plot_threshold = 0

    n_row = ceil(len(groups) / 2)
    fig, all_axes = plt.subplots(n_row, 2, figsize=(20, 9 * n_row))
    ax = all_axes.flat

    for i in range(len(groups)):
        G = nx.from_numpy_array(trans_mats[i] * 100)
        widths = nx.get_edge_attributes(G, "weight")
        if layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        # get node list
        nodelist = G.nodes()
        # normalize the usage values
        sum_usages = sum(abs_usages[i])
        normalized_usages = np.array([u / sum_usages for u in abs_usages[i]]) * node_scaling + 800
        node_color =[]
        for a in usages[i]:
            if np.abs(a) < plot_threshold:
                node_color.append('white')
            else:
                if a < 0:
                    node_color.append('red')
                else:
                    node_color.append('blue')
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            #node_size=normalized_usages,
            node_size=500,
            # node_color=node_color,
            node_color='white',
            edgecolors="black",
            ax=ax[i],
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=widths.keys(),
            width=[np.abs(v/2) for v in widths.values()],
            edge_color=["red" if v < 0 else "blue" for v in widths.values()],
            ax=ax[i],
            alpha=0.6,
        )
        nx.draw_networkx_labels(
            G,
            pos=pos,
            labels=dict(zip(nodelist, syll_names)),
            font_color="black",
            ax=ax[i],
        )
        ax[i].set_title(groups[i])
        print(np.array(list(widths.values())).max())
    # turn off the axis spines
    for sub_ax in ax:
        sub_ax.axis("off")

    return fig


syll_include = stats_df.syllable.unique()
diff_usages = [np.array(list(usages[0].values())) - np.array(list(usages[2].values())), 
               np.array(list(usages[1].values())) - np.array(list(usages[3].values()))]

diff_1 = trans_mats[0] - trans_mats[2]
diff_2 = trans_mats[1] -trans_mats[3]

abs_diff_trans_mats = [np.abs(trans_mats[0] - trans_mats[2]) > 0.05,np.abs(trans_mats[1] -trans_mats[3]) > 0.05]
diff_1[~abs_diff_trans_mats[0]] = 0
diff_2[~abs_diff_trans_mats[1]] = 0
diff_trans_mats = [diff_1, diff_2]

new_group = ['day1', 'day4']
plot_transition_graph_group_diff(
new_group, diff_trans_mats, diff_usages, syll_include,
save_dir=None,
layout="circular",
node_scaling=200,
show_syllable_names=False)






ahmet = stats_df[(stats_df['group'] == 'G')][['usage','syllable','velocity_2d_mm_mean']]

ahmet.sort_values('velocity_2d_mm_mean', inplace = True)

sait = genotype_kinematics[(genotype_kinematics['group'] == 'g')]['syllable']

plt.figure()
asan_mean = ahmet.groupby(['syllable']).mean()
asan_mean.sort_values('velocity_2d_mm_mean', inplace = True)
asan_sem = ahmet.groupby(['syllable']).sem()
asan_sem.sort_values('velocity_2d_mm_mean', inplace = True)

plt.plot(asan_mean.index,asan_mean['usage'])
plt.errorbar(asan_sem.reset_index().index,asan_mean['usage'],asan_sem['usage'])
plt.xticks(asan_sem.reset_index().index,asan_sem.index)












