import json
trans_mats = np.load('moseq_trans_mats.npy')
with open('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/plots/moseq_usages_dict.json', 'r') as f:
    usages= json.load(f)
    
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
syll_include = [i for i in syll_include if i < 58]

diff_usages = [np.array(list(usages[0].values())) - np.array(list(usages[1].values())),  
               np.array(list(usages[2].values())) - np.array(list(usages[1].values()))] 

diff_1 = trans_mats[0] - trans_mats[1] # G vs. J
diff_2 = trans_mats[2] -trans_mats[1] # G vs. J

abs_diff_trans_mats = [np.abs(trans_mats[0] - trans_mats[1]) > 0.05,np.abs(trans_mats[2] -trans_mats[1]) > 0.05]
diff_1[~abs_diff_trans_mats[0]] = 0
diff_2[~abs_diff_trans_mats[1]] = 0
diff_trans_mats = [diff_1, diff_2]


# Now safely subset transition matrices
trans_mats = [tm[syll_include, :][:, syll_include] for tm in trans_mats]

# Proceed with your plot function
new_group = ["G vs. J", "K vs. J"]
plot_transition_graph_group_diff(
    new_group, diff_trans_mats, diff_usages, syll_include,
    save_dir=None,
    layout="circular",
    node_scaling=200,
    show_syllable_names=False
)




# heatmap

from os.path import join, dirname, abspath
from moseq2_app.gui.progress import update_progress, restore_progress_vars, progress_path_sanity_check

progress_filepath = 'progress.yaml' # Add the path to your progress.yaml here.

# set to True if you are coming from the CLI and did not use the extraction and modeling notebook.
# keep False if you came from the extraction and modeling notebook. Most users will keep this False
from_CLI = True
# keep False if this is not a model object from applying a pre-trained model to new data. Most users will keep this False
pretrained_model_used = False

progress_paths = restore_progress_vars(progress_file=progress_filepath, init=from_CLI, overwrite=False)



from moseq2_viz.model.fingerprint_classifier import create_fingerprint_dataframe, plotting_fingerprint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

moseq_df=pd.read_csv('./model_dir/my_model_1000/moseq_df.csv')
stats_df=pd.read_csv('./model_dir/my_model_1000/stats_df.csv')

stat_type = 'mean'
n_bins = 200  # resolution of distribution 
range_type = 'robust'  # robust or full
preprocessor = MinMaxScaler()

summary, range_dict = create_fingerprint_dataframe(moseq_df, stats_df, stat_type=stat_type, n_bins=n_bins, range_type=range_type)
summary_new = summary.loc[['J','G','K']]

# custom_order = ['J', 'G', 'K']
# df = summary.loc[summary.index.get_level_values(0).isin(custom_order)]  # Keep only relevant rows
# df = df.sort_index(level=0, key=lambda x: x.map({v: i for i, v in enumerate(custom_order)}))

plotting_fingerprint(summary_new, progress_paths['plot_path'], range_dict, preprocessor=preprocessor)




# =============================================================================
# # PCA
# =============================================================================

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# moseq dataframes
# moseq_df=pd.read_csv('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/moseq_df.csv')
stats_df=pd.read_csv('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/stats_df.csv')
stats_df.dropna(inplace = True)

# Select the relevant features for PCA
X = stats_df.groupby(['group','uuid']).mean(numeric_only=True)

#X = X.iloc[:,3:-1]
# X = X.loc[:, ['usage', X.columns.str.contains('velocity', case=False, na=False)]]
X = X.loc[:, ['usage'] + X.columns[X.columns.str.contains('velocity', case=False, na=False)].tolist()]

X['usage']
X.dropna(inplace = True)
# X = X['usage']
# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Create a new DataFrame with PCA results
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# You can also add the group or syllable information for visualization purposes
pca_df['syllable'] = stats_df['syllable']

# Add the group information for visualization
pca_df['group'] = stats_df['group']  # Assuming you have a column named 'group' in stats_df

clrs = ['black','green','orange']

# Visualize the PCA result with color based on the group
# group_names = pca_df['group'].unique()
group_names = ['J','G','K']
plt.figure(figsize=(8, 6))
# for group in pca_df['group'].unique():
for grp in range(len(group_names)):
    group = group_names[grp]
    subset = pca_df[pca_df['group'] == group]
    plt.scatter(subset['PC1'], subset['PC2'], label=group, color=clrs[grp], alpha=0.7)



# Add labels and title
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Syllable Statistics')

# Add legend to differentiate groups
plt.legend(title="Group")

# Optionally, add color bar if you want color-coding for the groups
# plt.colorbar(label='Group')

plt.show()


# =============================================================================
# # LDA
# =============================================================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# moseq dataframes
# moseq_df=pd.read_csv('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/moseq_df.csv')
stats_df=pd.read_csv('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/stats_df.csv')
stats_df.dropna(inplace = True)

# Select the relevant features for PCA
X = stats_df.groupby(['group','uuid']).mean(numeric_only=True)
X.reset_index(inplace = True)

# LDA assumes you have the group labels to separate the data
y = X['group']  # Assuming the group column is named 'group'

#X = X.iloc[:,3:-1]
# X = X.loc[:, ['usage', X.columns.str.contains('velocity', case=False, na=False)]]
# X = X.loc[:, ['usage'] + X.columns[X.columns.str.contains('velocity', case=False, na=False)].tolist()]
X = X.loc[:, ['usage'] + X.columns[X.columns.str.contains('velocity', case=False, na=False) & X.columns.str.contains('mm', case=False, na=False)].tolist()]
# X = X.loc[:, ['usage'] + X.columns[X.columns.str.contains('centroid', case=False, na=False)].tolist()]
# X = X.loc[:, ['usage'] + X.columns[X.columns.str.contains('length', case=False, na=False)].tolist()]
# X = X.loc[:, ['usage'] + X.columns[X.columns.str.contains('height', case=False, na=False)].tolist()]
# X = X.loc[:, ['usage'] + X.columns[X.columns.str.contains('area', case=False, na=False)].tolist()]
# X = X.loc[:, ['usage'] + X.columns[X.columns.str.contains('dist', case=False, na=False)].tolist()]
# X = X.loc[:, ['usage'] + X.columns[X.columns.str.contains('width', case=False, na=False)].tolist()]


# X.dropna(inplace = True)
# X = X['usage']
# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Apply LDA
lda = LDA(n_components=2)  # Reduce to 2 components for visualization
X_lda = lda.fit_transform(X_scaled, y)

# Create a new DataFrame with LDA results
lda_df = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])

# Add the group information for visualization
lda_df['group'] = y
clrs = ['black','green','orange']
# Define colors for groups
clrs = {'J': 'black', 'G': 'green', 'K': 'orange'}

# Visualize the LDA result with color based on the group
plt.figure(figsize=(5, 4))
plt.rcParams['font.family'] = 'Arial'

# group_names = lda_df['group'].unique()
group_names = ['J','G','K']

for group in group_names:
    subset = lda_df[lda_df['group'] == group]
    
    # Plot the scatter plot for the group
    plt.scatter(subset['LD1'], subset['LD2'], label=group, color=clrs[group], alpha=0.7, s=64)
    
    # KDE plot for group-specific density cloud (using seaborn's kdeplot)
    try:
        sns.kdeplot(data=subset, x="LD1", y="LD2", color=clrs[group], fill=True, alpha=0.3, contour=False, linewidth=0)
    except AttributeError:
        # If there's an error with contour (QuadContourSet), continue with the next iteration
        print(f"Error occurred while plotting KDE for group {group}. Continuing without KDE.")
        continue

# Add labels and title
plt.xlabel('LD1', fontsize=12)

plt.ylabel('LD2', fontsize=12)
plt.title('LDA of Syllable Statistics', fontsize=12)

plt.style.use('default')    
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_position(('outward', 5))    
# plt.gca().spines['bottom'].set_position(('outward', 5))    
plt.gca().set_facecolor('white')

# Add legend to differentiate groups
plt.legend(title="Group")

# Optionally, add color bar if you want color-coding for the groups
# plt.colorbar(label='Group')

plt.tight_layout()
plt.show()


# =============================================================================
# # volcano plot
# =============================================================================
### first part calculate the statistics

import numpy as np
import scipy.stats as stats
import pandas as pd

stats_df=pd.read_csv('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/stats_df.csv')

master_stats_df = stats_df.copy()
master_stats_df = master_stats_df[master_stats_df['syllable'] < 59]

unique_syllables = master_stats_df['syllable'].unique()

# Initialize dictionary to store Kruskal-Wallis results
kruskal_results = {('G', 'J'): [], ('K', 'J'): []}

# Loop through each syllable to perform Kruskal-Wallis test for HET vs. WT and KO vs. WT
for syll in unique_syllables:
    df_syll = master_stats_df[master_stats_df['syllable'] == syll]

    # Get the data for HET, WT, and KO groups
    wt_data = df_syll[df_syll['group'] == 'J']['usage']
    het_data = df_syll[df_syll['group'] == 'G']['usage']
    ko_data = df_syll[df_syll['group'] == 'K']['usage']

    _, kruskal_stat_group = stats.kruskal(het_data, wt_data,ko_data)

    # Perform Kruskal-Wallis test for HET vs WT (G vs J)    
    _, pval_het_wt = stats.kruskal(het_data, wt_data)
    
    # Perform Kruskal-Wallis test for KO vs WT (K vs J)
    _, pval_ko_wt = stats.kruskal(ko_data, wt_data)
    
    # For directionality, let's compare the medians of the groups
    # Direction for HET vs WT (G vs J)
    if np.median(het_data) > np.median(wt_data):
        direction_het_wt = 'greater'
    else:
        direction_het_wt = 'less'

    # Direction for KO vs WT (K vs J)
    if np.median(ko_data) > np.median(wt_data):
        direction_ko_wt = 'greater'
    else:
        direction_ko_wt = 'less'

    # Store results in the dictionary
    kruskal_results[('G', 'J')].append({
        'syllable': syll,
        'kruskal_stat': kruskal_stat_group,
        'p_value': pval_het_wt,
        'direction': direction_het_wt
    })

    kruskal_results[('K', 'J')].append({
        'syllable': syll,
        'kruskal_stat': kruskal_stat_group,
        'p_value': pval_ko_wt,
        'direction': direction_ko_wt
    })

# Now extract the data into a list
data = []

# Loop through the dictionary to extract data
for group, results in kruskal_results.items():
    for result in results:
        # Append each result to the data list
        data.append({
            'group': group,
            'syllable': result['syllable'],
            'kruskal_stat': result['kruskal_stat'],
            'p_value': result['p_value'],
            'direction': result['direction']
        })

# Convert the list of results into a pandas DataFrame
df_kruskal_results = pd.DataFrame(data)



### this part generates the volcano figure

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from adjustText import adjust_text  # Import adjustText

# Filter the data for the group ('G', 'J')
df_kruskal_results_gj = df_kruskal_results[df_kruskal_results['group'] == ('G', 'J')]

# Get the HET and WT data for the fold change calculation
for syll in df_kruskal_results_gj['syllable'].unique():
    # Filter data for this syllable
    syll_data = master_stats_df[master_stats_df['syllable'] == syll]
    
    # Extract the HET (G) and WT (J) data for the usage values
    wt_data = syll_data[syll_data['group'] == 'J']['usage']
    het_data = syll_data[syll_data['group'] == 'G']['usage']
    
    # Calculate the fold change (median of HET vs WT)
    # median_wt = np.median(wt_data)
    # median_het = np.median(het_data)
    
    median_wt = np.mean(wt_data)
    median_het = np.mean(het_data)
    
    # Calculate fold change (log scale for better interpretation)
    fold_change = np.log2(median_het / median_wt)  # Use log2 for fold change
    
    # Add the fold change to the DataFrame
    df_kruskal_results_gj.loc[df_kruskal_results_gj['syllable'] == syll, 'fold_change'] = fold_change

# Calculate the -log10(p_value) for the volcano plot y-axis
df_kruskal_results_gj['neg_log10_p_value'] = -np.log10(df_kruskal_results_gj['p_value'])

# Create custom colormap where blue represents negative fold change, red represents positive
cmap = cm.RdBu_r  # Diverging colormap (blue for neg, red for pos)
# cmap = cm.RdBu  # Now blue represents positive values and red represents negative values

# Normalize colors based on fold change for hue (X-axis) and -log10(p) for intensity (Y-axis)
# norm_x = Normalize(vmin=df_kruskal_results_gj['fold_change'].min(), 
#                    vmax=df_kruskal_results_gj['fold_change'].max())  # Hue (color)
norm_x = Normalize(vmin=-1, vmax=1)

norm_y = Normalize(vmin=0, vmax=np.max(df_kruskal_results_gj['neg_log10_p_value']))  # Intensity (brightness)

# Generate colors for each point
colors = [cmap(norm_x(fc))[:3] + (norm_y(nlp),)  # Adjust transparency based on significance
          for fc, nlp in zip(df_kruskal_results_gj['fold_change'], df_kruskal_results_gj['neg_log10_p_value'])]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
plt.rcParams['font.family'] = 'Arial'

# Plot the points with the customized colormap
scatter = ax.scatter(
    df_kruskal_results_gj['fold_change'], 
    df_kruskal_results_gj['neg_log10_p_value'], 
    c=colors,  # Custom colors based on X (hue) and Y (intensity)
    edgecolor='black', 
    s=100
)

# Add color bar for fold change
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_x)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)  # Fix: Assign the colorbar to the same axis
cbar.set_label('Fold Change (log2)')

# Add labels and title
ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p = 0.05 significance')
ax.axvline(x=0, color='black', linestyle='--', label='No change (log fold = 0)')

# ax.set_title('HET vs WT (G vs J)')
ax.set_title('HT vs WT (G vs J)')
ax.set_xlabel('Fold Change (log2)')
ax.set_ylabel('-log10(p-value)')

# Annotate significant points
texts = []
for index, row in df_kruskal_results_gj.iterrows():
    if row['p_value'] < 0.05:
        texts.append(ax.text(
            row['fold_change'], row['neg_log10_p_value'], 
            str(row['syllable']), 
            color='red', ha='center', va='center', fontsize=10, fontweight='bold'
        ))

# Adjust text to prevent overlap
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

# Show plot
plt.show()





# =============================================================================
# # feature comparison spyder plot
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.
    This function creates a RadarAxes projection and registers it.
    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta



df_stat = stats_df.loc[:,~stats_df.columns.str.contains('timestamp|frame index')] 

# control_stat = stats_df[stats_df['group'] == 'J']
# control_stat = control_stat.loc[:,~control_stat.columns.str.contains('timestamp|frame index')] 

significant_syllables = {'increased' : [15,17,18,38],
                         'decreased' : [23,57,58]}

inc_stat = df_stat.loc[df_stat['syllable'].isin(significant_syllables['increased']),:]
dec_stat = df_stat.loc[df_stat['syllable'].isin(significant_syllables['decreased']),:]

# Calculate z-scores relative to the WT group
wt_data = inc_stat[inc_stat['group'] == 'J']
wt_data.drop(columns = ['group','uuid','syllable','syllable key'], inplace = True)

wt_mean = wt_data.mean()
wt_std = wt_data.std()

z_scores = (inc_stat.iloc[:,3:-1] - wt_mean) / wt_std
z_scores_final = pd.concat([inc_stat[['group','syllable']],z_scores],1)

categories_final = z_scores.columns  # these are macro features

groups = ['J', 'G', 'K']
clrs = ['black', 'green','orange']
# clrs = ['black','red']

# Prepare data for plotting
values = []
for group in groups:
    group_data = z_scores[z_scores_final['group'] == group]
    values.append(group_data.mean())

# Number of variables
num_vars = len(categories_final)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# Create radar chart
theta = radar_factory(num_vars, frame='polygon')

# Plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))

for i, group_data in enumerate(values):
    # Make the plot close to a circle
    values_plot = group_data.tolist()  # Remove the last element
    # Plot
    ax.plot(theta, values_plot, color=clrs[i], linewidth=2)

ax.set_ylim(-1.5, 1.5)  # Set y-axis limits
ax.set_rgrids([-3, -2, -1, 0,1])  # Set radial grids

# ax.set_xticks(angles)
ax.set_xticks(theta)



xtick_locs = ax.get_xticks()
# xtick_labels = ax.get_xticklabels()
import math
degrees_value = [math.degrees(i) for i in xtick_locs]

ax.set_xticklabels([])
for i in range(len(categories_final)):
    rotation_angle = degrees_value[i]
    ha = 'center'  # Default horizontal alignment
    lblss = categories_final[i]
    if 0 <= rotation_angle <= 180:
        rotation_angle += 180
        ha = 'right'  # Change horizontal alignment for angles between 90 and 270
        plt.text(angles[i]+0.1, 2.1, lblss, rotation=rotation_angle+90, ha='right', rotation_mode='anchor', fontsize=13)
    else:
        plt.text(angles[i]-0.1, 2.1, lblss, rotation=rotation_angle+90, ha='left', rotation_mode='anchor', fontsize = 13)

# ax.set_varlabels(categories_final)
# ax.tick_params(axis='x', labelsize=13)

plt.tight_layout()


# =============================================================================
# # horizontal bar plot
# =============================================================================

# df_stat = stats_df.loc[:,~stats_df.columns.str.contains('timestamps|frame index')] 
df_stat = stats_df.loc[:, ~stats_df.columns.str.contains('timestamps|frame index', case=False)]
df_stat = df_stat.loc[:,df_stat.columns.str.contains('group|uuid|syllable|mean')] 

# significant_syllables = {'increased' : [15,17,18,38],
#                          'decreased' : [23,57,58]}

inc_stat = df_stat.loc[df_stat['syllable'].isin([15,17,18,33,38,43,58]),:]
# inc_stat = df_stat.loc[df_stat['syllable'].isin([6,23,24,25,28,36,49,57,58]),:]

# significant_syllables = {'increased' : [15,17,18,33,38,43,58],
#                          'decreased' : [6,23,24,,25,28,36,49,57,58]}

# Calculate z-scores relative to the WT group
wt_data = inc_stat[inc_stat['group'] == 'J']
wt_data.drop(columns = ['group','uuid','syllable','syllable key'], inplace = True)

wt_mean = wt_data.mean()
wt_std = wt_data.std()

z_scores = (inc_stat.iloc[:,3:-1] - wt_mean) / wt_std
z_scores_final = pd.concat([inc_stat[['group','syllable']],z_scores],1)


# mean_values = z_scores_final[(z_scores_final['group'] == 'G') & (z_scores_final['syllable'] == 15)].mean().sort_values()
# z_scores_final_new = z_scores_final.loc[z_scores_final['syllable'].isin([23,57,58]),:]
mean_values = z_scores_final[(z_scores_final['group'] == 'K')].groupby('syllable').mean().mean().sort_values()

# plt.figure(figsize=(4, 10))  # Set figure size
plt.figure(figsize=(4, 4))  # Set figure size
plt.rcParams['font.family'] = 'Arial'
sns.barplot(x=mean_values.values, y=mean_values.index, palette=['blue' if x < 0 else 'red' for x in mean_values])
plt.xlim(-1.2, 1.2)
plt.tight_layout()



# =============================================================================
# # 2d vs 3d velocity
# =============================================================================

import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# df_stat = stats_df.loc[:,~stats_df.columns.str.contains('timestamps|frame index')] 
velo_2d = stats_df.loc[:,stats_df.columns.str.contains('group|uuid|2d', case = False)]
velo_3d = stats_df.loc[:,stats_df.columns.str.contains('group|uuid|3d', case = False)]

stat_2d = velo_2d.groupby(['group','uuid']).mean()['velocity_2d_mm_mean']
_, p_value_2d = stats.f_oneway(stat_2d['J'].values, stat_2d['G'].values, stat_2d['K'].values)
tukey_results_2d = pairwise_tukeyhsd(stat_2d.values, stat_2d.index.get_level_values('group'), alpha=0.05)
print(tukey_results_2d.summary())

stat_3d = velo_3d.groupby(['group','uuid']).mean()['velocity_3d_mm_mean']
_, p_value_3d = stats.f_oneway(stat_3d['J'].values, stat_3d['G'].values, stat_3d['K'].values)
tukey_results_3d = pairwise_tukeyhsd(stat_3d.values, stat_3d.index.get_level_values('group'), alpha=0.05)
print(tukey_results_3d.summary())

plt.figure(figsize=[5,3])
plt.subplot(1,2,1)
jt = np.random.uniform(-.1,0.1,len(stat_2d['J']))
plt.scatter(np.ones(len(stat_2d['J']))+jt,stat_2d['J'], color = 'black')
jt = np.random.uniform(-.1,0.1,len(stat_2d['G']))
plt.scatter(np.ones(len(stat_2d['G']))*2+jt,stat_2d['G'], color = 'green')
jt = np.random.uniform(-.1,0.1,len(stat_2d['K']))
plt.scatter(np.ones(len(stat_2d['K']))*3+jt,stat_2d['K'], color = 'orange')

plt.plot([1,3],[5.2,5.2], color='orange')
plt.text(1.75,5.3,'0.0076', fontsize = 12, color= 'orange')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_position(('outward', 5))    
# plt.gca().spines['bottom'].set_position(('outward', 5)) 

plt.ylim([0.5,6])
plt.title('Velocity 2D\np-value:'+str(round(p_value_2d,2)))
plt.xticks(ticks=[1,2,3], labels=['WT','HT','KO'])

plt.subplot(1,2,2)
jt = np.random.uniform(-.1,0.1,len(stat_3d['J']))
plt.scatter(np.ones(len(stat_3d['J']))+jt,stat_3d['J'], color = 'black')
jt = np.random.uniform(-.1,0.1,len(stat_3d['G']))
plt.scatter(np.ones(len(stat_3d['G']))*2+jt,stat_3d['G'], color = 'green')
jt = np.random.uniform(-.1,0.1,len(stat_3d['K']))
plt.scatter(np.ones(len(stat_3d['K']))*3+jt,stat_3d['K'], color = 'orange')

plt.plot([1,3],[5.2,5.2], color='orange')
plt.text(1.75,5.3,'0.0178', fontsize = 12, color= 'orange')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_position(('outward', 5))    
# plt.gca().spines['bottom'].set_position(('outward', 5)) 

plt.ylim([0.5,6])
plt.title('Velocity 3D\np-value:'+str(round(p_value_3d,2)))
plt.xticks(ticks=[1,2,3], labels=['WT','HT','KO'])


plt.tight_layout()








# =============================================================================
# # syllable analysis
# =============================================================================

import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# moseq dataframes
# moseq_df=pd.read_csv('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/moseq_df.csv')
stats_df=pd.read_csv('/Users/asan/Desktop/moseq/new/model_dir/my_model_1000/stats_df.csv')

# sylable_analysis = stats_df.groupby(['group','syllable','uuid']).mean().reset_index()
# g_0 = sylable_analysis[(sylable_analysis['group'] == 'G') & (sylable_analysis['syllable'] == 0)]

stats_df = stats_df[(stats_df['group'] == 'J') | (stats_df['group'] == 'K')]
# stats_df = stats_df[(stats_df['group'] == 'J')]

syllable_analysis = stats_df.groupby(['syllable']).mean().dropna()
# mean_features = ['angle_mean','area_mm_mean','centroid_x_mm_mean','centroid_y_mm_mean','height_ave_mm_mean','length_mm_mean',
#                  'velocity_2d_mm_mean','velocity_3d_mm_mean','velocity_theta_mean','width_mm_mean','dist_to_center_px_mean']
mean_features = ['angle_mean','area_mm_mean','centroid_x_mm_mean','centroid_y_mm_mean','height_ave_mm_mean','length_mm_mean',
                 'velocity_2d_mm_mean','velocity_3d_mm_mean','velocity_theta_mean','width_mm_mean','dist_to_center_px_mean']

syllable_analysis = syllable_analysis[mean_features]

# ## only wt and ht
# increased_syllables = syllable_analysis.iloc[[15,17,18,38],:]
# decreased_syllables = syllable_analysis.iloc[[23,57,58],:]

## only wt and ko
increased_syllables = syllable_analysis.iloc[[15,18,33,43],:]
decreased_syllables = syllable_analysis.iloc[[6,23,24,25,28,36,49,58],:]
rest_syllables = syllable_analysis.loc[~syllable_analysis.index.isin([15,18,33,43,6,23,24,25,28,36,49,58]), :]

# plt.figure();
plt.subplots(2,6, figsize=[12,5])
count = 0
for i in mean_features:
    count = count+1
    plt.subplot(2,6,count)
    jt = np.random.uniform(-.1,0.1,len(increased_syllables))
    plt.scatter(np.ones(len(increased_syllables))+jt,increased_syllables[i], alpha = 0.6, color = 'red')
    jt = np.random.uniform(-.1,0.1,len(rest_syllables))
    plt.scatter(np.ones(len(rest_syllables))*2+jt,rest_syllables[i], alpha = 0.6, color = 'black')
    jt = np.random.uniform(-.1,0.1,len(decreased_syllables))
    plt.scatter(np.ones(len(decreased_syllables))*3+jt,decreased_syllables[i], alpha = 0.6, color = 'blue')
    if count<7:
        plt.xticks([])
    else:
        plt.xticks([1,2,3],labels = ['incresed','rest','decreased'],rotation=60)
    plt.title(i)
    plt.xlim([.8,3.2])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
plt.tight_layout()




# significant_syllables = {'increased' : [15,17,18,38],
#                          'decreased' : [23,57,58]}


# df_stat = stats_df.loc[:,~stats_df.columns.str.contains('timestamps|frame index')] 
velo_2d = stats_df.loc[:,stats_df.columns.str.contains('group|uuid|2d', case = False)]
velo_3d = stats_df.loc[:,stats_df.columns.str.contains('group|uuid|3d', case = False)]

stat_2d = velo_2d.groupby(['group','uuid']).mean()['velocity_2d_mm_mean']
_, p_value_2d = stats.f_oneway(stat_2d['J'].values, stat_2d['G'].values, stat_2d['K'].values)
tukey_results_2d = pairwise_tukeyhsd(stat_2d.values, stat_2d.index.get_level_values('group'), alpha=0.05)
print(tukey_results_2d.summary())

stat_3d = velo_3d.groupby(['group','uuid']).mean()['velocity_3d_mm_mean']
_, p_value_3d = stats.f_oneway(stat_3d['J'].values, stat_3d['G'].values, stat_3d['K'].values)
tukey_results_3d = pairwise_tukeyhsd(stat_3d.values, stat_3d.index.get_level_values('group'), alpha=0.05)
print(tukey_results_3d.summary())

plt.figure(figsize=[5,3])
plt.subplot(1,2,1)
jt = np.random.uniform(-.1,0.1,len(stat_2d['J']))
plt.scatter(np.ones(len(stat_2d['J']))+jt,stat_2d['J'], color = 'black')
jt = np.random.uniform(-.1,0.1,len(stat_2d['G']))
plt.scatter(np.ones(len(stat_2d['G']))*2+jt,stat_2d['G'], color = 'green')
jt = np.random.uniform(-.1,0.1,len(stat_2d['K']))
plt.scatter(np.ones(len(stat_2d['K']))*3+jt,stat_2d['K'], color = 'orange')

plt.plot([1,3],[5.2,5.2], color='orange')
plt.text(1.75,5.3,'0.0076', fontsize = 12, color= 'orange')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_position(('outward', 5))    
# plt.gca().spines['bottom'].set_position(('outward', 5)) 

plt.ylim([0.5,6])
plt.title('Velocity 2D\np-value:'+str(round(p_value_2d,2)))
plt.xticks(ticks=[1,2,3], labels=['WT','HT','KO'])

plt.subplot(1,2,2)
jt = np.random.uniform(-.1,0.1,len(stat_3d['J']))
plt.scatter(np.ones(len(stat_3d['J']))+jt,stat_3d['J'], color = 'black')
jt = np.random.uniform(-.1,0.1,len(stat_3d['G']))
plt.scatter(np.ones(len(stat_3d['G']))*2+jt,stat_3d['G'], color = 'green')
jt = np.random.uniform(-.1,0.1,len(stat_3d['K']))
plt.scatter(np.ones(len(stat_3d['K']))*3+jt,stat_3d['K'], color = 'orange')

plt.plot([1,3],[5.2,5.2], color='orange')
plt.text(1.75,5.3,'0.0178', fontsize = 12, color= 'orange')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_position(('outward', 5))    
# plt.gca().spines['bottom'].set_position(('outward', 5)) 

plt.ylim([0.5,6])
plt.title('Velocity 3D\np-value:'+str(round(p_value_3d,2)))
plt.xticks(ticks=[1,2,3], labels=['WT','HT','KO'])


plt.tight_layout()


















































