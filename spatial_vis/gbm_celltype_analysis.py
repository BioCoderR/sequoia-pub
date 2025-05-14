import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from scipy.stats import percentileofscore
import os
from tqdm import tqdm
import math
import seaborn as sns
import matplotlib.colors

# Load subtype annotation dataframe 
subtype_df = pd.read_csv('/projects/conco/gundla/root/uniglacier/data/cohortlevelData/2025-04-24T-TCGA_task2-subtype.csv')
subtype_dict = dict(zip(subtype_df['slide_id'], subtype_df['label']))

def score2percentile(score, ref):
    if np.isnan(score):
        return score # deal with nans in visualization (set to black)
    percentile = percentileofscore(ref, score)
    return percentile


if __name__=='__main__':
    root = '/projects/conco/gundla/root/uniglacier/models/trained/image2st/'
    src_path = '/projects/conco/gundla/root/uniglacier/models/trained/image2st/vis_all'
    
    draw_heatmaps = True
    all_genes = np.load(os.path.join(root, 'generated_markers', 'all.npy'), allow_pickle=True)

    slide_names = [
        i for i in os.listdir(src_path)
        if os.path.isdir(os.path.join(src_path, i))
        and i.startswith('TCGA-')
    ]
    all_corr_dfs = []

    dests = [os.path.join(src_path, 'gbm_celltypes', 'corr_maps'),
             os.path.join(src_path, 'gbm_celltypes', 'spatial_maps')]
    for dest in dests:
        if not os.path.exists(dest):
            os.makedirs(dest)

    ac = np.load(os.path.join(root, 'generated_markers', 'AC.npy'),allow_pickle=True)
    g1s = np.load(os.path.join(root, 'generated_markers', 'G1S.npy'),allow_pickle=True)
    g2m = np.load(os.path.join(root, 'generated_markers', 'G2M.npy'),allow_pickle=True)
    mes1 = np.load(os.path.join(root, 'generated_markers', 'MES1.npy'),allow_pickle=True)
    mes2 = np.load(os.path.join(root, 'generated_markers', 'MES2.npy'),allow_pickle=True)
    npc1 = np.load(os.path.join(root, 'generated_markers', 'NPC1.npy'),allow_pickle=True)
    npc2 = np.load(os.path.join(root, 'generated_markers', 'NPC2.npy'),allow_pickle=True)
    opc = np.load(os.path.join(root, 'generated_markers', 'OPC.npy'),allow_pickle=True)
    mapper = {}

    green = '#CEBC36' # ---> Shade of yellow [NPC1,NPC2,OPC]
    red = '#CE3649' # ---> Shade of pink red [G1S,G1M]
    blue = '#3648CE' # --> Shade of blue [MES1,MES2]
    purple = '#36CEBC' # Shade of cyan [AC type]

    mapper.update(dict.fromkeys(ac, matplotlib.colors.to_rgb(purple))) # purple
    mapper.update(dict.fromkeys(g1s, matplotlib.colors.to_rgb(red))) # red
    mapper.update(dict.fromkeys(g2m, matplotlib.colors.to_rgb(red)))
    mapper.update(dict.fromkeys(mes1, matplotlib.colors.to_rgb(blue))) #blue
    mapper.update(dict.fromkeys(mes2, matplotlib.colors.to_rgb(blue)))
    mapper.update(dict.fromkeys(npc1, matplotlib.colors.to_rgb(green))) #green
    mapper.update(dict.fromkeys(npc2, matplotlib.colors.to_rgb(green)))
    mapper.update(dict.fromkeys(opc, matplotlib.colors.to_rgb(green)))

    max_lim = 0
    # slide_names = [i for i in os.listdir(src_path)
    #                     if os.path.isdir(os.path.join(src_path, i))
    #                     and i not in ['corr_maps', 'spatial_maps']
    #     ]
    for slide_name in tqdm(slide_names):

        # source_path = src_path + '/' + slide_name
        #/projects/conco/gundla/root/uniglacier/models/trained/image2st/vis_all/TCGA-02-0047-01Z-00-DX1/TCGA-02-0047-01Z-00-DX1stride-10.csv
        
        import re
        import glob

        slide_dir = os.path.join(src_path, slide_name)
        stride_csvs = glob.glob(os.path.join(slide_dir, f'{slide_name}_stride-*.csv'))
        stride_csvs = [f for f in stride_csvs if re.search(r'stride-\d+\.csv$', f)]
        if not stride_csvs:
            continue  # skip if no stride file found
        path = stride_csvs[0]  # use first match
        df = pd.read_csv(path)
        df_max = max_lim = max(max(df.xcoord_tf), max(df.ycoord_tf))
        if df_max > max_lim:
            max_lim = df_max

        all_genes = list(set(all_genes)&set(df.columns))
        
        df = df.dropna(axis=0, how='any')
        df = df[['xcoord_tf','ycoord_tf']+all_genes]
        
        corrdf = df[all_genes].corr()
        kind = corrdf.columns.map(mapper)
        all_corr_dfs.append(corrdf)

        # UMAP embedding of correlation matrix
        from umap import UMAP

        reducer = UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(corrdf.values)

        umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        umap_df['gene'] = corrdf.columns
        umap_df['color'] = umap_df['gene'].map(mapper)
        umap_df[['UMAP1', 'UMAP2', 'gene', 'color']].to_csv(
            os.path.join(src_path, 'gbm_celltypes', 'corr_maps', f'{slide_name}_umap_v1.csv'),
            index=False
        )
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c=umap_df['color'], s=10)
        subtype = subtype_dict.get(slide_name, "Unknown")
        plt.title(f'{slide_name} - Subtype: {subtype}')
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(src_path, 'gbm_celltypes', 'corr_maps', f'{slide_name}_umap_v1.png'), dpi=300)
        plt.close()

        plt.close()
        plt.figure()
        pl = sns.clustermap(corrdf, row_colors=kind, cmap='magma') #, yticklabels=True, xticklabels=True, figsize=(50,50))
        pl.ax_row_dendrogram.set_visible(False)
        pl.ax_col_dendrogram.set_visible(False)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=purple, label='astrocyte'),
            Patch(facecolor=red, label='cell cycle'),
            Patch(facecolor=blue, label='mesenchymal'),
            Patch(facecolor=green, label='lineage')
        ]
        # Move legend to bottom center
        pl.fig.legend(handles=legend_elements,
                      loc='lower center',
                      bbox_to_anchor=(0.5, -0.05),
                      ncol=4,
                      title='Cell Type',
                      frameon=False)
        save_path = os.path.join(src_path, 'gbm_celltypes', 'corr_maps', f'{slide_name}_clustered_v1.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if draw_heatmaps:

        scaling_factor = 2
        max_lim += scaling_factor*5

        for slide_name in tqdm(slide_names):

            slide_dir = os.path.join(src_path, slide_name)
            stride_csvs = glob.glob(os.path.join(slide_dir, f'{slide_name}_stride-*.csv'))
            stride_csvs = [f for f in stride_csvs if re.search(r'stride-\d+\.csv$', f)]
            if not stride_csvs:
                continue  # skip if no stride file found
            path = stride_csvs[0]
            df = pd.read_csv(path)
            
            all_genes = list(set(all_genes)&set(df.columns))
            df = df.dropna(axis=0, how='any')
            df = df[['xcoord_tf','ycoord_tf']+all_genes]
        
            categories = [ac.tolist(), g1s.tolist()+g2m.tolist(), mes1.tolist()+mes2.tolist(), npc1.tolist()+npc2.tolist()+opc.tolist()]
            labels = ['ac', 'cc', 'mes', 'lin']
            colors = {'ac':purple, 'cc':red, 'mes':blue, 'lin':green}
            
            for j,label in enumerate(labels):
                df[label] = df[[i for i in categories[j] if i in df.columns]].mean(axis=1)
                ref = df[label].values
                df[label + '_perc'] = df.apply(lambda row: score2percentile(row[label], ref), axis=1)

            df['color'] = df[[i+'_perc' for i in labels]].idxmax(axis=1)
            df['color'] = df['color'].str.replace('_perc', '')
            df['color'] = df['color'].map(colors)

            plt.close()
            fig, ax = plt.subplots()
            x_padding = int((max_lim-max(df.xcoord_tf))/2)
            y_padding = int((max_lim-max(df.ycoord_tf))/2)
            df['xcoord_tf'] += x_padding
            df['ycoord_tf'] += y_padding
            
            ax.scatter(df['xcoord_tf']*scaling_factor,
                        df['ycoord_tf']*scaling_factor, 
                        s=2, #changing it to have smaller spots
                        alpha=0.7, #alpha channel to see the spots overlapping 
                        c=df['color'])
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=purple, label='astrocyte'),
                Patch(facecolor=red, label='cell cycle'),
                Patch(facecolor=blue, label='mesenchymal'),
                Patch(facecolor=green, label='lineage')
            ]
            # Move legend to bottom center
            fig.legend(handles=legend_elements,
                       loc='lower center',
                       bbox_to_anchor=(0.5, -0.05),
                       ncol=4,
                       title='Cell Type',
                       frameon=False)

            ax.set_xlim([0,max_lim*scaling_factor])
            ax.set_ylim([0,max_lim*scaling_factor])
            ax.set_facecolor("#F1EFF0")
            for p in ['top', 'right', 'bottom', 'left']:
                ax.spines[p].set_color('gray') #.set_visible(False)
                ax.spines[p].set_linewidth(1)
            ax.invert_yaxis()
            subtype = subtype_dict.get(slide_name, "Unknown")
            ax.set_title(f'{slide_name}\nSubtype: {subtype}', fontsize=12, pad=15)
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='both', length=0, labelsize=0)

            save_path = os.path.join(src_path, 'gbm_celltypes', 'spatial_maps', f'{slide_name}_v1.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out_df = df[['xcoord_tf', 'ycoord_tf', 'color']]
            out_csv_path = os.path.join(src_path, 'gbm_celltypes', 'spatial_maps', f'{slide_name}_v1.csv')
            out_df.to_csv(out_csv_path, index=False)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    sum_df = all_corr_dfs[0]
    for i in range(1,len(all_corr_dfs)): 
        sum_df += all_corr_dfs[i]
    sum_df = sum_df / len(all_corr_dfs)
    
    plt.close()
    plt.figure()
    kind = sum_df.columns.map(mapper)
    pl = sns.clustermap(sum_df, row_colors=kind, col_colors=kind, cmap='magma')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=purple, label='astrocyte'),
        Patch(facecolor=red, label='cell cycle'),
        Patch(facecolor=blue, label='mesenchymal'),
        Patch(facecolor=green, label='lineage')
    ]
    # Move legend to bottom center for total clustermap
    pl.fig.legend(handles=legend_elements,
                  loc='lower center',
                  bbox_to_anchor=(0.5, -0.05),
                  ncol=4,
                  title='Cell Type',
                  frameon=False)
    pl.ax_row_dendrogram.set_visible(False)
    pl.ax_col_dendrogram.set_visible(False)
    save_path = os.path.join(src_path, 'gbm_celltypes', 'corr_maps', 'total_clustered_v1.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
