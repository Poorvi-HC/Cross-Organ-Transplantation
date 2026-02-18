import pandas as pd
import numpy as np
import networkx as nx
import leidenalg as la
import igraph as ig
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os
import random

random.seed(42)
np.random.seed(42)


def run_leiden_clustering(G, resolution_parameter=1.0):
    if G is None or G.number_of_nodes() == 0:
        print("Empty network")
        return {}

    edges = list(G.edges())
    weights = [G[u][v]['weight'] for u, v in edges]

    g = ig.Graph()
    g.add_vertices(list(G.nodes()))
    g.add_edges(edges)
    g.es['weight'] = weights

    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution_parameter,
        seed=42
    )

    clusters = {}
    for i, cluster in enumerate(partition):
        for node_idx in cluster:
            node_name = g.vs[node_idx]['name']
            clusters[node_name] = i

    print(f"Found {len(partition)} clusters")
    return clusters


def visualize_modules(expr_data, clusters, meta, tissue, run_dir):

    cluster_df = pd.DataFrame(
        {'gene': list(clusters.keys()), 'module': list(clusters.values())}
    )

    modules = sorted(cluster_df['module'].unique())
    os.makedirs(f"../../results/{tissue}/{run_dir}", exist_ok=True)

    if 'character' in meta.columns:
        character_classes = sorted(meta['character'].unique())
        palette = plt.cm.tab20(np.linspace(0, 1, len(character_classes)))
        character_colors = dict(zip(character_classes, palette))
    else:
        character_colors = {}

    for module in modules:
        module_genes = cluster_df[cluster_df['module'] == module]['gene'].tolist()
        if len(module_genes) <= 5:
            continue

        try:
            module_expr = expr_data[module_genes]
            module_expr_T = module_expr.T

            gene_linkage = linkage(
                pdist(module_expr_T, metric='correlation'),
                method='average'
            )

            sample_linkage = linkage(
                pdist(module_expr_T.T, metric='correlation'),
                method='average'
            )

            sample_order = dendrogram(sample_linkage, no_plot=True)['leaves']
            gene_order = dendrogram(gene_linkage, no_plot=True)['leaves']

            ordered_genes = [module_expr_T.index[i] for i in gene_order]
            ordered_samples = [module_expr_T.columns[i] for i in sample_order]
            ordered_data = module_expr_T.loc[ordered_genes, ordered_samples]

            n_genes = ordered_data.shape[0]

            fig = plt.figure(figsize=(14, min(len(module_genes), 100) * 0.22 + 4))
            fig.suptitle(f"Module {module} ({len(module_genes)} genes)", fontsize=16)

            gs = gridspec.GridSpec(
                3, 2,
                width_ratios=[0.18, 0.82],
                height_ratios=[0.20, 0.03, 0.77],
                hspace=0.02,
                wspace=0.02
            )

            # ---------------- Sample dendrogram ----------------
            ax_sample = fig.add_subplot(gs[0, 1])

            dendrogram(
                sample_linkage,
                orientation='top',
                no_labels=True,
                color_threshold=0,
                ax=ax_sample
            )

            dendro_xlim = ax_sample.get_xlim()

            ax_sample.set_xticks([])
            ax_sample.set_yticks([])

            # ---------------- Annotation bar ----------------
            ax_annot = fig.add_subplot(gs[1, 1])

            if 'character' in meta.columns and all(s in meta.index for s in ordered_samples):
                character_values = meta.loc[ordered_samples, 'character']

                color_array = np.array([
                    mcolors.to_rgb(character_colors[val])
                    for val in character_values
                ]).reshape(1, len(ordered_samples), 3)

                ax_annot.imshow(
                    color_array,
                    aspect='auto',
                    extent=[dendro_xlim[0], dendro_xlim[1], 0, 1]
                )

                ax_annot.set_xlim(dendro_xlim)
                ax_annot.set_xticks([])
                ax_annot.set_yticks([])

                # ---- Annotation legend ----
                legend_elements = [
                    Patch(facecolor=character_colors[val], label=str(val))
                    for val in character_classes
                ]

                ax_annot.legend(
                    handles=legend_elements,
                    loc='center left',
                    bbox_to_anchor=(1.01, 0.5),
                    frameon=False,
                    title="Character"
                )
            else:
                ax_annot.axis('off')

            # ---------------- Gene dendrogram ----------------
            ax_gene = fig.add_subplot(gs[2, 0])

            dendrogram(
                gene_linkage,
                orientation='left',
                no_labels=True,
                color_threshold=0,
                ax=ax_gene
            )

            ax_gene.set_xticks([])
            ax_gene.set_yticks([])

            # ---------------- Heatmap ----------------
            ax_heat = fig.add_subplot(gs[2, 1])

            im = ax_heat.imshow(
                ordered_data.values,
                aspect='auto',
                cmap='viridis',
                interpolation='nearest',
                extent=[dendro_xlim[0], dendro_xlim[1], 0, n_genes]
            )

            ax_heat.set_xlim(dendro_xlim)
            ax_heat.set_xticks([])
            ax_heat.set_yticks([])

            # ---------------- Manual colorbar ----------------
            cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.4])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Expression Level")

            fig.subplots_adjust(
                left=0.05,
                right=0.92,
                top=0.93,
                bottom=0.05
            )

            plt.savefig(
                f"../../results/{tissue}/{run_dir}/module_{module}_clustered_heatmap.png",
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

            ordered_data.to_csv(
                f"../../results/{tissue}/{run_dir}/module_{module}_clustered_data.csv"
            )

        except Exception as e:
            print(f"Error processing module {module}: {str(e)}")
            plt.close()
            continue
