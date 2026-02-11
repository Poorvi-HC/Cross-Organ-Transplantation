import pandas as pd
import numpy as np
import networkx as nx
import leidenalg as la
import igraph as ig
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from matplotlib import gridspec
import matplotlib.colors as mcolors
import os

def run_leiden_clustering(G, resolution_parameter=1.0):
    if G is None or G.number_of_nodes() == 0:
        print("Empty network, cannot perform clustering")
        return {}
    
    print("Converting networkx graph to igraph...")
    # Convert networkx graph to igraph
    edges = list(G.edges())
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Create igraph from edge list
    g = ig.Graph()
    g.add_vertices(list(G.nodes()))
    g.add_edges(edges)
    g.es['weight'] = weights
    
    print(f"Running Leiden clustering with resolution parameter {resolution_parameter}...")
    # Run Leiden algorithm
    partition = la.find_partition(
        g, 
        la.RBConfigurationVertexPartition, 
        weights='weight',
        resolution_parameter=resolution_parameter
    )
    
    # Create a dictionary mapping node names to cluster assignments
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
    os.makedirs(f"./{tissue}/{run_dir}", exist_ok=True)
    
    # Define a standard color palette that works for any tissue
    if 'character' in meta.columns:
        character_classes = meta['character'].unique()
        # Create a color mapping based on number of unique labels
        if len(character_classes) <= 10:
            # Use a predefined colorblind-friendly palette for up to 10 categories
            palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            character_colors = dict(zip(character_classes, palette[:len(character_classes)]))
        else:
            # For more categories, use a colormap that scales well
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab20')  # Good for up to 20 categories
            if len(character_classes) > 20:
                cmap = cm.get_cmap('viridis')  # Will work for any number
            
            colors = [cmap(i/len(character_classes)) for i in range(len(character_classes))]
            character_colors = dict(zip(character_classes, colors))
    else:
        print(f"Warning: 'character' column not found in metadata for {tissue}")
        character_colors = {}
    
    for module in modules:
        module_genes = cluster_df[cluster_df['module'] == module]['gene'].tolist()
        
        if len(module_genes) > 5:
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
                
                fig = plt.figure(figsize=(14, min(len(module_genes), 100) * 0.25 + 5))
                plt.suptitle(f"Module {module} ({len(module_genes)} genes)", fontsize=16)
                
                gs = gridspec.GridSpec(3, 2, width_ratios=[0.15, 0.85], 
                                      height_ratios=[0.15, 0.1, 0.85])
                
                ax_sample_dendrogram = plt.subplot(gs[0, 1])
                dendrogram(
                    sample_linkage,
                    orientation='top',
                    labels=module_expr.index if len(module_expr.index) < 30 else None,
                    distance_sort='descending',
                    show_leaf_counts=True,
                    no_labels=len(module_expr.index) >= 30,
                    color_threshold=0
                )
                ax_sample_dendrogram.set_xticks([])
                ax_sample_dendrogram.set_yticks([])
                
                ax_character = plt.subplot(gs[1, 1])
                
                # Check if we have character information for this tissue
                if 'character' in meta.columns and all(sample in meta.index for sample in ordered_samples):
                    character_values = meta.loc[ordered_samples, 'character']
                    
                    # Only proceed if we have character colors defined
                    if character_colors:
                        color_list = [character_colors[char] for char in character_values]
                        
                        for i, color in enumerate(color_list):
                            ax_character.fill_between([i, i+1], 0, 1, color=color)
                        
                        ax_character.set_xlim(0, len(ordered_samples))
                        ax_character.set_ylim(0, 1)
                        
                        ax_character.set_xticks([])
                        ax_character.set_yticks([])
                        
                        from matplotlib.patches import Patch
                        legend_elements = [Patch(facecolor=color, label=char) 
                                          for char, color in character_colors.items()]
                        ax_character.legend(handles=legend_elements, loc='center left', 
                                            bbox_to_anchor=(1.05, 0.5), frameon=False)
                    else:
                        print(f"Warning: No color mapping defined for character values in {tissue}")
                        ax_character.set_visible(False)
                else:
                    print(f"Warning: Some samples in module {module} are not in the metadata")
                    ax_character.set_visible(False)
                
                ax_gene_dendrogram = plt.subplot(gs[2, 0])
                dendrogram(
                    gene_linkage,
                    orientation='left',
                    labels=None,  # Too many genes to show labels
                    distance_sort='descending',
                    show_leaf_counts=True,
                    no_labels=True,
                    color_threshold=0
                )
                ax_gene_dendrogram.set_xticks([])
                ax_gene_dendrogram.set_yticks([])
                
                ax_heatmap = plt.subplot(gs[2, 1])
                sns.heatmap(
                    ordered_data,
                    cmap='viridis',
                    xticklabels=False,
                    yticklabels=False,
                    cbar_kws={"shrink": 0.5, "label": "Expression Level"}
                )
                
                plt.xlabel("Samples", fontsize=12)
                plt.ylabel("Genes", fontsize=12)
                
                cbar = ax_heatmap.collections[0].colorbar
                cbar.set_label("Expression Level", fontsize=10)
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.9, wspace=0.05, hspace=0.05)
                plt.savefig(f"./{tissue}/{run_dir}/module_{module}_clustered_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                ordered_data.to_csv(f"./{tissue}/{run_dir}/module_{module}_clustered_data.csv")
                
                # Save the ordering information
                with open(f"./{tissue}/{run_dir}/module_{module}_clustering_order.txt", 'w') as f:
                    f.write("Gene order (from top to bottom):\n")
                    for gene in ordered_genes:
                        f.write(f"{gene}\n")
                    f.write("\nSample order (from left to right):\n")
                    for sample in ordered_samples:
                        f.write(f"{sample}\n")
                    
                    # Save character values if available
                    if 'character' in meta.columns and all(sample in meta.index for sample in ordered_samples):
                        f.write("\nCharacter values for ordered samples:\n")
                        for sample, char in zip(ordered_samples, character_values):
                            f.write(f"{sample}: {char}\n")
            
            except Exception as e:
                print(f"Error processing module {module}: {str(e)}")
                import traceback
                traceback.print_exc()
                plt.close()  # Close any open figures
                continue