import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pickle

def build_coexpression_network(expression_file, tissue, network_dir, intensity_threshold=0, variance_threshold_percentile=75,
                              correlation_threshold_percentile=93, fdr_threshold=0.05):
    save_dir = f"{network_dir}/network_meta_{tissue}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading microarray expression data...")
    expr_data = pd.read_csv(expression_file, index_col=0)
    
    print(f"Filtering genes with mean intensity < {intensity_threshold}...")
    gene_means = expr_data.mean(axis=0)  
    genes_to_keep = gene_means[gene_means >= intensity_threshold].index
    expr_filtered = expr_data[genes_to_keep]
    
    plt.figure(figsize=(10, 6))
    plt.hist(gene_means, bins=50)
    plt.title(f"Distribution of Mean Gene Expression (threshold={intensity_threshold})")
    plt.xlabel("Mean Expression")
    plt.ylabel("Count")
    plt.axvline(intensity_threshold, color='red', linestyle='dashed',
                label=f'Threshold: {intensity_threshold:.2f}')
    plt.legend()
    plt.savefig(f"{save_dir}/gene_means_histogram.png", dpi=300)
    plt.close()
    
    # Apply variance filter
    print(f"Filtering genes with variance below {variance_threshold_percentile}th percentile...")
    gene_variances = expr_filtered.var(axis=0)
    variance_threshold = np.percentile(gene_variances, variance_threshold_percentile)
    genes_to_keep = gene_variances[gene_variances >= variance_threshold].index
    expr_filtered = expr_filtered[genes_to_keep]
    
    plt.figure(figsize=(10, 6))
    plt.hist(gene_variances, bins=50)
    plt.axvline(variance_threshold, color='red', linestyle='dashed', 
                label=f'{variance_threshold_percentile}th Percentile: {variance_threshold:.2f}')
    plt.title("Distribution of Gene Expression Variance")
    plt.xlabel("Variance")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f"{save_dir}/gene_variance_histogram.png", dpi=300)
    plt.close()
    
    print(f"Retained {len(genes_to_keep)} genes out of {expr_data.shape[1]} after filtering")
    
    # Save filtered expression data
    expr_filtered.to_csv(f"{save_dir}/filtered_expression_data.csv")
    
    print("Calculating Spearman correlations between all gene pairs...")
    correlation_matrix = expr_filtered.corr(method='spearman')
    
    pairs = []
    pvalues = []
    correlations = []
    genes = correlation_matrix.columns
    
    print("Calculating p-values for correlations...")
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            gene1 = genes[i]
            gene2 = genes[j]
            r = correlation_matrix.loc[gene1, gene2]
            
            if r <= 0:
                continue
                
            n = expr_filtered.shape[0]  
            t_stat = r * np.sqrt((n-2) / (1-r**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
            
            pairs.append((gene1, gene2))
            correlations.append(r)
            pvalues.append(p_value)
    
    print("Performing FDR correction...")
    min_correlation = 0.65
    if len(pvalues) > 0:
        rejected, pvals_corrected, _, _ = multipletests(pvalues, alpha=fdr_threshold, method='fdr_bh')
        
        significant_pairs = [pairs[i] for i in range(len(pairs)) if 
                    rejected[i] and correlations[i] >= min_correlation]
        significant_correlations = [correlations[i] for i in range(len(correlations)) if 
                    rejected[i] and correlations[i] >= min_correlation]
        
        if len(significant_correlations) > 0:
            threshold = np.percentile(significant_correlations, correlation_threshold_percentile)
            
            plt.figure(figsize=(10, 6))
            plt.hist(significant_correlations, bins=50)
            plt.axvline(threshold, color='red', linestyle='dashed', 
                        label=f'{correlation_threshold_percentile}th Percentile: {threshold:.3f}')
            plt.title(f"Selecting Correlations > {threshold:.3f}")
            plt.xlabel("Correlation Coefficient")
            plt.ylabel("Count")
            plt.legend()
            plt.savefig(f"{save_dir}/correlation_threshold.png", dpi=300)
            plt.close()
            
            print("Building network...")
            G = nx.Graph()
            
            edge_count = 0
            for i in range(len(significant_pairs)):
                if significant_correlations[i] >= threshold:
                    G.add_edge(significant_pairs[i][0], significant_pairs[i][1], 
                              weight=significant_correlations[i])
                    edge_count += 1
            
            print(f"Network built with {len(G.nodes())} nodes and {edge_count} edges")
            
            print(f"Saving network to {save_dir}/network.graphml")
            nx.write_graphml(G, f"{save_dir}/network.graphml")
            
            with open(f"{save_dir}/network.pkl", 'wb') as f:
                pickle.dump(G, f)
            
            with open(f"{save_dir}/network_stats.txt", 'w') as f:
                f.write(f"Number of genes (nodes): {len(G.nodes())}\n")
                f.write(f"Number of connections (edges): {edge_count}\n")
                f.write(f"Mean intensity threshold: {intensity_threshold}\n")
                f.write(f"Variance threshold ({variance_threshold_percentile}th percentile): {variance_threshold}\n")
                f.write(f"FDR threshold: {fdr_threshold}\n")
                f.write(f"Final correlation threshold: {threshold}\n")
                
                f.write("\nNetwork Metrics:\n")
                if nx.is_connected(G):
                    f.write("Network is connected\n")
                    f.write(f"Network diameter: {nx.diameter(G)}\n")
                else:
                    f.write("Network is not connected\n")
                    components = list(nx.connected_components(G))
                    f.write(f"Number of connected components: {len(components)}\n")
                    largest_cc = max(components, key=len)
                    f.write(f"Size of largest connected component: {len(largest_cc)} nodes\n")
                
                f.write(f"Average clustering coefficient: {nx.average_clustering(G)}\n")
                f.write(f"Network density: {nx.density(G)}\n")
            
            # Plot network visualization for smaller networks
            if len(G.nodes()) <= 500:  
                plt.figure(figsize=(12, 12))
                pos = nx.spring_layout(G, seed=42)
                nx.draw_networkx(G, pos=pos, node_size=30, with_labels=False, 
                                 edge_color='gray', alpha=0.6)
                plt.title(f"Co-expression Network (Graph {graph_no})")
                plt.axis('off')
                plt.savefig(f"{save_dir}/network_visualization.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            return G, expr_filtered
        else:
            print("No significant correlations found")
            with open(f"{save_dir}/error_log.txt", 'w') as f:
                f.write("No significant correlations found after applying thresholds\n")
            return None, expr_filtered
    else:
        print("No positive correlations found")
        with open(f"{save_dir}/error_log.txt", 'w') as f:
            f.write("No positive correlations found\n")
        return None, expr_filtered