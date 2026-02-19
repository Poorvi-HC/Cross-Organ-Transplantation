import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn2
import networkx as nx
import pickle
from collections import defaultdict, Counter
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import itertools
import matplotlib.cm as cm

class ModuleComparison:
    def __init__(self, tissues=["liver", "kidney", "heart"], output_dir='../../results/module_comparison'):
        """
        Initialize the module comparison analysis.
        
        Parameters:
        -----------
        tissues : list
            List of tissue names to compare
        output_dir : str
            Directory to save the module comparison results
        """
        self.tissues = tissues
        self.networks = {}
        self.clusters = {}
        self.genes_by_module = {}
        self.module_genes = {}
        self.conserved_modules = {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load network and cluster data for all tissues"""
        print("Loading network and cluster data...")
        
        for tissue in self.tissues:
            # Load clusters from CSV
            cluster_file = f"../../results/networks/{tissue}_clusters.csv"
            if os.path.exists(cluster_file):
                clusters_df = pd.read_csv(cluster_file)
                self.clusters[tissue] = dict(zip(clusters_df['gene'], clusters_df['module']))
                print(f"  ✓ Loaded clusters for {tissue}: {len(self.clusters[tissue])} genes")
                
                # Organize genes by module
                self.genes_by_module[tissue] = defaultdict(list)
                for gene, module in self.clusters[tissue].items():
                    self.genes_by_module[tissue][module].append(gene)
                
                print(f"  ✓ Found {len(self.genes_by_module[tissue])} modules in {tissue}")
            else:
                print(f"  ✗ Cluster file not found for {tissue}: {cluster_file}")
                
            # Load network from pickle
            network_files = [f for f in os.listdir("../../results/networks") if f.startswith(f"network_meta_") and f.endswith(".pkl")]
            for network_file in network_files:
                try:
                    with open(f"../../results/networks/{network_file}/network.pkl", 'rb') as f:
                        G = pickle.load(f)

                        if tissue in self.clusters and len(set(G.nodes).intersection(self.clusters[tissue].keys())) > 0:
                            self.networks[tissue] = G
                            print(f"  ✓ Loaded network for {tissue}: {len(G.nodes())} nodes, {len(G.edges())} edges")
                            break
                except:
                    continue
                    
        if not self.clusters:
            print("\nNo cluster data found. Please run main.py first.")
            return False
            
        return True
    
    def get_all_genes(self):
        """Get all unique genes across all tissues"""
        all_genes = set()
        for tissue in self.tissues:
            if tissue in self.clusters:
                all_genes.update(self.clusters[tissue].keys())
        return all_genes
    
    def analyze_gene_overlap(self):
        """Analyze gene overlap between tissues"""
        print("\nAnalyzing gene overlap between tissues...")
        
        # Get genes in each tissue
        tissue_genes = {t: set(self.clusters[t].keys()) for t in self.tissues if t in self.clusters}
        
        # Get pairwise overlaps
        pairs = list(itertools.combinations(tissue_genes.keys(), 2))
        overlaps = {f"{t1}_{t2}": tissue_genes[t1].intersection(tissue_genes[t2]) 
                   for t1, t2 in pairs}
        
        # Get triple overlap if we have 3 tissues
        if len(tissue_genes) >= 3:
            all_tissues = list(tissue_genes.keys())
            triple_overlap = set.intersection(*[tissue_genes[t] for t in all_tissues])
            print(f"  ✓ {len(triple_overlap)} genes are shared across all tissues")
            
            # Save shared genes
            with open(f"{self.output_dir}/shared_genes_all_tissues.txt", 'w') as f:
                for gene in sorted(triple_overlap):
                    f.write(f"{gene}\n")
        
        # Create Venn diagram
        plt.figure(figsize=(10, 8))
        if len(tissue_genes) == 3:
            venn3([tissue_genes[t] for t in self.tissues], 
                 set_labels=self.tissues,
                 alpha=0.7)
        elif len(tissue_genes) == 2:
            venn2([tissue_genes[t] for t in tissue_genes.keys()],
                 set_labels=list(tissue_genes.keys()),
                 alpha=0.7)
        
        plt.title("Gene Overlap Between Tissues", fontsize=16)
        plt.savefig(f"{self.output_dir}/gene_overlap_venn.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary table
        overlap_data = []
        for t1 in tissue_genes:
            for t2 in tissue_genes:
                if t1 >= t2:  # Skip duplicates and self-comparisons
                    continue
                overlap = len(tissue_genes[t1].intersection(tissue_genes[t2]))
                pct1 = 100 * overlap / len(tissue_genes[t1])
                pct2 = 100 * overlap / len(tissue_genes[t2])
                overlap_data.append({
                    'Tissue1': t1,
                    'Tissue2': t2,
                    'Genes1': len(tissue_genes[t1]),
                    'Genes2': len(tissue_genes[t2]),
                    'Overlap': overlap,
                    'Overlap%_of_1': f"{pct1:.1f}%",
                    'Overlap%_of_2': f"{pct2:.1f}%"
                })
        
        # If we have 3 tissues, add the triple overlap
        if len(tissue_genes) >= 3:
            overlap_data.append({
                'Tissue1': 'All',
                'Tissue2': '',
                'Genes1': '',
                'Genes2': '',
                'Overlap': len(triple_overlap),
                'Overlap%_of_1': '',
                'Overlap%_of_2': ''
            })
        
        overlap_df = pd.DataFrame(overlap_data)
        overlap_df.to_csv(f"{self.output_dir}/gene_overlap_summary.csv", index=False)
        print(f"  ✓ Gene overlap analysis saved to {self.output_dir}/gene_overlap_summary.csv")
        return True
    
    def identify_conserved_modules(self, min_overlap=10, p_threshold=0.05):
        print("\nIdentifying conserved modules across tissues...")
        
        # Get all possible tissue pairs
        tissue_pairs = list(itertools.combinations(self.tissues, 2))
        
        # Setup for hypergeometric test
        all_genes = self.get_all_genes()
        universe_size = len(all_genes)
        
        # Store module associations
        module_associations = []
        p_values = []
        
        # For each tissue pair
        for t1, t2 in tissue_pairs:
            if t1 not in self.genes_by_module or t2 not in self.genes_by_module:
                continue
                
            print(f"  Comparing modules: {t1} vs {t2}")
            t1_modules = self.genes_by_module[t1]
            t2_modules = self.genes_by_module[t2]
            
            # Test all module pairs
            for m1, genes1 in t1_modules.items():
                for m2, genes2 in t2_modules.items():
                    # Calculate overlap
                    overlap = set(genes1).intersection(genes2)
                    n_overlap = len(overlap)
                    
                    # Only consider modules with sufficient overlap
                    if n_overlap >= min_overlap:
                        # Contingency table for Fisher's exact test
                        # (overlap, in m1 but not m2, in m2 but not m1, in neither)
                        table = [
                            [n_overlap, len(genes1) - n_overlap],
                            [len(genes2) - n_overlap, universe_size - len(genes1) - len(genes2) + n_overlap]
                        ]
                        
                        # Perform Fisher's exact test
                        odds_ratio, p_value = fisher_exact(table, alternative='greater')
                        
                        module_associations.append({
                            'tissue1': t1,
                            'tissue2': t2,
                            'module1': m1,
                            'module2': m2,
                            'module1_size': len(genes1),
                            'module2_size': len(genes2),
                            'overlap': n_overlap,
                            'overlap_pct1': 100 * n_overlap / len(genes1),
                            'overlap_pct2': 100 * n_overlap / len(genes2),
                            'p_value': p_value,
                            'odds_ratio': odds_ratio,
                            'genes': list(overlap)
                        })
                        p_values.append(p_value)
        
        # FDR correction
        if p_values:
            rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=p_threshold, method='fdr_bh')
            
            # Add corrected p-values and filter significant associations
            for i, assoc in enumerate(module_associations):
                assoc['p_adjusted'] = pvals_corrected[i]
                assoc['significant'] = rejected[i]
                
            # Filter significant associations
            significant_associations = [a for a in module_associations if a['significant']]
            
            # Create DataFrame for results
            assoc_df = pd.DataFrame(module_associations)
            sig_df = pd.DataFrame(significant_associations)
            
            # Save results
            assoc_df.to_csv(f"{self.output_dir}/module_associations_all.csv", index=False)
            if not sig_df.empty:
                sig_df.to_csv(f"{self.output_dir}/module_associations_significant.csv", index=False)
                
                # Organize conserved modules
                print(f"  ✓ Found {len(significant_associations)} significantly conserved module pairs")
                
                # Build adjacency list of conserved modules
                # Use string representation of "tissue:module" as node names
                adj_list = defaultdict(list)
                for assoc in significant_associations:
                    node1 = f"{assoc['tissue1']}:{assoc['module1']}"
                    node2 = f"{assoc['tissue2']}:{assoc['module2']}"
                    adj_list[node1].append((node2, assoc))
                    adj_list[node2].append((node1, assoc))
                
                # Find connected components (conserved module groups)
                visited = set()
                conserved_groups = []
                
                for node in adj_list:
                    if node not in visited:
                        # Find all connected nodes (BFS)
                        queue = [node]
                        component = []
                        component_edges = []
                        visited.add(node)
                        
                        while queue:
                            curr = queue.pop(0)
                            component.append(curr)
                            
                            for neighbor, assoc in adj_list[curr]:
                                component_edges.append(assoc)
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append(neighbor)
                        
                        # Store component (connected modules)
                        conserved_groups.append({
                            'modules': component,
                            'associations': component_edges
                        })
                
                # Analyze each conserved group
                for i, group in enumerate(conserved_groups):
                    group_id = f"conserved_group_{i+1}"
                    
                    # Get all genes in this conserved group
                    all_genes_in_group = set()
                    for assoc in group['associations']:
                        all_genes_in_group.update(assoc['genes'])
                    
                    # Count tissues represented in this group
                    tissues_in_group = set()
                    for module in group['modules']:
                        tissue = module.split(':')[0]
                        tissues_in_group.add(tissue)
                    
                    # Save this conserved group
                    self.conserved_modules[group_id] = {
                        'modules': group['modules'],
                        'associations': group['associations'],
                        'genes': list(all_genes_in_group),
                        'tissues': list(tissues_in_group),
                        'num_tissues': len(tissues_in_group),
                        'num_modules': len(group['modules']),
                        'num_genes': len(all_genes_in_group)
                    }
                
                # Print summary of conserved modules
                print(f"\nConserved module groups summary:")
                for group_id, group_data in self.conserved_modules.items():
                    tissues_str = ", ".join(group_data['tissues'])
                    print(f"  {group_id}: {group_data['num_modules']} modules across {group_data['num_tissues']} tissues ({tissues_str})")
                    print(f"     {group_data['num_genes']} genes conserved")
                
                # Save detailed conserved module information
                self._visualize_conserved_modules()
                
                return True
            else:
                print("  ✗ No significantly conserved module pairs found")
                return False
        else:
            print("  ✗ No module pairs to compare")
            return False
    
    def _visualize_conserved_modules(self):
        """Create visualizations for conserved modules"""
        if not self.conserved_modules:
            return
            
        # Create summary dataframe
        summary_data = []
        for group_id, group in self.conserved_modules.items():
            summary_data.append({
                'Group_ID': group_id,
                'Modules': ", ".join(group['modules']),
                'Tissues': ", ".join(group['tissues']),
                'Num_Tissues': group['num_tissues'],
                'Num_Modules': group['num_modules'],
                'Num_Genes': group['num_genes']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.output_dir}/conserved_modules_summary.csv", index=False)
        
        # Save gene lists for each conserved group
        for group_id, group in self.conserved_modules.items():
            with open(f"{self.output_dir}/{group_id}_genes.txt", 'w') as f:
                for gene in sorted(group['genes']):
                    f.write(f"{gene}\n")
        
        # Create heatmap of module overlaps
        self._create_module_overlap_heatmap()
        
        # Create network visualization of conserved modules
        self._create_conserved_module_network()
    
    def _create_module_overlap_heatmap(self):
        """Create a heatmap of module overlaps"""
        # Only proceed if we have at least two tissues with modules
        tissue_with_modules = [t for t in self.tissues if t in self.genes_by_module]
        if len(tissue_with_modules) < 2:
            return
            
        # Create a combined dataframe of all modules
        all_modules = {}
        for tissue in tissue_with_modules:
            for module, genes in self.genes_by_module[tissue].items():
                all_modules[f"{tissue}:{module}"] = genes
        
        # Calculate Jaccard similarities between all module pairs
        module_names = list(all_modules.keys())
        n_modules = len(module_names)
        
        similarity_matrix = np.zeros((n_modules, n_modules))
        
        for i in range(n_modules):
            for j in range(n_modules):
                genes_i = set(all_modules[module_names[i]])
                genes_j = set(all_modules[module_names[j]])
                
                jaccard = len(genes_i.intersection(genes_j)) / len(genes_i.union(genes_j)) if genes_i or genes_j else 0
                similarity_matrix[i, j] = jaccard
        
        # Create heatmap
        plt.figure(figsize=(max(12, n_modules//4), max(10, n_modules//4)))
        sns.heatmap(similarity_matrix, 
                   xticklabels=module_names,
                   yticklabels=module_names,
                   cmap="YlGnBu",
                   vmin=0, vmax=1)
        plt.title("Module Similarity (Jaccard Index)", fontsize=16)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/module_similarity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_conserved_module_network(self):
        """Create a network visualization of conserved modules"""
        if not self.conserved_modules:
            return
            
        # Create network
        G = nx.Graph()
        
        # Add nodes (modules)
        tissue_colors = {
            'liver': '#e41a1c',
            'kidney': '#377eb8',
            'heart': '#4daf4a'
        }
        
        # Add custom colors for any unlisted tissues
        for tissue in self.tissues:
            if tissue not in tissue_colors:
                # Generate a random color
                tissue_colors[tissue] = '#%02x%02x%02x' % (
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256)
                )
        
        # Add modules as nodes
        for group_id, group in self.conserved_modules.items():
            for module in group['modules']:
                tissue, module_id = module.split(':')
                G.add_node(module, 
                          tissue=tissue, 
                          module_id=module_id, 
                          group=group_id,
                          color=tissue_colors[tissue])
        
        # Add edges between modules
        for group_id, group in self.conserved_modules.items():
            for assoc in group['associations']:
                node1 = f"{assoc['tissue1']}:{assoc['module1']}"
                node2 = f"{assoc['tissue2']}:{assoc['module2']}"
                G.add_edge(node1, node2, 
                          weight=assoc['overlap'],
                          jaccard=min(assoc['overlap_pct1'], assoc['overlap_pct2']) / 100,
                          p_value=assoc['p_adjusted'])
        
        # Visualize the network
        plt.figure(figsize=(12, 12))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=2.0/np.sqrt(G.number_of_nodes()), iterations=50, seed=42)
        
        # Draw nodes grouped by tissue
        for tissue in self.tissues:
            if tissue in tissue_colors:
                tissue_nodes = [n for n, attr in G.nodes(data=True) if attr.get('tissue') == tissue]
                nx.draw_networkx_nodes(
                    G, pos, 
                    nodelist=tissue_nodes,
                    node_color=tissue_colors[tissue],
                    node_size=200,
                    alpha=0.8,
                    label=tissue
                )
        
        # Draw edges with width proportional to overlap
        edge_weights = [G[u][v]['jaccard'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            width=edge_weights,
            alpha=0.5,
            edge_color='gray'
        )
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title("Conserved Module Network", fontsize=16)
        plt.legend(scatterpoints=1)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/conserved_module_network.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save network as GraphML for further analysis
        nx.write_graphml(G, f"{self.output_dir}/conserved_module_network.graphml", infer_numeric_types=True)
    
    def analyze_top_conserved_modules(self, top_n=5):
        """
        Analyze the top conserved modules
        
        Parameters:
        -----------
        top_n : int
            Number of top module groups to analyze in detail
        """
        if not self.conserved_modules:
            print("\nNo conserved modules found to analyze")
            return False
            
        print(f"\nAnalyzing top {top_n} conserved module groups...")
        
        # Sort conserved modules by: 1) number of tissues, 2) number of genes
        sorted_groups = sorted(
            self.conserved_modules.items(),
            key=lambda x: (x[1]['num_tissues'], x[1]['num_genes']),
            reverse=True
        )
        
        # Analyze top N groups
        top_groups = sorted_groups[:min(top_n, len(sorted_groups))]
        
        for group_id, group in top_groups:
            print(f"\n  {group_id}: {group['num_modules']} modules across {group['num_tissues']} tissues")
            print(f"    Tissues: {', '.join(group['tissues'])}")
            print(f"    Conserved genes: {group['num_genes']}")
            
            # Save list of genes with their module assignment for each tissue
            gene_modules = defaultdict(dict)
            
            for module in group['modules']:
                tissue, module_id = module.split(':')
                for gene in self.genes_by_module[tissue][int(module_id)]:
                    if gene in group['genes']:
                        gene_modules[gene][tissue] = module_id
            
            # Create a table of genes and their module assignments
            gene_data = []
            for gene in group['genes']:
                row = {'Gene': gene}
                for tissue in group['tissues']:
                    row[f"{tissue}_module"] = gene_modules[gene].get(tissue, "")
                gene_data.append(row)
            
            gene_df = pd.DataFrame(gene_data)
            gene_df.to_csv(f"{self.output_dir}/{group_id}_gene_modules.csv", index=False)
            
            print(f"    Gene-module assignments saved to {self.output_dir}/{group_id}_gene_modules.csv")
        
        return True
    
    def run_all_analyses(self):
        """Run all module comparison analyses"""
        # Load data
        if not self.load_data():
            return
        
        # Run analyses
        self.analyze_gene_overlap()
        self.identify_conserved_modules()
        self.analyze_top_conserved_modules()
        
        print(f"\nAll analyses complete! Results saved to {self.output_dir}")

# Main function to run the module comparison
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare gene co-expression modules across tissues')
    parser.add_argument('--tissues', nargs='+', default=['liver', 'kidney', 'heart'], 
                        help='List of tissues to compare (default: liver kidney heart)')
    parser.add_argument('--min-overlap', type=int, default=10,
                        help='Minimum number of genes to consider an overlap significant (default: 10)')
    parser.add_argument('--p-threshold', type=float, default=0.05,
                        help='P-value threshold after FDR correction (default: 0.05)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top conserved module groups to analyze in detail (default: 5)')
    parser.add_argument('--module_comp_dir', type=str, default='../../results/module_comparison', help='Directory to save module comparison results')
    args = parser.parse_args()
    
    # Run the module comparison
    mc = ModuleComparison(tissues=args.tissues, output_dir=args.module_comp_dir)
    mc.load_data()
    mc.analyze_gene_overlap()
    mc.identify_conserved_modules(min_overlap=args.min_overlap, p_threshold=args.p_threshold)
    mc.analyze_top_conserved_modules(top_n=args.top_n)
    
    print(f"\nModule comparison complete! Results saved to {mc.output_dir}")