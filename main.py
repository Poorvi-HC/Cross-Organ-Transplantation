import os
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from network import build_coexpression_network
from leiden import run_leiden_clustering, visualize_modules
import argparse

def read_meta(meta_file):
    """Read metadata file"""
    try:
        if meta_file.endswith('.xlsx'):
            df = pd.read_excel(meta_file, index_col=0)
        else:
            df = pd.read_csv(meta_file, index_col=0)
        return df
    except Exception as e:
        print(f"Error reading metadata file: {str(e)}")
        return None

def load_existing_network(tissue, network_dir):
    """
    Load existing network from GraphML or pickle file
    
    Parameters:
    -----------
    tissue : str
        Name of the tissue
    graph_no : int
        Index of the graph
        
    Returns:
    --------
    G : networkx.Graph or None
        Loaded network graph, or None if files not found
    expr_filtered : pandas.DataFrame or None
        Filtered expression data, or None if file not found
    """
    network_dir_tissue = f"{network_dir}/{tissue}"
    network_graphml = f"{network_dir_tissue}/network.graphml"
    network_pkl = f"{network_dir_tissue}/network.pkl"
    expr_file = f"{network_dir_tissue}/filtered_expression_data.csv"
    
    G = None
    expr_filtered = None
    
    # Try loading network from GraphML first
    if os.path.exists(network_graphml):
        try:
            print(f"Loading network from GraphML: {network_graphml}")
            G = nx.read_graphml(network_graphml)
            print(f"  ✓ Loaded {tissue} network with {len(G.nodes())} nodes and {len(G.edges())} edges")
        except Exception as e:
            print(f"  ✗ Error loading GraphML: {str(e)}")
            G = None
    
    # Try loading from pickle if GraphML failed or doesn't exist
    if G is None and os.path.exists(network_pkl):
        try:
            print(f"Loading network from pickle: {network_pkl}")
            with open(network_pkl, 'rb') as f:
                G = pickle.load(f)
            print(f"  ✓ Loaded {tissue} network with {len(G.nodes())} nodes and {len(G.edges())} edges")
        except Exception as e:
            print(f"  ✗ Error loading pickle: {str(e)}")
            G = None
    
    # Load filtered expression data
    if os.path.exists(expr_file):
        try:
            print(f"Loading filtered expression data: {expr_file}")
            expr_filtered = pd.read_csv(expr_file, index_col=0)
            print(f"  ✓ Loaded expression data: {expr_filtered.shape[0]} samples, {expr_filtered.shape[1]} genes")
        except Exception as e:
            print(f"  ✗ Error loading expression data: {str(e)}")
            expr_filtered = None
            
    return G, expr_filtered

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build or load co-expression networks and run module detection')
    parser.add_argument('--load', action='store_true', help='Load existing networks instead of building new ones')
    parser.add_argument('--tissues', nargs='+', default=['liver', 'kidney', 'heart'], 
                        help='List of tissues to process (default: liver kidney heart)')
    parser.add_argument('--network_dir', type=str, default='networks', help='Directory to store networks')  
    parser.add_argument('--module_dir', type=str, default='run_1', help='Directory to store modules under each organ directory')
    args = parser.parse_args()
    
    # Tissue-specific data files
    expression_files = {
        "liver": "./data/GSE145780_liver.csv",
        "kidney": "./data/GSE192444_kidney.csv",
        "heart": "./data/GSE272655_heart.csv"
    }
    
    # Metadata files - update paths as needed
    meta_files = {
        "liver": "./data/liver_meta.xlsx",
        "kidney": "./data/kidney_meta.xlsx",
        "heart": "./data/heart_meta.xlsx"
    }
    
    # Ensure all requested tissues have valid file paths
    tissues = [t for t in args.tissues if t in expression_files]
    if not tissues:
        print("Error: No valid tissues specified")
        return
    
    # Populate the variables given in the parser
    network_dir = args.network_dir
    module_dir = args.module_dir

    # Consistent hyperparameters for all tissues to ensure fair comparative analysis
    common_params = {
        "intensity_threshold": 3.0,              # Filter out lowly expressed genes
        "variance_threshold_percentile": 75,     # Keep top 25% most variable genes
        "correlation_threshold_percentile": 93,  # Keep top 2% strongest correlations
        "fdr_threshold": 0.01,                   # Strict FDR control
        "leiden_resolution": 0.2                 # Balanced module size
    }
    
    # Use the same parameters for all tissues
    hyperparams = {tissue: common_params for tissue in tissues}
    
    # Create networks directory if it doesn't exist
    os.makedirs(network_dir, exist_ok=True)
    
    # Build or load networks for each tissue
    networks = {}
    filtered_expr = {}
    
    for i, tissue in enumerate(tissues, 1):
        print(f"\n{'='*50}")
        print(f"Processing {tissue.upper()}")
        print(f"{'='*50}")
        
        # Get tissue-specific parameters
        params = hyperparams[tissue]
        
        if args.load:
            # Try to load existing network
            networks[tissue], filtered_expr[tissue] = load_existing_network(tissue, network_dir)
            
            # If loading failed, offer to build a new network
            if networks[tissue] is None or filtered_expr[tissue] is None:
                print(f"Failed to load existing network for {tissue}")
                user_input = input(f"Would you like to build a new network for {tissue}? (y/n): ")
                
                if user_input.lower() == 'y':
                    print(f"Building new network for {tissue}...")
                    networks[tissue], filtered_expr[tissue] = build_coexpression_network(
                        expression_files[tissue],
                        tissue, 
                        network_dir,
                        intensity_threshold=params["intensity_threshold"],
                        variance_threshold_percentile=params["variance_threshold_percentile"],
                        fdr_threshold=params["fdr_threshold"],
                    )
                else:
                    print(f"Skipping {tissue}")
                    continue
        else:
            print(f"Building new network for {tissue}...")
            networks[tissue], filtered_expr[tissue] = build_coexpression_network(
                expression_files[tissue],
                tissue, 
                network_dir,
                intensity_threshold=params["intensity_threshold"],
                variance_threshold_percentile=params["variance_threshold_percentile"],
                fdr_threshold=params["fdr_threshold"],
            )
    
    # Run Leiden clustering and visualization for each tissue
    clusters = {}
    for i, tissue in enumerate(tissues, 1):
        if tissue not in networks or networks[tissue] is None:
            print(f"Skipping {tissue}: No network available")
            continue
        
        print(f"\nRunning Leiden clustering for {tissue} with resolution {params['leiden_resolution']}...")
        clusters[tissue] = run_leiden_clustering(
            networks[tissue], 
            resolution_parameter=params["leiden_resolution"]
        )
        
        # Read metadata if available
        meta = None
        if tissue in meta_files and os.path.exists(meta_files[tissue]):
            meta = read_meta(meta_files[tissue])
        if "character" not in meta.columns:
            meta["character"] = meta["rej_status"]
            
        # Create directory for this tissue
        os.makedirs(f"./{tissue}", exist_ok=True)
        
        # Visualize modules
        run_dir = module_dir
        if meta is not None:
            visualize_modules(filtered_expr[tissue], clusters[tissue], meta, tissue, run_dir)
        else:
            print(f"Warning: No metadata found for {tissue}, skipping visualize_modules")
            
    # Save network and cluster data for module_comparison.py
    for tissue in networks:
        if networks[tissue] is not None and tissue in clusters:
            # Save cluster assignments
            cluster_df = pd.DataFrame({
                'gene': list(clusters[tissue].keys()),
                'module': list(clusters[tissue].values())
            })
            cluster_df.to_csv(f"{network_dir}/{tissue}_clusters.csv", index=False)

    print("\nNetwork analysis complete!")
    print("Next, run module_comparison.py to compare modules across tissues")

if __name__ == "__main__":
    main()