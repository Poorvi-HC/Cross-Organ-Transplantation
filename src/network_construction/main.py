import os
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from network import build_coexpression_network
from leiden import run_leiden_clustering, visualize_modules
import argparse


def read_meta(meta_file):
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
    network_dir_tissue = f"{network_dir}/network_meta_{tissue}"
    network_graphml = f"{network_dir_tissue}/network.graphml"
    network_pkl = f"{network_dir_tissue}/network.pkl"
    expr_file = f"{network_dir_tissue}/filtered_expression_data.csv"

    G = None
    expr_filtered = None

    if os.path.exists(network_graphml):
        try:
            print(f"Loading network from GraphML: {network_graphml}")
            G = nx.read_graphml(network_graphml)
            print(f"  ✓ Loaded {tissue} network with {len(G.nodes())} nodes and {len(G.edges())} edges")
        except Exception as e:
            print(f"  ✗ Error loading GraphML: {str(e)}")

    if G is None and os.path.exists(network_pkl):
        try:
            print(f"Loading network from pickle: {network_pkl}")
            with open(network_pkl, 'rb') as f:
                G = pickle.load(f)
            print(f"  ✓ Loaded {tissue} network with {len(G.nodes())} nodes and {len(G.edges())} edges")
        except Exception as e:
            print(f"  ✗ Error loading pickle: {str(e)}")

    if os.path.exists(expr_file):
        try:
            print(f"Loading filtered expression data: {expr_file}")
            expr_filtered = pd.read_csv(expr_file, index_col=0)
            print(f"  ✓ Loaded expression data: {expr_filtered.shape[0]} samples, {expr_filtered.shape[1]} genes")
        except Exception as e:
            print(f"  ✗ Error loading expression data: {str(e)}")

    return G, expr_filtered


def main():
    parser = argparse.ArgumentParser(description='Build or load co-expression networks and run module detection')
    parser.add_argument('--load', action='store_true', help='Load existing networks and clusters')
    parser.add_argument('--tissues', nargs='+', default=['liver', 'kidney', 'heart'])
    parser.add_argument('--network_dir', type=str, default='networks')
    parser.add_argument('--module_dir', type=str, default='modules')
    args = parser.parse_args()

    expression_files = {
        "liver": "../../data/GSE145780_liver.csv",
        "kidney": "../../data/GSE192444_kidney.csv",
        "heart": "../../data/GSE272655_heart.csv"
    }

    meta_files = {
        "liver": "../../data/liver_meta.xlsx",
        "kidney": "../../data/kidney_meta.xlsx",
        "heart": "../../data/heart_meta.xlsx"
    }

    tissues = [t for t in args.tissues if t in expression_files]
    if not tissues:
        print("Error: No valid tissues specified")
        return

    network_dir = f"../../results/{args.network_dir}"
    os.makedirs(network_dir, exist_ok=True)

    common_params = {
        "intensity_threshold": 3.0,
        "variance_threshold_percentile": 75,
        "fdr_threshold": 0.01,
        "leiden_resolution": 0.2
    }

    networks = {}
    filtered_expr = {}
    clusters = {}

    for tissue in tissues:

        print(f"\n{'='*50}")
        print(f"Processing {tissue.upper()}")
        print(f"{'='*50}")

        if args.load:
            # Load network + expression
            G, expr = load_existing_network(tissue, network_dir)

            if G is None or expr is None:
                print(f"  ✗ Failed to load network or expression for {tissue}")
                continue

            networks[tissue] = G
            filtered_expr[tissue] = expr

            # ---- Load clusters ----
            cluster_file = f"{network_dir}/{tissue}_clusters.csv"
            if os.path.exists(cluster_file):
                print(f"Loading clusters from: {cluster_file}")
                cluster_df = pd.read_csv(cluster_file)
                clusters[tissue] = dict(zip(cluster_df['gene'], cluster_df['module']))
                print(f"  ✓ Loaded {len(clusters[tissue])} cluster assignments")
            else:
                print(f"  ✗ Cluster file not found for {tissue}")
                continue

        else:
            # Build new network
            print(f"Building new network for {tissue}...")
            G, expr = build_coexpression_network(
                expression_files[tissue],
                tissue,
                network_dir,
                intensity_threshold=common_params["intensity_threshold"],
                variance_threshold_percentile=common_params["variance_threshold_percentile"],
                fdr_threshold=common_params["fdr_threshold"],
            )

            networks[tissue] = G
            filtered_expr[tissue] = expr

            # Run Leiden
            print(f"Running Leiden clustering for {tissue}...")
            clusters[tissue] = run_leiden_clustering(
                G,
                resolution_parameter=common_params["leiden_resolution"]
            )

            # Save clusters
            cluster_df = pd.DataFrame({
                'gene': list(clusters[tissue].keys()),
                'module': list(clusters[tissue].values())
            })
            cluster_df.to_csv(f"{network_dir}/{tissue}_clusters.csv", index=False)
            print(f"  ✓ Saved cluster assignments")

        # ---- Visualization ----
        meta = None
        if tissue in meta_files and os.path.exists(meta_files[tissue]):
            meta = read_meta(meta_files[tissue])

        if meta is not None:
            if "character" not in meta.columns and "rej_status" in meta.columns:
                meta["character"] = meta["rej_status"]

            visualize_modules(
                filtered_expr[tissue],
                clusters[tissue],
                meta,
                tissue,
                args.module_dir
            )
        else:
            print(f"Warning: No metadata found for {tissue}")

    print("\nNetwork analysis complete!")


if __name__ == "__main__":
    main()
