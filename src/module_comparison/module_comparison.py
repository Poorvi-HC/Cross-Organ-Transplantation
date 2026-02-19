import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn2
import networkx as nx
from collections import defaultdict
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import itertools


class ModuleComparison:

    def __init__(self, tissues=["liver", "kidney", "heart"],
                 output_dir='../../results/module_comparison'):
        self.tissues = tissues
        self.clusters = {}
        self.genes_by_module = {}
        self.conserved_modules = {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------

    def load_data(self):

        for tissue in self.tissues:

            cluster_file = f"../../results/networks/{tissue}_clusters.csv"

            if os.path.exists(cluster_file):
                df = pd.read_csv(cluster_file)

                self.clusters[tissue] = dict(zip(df["gene"], df["module"]))

                self.genes_by_module[tissue] = defaultdict(list)

                for gene, module in self.clusters[tissue].items():
                    self.genes_by_module[tissue][module].append(gene)

            else:
                print(f"Missing cluster file for {tissue}")

        return bool(self.clusters)

    def get_all_genes(self):
        genes = set()
        for t in self.clusters:
            genes.update(self.clusters[t].keys())
        return genes

    # ----------------------------------------------------------
    # GENE OVERLAP
    # ----------------------------------------------------------

    def analyze_gene_overlap(self):

        tissue_genes = {t: set(self.clusters[t].keys())
                        for t in self.clusters}

        if len(tissue_genes) >= 3:
            triple = set.intersection(*tissue_genes.values())
            with open(f"{self.output_dir}/shared_genes_all_tissues.txt", "w") as f:
                for g in sorted(triple):
                    f.write(g + "\n")

        plt.figure(figsize=(8, 6))

        if len(tissue_genes) == 3:
            venn3([tissue_genes[t] for t in self.tissues],
                  set_labels=self.tissues)

        elif len(tissue_genes) == 2:
            venn2([tissue_genes[t] for t in tissue_genes],
                  set_labels=list(tissue_genes.keys()))

        plt.title("Gene Overlap Between Tissues")
        plt.savefig(f"{self.output_dir}/gene_overlap_venn.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    # ----------------------------------------------------------
    # 2-WAY CONSERVED MODULES (Hypergeometric)
    # ----------------------------------------------------------

    def identify_conserved_modules(self,
                                   min_overlap=10,
                                   p_threshold=0.05):

        tissue_pairs = list(itertools.combinations(self.tissues, 2))

        universe = self.get_all_genes()
        N = len(universe)

        module_associations = []
        pvals = []

        for t1, t2 in tissue_pairs:

            if t1 not in self.genes_by_module or \
               t2 not in self.genes_by_module:
                continue

            for m1, g1 in self.genes_by_module[t1].items():
                for m2, g2 in self.genes_by_module[t2].items():

                    g1 = set(g1)
                    g2 = set(g2)

                    overlap = g1.intersection(g2)
                    k = len(overlap)

                    if k >= min_overlap:

                        pval = hypergeom.sf(
                            k - 1,
                            N,
                            len(g1),
                            len(g2)
                        )

                        expected = (len(g1) * len(g2)) / N
                        fold = k / expected if expected > 0 else np.inf

                        module_associations.append({
                            "tissue1": t1,
                            "tissue2": t2,
                            "module1": m1,
                            "module2": m2,
                            "overlap": k,
                            "expected": expected,
                            "fold_enrichment": fold,
                            "genes": list(overlap),
                            "p_value": pval
                        })

                        pvals.append(pval)

        if not pvals:
            return False

        reject, p_adj, _, _ = multipletests(
            pvals, alpha=p_threshold, method="fdr_bh")

        for i, assoc in enumerate(module_associations):
            assoc["p_adjusted"] = p_adj[i]
            assoc["significant"] = reject[i]

        df_all = pd.DataFrame(module_associations)
        df_all.to_csv(f"{self.output_dir}/module_associations_all.csv",
                      index=False)

        df_sig = df_all[df_all["significant"]]

        if df_sig.empty:
            return False

        df_sig.to_csv(
            f"{self.output_dir}/module_associations_significant.csv",
            index=False)

        adj = defaultdict(list)

        for _, a in df_sig.iterrows():
            n1 = f"{a['tissue1']}:{a['module1']}"
            n2 = f"{a['tissue2']}:{a['module2']}"
            adj[n1].append((n2, a.to_dict()))
            adj[n2].append((n1, a.to_dict()))

        visited = set()
        groups = []

        for node in adj:
            if node not in visited:

                queue = [node]
                component = []
                edges = []
                visited.add(node)

                while queue:
                    cur = queue.pop(0)
                    component.append(cur)

                    for neigh, assoc in adj[cur]:
                        edges.append(assoc)
                        if neigh not in visited:
                            visited.add(neigh)
                            queue.append(neigh)

                groups.append({"modules": component,
                               "associations": edges})

        for i, g in enumerate(groups):

            gid = f"conserved_group_{i+1}"
            genes = set()
            tissues = set()

            for a in g["associations"]:
                genes.update(a["genes"])
                tissues.add(a["tissue1"])
                tissues.add(a["tissue2"])

            self.conserved_modules[gid] = {
                "modules": g["modules"],
                "associations": g["associations"],
                "genes": list(genes),
                "tissues": list(tissues),
                "num_tissues": len(tissues),
                "num_modules": len(g["modules"]),
                "num_genes": len(genes)
            }

        self._visualize_conserved_modules()
        return True

    # ----------------------------------------------------------
    # 3-WAY CONSERVED MODULES (Sequential Hypergeometric)
    # ----------------------------------------------------------

    def identify_three_way_conserved_modules(self,
                                             min_overlap=10,
                                             p_threshold=0.05):

        if len(self.tissues) < 3:
            return False

        t1, t2, t3 = self.tissues[:3]

        universe = self.get_all_genes()
        N = len(universe)

        results = []
        pvals = []

        for m1, g1 in self.genes_by_module[t1].items():
            for m2, g2 in self.genes_by_module[t2].items():
                for m3, g3 in self.genes_by_module[t3].items():

                    g1 = set(g1)
                    g2 = set(g2)
                    g3 = set(g3)

                    triple = g1.intersection(g2).intersection(g3)
                    k = len(triple)

                    if k >= min_overlap:

                        g23 = g2.intersection(g3)
                        n23 = len(g23)

                        if n23 == 0:
                            continue

                        pval = hypergeom.sf(
                            k - 1,
                            N,
                            len(g1),
                            n23
                        )

                        expected = (len(g1) * n23) / N
                        fold = k / expected if expected > 0 else np.inf

                        results.append({
                            "module1": m1,
                            "module2": m2,
                            "module3": m3,
                            "triple_overlap": k,
                            "expected": expected,
                            "fold_enrichment": fold,
                            "genes": list(triple),
                            "p_value": pval
                        })

                        pvals.append(pval)

        if not pvals:
            return False

        reject, p_adj, _, _ = multipletests(
            pvals, alpha=p_threshold, method="fdr_bh")

        for i, r in enumerate(results):
            r["p_adjusted"] = p_adj[i]
            r["significant"] = reject[i]

        df_all = pd.DataFrame(results)
        df_all.to_csv(
            f"{self.output_dir}/three_way_module_associations_all.csv",
            index=False)

        df_sig = df_all[df_all["significant"]]

        if df_sig.empty:
            return False

        df_sig.to_csv(
            f"{self.output_dir}/three_way_module_associations_significant.csv",
            index=False)

        for i, row in df_sig.iterrows():
            fname = f"{self.output_dir}/three_way_conserved_genes_{i+1}.txt"
            with open(fname, "w") as f:
                for g in sorted(row["genes"]):
                    f.write(g + "\n")

        return True

    # ----------------------------------------------------------
    # VISUALIZATION
    # ----------------------------------------------------------

    def _visualize_conserved_modules(self):

        if not self.conserved_modules:
            return

        summary = []

        for gid, g in self.conserved_modules.items():

            summary.append({
                "Group_ID": gid,
                "Modules": ", ".join(g["modules"]),
                "Tissues": ", ".join(g["tissues"]),
                "Num_Tissues": g["num_tissues"],
                "Num_Modules": g["num_modules"],
                "Num_Genes": g["num_genes"]
            })

            with open(f"{self.output_dir}/{gid}_genes.txt", "w") as f:
                for gene in sorted(g["genes"]):
                    f.write(gene + "\n")

        pd.DataFrame(summary).to_csv(
            f"{self.output_dir}/conserved_modules_summary.csv",
            index=False)

        self._create_module_overlap_heatmap()
        self._create_conserved_module_network()

    def _create_module_overlap_heatmap(self):

        tissues = [t for t in self.tissues
                   if t in self.genes_by_module]

        all_modules = {}

        for tissue in tissues:
            for module, genes in self.genes_by_module[tissue].items():
                all_modules[f"{tissue}:{module}"] = genes

        names = list(all_modules.keys())
        n = len(names)

        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                gi = set(all_modules[names[i]])
                gj = set(all_modules[names[j]])
                jaccard = len(gi & gj) / len(gi | gj) if gi | gj else 0
                matrix[i, j] = jaccard

        plt.figure(figsize=(max(10, n//4), max(8, n//4)))
        sns.heatmap(matrix,
                    xticklabels=names,
                    yticklabels=names,
                    cmap="YlGnBu",
                    vmin=0, vmax=1)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/module_similarity_heatmap.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_conserved_module_network(self):

        G = nx.Graph()

        colors = {"liver": "#e41a1c",
                  "kidney": "#377eb8",
                  "heart": "#4daf4a"}

        for gid, g in self.conserved_modules.items():
            for module in g["modules"]:
                tissue, mid = module.split(":")
                G.add_node(module,
                           tissue=tissue,
                           group=gid,
                           color=colors.get(tissue, "#999999"))

        for gid, g in self.conserved_modules.items():
            for assoc in g["associations"]:
                n1 = f"{assoc['tissue1']}:{assoc['module1']}"
                n2 = f"{assoc['tissue2']}:{assoc['module2']}"
                G.add_edge(n1, n2,
                           weight=assoc["fold_enrichment"])

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, seed=42)

        for tissue in self.tissues:
            nodes = [n for n, d in G.nodes(data=True)
                     if d["tissue"] == tissue]
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=colors.get(tissue, "#999999"),
                node_size=200,
                label=tissue)

        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.legend()
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/conserved_module_network.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

        nx.write_graphml(G,
                         f"{self.output_dir}/conserved_module_network.graphml")

    # ----------------------------------------------------------

    def run_all_analyses(self,
                         min_overlap=10,
                         p_threshold=0.05):

        if not self.load_data():
            return

        self.analyze_gene_overlap()
        self.identify_conserved_modules(min_overlap, p_threshold)
        self.identify_three_way_conserved_modules(
            min_overlap, p_threshold)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tissues", nargs="+",
                        default=["liver", "kidney", "heart"])
    parser.add_argument("--min-overlap", type=int, default=10)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--module_comp_dir", type=str,
                        default='../../results/module_comparison')

    args = parser.parse_args()

    mc = ModuleComparison(args.tissues,
                          args.module_comp_dir)

    mc.run_all_analyses(args.min_overlap,
                        args.p_threshold)
