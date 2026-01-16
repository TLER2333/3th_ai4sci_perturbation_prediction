import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import rankdata
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
import math
import seaborn as sns
from scipy.stats import pearsonr
import gseapy as gp 

class BiologicalEvaluator:
    def __init__(self, data_dict, gene_names, result_dir='./output'):
        """
        初始化评估器
        :param data_dict: 包含 'ctrl', 'pert', 'pred', 'cell_type' 的字典 (Tensor or Numpy)
        :param gene_names: 长度为 1000 的 list，对应 HVG 的基因名称 (用于富集分析)
        :param result_dir: 结果保存路径
        """
        self.gene_names = np.array(gene_names)
        self.result_dir = result_dir
        
        
        self.ctrl = self._to_numpy(data_dict['ctrl'])
        self.pert = self._to_numpy(data_dict['pert'])
        self.pred = self._to_numpy(data_dict['pred'])
        self.cell_types = np.array(data_dict['cell_type'])
        

        self.true_lfc = self.pert - self.ctrl
        self.pred_lfc = self.pred - self.ctrl
        
 
        self.df_lfc_true = pd.DataFrame(self.true_lfc, columns=self.gene_names)
        self.df_lfc_true['cell_type'] = self.cell_types
        
        self.df_lfc_pred = pd.DataFrame(self.pred_lfc, columns=self.gene_names)
        self.df_lfc_pred['cell_type'] = self.cell_types

    def _to_numpy(self, data):
        if hasattr(data, 'detach'):
            return data.detach().cpu().numpy()
        return np.array(data)

 
    def analyze_des_correlation(self, top_n=50):

      
        mean_lfc_true = self.df_lfc_true.drop(columns='cell_type').mean(axis=0)
        mean_lfc_pred = self.df_lfc_pred.drop(columns='cell_type').mean(axis=0)
        
     
        top_indices = np.argsort(np.abs(mean_lfc_true))[-top_n:]
        top_genes = self.gene_names[top_indices]
        
        y_true = mean_lfc_true.values[top_indices]
        y_pred = mean_lfc_pred.values[top_indices]
        

        pcc, _ = pearsonr(y_true, y_pred)
        
     
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, c='blue', alpha=0.6, edgecolors='w', s=80)
        

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Fit')
        
        plt.title(f"LFC Correlation (Top {top_n} DEGs)\nPCC = {pcc:.4f}")
        plt.xlabel("True Log Fold Change")
        plt.ylabel("Predicted Log Fold Change")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        
        save_path = f"{self.result_dir}/des_correlation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[DES] Top {top_n} DEGs Correlation: {pcc:.4f}")
        
        return top_genes 

    def analyze_pathway_enrichment(self, top_genes_list=None, top_n=100):

        print("[Enrichment] Running GSEA (this may take a moment)...")
        
      
        mean_lfc_true = self.df_lfc_true.drop(columns='cell_type').mean(axis=0)
        mean_lfc_pred = self.df_lfc_pred.drop(columns='cell_type').mean(axis=0)
     
        genes_true = mean_lfc_true.nlargest(top_n).index.tolist()
        genes_pred = mean_lfc_pred.nlargest(top_n).index.tolist()
        
        db = 'GO_Biological_Process_2021' 
        
        try:
            res_true = gp.enrichr(gene_list=genes_true, gene_sets=db, outdir=None).res2d
            res_pred = gp.enrichr(gene_list=genes_pred, gene_sets=db, outdir=None).res2d
        except Exception as e:
            print(f"GSEA Error: {e}")
            return

    
        res_true = res_true[res_true['Adjusted P-value'] < 0.05].head(10)
        res_pred = res_pred[res_pred['Adjusted P-value'] < 0.05].head(10)
        
        if res_true.empty or res_pred.empty:
            print("No significant pathways found.")
            return

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
        

        def plot_dot(df, ax, title):
            if df.empty: return
            df['-log10(P-adj)'] = -np.log10(df['Adjusted P-value'])
            sns.scatterplot(
                data=df, x='-log10(P-adj)', y='Term', 
                size='Overlap', hue='-log10(P-adj)', 
                sizes=(50, 400), palette='viridis', ax=ax, legend=False
            )
            ax.set_title(title)
            ax.set_xlabel("-log10(Adj. P-value)")
            ax.set_ylabel("")

        plot_dot(res_true, ax[0], "Enriched Pathways (Ground Truth)")
        plot_dot(res_pred, ax[1], "Enriched Pathways (Predicted)")
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/pathway_comparison.png", dpi=300)
        plt.show()

    def analyze_cell_specificity(self, n_top_genes=20):
        avg_lfc_true = self.df_lfc_true.groupby('cell_type').mean()
        avg_lfc_pred = self.df_lfc_pred.groupby('cell_type').mean()
      
        gene_vars = avg_lfc_true.var(axis=0)
        top_var_genes = gene_vars.nlargest(n_top_genes).index
     
        plot_data_true = avg_lfc_true[top_var_genes].T 
        plot_data_pred = avg_lfc_pred[top_var_genes].T
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
      
        vmin = min(plot_data_true.min().min(), plot_data_pred.min().min())
        vmax = max(plot_data_true.max().max(), plot_data_pred.max().max())
        
        sns.heatmap(plot_data_true, ax=axes[0], cmap="RdBu_r", center=0, 
                    vmin=vmin, vmax=vmax, cbar=False, annot=False)
        axes[0].set_title("True LFC per Cell Type")
        axes[0].set_xlabel("Cell Type")
        axes[0].set_ylabel("Genes (High Variance)")
        
        sns.heatmap(plot_data_pred, ax=axes[1], cmap="RdBu_r", center=0, 
                    vmin=vmin, vmax=vmax, cbar=True, annot=False)
        axes[1].set_title("Predicted LFC per Cell Type")
        axes[1].set_xlabel("Cell Type")
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/cell_specificity_heatmap.png", dpi=300)
        plt.show()

def batch_pearson_corr(A, B):
    A_mean = A - A.mean(dim=1, keepdim=True)
    B_mean = B - B.mean(dim=1, keepdim=True)
    
   
    covariance = (A_mean * B_mean).sum(dim=1)
    
    A_std = torch.sqrt((A_mean**2).sum(dim=1))
    B_std = torch.sqrt((B_mean**2).sum(dim=1))
    

    denominator = A_std * B_std
    denominator[denominator == 0] = 1e-6 
    
    pearson_corr = covariance / denominator
    return pearson_corr

def batch_spearman_corr(A, B):

    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()

    from scipy.stats import rankdata

    scc = []
    for a, b in zip(A, B):
        ra = rankdata(a)
        rb = rankdata(b)
        scc.append(np.corrcoef(ra, rb)[0, 1])

    return torch.tensor(scc, dtype=torch.float32)

def batch_r2(pred, target, eps=1e-8):

    ss_res = ((target - pred) ** 2).sum(dim=1)
    ss_tot = ((target - target.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    return 1 - ss_res / (ss_tot + eps)


def batch_mse(A, B):
    return ((A - B) ** 2).mean(dim=1)


def loss_mse(pred, target):
    return ((pred - target) ** 2).mean()

def loss_MMD(pred, target, blur=0.1):
    energy_loss = SamplesLoss(loss="energy", blur=blur)
    return energy_loss(pred, target)


# calc pcc Δpcc mse between pred and target
def eval_metrics(data):


    # ========= 1. HVG level =========
    ctrl = data["ctrl"]
    pert = data["pert"]
    pred = data["pred"]

    # PCC
    pcc_hvg = batch_pearson_corr(pred, pert)

    # Delta
    delta_pred = pred - ctrl
    delta_pert = pert - ctrl

    delta_pcc_hvg = batch_pearson_corr(delta_pred, delta_pert)
    delta_mse_hvg = batch_mse(delta_pred, delta_pert)

    # ========= 2. Full-gene level =========
    ctrl_full = data["ctrl_full"]
    pert_full = data["pert_full"]


    # ---- 构造 pred_full ----
    # 默认：所有基因 = ctrl_full
    pred_full = ctrl_full.clone()
    # use hvg_genes
    # print(data["hvg_idx"])
    hvg_idx = torch.cat(data['hvg_idx']).unique()   
    pred_full[:, hvg_idx] = pred


    # pcc 
    pcc_full = batch_pearson_corr(pred_full, pert_full)

    # ---- Delta (full) ----
    delta_pred_full = pred_full - ctrl_full
    delta_pert_full = pert_full - ctrl_full

    delta_pcc_full = batch_pearson_corr(delta_pred_full, delta_pert_full)
    delta_mse_full = batch_mse(delta_pred_full, delta_pert_full)


    return {
        # HVG
        "pcc_hvg": torch.mean(pcc_hvg).item(),
        "delta_pcc_hvg": torch.mean(delta_pcc_hvg).item(),
        "delta_mse_hvg": torch.mean(delta_mse_hvg).item(),

        # Full gene
        "pcc_full": torch.mean(pcc_full).item(),
        "delta_pcc_full": torch.mean(delta_pcc_full).item(),
        "delta_mse_full": torch.mean(delta_mse_full).item(),
    }


def eval_metrics_simple(data):
    ctrl = data["ctrl"]
    pert = data["pert"]
    pred = data["pred"]

    delta_pred = pred - ctrl
    delta_pert = pert - ctrl

    
    mse_hvg = batch_mse(pred, pert)
    pcc_hvg = batch_pearson_corr(pred, pert)
    delta_pcc_hvg = batch_pearson_corr(delta_pred, delta_pert)
    delta_mse_hvg = batch_mse(delta_pred, delta_pert)

    r2_hvg = batch_r2(pred, pert)
    spearman_hvg = batch_spearman_corr(pert, pred)
    delta_spearman_hvg = batch_spearman_corr(delta_pert, delta_pred)
    return {
        # HVG
        "mse_hvg": torch.mean(mse_hvg).item(),
        "pcc_hvg": torch.mean(pcc_hvg).item(),
        "delta_pcc_hvg": torch.mean(delta_pcc_hvg).item(),
        "delta_mse_hvg": torch.mean(delta_mse_hvg).item(),
        "r2_hvg": torch.mean(r2_hvg).item(),
        "spearman_hvg": torch.mean(spearman_hvg).item(),
        "delta_spearman_hvg":torch.mean(delta_spearman_hvg).item(),

        "pcc_full": torch.mean(pcc_hvg).item(),
        "delta_pcc_full": torch.mean(delta_pcc_hvg).item(),
        "delta_mse_full": torch.mean(delta_mse_hvg).item(),
    }


    


def visualize_umap_hvg(data, save_path=None, n_neighbors=15, min_dist=0.3):


    ctrl = data["ctrl"].detach().cpu().numpy()
    pert = data["pert"].detach().cpu().numpy()
    pred = data["pred"].detach().cpu().numpy()

    B = ctrl.shape[0]

    X = np.concatenate([ctrl, pert, pred], axis=0)

    labels = (
        ["Control"] * B
        + ["Perturbed"] * B
        + ["Predicted"] * B
    )

    cell_types = data.get("cell_type", ["Unknown"] * B)
    cell_types = cell_types * 3

    adata = sc.AnnData(X)
    adata.obs["group"] = labels
    adata.obs["cell_type"] = cell_types


    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist)

    plt.figure(figsize=(6, 5))
    sc.pl.umap(
        adata,
        color="group",
        size=20,
        alpha=0.8,
        frameon=False,
        show=False
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # plt.show()



def visualize_celltypes_grid(data, save_path="umap_grid.png", n_cols=3):

    ctrl = data["ctrl"].detach().cpu().numpy()
    pert = data["pert"].detach().cpu().numpy()
    pred = data["pred"].detach().cpu().numpy()
    cell_types = np.array(data["cell_type"])
    
    unique_cts = np.unique(cell_types)
    n_cts = len(unique_cts)
    
  
    n_rows = math.ceil(n_cts / n_cols)
    
   
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()  
    
    print(f"Plotting {n_cts} cell types in a {n_rows}x{n_cols} grid...")


    for i, ct in enumerate(unique_cts):
        ax = axes[i]
      
        idx = np.where(cell_types == ct)[0]
        
        c_sub = ctrl[idx]
        p_sub = pert[idx]
        pr_sub = pred[idx]
    
        X_sub = np.concatenate([c_sub, p_sub, pr_sub], axis=0)
        
      
        n_sub = len(idx)
        labels = ["Control"] * n_sub + ["True Pert"] * n_sub + ["Predicted"] * n_sub
        
      
        adata_sub = sc.AnnData(X_sub)
        adata_sub.obs["Condition"] = labels
        

        sc.pp.pca(adata_sub, n_comps=min(50, len(X_sub) - 1)) 
        sc.pp.neighbors(adata_sub, n_neighbors=min(15, len(X_sub) - 1))
        sc.tl.umap(adata_sub)
        
    
        sc.pl.umap(
            adata_sub,
            color="Condition",
            ax=ax,                  
            show=False,             
            frameon=False,          
            title=f"{ct} (n={n_sub})", 
            legend_loc="right margin", 
            palette={"Control": "gray", "True Pert": "blue", "Predicted": "red"}, 
            alpha=0.7,
            size=50
        )
        
        
        if i != n_cols - 1: 
             ax.legend().set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.close()