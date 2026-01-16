import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc
from scipy import sparse

class PerturbationDataset(Dataset):
    def __init__(self, adata, 
                 control_key='control', 
                 condition_key='condition', 
                 cell_type_key='cell_type',
                 pct_near_avg=1.0, 
                 hvg_genes=None):

        self.raw_adata = adata

        if hvg_genes is not None:
            self.hvg_idx = [self.raw_adata.var_names.get_loc(g) for g in hvg_genes]
            
            self.adata = self.raw_adata[:, hvg_genes].copy()
        else:
            self.adata = self.raw_adata.copy()

        if pct_near_avg < 1.0:
            self.adata = self._filter_outliers(self.adata, 
                                               cell_type_key, 
                                               condition_key, 
                                               pct_near_avg)

        adata_full = self.raw_adata[self.adata.obs.index, :]
        X_full = adata_full.X
        if sparse.issparse(X_full):
            X_full = X_full.toarray()
        self.X_full = X_full

        X = self.adata.X
        if sparse.issparse(X):
            X = X.toarray()
        obs = self.adata.obs

        ctrl_mask = (obs[condition_key] == control_key).values
        stim_mask = ~ctrl_mask

        self.ctrl_dict = {}
        self.ctrl_dict_full = {}

        unique_ctypes = obs[cell_type_key].unique()
        for ct in unique_ctypes:
            ct_ctrl_mask = (obs[cell_type_key] == ct) & ctrl_mask
            if np.sum(ct_ctrl_mask) > 0:
                self.ctrl_dict[ct] = X[ct_ctrl_mask]
                self.ctrl_dict_full[ct] = self.X_full[ct_ctrl_mask]
            else:
                print(f"Warning: Cell type '{ct}' has no control samples.")

        valid_stim_indices = []
        for i in np.where(stim_mask)[0]:
            ct = obs[cell_type_key].iloc[i]
            if ct in self.ctrl_dict:
                valid_stim_indices.append(i)

        self.stim_data = X[valid_stim_indices]
        self.stim_data_full = self.X_full[valid_stim_indices]
        self.stim_cell_types = obs[cell_type_key].iloc[valid_stim_indices].values
        print("total data shape:", X.shape)
        print(f"Dataset initialized: {len(self.stim_data)} stimulated cells paired with matching controls.")

    def _filter_outliers(self, adata, c_key, cond_key, pct):
        keep_indices = []
        groups = adata.obs.groupby([c_key, cond_key], observed=False)
        for (ct, cond), group_df in groups:
            if len(group_df) == 0: continue
            curr_indices = group_df.index
            X_group = adata[curr_indices].X
            if sparse.issparse(X_group):
                X_group = X_group.toarray()
            mean_vec = X_group.mean(axis=0)
            dists = np.linalg.norm(X_group - mean_vec, axis=1)
            n_keep = max(1, int(len(dists) * pct))
            sorted_local_idx = np.argsort(dists)[:n_keep]
            keep_indices.extend(curr_indices[sorted_local_idx])
        return adata[keep_indices].copy()

    def __len__(self):
        return len(self.stim_data)

    def __getitem__(self, idx):
        cell_type = self.stim_cell_types[idx]
        x_pert = self.stim_data[idx]
        x_pert_full = self.stim_data_full[idx]
        ctrl_pool = self.ctrl_dict[cell_type]
        ctrl_pool_full = self.ctrl_dict_full[cell_type]
        rand_idx = np.random.randint(len(ctrl_pool))
        x_ctrl = ctrl_pool[rand_idx]
        x_ctrl_full = ctrl_pool_full[rand_idx]
        return {
            "ctrl": torch.tensor(x_ctrl, dtype=torch.float32),
            "pert": torch.tensor(x_pert, dtype=torch.float32),
            "ctrl_full": torch.tensor(x_ctrl_full, dtype=torch.float32),
            "pert_full": torch.tensor(x_pert_full, dtype=torch.float32),
            "cell_type": cell_type,
            "hvg_idx": self.hvg_idx
        }
    



class PseudoBulkDataset(Dataset):
    def __init__(self, adata, 
                 control_key='control', 
                 condition_key='condition', 
                 cell_type_key='cell_type',
                 hvg_genes=None,
                 n_cells_per_sample=50,  
                 samples_per_epoch=5000):

        self.raw_adata = adata
        self.n_cells_per_sample = n_cells_per_sample
        self.samples_per_epoch = samples_per_epoch

        if hvg_genes is not None:
            self.hvg_idx = [self.raw_adata.var_names.get_loc(g) for g in hvg_genes]
            self.adata = self.raw_adata[:, hvg_genes].copy()
        else:
            self.adata = self.raw_adata.copy()
            self.hvg_idx = list(range(self.adata.n_vars))
        print("Loading data into memory for fast sampling...")
        
      
        X = self.adata.X
        if sparse.issparse(X): X = X.toarray()
        
        X_full = self.raw_adata.X
        if sparse.issparse(X_full): X_full = X_full.toarray()
        
        obs = self.adata.obs
        unique_ctypes = obs[cell_type_key].unique()
        self.cell_types = unique_ctypes

    
        self.pools = {}
        
        for ct in unique_ctypes:
        
            ct_mask = (obs[cell_type_key] == ct)
            
          
            ctrl_mask = ct_mask & (obs[condition_key] == control_key)
            stim_mask = ct_mask & (obs[condition_key] != control_key)
            
            ctrl_indices = np.where(ctrl_mask)[0]
            stim_indices = np.where(stim_mask)[0]
            
           
            if len(ctrl_indices) > 0 and len(stim_indices) > 0:
                self.pools[ct] = {
                    'ctrl_idx': ctrl_indices,
                    'stim_idx': stim_indices
                }
            else:
                print(f"Warning: Cell type '{ct}' missing control or condition samples. Skipped.")

        self.valid_cell_types = list(self.pools.keys())
        
       
        self.X = X
        self.X_full = X_full
        
        print(f"PseudoBulkDataset Initialized.")
        print(f" - Mode: Level 1 (Population Average Prediction)")
        print(f" - Aggregation: Mean of {n_cells_per_sample} random cells")
        print(f" - Epoch Length: {samples_per_epoch} synthetic samples")
        print(f" - Valid Cell Types: {len(self.valid_cell_types)}")

    def __len__(self):
       
        return self.samples_per_epoch

    def __getitem__(self, idx):

        
        ct_idx = np.random.randint(len(self.valid_cell_types))
        cell_type = self.valid_cell_types[ct_idx]
        
        pool = self.pools[cell_type]
        
      
        idx_ctrl = np.random.choice(pool['ctrl_idx'], 
                                    size=self.n_cells_per_sample, 
                                    replace=True)
        

        idx_stim = np.random.choice(pool['stim_idx'], 
                                    size=self.n_cells_per_sample, 
                                    replace=True)
        
     
        
        # HVG
        x_ctrl_mean = np.mean(self.X[idx_ctrl], axis=0)
        x_pert_mean = np.mean(self.X[idx_stim], axis=0)
        
        # Full Genes
        x_ctrl_full_mean = np.mean(self.X_full[idx_ctrl], axis=0)
        x_pert_full_mean = np.mean(self.X_full[idx_stim], axis=0)


        return {
            "ctrl": torch.tensor(x_ctrl_mean, dtype=torch.float32),
            "pert": torch.tensor(x_pert_mean, dtype=torch.float32),
            "ctrl_full": torch.tensor(x_ctrl_full_mean, dtype=torch.float32),
            "pert_full": torch.tensor(x_pert_full_mean, dtype=torch.float32),
            "cell_type": cell_type,
            "hvg_idx": self.hvg_idx
        }