import os
import torch
import logging
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict


from dataloader import PseudoBulkDataset  
from model import PerturbationModel
from utils import * 

# level1 : train in 7 cell types, test in held-out cell type
# for each cell type, do one training and evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 42
data_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/train_data.h5ad"
hvg_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/hvg_genes.txt"


save_root = "output/task1_LOCO_Experiment" 
os.makedirs(save_root, exist_ok=True)

epoch = 200
TRAIN_SAMPLES_PER_EPOCH = 2000 
TEST_SAMPLES_PER_EPOCH = 200

all_cell_types = [
    'NK cells', 'Dendritic cells', 'CD4 T cells', 'B cells', 
    'FCGR3A+ Monocytes', 'CD14+ Monocytes', 'CD8 T cells'
]



def setup_logger(save_dir, cell_type):
    log_file = os.path.join(save_dir, f"{cell_type.replace(' ', '_')}_train.log")
    

    logger = logging.getLogger(cell_type)
    logger.setLevel(logging.INFO)
    logger.handlers = [] 

    # File Handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger

def split_train_by_cell_type(adata, val_cell_types):

    val_mask = adata.obs['cell_type'].isin(val_cell_types)
    adata_train = adata[~val_mask].copy()
    adata_val = adata[val_mask].copy()
    return adata_train, adata_val


print(f"Loading raw data from {data_path}...")
adata = sc.read_h5ad(data_path)
with open(hvg_path, "r") as f:
    hvg_genes = [line.strip() for line in f.readlines()]
print("Data loaded.")


for test_ct in all_cell_types:

    current_save_dir = os.path.join(save_root, test_ct.replace(' ', '_'))
    os.makedirs(current_save_dir, exist_ok=True)
    
    
    logger = setup_logger(current_save_dir, test_ct)
    logger.info(f"========== Start Training: Test Set = {test_ct} ==========")


    train_data, test_data = split_train_by_cell_type(adata, [test_ct])
    
    logger.info(f"Train Cells: {train_data.n_obs}, Test Cells ({test_ct}): {test_data.n_obs}")


    train_dataset = PseudoBulkDataset(
        train_data, hvg_genes=hvg_genes, 
        samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH
    )
    test_dataset = PseudoBulkDataset(
        test_data, hvg_genes=hvg_genes, 
        samples_per_epoch=TEST_SAMPLES_PER_EPOCH
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)


    model = PerturbationModel(input_dim=len(hvg_genes), hidden_dim=512)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


    train_losses = []
    eval_epochs = []
    delta_pcc_list = []
    

    last_all_data = None 

    for ep in range(epoch):
        # Train Step
        model.train()
        batch_losses = []
        for batch in train_dataloader:
            ctrl = batch["ctrl"].to(device)
            pert = batch["pert"].to(device)

            pred = model(ctrl)
            loss = ((pred - pert) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)

        # Eval Step
        if (ep + 1) % 5 == 0:
            model.eval()
            epoch_metrics = defaultdict(list)
         
            temp_all_data = {
                "ctrl": [], "pert": [], "pred": [],
                "ctrl_full": [], "pert_full": [],
                "cell_type": [], "hvg_idx": None
            }

            with torch.no_grad():
                for batch in test_dataloader:
                    ctrl = batch["ctrl"].to(device)
                    pert = batch["pert"].to(device)
                    pred = model(ctrl)

                 
                    temp_all_data["ctrl"].append(ctrl)
                    temp_all_data["pert"].append(pert)
                    temp_all_data["pred"].append(pred)
                    temp_all_data["ctrl_full"].append(batch["ctrl_full"].to(device))
                    temp_all_data["pert_full"].append(batch["pert_full"].to(device))
                    temp_all_data["cell_type"].extend(batch["cell_type"])

                    if temp_all_data["hvg_idx"] is None:
                        temp_all_data["hvg_idx"] = batch["hvg_idx"]

                    # ===== batch-level metrics =====
                    data_batch = {
                        "ctrl": ctrl,
                        "pert": pert,
                        "pred": pred,
                        "ctrl_full": batch["ctrl_full"].to(device),
                        "pert_full": batch["pert_full"].to(device),
                        "cell_type": batch["cell_type"],
                        "hvg_idx": batch["hvg_idx"],
                    }

                    batch_res = eval_metrics_simple(data_batch)

                    for k, v in batch_res.items():
                        epoch_metrics[k].append(v)

            
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            eval_epochs.append(ep + 1)
            delta_pcc_list.append(avg_metrics["delta_pcc_hvg"])
            
            logger.info(
                f"Epoch {ep+1}/{epoch} | "
                f"Loss: {epoch_loss:.4f} | "
                f"[HVG] "
                f"PCC: {avg_metrics['pcc_hvg']:.4f}, "
                f"SCC: {avg_metrics['spearman_hvg']:.4f}, "
                f"R2: {avg_metrics['r2_hvg']:.4f} | "
                f"[Δ] "
                f"PCC: {avg_metrics['delta_pcc_hvg']:.4f}, "
                f"SCC: {avg_metrics['delta_spearman_hvg']:.4f}, "
                f"MSE: {avg_metrics['delta_mse_hvg']:.4f}"
            )           

           
            if (ep + 1) == epoch: 
                for k in ["ctrl", "pert", "pred", "ctrl_full", "pert_full"]:
                    temp_all_data[k] = torch.cat(temp_all_data[k], dim=0)
                last_all_data = temp_all_data


    

    model_save_path = os.path.join(current_save_dir, "model_final.pth")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    fig_save_path = os.path.join(current_save_dir, "final_umap.png")
    visualize_umap_hvg(temp_all_data, fig_save_path)

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.title(f"Training Loss - Test: {test_ct}")
    plt.tight_layout()
    plt.savefig(os.path.join(current_save_dir, "training_loss.png"))
    plt.close()

    # Delta PCC Curve
    plt.figure()
    plt.plot(eval_epochs, delta_pcc_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Delta PCC")
    plt.title(f"Delta PCC - Test: {test_ct}")
    plt.tight_layout()
    plt.savefig(os.path.join(current_save_dir, "delta_pcc.png"))
    plt.close()


    if last_all_data is not None:
        logger.info("Running Biological Evaluation...")
        
      
        evaluator = BiologicalEvaluator(
            data_dict=last_all_data,
            gene_names=hvg_genes,
            result_dir=current_save_dir 
        )


        evaluator.analyze_des_correlation(top_n=50)


        evaluator.analyze_pathway_enrichment(top_n=100)

        logger.info("Skipping analyze_cell_specificity (Test set contains only single cell type).")

    logger.info(f"========== Finished: {test_ct} ==========\n")