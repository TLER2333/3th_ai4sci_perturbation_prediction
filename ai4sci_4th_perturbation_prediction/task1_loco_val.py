import os
import torch
import logging
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader
from collections import defaultdict

from dataloader import PseudoBulkDataset
from model import PerturbationModel
from utils import eval_metrics_simple, visualize_celltypes_grid

# ================= Config =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/train_data.h5ad"
hvg_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/hvg_genes.txt"

save_root = "output/task1"
ckpt_root = "output/task1_LOCO_Experiment"   # 训练时的根目录

os.makedirs(save_root, exist_ok=True)

# ⚠️ 评估时显著增大采样量
VAL_SAMPLES_PER_EPOCH = 2000

all_cell_types = [
    'NK cells', 'Dendritic cells', 'CD4 T cells', 'B cells',
    'FCGR3A+ Monocytes', 'CD14+ Monocytes', 'CD8 T cells'
]


# ================= Logger =================
def setup_logger(save_dir, tag):
    log_file = os.path.join(save_dir, f"{tag}_eval.log")

    logger = logging.getLogger(tag)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger

def split_train_by_cell_type(adata, val_cell_types):
    mask = adata.obs["cell_type"].isin(val_cell_types)
    return adata[~mask].copy(), adata[mask].copy()

# ================= Load Data =================
adata = sc.read_h5ad(data_path)
with open(hvg_path) as f:
    hvg_genes = [l.strip() for l in f]


all_data = {
    "ctrl": [],
    "pert": [],
    "pred": [],
    "ctrl_full": [],
    "pert_full": [],
    "cell_type": [],
    "hvg_idx": None
}


# ================= LOCO Evaluation =================
for test_ct in all_cell_types:

    # ---------- dirs & logger ----------
    save_dir = os.path.join(save_root, test_ct.replace(" ", "_"))
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir, test_ct)
    logger.info(f"========== LOCO Evaluation | Test = {test_ct} ==========")

    # ---------- data ----------
    _, test_data = split_train_by_cell_type(adata, [test_ct])

    test_dataset = PseudoBulkDataset(
        test_data,
        hvg_genes=hvg_genes,
        samples_per_epoch=VAL_SAMPLES_PER_EPOCH
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False
    )

    logger.info(f"Validation samples (pseudo-bulk): {len(test_dataset)}")

    # ---------- model ----------
    ckpt_path = os.path.join(
        ckpt_root,
        test_ct.replace(" ", "_"),
        "model_final.pth"
    )

    model = PerturbationModel(
        input_dim=len(hvg_genes),
        hidden_dim=512
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # ---------- collect all_data ----------


    with torch.no_grad():
        for batch in test_loader:
            ctrl = batch["ctrl"].to(device)
            pert = batch["pert"].to(device)
            pred = model(ctrl)

            all_data["ctrl"].append(ctrl)
            all_data["pert"].append(pert)
            all_data["pred"].append(pred)
            all_data["ctrl_full"].append(batch["ctrl_full"].to(device))
            all_data["pert_full"].append(batch["pert_full"].to(device))
            all_data["cell_type"].extend(batch["cell_type"])

            if all_data["hvg_idx"] is None:
                all_data["hvg_idx"] = batch["hvg_idx"]




    logger.info(f"Appended results for {test_ct}")
    logger.info(f"========== Finished: {test_ct} ==========\n")

for k in ["ctrl", "pert", "pred", "ctrl_full", "pert_full"]:
    all_data[k] = torch.cat(all_data[k], dim=0)
final_fig_path = os.path.join(save_root, "all_celltypes_grid.png")

visualize_celltypes_grid(
    all_data,
    save_path=final_fig_path
)

logger.info(f"Saved final cell-type grid to {final_fig_path}")