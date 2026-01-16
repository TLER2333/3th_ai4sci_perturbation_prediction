import os
import torch
import logging
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader
from collections import defaultdict

# ===== 自定义模块 =====
from dataloader import PerturbationDataset
from model import PerturbationModel
from utils import eval_metrics_simple

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

# ================= Config =================
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
data_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/train_data.h5ad"
hvg_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/hvg_genes.txt"
ckpt_path = "/home/yuzq/project2/ai4sci_4th/output/task2/model_final.pth"  

save_dir = "output/task2_eval"
os.makedirs(save_dir, exist_ok=True)
logger = setup_logger(save_dir, "celltype")

# ================= Load HVG =================
with open(hvg_path, "r") as f:
    hvg_genes = [line.strip() for line in f]

# ================= Load Data =================
adata = sc.read_h5ad(data_path)
test_data = adata[adata.obs["split"] == "test"]

test_dataset = PerturbationDataset(
    test_data,
    pct_near_avg=1,
    hvg_genes=hvg_genes
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False
)

logger.info(f"Test samples: {len(test_dataset)}")

# ================= Load Model =================
model = PerturbationModel(
    input_dim=len(hvg_genes),
    hidden_dim=512
)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device)
model.eval()

logger.info(f"Loaded checkpoint: {ckpt_path}")
logger.info("========== Start cell-type-wise evaluation ==========")

# ================= Collect by Cell Type =================
celltype_data = defaultdict(lambda: {
    "ctrl": [],
    "pert": [],
    "pred": [],
    "ctrl_full": [],
    "pert_full": [],
    "cell_type": [],
    "hvg_idx": None
})

with torch.no_grad():
    for batch in test_dataloader:
        ctrl = batch["ctrl"].to(device)
        pert = batch["pert"].to(device)
        pred = model(ctrl)

        cell_types = np.array(batch["cell_type"])

        for ct in np.unique(cell_types):
            mask = (cell_types == ct)

            celltype_data[ct]["ctrl"].append(ctrl[mask])
            celltype_data[ct]["pert"].append(pert[mask])
            celltype_data[ct]["pred"].append(pred[mask])
            celltype_data[ct]["ctrl_full"].append(
                batch["ctrl_full"][mask].to(device)
            )
            celltype_data[ct]["pert_full"].append(
                batch["pert_full"][mask].to(device)
            )
            celltype_data[ct]["cell_type"].extend([ct] * int(mask.sum()))

            if celltype_data[ct]["hvg_idx"] is None:
                celltype_data[ct]["hvg_idx"] = batch["hvg_idx"]

# ================= Per-Cell-Type Metrics =================
logger.info("========== Per Cell Type Results ==========")

for ct, data in celltype_data.items():

    # concat
    for k in ["ctrl", "pert", "pred", "ctrl_full", "pert_full"]:
        data[k] = torch.cat(data[k], dim=0)

    metrics = eval_metrics_simple(data)

    logger.info(
        f"[Cell Type: {ct}] "
        f"HVG | "
        f"PCC: {metrics['pcc_hvg']:.4f}, "
        f"SCC: {metrics['spearman_hvg']:.4f}, "
        f"R2: {metrics['r2_hvg']:.4f} || "
        f"Δ | "
        f"PCC: {metrics['delta_pcc_hvg']:.4f}, "
        f"SCC: {metrics['delta_spearman_hvg']:.4f}, "
        f"MSE: {metrics['delta_mse_hvg']:.4f}"
    )

# ================= Overall Metrics =================
logger.info("========== Overall Results ==========")

all_data = {
    "ctrl": torch.cat([v["ctrl"] for v in celltype_data.values()], dim=0),
    "pert": torch.cat([v["pert"] for v in celltype_data.values()], dim=0),
    "pred": torch.cat([v["pred"] for v in celltype_data.values()], dim=0),
    "ctrl_full": torch.cat([v["ctrl_full"] for v in celltype_data.values()], dim=0),
    "pert_full": torch.cat([v["pert_full"] for v in celltype_data.values()], dim=0),
    "cell_type": sum([v["cell_type"] for v in celltype_data.values()], []),
    "hvg_idx": next(iter(celltype_data.values()))["hvg_idx"]
}

overall_metrics = eval_metrics_simple(all_data)

logger.info(
    f"[Overall] "
    f"HVG | "
    f"PCC: {overall_metrics['pcc_hvg']:.4f}, "
    f"SCC: {overall_metrics['spearman_hvg']:.4f}, "
    f"R2: {overall_metrics['r2_hvg']:.4f} || "
    f"Δ | "
    f"PCC: {overall_metrics['delta_pcc_hvg']:.4f}, "
    f"SCC: {overall_metrics['delta_spearman_hvg']:.4f}, "
    f"MSE: {overall_metrics['delta_mse_hvg']:.4f}"
)

logger.info("========== Evaluation Finished ==========")
