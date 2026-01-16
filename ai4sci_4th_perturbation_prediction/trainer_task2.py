import os
import torch
import logging
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict

# 引入你自己的模块
from dataloader import PerturbationDataset  
from model import PerturbationModel
from utils import *

def setup_logger(save_dir, cell_type):
    """配置 Logger，同时输出到文件和控制台"""
    log_file = os.path.join(save_dir, f"{cell_type.replace(' ', '_')}_train.log")
    
    # 创建 logger
    logger = logging.getLogger(cell_type)
    logger.setLevel(logging.INFO)
    logger.handlers = [] # 清空之前的 handler 防止重复打印

    # File Handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
random_seed = 42
data_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/train_data.h5ad"
hvg_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/hvg_genes.txt"
epoch = 300

save_dir = "output/task2"
os.makedirs(save_dir, exist_ok=True)
logger = setup_logger(save_dir, "all_celltypes")
train_losses = []
eval_epochs = []
delta_pcc_list = []


# load hvg genes
with open(hvg_path, "r") as f:
    hvg_genes = [line.strip() for line in f.readlines()]
    
adata = sc.read_h5ad(data_path)

# 取split == train的数据
train_data = adata[adata.obs["split"] == "train"]
test_data = adata[adata.obs["split"] == "test"]


train_dataset = PerturbationDataset(train_data, pct_near_avg=1, hvg_genes=hvg_genes)
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)

test_dataset = PerturbationDataset(test_data, pct_near_avg=1, hvg_genes=hvg_genes)
test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

print(len(train_dataset))
print(len(test_dataset))

model = PerturbationModel(input_dim=len(hvg_genes), hidden_dim=512)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
model.to(device)
for ep in range(epoch):
    model.train()
    for batch in train_dataloader:
        ctrl = batch["ctrl"].to(device)
        pert = batch["pert"].to(device)
        cell_types = np.array(batch['cell_type']) # 转为 numpy 方便操作
        
        pred = model(ctrl)

        loss_mse = ((pred - pert) ** 2).mean() 
        loss_mmd = 0.0
        unique_cts = np.unique(cell_types)
        valid_groups = 0
    
        for ct in unique_cts:
            # 生成掩码：找出当前 batch 里属于这个细胞系的样本
            mask = (cell_types == ct)
            
            # MMD 至少需要 2 个样本才能计算方差/分布
            if np.sum(mask) > 1:
                # 提取属于该细胞系的预测值和真实值
                pred_subset = pred[mask]
                target_subset = pert[mask]
                
                # 计算该细胞系的 MMD 并累加
                loss_mmd += loss_MMD(pred_subset, target_subset)
                valid_groups += 1
            
        if valid_groups > 0:
            loss_mmd = loss_mmd / valid_groups  # 取平均
            
        # 5. 总 Loss
        total_loss = loss_mse + 0 * loss_mmd

        optimizer.zero_grad()
        total_loss.backward()
        train_losses.append(total_loss.item())
        optimizer.step()
    

    if ep % 5 == 0:
        model.eval()
        epoch_metrics = defaultdict(list)

        all_data = {
            "ctrl": [], "pert": [], "pred": [],
            "ctrl_full": [], "pert_full": [],
            "cell_type": [], "hvg_idx": None
        }

        with torch.no_grad():
            for batch in test_dataloader:
                ctrl = batch["ctrl"].to(device)
                pert = batch["pert"].to(device)
                pred = model(ctrl)

                # ===== 收集数据 =====
                all_data["ctrl"].append(ctrl)
                all_data["pert"].append(pert)
                all_data["pred"].append(pred)
                all_data["ctrl_full"].append(batch["ctrl_full"].to(device))
                all_data["pert_full"].append(batch["pert_full"].to(device))
                all_data["cell_type"].extend(batch["cell_type"])

                if all_data["hvg_idx"] is None:
                    all_data["hvg_idx"] = batch["hvg_idx"]

                # ===== batch metrics =====
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

        # ===== 平均指标 =====
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        eval_epochs.append(ep + 1)
        delta_pcc_list.append(avg_metrics["delta_pcc_hvg"])

        logger.info(
            f"Epoch {ep+1}/{epoch} | "
            f"Loss: {total_loss:.4f} | "
            f"[HVG] "
            f"PCC: {avg_metrics['pcc_hvg']:.4f}, "
            f"SCC: {avg_metrics['spearman_hvg']:.4f}, "
            f"R2: {avg_metrics['r2_hvg']:.4f} | "
            f"[Δ] "
            f"PCC: {avg_metrics['delta_pcc_hvg']:.4f}, "
            f"SCC: {avg_metrics['delta_spearman_hvg']:.4f}, "
            f"MSE: {avg_metrics['delta_mse_hvg']:.4f}"
        )
 
        for k in ["ctrl", "pert", "pred", "ctrl_full", "pert_full"]:
            all_data[k] = torch.cat(all_data[k], dim=0)
        last_all_data = all_data

        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        eval_epochs.append(ep + 1)
        delta_pcc_list.append(avg_metrics["delta_pcc_hvg"])

        # ===== 一次性可视化整个 validation =====
        
        # save_path = f"output/umap_task2/epoch{ep+1}.png"
        save_path = f"output/plot_test/epoch{ep+1}.png"
        # visualize_umap_hvg(all_data, save_path=save_path)
        # visualize_celltypes_grid(last_all_data, save_path=save_path)


model_save_path = os.path.join(save_dir, "model_final.pth")
torch.save(model.state_dict(), model_save_path)
logger.info(f"Model saved to {model_save_path}")

# -------- Loss 曲线 --------
plt.figure()
plt.plot(range(1, len(train_losses) + 1), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss (MSE)")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "training_loss.png"))
plt.close()

# -------- Delta PCC 曲线 --------
plt.figure()
plt.plot(eval_epochs, delta_pcc_list, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Delta PCC")
plt.title("Evaluation Delta PCC Curve")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "delta_pcc.png"))
plt.close()


    
current_save_dir = save_dir

save_path1 = os.path.join(save_dir, "umap_hvg_final.png")
save_path2 = os.path.join(save_dir, "umap_celltypes_final.png")

visualize_umap_hvg(last_all_data, save_path=save_path1)
visualize_celltypes_grid(last_all_data, save_path=save_path2)

logger.info("Final UMAP visualizations saved.")


evaluator = BiologicalEvaluator(
    data_dict=last_all_data,
    gene_names=hvg_genes,
    result_dir=current_save_dir # 结果保存在当前细胞系文件夹下
)

# 1. DES 分析 (关键！)
evaluator.analyze_des_correlation(top_n=50)

# 2. 通路富集分析 (关键！)
evaluator.analyze_pathway_enrichment(top_n=100)

# 3. 细胞特异性分析 (SKIPPED)
# 原因：Test set 只有一个细胞系，无法画出 heatmap 对比不同细胞系。
logger.info("Skipping analyze_cell_specificity (Test set contains only single cell type).")

logger.info(f"========== Finished==========\n")