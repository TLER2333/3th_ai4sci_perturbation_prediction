from torch.utils.data import DataLoader
import scanpy as sc
import numpy as np
from dataloader import PseudoBulkDataset  
from model import PerturbationModel
from utils import *
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def split_train_by_cell_type(adata, train_cell_types):
    train_mask = adata.obs['cell_type'].isin(train_cell_types)
    adata_train = adata[~train_mask].copy()
    adata_val = adata[train_mask].copy()
    return adata_train, adata_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 42
data_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/train_data.h5ad"
hvg_path = "/bigdat2/user/yuzq/dataset/ai4sci_4th_perturbation/hvg_genes.txt"
epoch = 200

save_dir = "output/task1"
os.makedirs(save_dir, exist_ok=True)

train_losses = []
eval_epochs = []
delta_pcc_list = []

all_cell_types = ['NK cells', 'Dendritic cells', 'CD4 T cells', 'B cells', 'FCGR3A+ Monocytes', 'CD14+ Monocytes', 'CD8 T cells']
train_cell_type = ['Dendritic cells']

# load hvg genes
with open(hvg_path, "r") as f:
    hvg_genes = [line.strip() for line in f.readlines()]
    

adata = sc.read_h5ad(data_path)

train_data, test_data = split_train_by_cell_type(adata, train_cell_type)

train_dataset = PseudoBulkDataset(train_data, hvg_genes=hvg_genes,samples_per_epoch = 200)
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)

test_dataset = PseudoBulkDataset(test_data, hvg_genes=hvg_genes, samples_per_epoch = 20)
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

        pred = model(ctrl)

        loss = ((pred - pert) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()
        

    if ep % 5 == 0:
        model.eval()
        epoch_metrics = defaultdict(list)

        all_data = {
            "ctrl": [],
            "pert": [],
            "pred": [],
            "ctrl_full": [],
            "pert_full": [],
            "cell_type": [],
            "hvg_idx": batch["hvg_idx"]  # 假设所有 batch 相同
        }

        with torch.no_grad():
            for batch in test_dataloader:
                ctrl = batch["ctrl"].to(device)
                pert = batch["pert"].to(device)

                pred = model(ctrl)

                # 累加整个 validation 数据
                all_data["ctrl"].append(ctrl)
                all_data["pert"].append(pert)
                all_data["pred"].append(pred)
                all_data["ctrl_full"].append(batch["ctrl_full"].to(device))
                all_data["pert_full"].append(batch["pert_full"].to(device))
                all_data["cell_type"].extend(batch["cell_type"])

                # 计算 batch metrics
                data_batch = {
                    "ctrl": ctrl,
                    "pert": pert,
                    "pred": pred,
                    "ctrl_full": batch["ctrl_full"].to(device),
                    "pert_full": batch["pert_full"].to(device),
                    "cell_type": batch["cell_type"],
                    "hvg_idx": batch["hvg_idx"]
                }
                batch_res = eval_metrics_simple(data_batch)
                for key, value in batch_res.items():
                    epoch_metrics[key].append(value)

        # 合并所有 batch 的 Tensor
        for k in ["ctrl", "pert", "pred", "ctrl_full", "pert_full"]:
            all_data[k] = torch.cat(all_data[k], dim=0)

        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        eval_epochs.append(ep + 1)
        delta_pcc_list.append(avg_metrics["delta_pcc_hvg"])

        save_path = f"output/umap/epoch{ep+1}.png"
        visualize_umap_hvg(all_data, save_path=save_path)
        # visualize_celltypes_grid(all_data, save_path=save_path)

        print(f"Epoch {ep+1}/{epoch}, Loss: {loss.item():.4f}")
        print(
            f"  [HVG] "
            f"PCC: {avg_metrics['pcc_hvg']:.4f} | "
            f"Delta PCC: {avg_metrics['delta_pcc_hvg']:.4f} | "
            f"Delta MSE: {avg_metrics['delta_mse_hvg']:.4f}"
        )
        print(
            f"  [FULL] "
            f"PCC: {avg_metrics['pcc_full']:.4f} | "
            f"Delta PCC: {avg_metrics['delta_pcc_full']:.4f} | "
            f"Delta MSE: {avg_metrics['delta_mse_full']:.4f}"
        )


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


evaluator = BiologicalEvaluator(
    data_dict=all_data,   # 你的 data_batch 字典
    gene_names=hvg_genes,   # 必须是 list of strings
    result_dir='./output/task1'
)

# 2. 运行 DES 分析 (生成散点图)
# 这一步会告诉你模型对变化最大的那 50 个基因预测得准不准
top_genes = evaluator.analyze_des_correlation(top_n=50)

# 3. 运行通路富集分析 (生成气泡对比图)
# 需要联网下载 GO 库，或者你可以指定本地 GMT 文件
evaluator.analyze_pathway_enrichment(top_n=100)

# 4. 运行细胞特异性分析 (生成热图)
# 展示模型是否捕捉到了“不同细胞系对同一基因反应不同”这一现象
evaluator.analyze_cell_specificity(n_top_genes=20)

