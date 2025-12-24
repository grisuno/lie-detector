import os
import glob
import torch
import zipfile
import kagglehub
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops, degree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OPTIMIZED E8 LATTICE LAYER
# ============================================================================

class OptimizedE8Layer(nn.Module):
    """Optimized E8 with caching and efficiency improvements"""
    def __init__(self, in_features, out_features, edge_index, num_nodes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute normalized adjacency once
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
        self.register_buffer('adj', adj)
        
    def forward(self, x):
        x = torch.sparse.mm(self.adj, x)
        return F.linear(x, self.weight)

# ============================================================================
# STREAMLINED ARCHITECTURES
# ============================================================================

class RESMAv2Fast(nn.Module):
    """Fast version: E8 + GAT fusion with minimal overhead"""
    def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Single E8 + GAT per layer
        self.e8 = OptimizedE8Layer(hidden_dim, hidden_dim, edge_index, num_nodes)
        self.gat = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        
        # Simple fusion
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Lightweight readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.embedding(x)
        
        # Single fusion layer
        res = x
        x_e8 = self.e8(x)
        x_gat = self.gat(x, edge_index)
        x = self.fusion(torch.cat([x_e8, x_gat], dim=-1))
        x = F.relu(self.norm(x + res))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return torch.sigmoid(self.readout(x))

class RESMAv2Standard(nn.Module):
    """Standard version: 2 layers of E8 + GAT fusion"""
    def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Two E8 + GAT layers
        self.e8_1 = OptimizedE8Layer(hidden_dim, hidden_dim, edge_index, num_nodes)
        self.gat_1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.fusion_1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm_1 = nn.LayerNorm(hidden_dim)
        
        self.e8_2 = OptimizedE8Layer(hidden_dim, hidden_dim, edge_index, num_nodes)
        self.gat_2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
        self.fusion_2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.embedding(x)
        
        # Layer 1
        res = x
        x_e8 = self.e8_1(x)
        x_gat = self.gat_1(x, edge_index)
        x = self.fusion_1(torch.cat([x_e8, x_gat], dim=-1))
        x = F.relu(self.norm_1(x + res))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        res = x
        x_e8 = self.e8_2(x)
        x_gat = self.gat_2(x, edge_index)
        x = self.fusion_2(torch.cat([x_e8, x_gat], dim=-1))
        x = F.relu(self.norm_2(x + res))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return torch.sigmoid(self.readout(x))

class RESMAv2Deep(nn.Module):
    """Deeper version with 3 layers"""
    def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Three layers
        self.e8_layers = nn.ModuleList([
            OptimizedE8Layer(hidden_dim, hidden_dim, edge_index, num_nodes)
            for _ in range(3)
        ])
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
            for _ in range(3)
        ])
        self.fusion_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(3)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.embedding(x)
        
        for e8, gat, fusion, norm in zip(self.e8_layers, self.gat_layers, 
                                          self.fusion_layers, self.norms):
            res = x
            x_e8 = e8(x)
            x_gat = gat(x, edge_index)
            x = fusion(torch.cat([x_e8, x_gat], dim=-1))
            x = F.relu(norm(x + res))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return torch.sigmoid(self.readout(x))

class GAT_Baseline(nn.Module):
    """Optimized GAT baseline"""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim * 4)
        self.norm2 = nn.LayerNorm(hidden_dim * 4)
        self.readout = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.norm1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.norm2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        return torch.sigmoid(self.readout(x))

# ============================================================================
# DATA LOADING
# ============================================================================

def load_elliptic_data():
    import os, glob, zipfile
    import pandas as pd
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.utils import add_self_loops
    import kagglehub

    print("📥 Descargando Elliptic Data Set...")
    path = kagglehub.dataset_download("ellipticco/elliptic-data-set")

    zip_files = glob.glob(os.path.join(path, "*.zip"))
    if zip_files:
        with zipfile.ZipFile(zip_files[0], "r") as z:
            z.extractall(path)

    csv_files = glob.glob(
        os.path.join(path, "**", "elliptic_txs_features.csv"),
        recursive=True
    )
    if not csv_files:
        raise FileNotFoundError("❌ No se encontró elliptic_txs_features.csv")

    base_dir = os.path.dirname(csv_files[0])

    features_path = os.path.join(base_dir, "elliptic_txs_features.csv")
    classes_path  = os.path.join(base_dir, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(base_dir, "elliptic_txs_edgelist.csv")

    nodes   = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    edges   = pd.read_csv(edgelist_path)

    nodes.columns = ["txId", "timestep"] + [f"feat_{i}" for i in range(1, 166)]
    df = nodes.merge(classes, on="txId", how="left")

    df = df[df["class"].isin(["1", "2"])]
    df["class"] = df["class"].map({"1": 1, "2": 0}).astype(int)

    timestep = df["timestep"].values
    X = df.iloc[:, 2:-1].values
    y = df["class"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    tx_to_idx = {tx: i for i, tx in enumerate(df["txId"])}

    edge_df = edges[
        edges["txId1"].isin(tx_to_idx) &
        edges["txId2"].isin(tx_to_idx)
    ]

    edge_index = torch.tensor(
        [
            [tx_to_idx[tx] for tx in edge_df["txId1"]],
            [tx_to_idx[tx] for tx in edge_df["txId2"]],
        ],
        dtype=torch.long
    )

    edge_index, _ = add_self_loops(edge_index, num_nodes=len(X))

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    print(f"✅ Datos cargados: {len(X)} nodos, {edge_index.shape[1]} aristas")
    print(f"   Ilícitos: {y.sum().item()} ({(y.float().mean()*100):.2f}%)")

    return X, y, edge_index, timestep


# ============================================================================
# FAST TRAINING
# ============================================================================

def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, 
                       epochs=100, lr=0.001, name="Model", fold=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    best_auprc = 0
    best_auroc = 0
    best_f1 = 0
    patience = 0
    max_patience = 15
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(X, edge_index)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(out[train_idx].squeeze(), y[train_idx].float())

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X, edge_index)[val_idx].cpu().numpy()
            val_y = y[val_idx].cpu().numpy()
            
            auprc = average_precision_score(val_y, val_out)
            auroc = roc_auc_score(val_y, val_out)
            f1 = f1_score(val_y, (val_out > 0.5).astype(int))
            
        scheduler.step(auprc)
        
        if auprc > best_auprc:
            best_auprc = auprc
            best_auroc = auroc
            best_f1 = f1
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break
        
        # Progress update every 20 epochs
        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"    [{name} Fold {fold}] Epoch {epoch+1:3d} | "
                  f"AUPRC: {auprc:.4f} | Time: {elapsed:.1f}s", flush=True)


    return best_auprc, best_auroc, best_f1



def temporal_cross_validate_model(
    model_class, X, y, edge_index, timestep, num_nodes,
    name="Model", min_train_ts=3
):
    unique_ts = np.sort(np.unique(timestep))

    auprc_scores = []
    auroc_scores = []
    f1_scores = []

    print(f"  Temporal evaluation over {len(unique_ts)} timesteps...", flush=True)

    for t in unique_ts[min_train_ts:-1]:
        train_idx = np.where(timestep < t)[0]
        val_idx   = np.where(timestep == t)[0]

        if len(val_idx) < 50 or len(train_idx) < 200:
            continue

        train_idx = torch.tensor(train_idx)
        val_idx   = torch.tensor(val_idx)

        # 🔴 AQUÍ ESTÁ EL BUG QUE TENÍAS Y AQUÍ SE ARREGLA
        if model_class is GAT_Baseline:
            model = model_class(X.shape[1], 128)
        else:
            model = model_class(X.shape[1], 128, edge_index, num_nodes)

        auprc, auroc, f1 = train_and_evaluate(
            model, X, y, edge_index,
            train_idx, val_idx,
            name=name, fold=int(t)
        )

        auprc_scores.append(auprc)
        auroc_scores.append(auroc)
        f1_scores.append(f1)

        print(
            f"    ✓ T={t:02d} | AUPRC={auprc:.4f} | AUROC={auroc:.4f} | F1={f1:.4f}",
            flush=True
        )

    return {
        'auprc_mean': np.mean(auprc_scores),
        'auprc_std': np.std(auprc_scores),
        'auroc_mean': np.mean(auroc_scores),
        'auroc_std': np.std(auroc_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores)
    }






# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    print("🚀 RESMA v2: Optimized for Speed")
    print("=" * 70)
    
    X, y, edge_index, timestep = load_elliptic_data()
    num_nodes = X.shape[0]
    
    models = [
        ("GAT Baseline", GAT_Baseline, {}),
        ("RESMA v2 Fast", RESMAv2Fast, {}),
        ("RESMA v2 Standard", RESMAv2Standard, {}),
        ("RESMA v2 Deep", RESMAv2Deep, {}),
    ]
    
    results = []
    
    print("\n🔬 Ejecutando experimentos (5-fold CV)...\n")
    
    overall_start = time.time()
    
    for model_name, model_class, kwargs in models:
        print(f"📊 {model_name}")
        print("-" * 70)
        
        model_start = time.time()
        metrics = temporal_cross_validate_model(
            model_class, X, y, edge_index, timestep, num_nodes, name=model_name
        )

        model_time = time.time() - model_start
        
        results.append({'name': model_name, 'time': model_time, **metrics})
        
        print(f"  ✅ AUPRC: {metrics['auprc_mean']:.4f} ± {metrics['auprc_std']:.4f}")
        print(f"     AUROC: {metrics['auroc_mean']:.4f} ± {metrics['auroc_std']:.4f}")
        print(f"     F1:    {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
        print(f"     Time:  {model_time:.1f}s ({model_time/60:.1f} min)")
        print()
    
    total_time = time.time() - overall_start
    
    results.sort(key=lambda x: x['auprc_mean'], reverse=True)
    
    print("\n" + "=" * 70)
    print("🏆 RESULTADOS FINALES")
    print("=" * 70)
    
    for i, r in enumerate(results, 1):
        marker = "👑" if i == 1 else f"{i}."
        print(f"{marker:3} {r['name']:20} | AUPRC: {r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} | {r['time']/60:.1f}min")
    
    winner = results[0]
    gat = next(r for r in results if 'GAT' in r['name'])
    
    print("\n" + "=" * 70)
    print("📊 ANÁLISIS FINAL")
    print("=" * 70)
    
    improvement = (winner['auprc_mean'] - gat['auprc_mean']) * 100
    
    if winner['name'] != gat['name']:
        print(f"\n🎉 ¡{winner['name']} VENCIÓ A GAT!")
        print(f"   Mejora: +{improvement:.2f}%")
        print(f"   AUPRC: {winner['auprc_mean']:.4f} vs {gat['auprc_mean']:.4f}")
    else:
        print(f"\n😅 GAT sigue siendo el rey...")
        print(f"   Pero el mejor RESMA alcanzó: {results[1]['auprc_mean']:.4f}")
        print(f"   Diferencia: {abs((gat['auprc_mean'] - results[1]['auprc_mean'])*100):.2f}%")
    
    print(f"\n⏱️  Tiempo total: {total_time/60:.1f} minutos")
    print("\n✅ Experimento completado!")
