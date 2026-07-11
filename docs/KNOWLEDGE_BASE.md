# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 3 | **Total Symbols Extracted:** 36 | **Total Imports:** 40

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
    app_timestamp_py["app_timestamp.py (py)"]
    class app_timestamp_py mod;
    app_timestamp_py_OptimizedE8Layer["OptimizedE8Layer"]
    class app_timestamp_py_OptimizedE8Layer cls;
    app_timestamp_py --> app_timestamp_py_OptimizedE8Layer
    app_timestamp_py_RESMAv2Fast["RESMAv2Fast"]
    class app_timestamp_py_RESMAv2Fast cls;
    app_timestamp_py --> app_timestamp_py_RESMAv2Fast
    app_timestamp_py_RESMAv2Standard["RESMAv2Standard"]
    class app_timestamp_py_RESMAv2Standard cls;
    app_timestamp_py --> app_timestamp_py_RESMAv2Standard
    app_timestamp_py_RESMAv2Deep["RESMAv2Deep"]
    class app_timestamp_py_RESMAv2Deep cls;
    app_timestamp_py --> app_timestamp_py_RESMAv2Deep
    app_timestamp_py_GAT_Baseline["GAT_Baseline"]
    class app_timestamp_py_GAT_Baseline cls;
    app_timestamp_py --> app_timestamp_py_GAT_Baseline
    app_py["app.py (py)"]
    class app_py mod;
    app_py_OptimizedE8Layer["OptimizedE8Layer"]
    class app_py_OptimizedE8Layer cls;
    app_py --> app_py_OptimizedE8Layer
    app_py_RESMAv2Fast["RESMAv2Fast"]
    class app_py_RESMAv2Fast cls;
    app_py --> app_py_RESMAv2Fast
    app_py_RESMAv2Standard["RESMAv2Standard"]
    class app_py_RESMAv2Standard cls;
    app_py --> app_py_RESMAv2Standard
    app_py_RESMAv2Deep["RESMAv2Deep"]
    class app_py_RESMAv2Deep cls;
    app_py --> app_py_RESMAv2Deep
    app_py_GAT_Baseline["GAT_Baseline"]
    class app_py_GAT_Baseline cls;
    app_py --> app_py_GAT_Baseline
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext_os["os"]
    class ext_os ext;
    app_py -.->|imports| ext_os
    ext_glob["glob"]
    class ext_glob ext;
    app_py -.->|imports| ext_glob
    ext_torch["torch"]
    class ext_torch ext;
    app_py -.->|imports| ext_torch
    ext_zipfile["zipfile"]
    class ext_zipfile ext;
    app_py -.->|imports| ext_zipfile
    ext_kagglehub["kagglehub"]
    class ext_kagglehub ext;
    app_py -.->|imports| ext_kagglehub
    ext_numpy["numpy"]
    class ext_numpy ext;
    app_py -.->|imports| ext_numpy
    ext_pandas["pandas"]
    class ext_pandas ext;
    app_py -.->|imports| ext_pandas
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    app_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    app_py -.->|imports| ext_torch_nn_functional
    ext_torch_geometric_nn["torch_geometric.nn"]
    class ext_torch_geometric_nn ext;
    app_py -.->|imports| ext_torch_geometric_nn
    ext_torch_geometric_utils["torch_geometric.utils"]
    class ext_torch_geometric_utils ext;
    app_py -.->|imports| ext_torch_geometric_utils
    ext_sklearn_preprocessing["sklearn.preprocessing"]
    class ext_sklearn_preprocessing ext;
    app_py -.->|imports| ext_sklearn_preprocessing
    ext_sklearn_model_selection["sklearn.model_selection"]
    class ext_sklearn_model_selection ext;
    app_py -.->|imports| ext_sklearn_model_selection
    ext_sklearn_metrics["sklearn.metrics"]
    class ext_sklearn_metrics ext;
    app_py -.->|imports| ext_sklearn_metrics
    ext_time["time"]
    class ext_time ext;
    app_py -.->|imports| ext_time
    ext_warnings["warnings"]
    class ext_warnings ext;
    app_py -.->|imports| ext_warnings
    app_timestamp_py -.->|imports| ext_os
    app_timestamp_py -.->|imports| ext_glob
    app_timestamp_py -.->|imports| ext_torch
    app_timestamp_py -.->|imports| ext_zipfile
    app_timestamp_py -.->|imports| ext_kagglehub
    app_timestamp_py -.->|imports| ext_numpy
    app_timestamp_py -.->|imports| ext_pandas
    app_timestamp_py -.->|imports| ext_torch_nn
    app_timestamp_py -.->|imports| ext_torch_nn_functional
    app_timestamp_py -.->|imports| ext_torch_geometric_nn
    app_timestamp_py -.->|imports| ext_torch_geometric_utils
    app_timestamp_py -.->|imports| ext_sklearn_preprocessing
    app_timestamp_py -.->|imports| ext_sklearn_model_selection
    app_timestamp_py -.->|imports| ext_sklearn_metrics
    app_timestamp_py -.->|imports| ext_time
    app_timestamp_py -.->|imports| ext_warnings
    app_timestamp_py -.->|imports| ext_os
    app_timestamp_py -.->|imports| ext_glob
    app_timestamp_py -.->|imports| ext_zipfile
    app_timestamp_py -.->|imports| ext_pandas
    app_timestamp_py -.->|imports| ext_torch
    app_timestamp_py -.->|imports| ext_sklearn_preprocessing
    app_timestamp_py -.->|imports| ext_torch_geometric_utils
    app_timestamp_py -.->|imports| ext_kagglehub
```

---

## Architecture Reference

### PY (2 files)

#### `app.py`
**Path:** `app.py`

**Classes:**
- `OptimizedE8Layer` (line 49) `class OptimizedE8Layer` - *Optimized E8 with caching and efficiency improvements*
- `RESMAv2Fast` (line 74) `class RESMAv2Fast` - *Fast version: E8 + GAT fusion with minimal overhead*
- `RESMAv2Standard` (line 112) `class RESMAv2Standard` - *Standard version: 2 layers of E8 + GAT fusion*
- `RESMAv2Deep` (line 160) `class RESMAv2Deep` - *Deeper version with 3 layers*
- `GAT_Baseline` (line 208) `class GAT_Baseline` - *Optimized GAT baseline*

**Functions:**
- `load_elliptic_data` (line 233) `def load_elliptic_data()`
- `train_and_evaluate` (line 290) `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
- `cross_validate_model` (line 347) `def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)`
- `__init__` (line 51) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `forward` (line 66) `def forward(self, x)`
- `__init__` (line 76) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (line 99) `def forward(self, x, edge_index)`
- `__init__` (line 114) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (line 139) `def forward(self, x, edge_index)`
- `__init__` (line 162) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (line 194) `def forward(self, x, edge_index)`
- `__init__` (line 210) `def __init__(self, input_dim, hidden_dim, dropout)`
- `forward` (line 220) `def forward(self, x, edge_index)`

#### `app_timestamp.py`
**Path:** `app_timestamp.py`

**Classes:**
- `OptimizedE8Layer` (line 23) `class OptimizedE8Layer` - *Optimized E8 with caching and efficiency improvements*
- `RESMAv2Fast` (line 48) `class RESMAv2Fast` - *Fast version: E8 + GAT fusion with minimal overhead*
- `RESMAv2Standard` (line 86) `class RESMAv2Standard` - *Standard version: 2 layers of E8 + GAT fusion*
- `RESMAv2Deep` (line 134) `class RESMAv2Deep` - *Deeper version with 3 layers*
- `GAT_Baseline` (line 182) `class GAT_Baseline` - *Optimized GAT baseline*

**Functions:**
- `load_elliptic_data` (line 207) `def load_elliptic_data()`
- `train_and_evaluate` (line 283) `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
- `temporal_cross_validate_model` (line 345) `def temporal_cross_validate_model(model_class, X, y, edge_index, timestep, num_nodes, name, min_train_ts)`
- `__init__` (line 25) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `forward` (line 40) `def forward(self, x)`
- `__init__` (line 50) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (line 73) `def forward(self, x, edge_index)`
- `__init__` (line 88) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (line 113) `def forward(self, x, edge_index)`
- `__init__` (line 136) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (line 168) `def forward(self, x, edge_index)`
- `__init__` (line 184) `def __init__(self, input_dim, hidden_dim, dropout)`
- `forward` (line 194) `def forward(self, x, edge_index)`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
