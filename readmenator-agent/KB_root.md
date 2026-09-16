# Subsystem: root

## app.py
- Layer: utility
- Doc: _*_ coding: utf8 _*_
- Language: py
- Symbols:
  - `OptimizedE8Layer` (class, line 49) `class OptimizedE8Layer(Module)`
  - `RESMAv2Fast` (class, line 74) `class RESMAv2Fast(Module)`
  - `RESMAv2Standard` (class, line 112) `class RESMAv2Standard(Module)`
  - `RESMAv2Deep` (class, line 160) `class RESMAv2Deep(Module)`
  - `GAT_Baseline` (class, line 208) `class GAT_Baseline(Module)`
  - `load_elliptic_data` (method, line 233) `def load_elliptic_data()`
  - `train_and_evaluate` (method, line 290) `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
  - `cross_validate_model` (method, line 347) `def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)`
  - `__init__` (method, line 51) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
  - `forward` (method, line 66) `def forward(self, x)`
  - `__init__` (method, line 76) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
  - `forward` (method, line 99) `def forward(self, x, edge_index)`
  - `__init__` (method, line 114) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
  - `forward` (method, line 139) `def forward(self, x, edge_index)`
  - `__init__` (method, line 162) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
  - `forward` (method, line 194) `def forward(self, x, edge_index)`
  - `__init__` (method, line 210) `def __init__(self, input_dim, hidden_dim, dropout)`
  - `forward` (method, line 220) `def forward(self, x, edge_index)`

## app_timestamp.py
- Layer: utility
- Language: py
- Symbols:
  - `OptimizedE8Layer` (class, line 23) `class OptimizedE8Layer(Module)`
  - `RESMAv2Fast` (class, line 48) `class RESMAv2Fast(Module)`
  - `RESMAv2Standard` (class, line 86) `class RESMAv2Standard(Module)`
  - `RESMAv2Deep` (class, line 134) `class RESMAv2Deep(Module)`
  - `GAT_Baseline` (class, line 182) `class GAT_Baseline(Module)`
  - `load_elliptic_data` (method, line 207) `def load_elliptic_data()`
  - `train_and_evaluate` (method, line 283) `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
  - `temporal_cross_validate_model` (method, line 345) `def temporal_cross_validate_model(model_class, X, y, edge_index, timestep, num_nodes, name, min_train_ts)`
  - `__init__` (method, line 25) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
  - `forward` (method, line 40) `def forward(self, x)`
  - `__init__` (method, line 50) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
  - `forward` (method, line 73) `def forward(self, x, edge_index)`
  - `__init__` (method, line 88) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
  - `forward` (method, line 113) `def forward(self, x, edge_index)`
  - `__init__` (method, line 136) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
  - `forward` (method, line 168) `def forward(self, x, edge_index)`
  - `__init__` (method, line 184) `def __init__(self, input_dim, hidden_dim, dropout)`
  - `forward` (method, line 194) `def forward(self, x, edge_index)`

## install.sh
- Layer: utility
- Language: sh
