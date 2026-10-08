# API

## app.py
- `OptimizedE8Layer.__init__` (method) `app.py:51` `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `OptimizedE8Layer.forward` (method) `app.py:66` `def forward(self, x)`
- `RESMAv2Fast.__init__` (method) `app.py:76` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `RESMAv2Fast.forward` (method) `app.py:99` `def forward(self, x, edge_index)`
- `RESMAv2Standard.__init__` (method) `app.py:114` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `RESMAv2Standard.forward` (method) `app.py:139` `def forward(self, x, edge_index)`
- `RESMAv2Deep.__init__` (method) `app.py:162` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `RESMAv2Deep.forward` (method) `app.py:194` `def forward(self, x, edge_index)`
- `GAT_Baseline.__init__` (method) `app.py:210` `def __init__(self, input_dim, hidden_dim, dropout)`
- `GAT_Baseline.forward` (method) `app.py:220` `def forward(self, x, edge_index)`
- `GAT_Baseline.load_elliptic_data` (method) `app.py:233` `def load_elliptic_data()`
- `GAT_Baseline.train_and_evaluate` (method) `app.py:290` `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
- `GAT_Baseline.cross_validate_model` (method) `app.py:347` `def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)`

## app_timestamp.py
- `OptimizedE8Layer.__init__` (method) `app_timestamp.py:25` `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `OptimizedE8Layer.forward` (method) `app_timestamp.py:40` `def forward(self, x)`
- `RESMAv2Fast.__init__` (method) `app_timestamp.py:50` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `RESMAv2Fast.forward` (method) `app_timestamp.py:73` `def forward(self, x, edge_index)`
- `RESMAv2Standard.__init__` (method) `app_timestamp.py:88` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `RESMAv2Standard.forward` (method) `app_timestamp.py:113` `def forward(self, x, edge_index)`
- `RESMAv2Deep.__init__` (method) `app_timestamp.py:136` `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `RESMAv2Deep.forward` (method) `app_timestamp.py:168` `def forward(self, x, edge_index)`
- `GAT_Baseline.__init__` (method) `app_timestamp.py:184` `def __init__(self, input_dim, hidden_dim, dropout)`
- `GAT_Baseline.forward` (method) `app_timestamp.py:194` `def forward(self, x, edge_index)`
- `GAT_Baseline.load_elliptic_data` (method) `app_timestamp.py:207` `def load_elliptic_data()`
- `GAT_Baseline.train_and_evaluate` (method) `app_timestamp.py:283` `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
- `GAT_Baseline.temporal_cross_validate_model` (method) `app_timestamp.py:345` `def temporal_cross_validate_model(model_class, X, y, edge_index, timestep, num_nodes, name, min_train_ts)`
