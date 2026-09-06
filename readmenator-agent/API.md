# API

## app.py

### load_elliptic_data `def load_elliptic_data()`
- Defined: `app.py:233`

### train_and_evaluate `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
- Defined: `app.py:290`

### cross_validate_model `def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)`
- Defined: `app.py:347`

### __init__ `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- Defined: `app.py:51`

### forward `def forward(self, x)`
- Defined: `app.py:66`

### __init__ `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `app.py:76`

### forward `def forward(self, x, edge_index)`
- Defined: `app.py:99`

### __init__ `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `app.py:114`

### forward `def forward(self, x, edge_index)`
- Defined: `app.py:139`

### __init__ `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `app.py:162`

### forward `def forward(self, x, edge_index)`
- Defined: `app.py:194`

### __init__ `def __init__(self, input_dim, hidden_dim, dropout)`
- Defined: `app.py:210`

### forward `def forward(self, x, edge_index)`
- Defined: `app.py:220`

## app_timestamp.py

### load_elliptic_data `def load_elliptic_data()`
- Defined: `app_timestamp.py:207`

### train_and_evaluate `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)`
- Defined: `app_timestamp.py:283`

### temporal_cross_validate_model `def temporal_cross_validate_model(model_class, X, y, edge_index, timestep, num_nodes, name, min_train_ts)`
- Defined: `app_timestamp.py:345`

### __init__ `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- Defined: `app_timestamp.py:25`

### forward `def forward(self, x)`
- Defined: `app_timestamp.py:40`

### __init__ `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `app_timestamp.py:50`

### forward `def forward(self, x, edge_index)`
- Defined: `app_timestamp.py:73`

### __init__ `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `app_timestamp.py:88`

### forward `def forward(self, x, edge_index)`
- Defined: `app_timestamp.py:113`

### __init__ `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- Defined: `app_timestamp.py:136`

### forward `def forward(self, x, edge_index)`
- Defined: `app_timestamp.py:168`

### __init__ `def __init__(self, input_dim, hidden_dim, dropout)`
- Defined: `app_timestamp.py:184`

### forward `def forward(self, x, edge_index)`
- Defined: `app_timestamp.py:194`
