# root

*Community 0 | 3 files | cohesion 1.00*

## Definition

This community groups 3 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `GAT_Baseline`, `OptimizedE8Layer`, `RESMAv2Deep`, `RESMAv2Fast`, `RESMAv2Standard`, `__init__`, `cross_validate_model`, `forward`. Core file: `app.py` (18 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: 24/12/2025 Licencia: GPL v3  Descripción: Lie Detector: E8-Divisio.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 18 | yes |
| `app_timestamp.py` | py | utility | 18 | no |
| `install.sh` | sh | utility | 0 | no |

## Key Symbols

- `OptimizedE8Layer` (class, `app.py:49`) `class OptimizedE8Layer(Module)` - Optimized E8 with caching and efficiency improvements
- `__init__` (method, `app.py:51`) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `forward` (method, `app.py:66`) `def forward(self, x)`
- `RESMAv2Fast` (class, `app.py:74`) `class RESMAv2Fast(Module)` - Fast version: E8 + GAT fusion with minimal overhead
- `__init__` (method, `app.py:76`) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (method, `app.py:99`) `def forward(self, x, edge_index)`
- `RESMAv2Standard` (class, `app.py:112`) `class RESMAv2Standard(Module)` - Standard version: 2 layers of E8 + GAT fusion
- `__init__` (method, `app.py:114`) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (method, `app.py:139`) `def forward(self, x, edge_index)`
- `RESMAv2Deep` (class, `app.py:160`) `class RESMAv2Deep(Module)` - Deeper version with 3 layers
- `__init__` (method, `app.py:162`) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (method, `app.py:194`) `def forward(self, x, edge_index)`
- `GAT_Baseline` (class, `app.py:208`) `class GAT_Baseline(Module)` - Optimized GAT baseline
- `__init__` (method, `app.py:210`) `def __init__(self, input_dim, hidden_dim, dropout)`
- `forward` (method, `app.py:220`) `def forward(self, x, edge_index)`
- `load_elliptic_data` (method, `app.py:233`) `def load_elliptic_data()`
- `train_and_evaluate` (method, `app.py:290`) `def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr,`
- `cross_validate_model` (method, `app.py:347`) `def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, see`
- `OptimizedE8Layer` (class, `app_timestamp.py:23`) `class OptimizedE8Layer(Module)` - Optimized E8 with caching and efficiency improvements
- `__init__` (method, `app_timestamp.py:25`) `def __init__(self, in_features, out_features, edge_index, num_nodes)`
- `forward` (method, `app_timestamp.py:40`) `def forward(self, x)`
- `RESMAv2Fast` (class, `app_timestamp.py:48`) `class RESMAv2Fast(Module)` - Fast version: E8 + GAT fusion with minimal overhead
- `__init__` (method, `app_timestamp.py:50`) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (method, `app_timestamp.py:73`) `def forward(self, x, edge_index)`
- `RESMAv2Standard` (class, `app_timestamp.py:86`) `class RESMAv2Standard(Module)` - Standard version: 2 layers of E8 + GAT fusion
- `__init__` (method, `app_timestamp.py:88`) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (method, `app_timestamp.py:113`) `def forward(self, x, edge_index)`
- `RESMAv2Deep` (class, `app_timestamp.py:134`) `class RESMAv2Deep(Module)` - Deeper version with 3 layers
- `__init__` (method, `app_timestamp.py:136`) `def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)`
- `forward` (method, `app_timestamp.py:168`) `def forward(self, x, edge_index)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 2 file(s) lack file-level docs (e.g. `app_timestamp.py`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `app_timestamp.py`
- `install.sh`
