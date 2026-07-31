# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM, Ruby, Swift, Kotlin, Scala, Lua, Elixir.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 3 | **Total Symbols Extracted:** 36 | **Total Imports:** 40

<!-- ranking_model: v1.0 | weights: {ppr:0.45,auth:0.2,test:0.15,doc:0.1,fresh:0.1} | alpha:0.85 | commit:e63a2e6 | date:2026-07-18 -->


## Table of Contents

1. [Statistics Dashboard](#statistics-dashboard)
2. [Architectural Layers](#architectural-layers)
3. [Ranked Context](#ranked-context)
4. [God Nodes](#god-nodes)
5. [Suggested Questions](#suggested-questions)
6. [Hotspot Analysis](#hotspot-analysis)
7. [Change Impact Analysis](#change-impact-analysis)
8. [Suggested Linting Rules](#suggested-linting-rules)
9. [Orphans](#orphans)
10. [Query Recipes](#query-recipes)
11. [Structural Knowledge Map](#structural-knowledge-map)
12. [Code Property Graph](#code-property-graph)
13. [Architecture Reference](#architecture-reference)
    - [PY (2 files)](#py-2-files)
    - [SH (1 files)](#sh-1-files)

---

## Statistics Dashboard

| Metric | Value |
|--------|-------|
| Total Files | 3 |
| Total Symbols | 36 |
| Total Imports | 40 |
| Call Edges | 415 |
| Inheritance Edges | 10 |
| Languages | 2 |
| Avg Symbols/File | 12.0 |
| Avg Imports/File | 13.3 |

### Top Files by Import Count (Fan-Out)

| File | Imports | Symbols | Language |
|------|---------|---------|----------|
| `app_timestamp.py` | 24 | 18 | py |
| `app.py` | 16 | 18 | py |

---

## Architectural Layers

Auto-detected from path patterns, naming conventions, and imported frameworks.

| Layer | Files |
|-------|-------|
| utility | 3 |

### utility

- `app.py` (py, 18 symbols)
- `app_timestamp.py` (py, 18 symbols)
- `install.sh` (sh, 0 symbols)

---

## Ranked Context

Files ranked by composite score for the current query context. The ranking combines Personalized PageRank (query relevance), global authority, test coverage, documentation coverage, and code freshness. Model: v1.0.

| Rank | File | Composite | PPR | Authority | Test | Doc |
|------|------|-----------|-----|-----------|------|-----|
| 1 | `app.py` | 0.0333 | 0.0000 | 0.0000 | 0.00 | 0.33 |
| 2 | `app_timestamp.py` | 0.0278 | 0.0000 | 0.0000 | 0.00 | 0.28 |
| 3 | `install.sh` | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 |

---

## God Nodes

Most architecturally central files ranked by combined import/export degree and symbol richness.

| File | Score | Connections | PageRank |
|------|-------|-------------|----------|
| `app.py` | 1.8 | | 0.0000 |
| `app_timestamp.py` | 1.8 | | 0.0000 |
| `install.sh` | 0.0 | | 0.0000 |

---

## Suggested Questions

Auto-generated exploration prompts based on graph structure:

- What does app.py depend on, and what depends on it? (0 connections)
- What does app_timestamp.py depend on, and what depends on it? (0 connections)
- What does install.sh depend on, and what depends on it? (0 connections)
- What is OptimizedE8Layer in app.py and how is it used?
- What is OptimizedE8Layer in app_timestamp.py and how is it used?

---

## Hotspot Analysis

Files ranked by combined complexity (symbol count) and centrality (connection count). High-scoring files are architecturally critical and may need refactoring attention.

| File | Complexity | Centrality | Combined | Symbols | Connections |
|------|-----------|------------|----------|---------|-------------|
| `app.py` | 1.000 | 0.667 | 0.800 | 18 | 16 |
| `app_timestamp.py` | 1.000 | 1.000 | 1.000 | 18 | 24 |
| `install.sh` | 0.000 | 0.000 | 0.000 | 0 | 0 |

---

## Change Impact Analysis

Files sorted by how many other files would be affected if they changed. High-impact files should be changed with caution.

| File | Direct Dependents | Transitive Dependents | Total Impact |
|------|------------------|----------------------|--------------|
| `app.py` | 0 | 0 | 0 |
| `app_timestamp.py` | 0 | 0 | 0 |
| `install.sh` | 0 | 0 | 0 |

---

## Suggested Linting Rules

Automatically suggested linting and security rules based on patterns detected in the codebase. These can be exported as Semgrep rules using the `--export-rules` flag.

| Rule ID | Severity | Description | Language | Matches |
|---------|----------|-------------|----------|---------|
| `RM001` | info | Large number of functions in py: 26 total | py | 26 |
| `RM002` | info | Print statement found (consider logging instead) | python | 62 |

---

## Orphans

Files with no documentation or low connectivity. These are candidates for documentation investment or cleanup.

- `install.sh` (0 symbols, no doc)

---

## Query Recipes

Example queries you can run against this knowledge base using the ranking engine:

```
# Find files most relevant to a concept
readmenator query "Where is the import resolver implemented?"

# Rank files by relevance to a topic
readmenator query "How does documentation generation work?"

# Explain why a file ranks highly
readmenator query "explain readmenator/_documentation.py"

# Trace dependency paths with ranked context
readmenator query "path from CLI to exporter"
```

The ranking model uses the following signals:

- **Personalized PageRank** (45% weight): query-specific relevance via seed propagation
- **Global Authority** (20% weight): structural importance via standard PageRank
- **Test Coverage** (15% weight): fraction of symbols referenced in test files
- **Doc Coverage** (10% weight): presence of docstrings and file-level docs
- **Freshness** (10% weight): recent modification activity

Results include score decomposition and justification paths for each ranked item.

---

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

## Code Property Graph

Machine-readable Code Property Graph (CPG) in JSON-LD format. This block allows AI agents to parse the full structural graph without additional file reads. Compatible with GraphRAG pipelines.

```json
{"@context": "https://readmenator.dev/cpg/v1", "analysis": {"communities": [], "god_nodes": [{"node_id": "app.py", "score": 1.8}, {"node_id": "app_timestamp.py", "score": 1.8}, {"node_id": "install.sh", "score": 0.0}], "surprising_connections": []}, "edges": [{"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "zipfile"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "kagglehub"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch_geometric.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch_geometric.utils"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "sklearn.preprocessing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "sklearn.model_selection"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "sklearn.metrics"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "zipfile"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "kagglehub"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "torch_geometric.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "torch_geometric.utils"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "sklearn.preprocessing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "sklearn.model_selection"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "sklearn.metrics"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "zipfile"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "sklearn.preprocessing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "torch_geometric.utils"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app_timestamp.py", "target": "kagglehub"}], "generator": "readmenator", "metadata": {"edge_count": 465, "file_count": 3, "language_count": 2, "symbol_count": 36}, "nodes": [{"doc": "_*_ coding: utf8 _*_", "id": "app.py", "kind": "module", "label": "app.py", "language": "py", "sha256": "1a0ba644cf1651db", "symbol_count": 18, "symbols": [{"doc": "Optimized E8 with caching and efficiency improvements", "kind": "class", "line": 49, "name": "OptimizedE8Layer", "signature": "class OptimizedE8Layer(Module)"}, {"doc": "Fast version: E8 + GAT fusion with minimal overhead", "kind": "class", "line": 74, "name": "RESMAv2Fast", "signature": "class RESMAv2Fast(Module)"}, {"doc": "Standard version: 2 layers of E8 + GAT fusion", "kind": "class", "line": 112, "name": "RESMAv2Standard", "signature": "class RESMAv2Standard(Module)"}, {"doc": "Deeper version with 3 layers", "kind": "class", "line": 160, "name": "RESMAv2Deep", "signature": "class RESMAv2Deep(Module)"}, {"doc": "Optimized GAT baseline", "kind": "class", "line": 208, "name": "GAT_Baseline", "signature": "class GAT_Baseline(Module)"}, {"kind": "method", "line": 233, "name": "load_elliptic_data", "signature": "def load_elliptic_data()"}, {"kind": "method", "line": 290, "name": "train_and_evaluate", "signature": "def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)"}, {"kind": "method", "line": 347, "name": "cross_validate_model", "signature": "def cross_validate_model(model_class, X, y, edge_index, num_nodes, n_splits, seed, name)"}, {"kind": "method", "line": 51, "name": "__init__", "signature": "def __init__(self, in_features, out_features, edge_index, num_nodes)"}, {"kind": "method", "line": 66, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 76, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)"}, {"kind": "method", "line": 99, "name": "forward", "signature": "def forward(self, x, edge_index)"}, {"kind": "method", "line": 114, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)"}, {"kind": "method", "line": 139, "name": "forward", "signature": "def forward(self, x, edge_index)"}, {"kind": "method", "line": 162, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)"}, {"kind": "method", "line": 194, "name": "forward", "signature": "def forward(self, x, edge_index)"}, {"kind": "method", "line": 210, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, dropout)"}, {"kind": "method", "line": 220, "name": "forward", "signature": "def forward(self, x, edge_index)"}]}, {"id": "app_timestamp.py", "kind": "module", "label": "app_timestamp.py", "language": "py", "sha256": "e93711220f2bdcff", "symbol_count": 18, "symbols": [{"doc": "Optimized E8 with caching and efficiency improvements", "kind": "class", "line": 23, "name": "OptimizedE8Layer", "signature": "class OptimizedE8Layer(Module)"}, {"doc": "Fast version: E8 + GAT fusion with minimal overhead", "kind": "class", "line": 48, "name": "RESMAv2Fast", "signature": "class RESMAv2Fast(Module)"}, {"doc": "Standard version: 2 layers of E8 + GAT fusion", "kind": "class", "line": 86, "name": "RESMAv2Standard", "signature": "class RESMAv2Standard(Module)"}, {"doc": "Deeper version with 3 layers", "kind": "class", "line": 134, "name": "RESMAv2Deep", "signature": "class RESMAv2Deep(Module)"}, {"doc": "Optimized GAT baseline", "kind": "class", "line": 182, "name": "GAT_Baseline", "signature": "class GAT_Baseline(Module)"}, {"kind": "method", "line": 207, "name": "load_elliptic_data", "signature": "def load_elliptic_data()"}, {"kind": "method", "line": 283, "name": "train_and_evaluate", "signature": "def train_and_evaluate(model, X, y, edge_index, train_idx, val_idx, epochs, lr, name, fold)"}, {"kind": "method", "line": 345, "name": "temporal_cross_validate_model", "signature": "def temporal_cross_validate_model(model_class, X, y, edge_index, timestep, num_nodes, name, min_train_ts)"}, {"kind": "method", "line": 25, "name": "__init__", "signature": "def __init__(self, in_features, out_features, edge_index, num_nodes)"}, {"kind": "method", "line": 40, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 50, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)"}, {"kind": "method", "line": 73, "name": "forward", "signature": "def forward(self, x, edge_index)"}, {"kind": "method", "line": 88, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)"}, {"kind": "method", "line": 113, "name": "forward", "signature": "def forward(self, x, edge_index)"}, {"kind": "method", "line": 136, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, edge_index, num_nodes, dropout)"}, {"kind": "method", "line": 168, "name": "forward", "signature": "def forward(self, x, edge_index)"}, {"kind": "method", "line": 184, "name": "__init__", "signature": "def __init__(self, input_dim, hidden_dim, dropout)"}, {"kind": "method", "line": 194, "name": "forward", "signature": "def forward(self, x, edge_index)"}]}, {"id": "install.sh", "kind": "module", "label": "install.sh", "language": "sh", "sha256": "c907d80fd6734993", "symbol_count": 0, "symbols": []}], "type": "CodePropertyGraph", "version": "1.0"}
```

---

## Architecture Reference

### PY (2 files)

#### `app.py`
**Path:** `app.py`
**File Doc:** *_*_ coding: utf8 _*_*

**Classes:**
- `OptimizedE8Layer` (line 49) `class OptimizedE8Layer(Module)` - *Optimized E8 with caching and efficiency improvements*
- `RESMAv2Fast` (line 74) `class RESMAv2Fast(Module)` - *Fast version: E8 + GAT fusion with minimal overhead*
- `RESMAv2Standard` (line 112) `class RESMAv2Standard(Module)` - *Standard version: 2 layers of E8 + GAT fusion*
- `RESMAv2Deep` (line 160) `class RESMAv2Deep(Module)` - *Deeper version with 3 layers*
- `GAT_Baseline` (line 208) `class GAT_Baseline(Module)` - *Optimized GAT baseline*

**Methods:**
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
- `OptimizedE8Layer` (line 23) `class OptimizedE8Layer(Module)` - *Optimized E8 with caching and efficiency improvements*
- `RESMAv2Fast` (line 48) `class RESMAv2Fast(Module)` - *Fast version: E8 + GAT fusion with minimal overhead*
- `RESMAv2Standard` (line 86) `class RESMAv2Standard(Module)` - *Standard version: 2 layers of E8 + GAT fusion*
- `RESMAv2Deep` (line 134) `class RESMAv2Deep(Module)` - *Deeper version with 3 layers*
- `GAT_Baseline` (line 182) `class GAT_Baseline(Module)` - *Optimized GAT baseline*

**Methods:**
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
