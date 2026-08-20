# B-Rep-KG: A Persistent B-Rep Knowledge Graph for Natural-Language Spatial Querying of BIM Models

This repository contains the implementation and evaluation data for a
persistent Boundary Representation Knowledge Graph (B-Rep-KG) framework
that enables natural-language spatial querying over Building Information
Models (BIM). Tessellated B-Rep geometry extracted from IFC files is
stored directly within a Neo4j property graph, allowing DE-9IM spatial
relationships between building elements to be computed once and
persisted as reusable, queryable graph edges. An LLM-powered multi-agent
pipeline translates natural-language queries into structured commands,
resolves the referenced elements, evaluates spatial predicates against
the graph, and returns interpretable natural-language answers.

## Architecture

The system is organized as a five-stage pipeline, orchestrated by a
Manager Agent:

1. **IFC Parsing** (`src/01_ifc_parser`) — extracts semantic and
   geometric data from IFC files using IfcOpenShell.
2. **Tessellation** (`src/02_tessellator`) — converts parsed IFC
   geometry into B-Rep primitives (vertices, faces, edges).
3. **Graph Construction** (`src/03_graph_builder`) — persists the
   semantic and B-Rep data as a Neo4j knowledge graph.
4. **Spatial Relationship Computation Engine** (`src/04_spatial_engine`)
   — evaluates the six DE-9IM predicates (DISJOINT, EQUALS, CONTAINS,
   COVERS, OVERLAPS, TOUCHES) against the persistent graph, using
   hierarchical candidate filtering and bounding-box pre-filtering, and
   caches each computed relationship as a typed graph edge for O(1)
   retrieval on subsequent queries.
5. **Multi-Agent Query Pipeline** (`src/05_agents`) — a Manager Agent
   coordinates four specialized agents:
   - **Understanding Agent** — parses natural-language queries into
     structured commands using an LLM.
   - **Retrieval Agent** — resolves the referenced element(s) from the
     graph.
   - **Computation Agent** — invokes the spatial engine to classify or
     retrieve cached spatial relationships.
   - **Response Agent** — formats the result into a natural-language
     answer.

## Multi-model evaluation

The framework is language-model-agnostic: the Understanding and
Response Agents accept any OpenAI-compatible LLM endpoint via a central
provider registry (`llm_providers.py`), allowing new models to be added
without modifying agent code. The included evaluation data covers three
language models — GPT-4, Llama 3.3 70B, and Mistral Small — each
evaluated on 90 natural-language spatial queries spanning three IFC
test models of increasing scale and complexity, under both cold-cache
and (for GPT-4) warm-cache conditions.

## Repository layout

```
BRepGraph-Article1/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py.example          # template configuration file
├── llm_providers.py            # LLM provider registry
│
├── src/
│   ├── 01_ifc_parser/
│   ├── 02_tessellator/
│   ├── 03_graph_builder/
│   ├── 04_spatial_engine/
│   │   └── spatial_engine.py   # DE-9IM predicate evaluation engine
│   └── 05_agents/
│       ├── manager_agent.py
│       ├── understanding_agent.py
│       ├── retrieval_agent.py
│       ├── computation_agent.py
│       └── response_agent.py
│
├── data/
│   ├── queries/                 # evaluation query sets with ground truth
│   └── results/                 # raw evaluation results (per model, per LLM, per cache state)
│
└── evaluation_summary/
    ├── master_results.csv       # all evaluation runs, one row per query
    └── summary_statistics.csv   # aggregated accuracy, timing, and caching statistics
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt --break-system-packages
   ```
2. Copy `config.py.example` to `config.py` and fill in your Neo4j
   credentials and IFC file path.
3. Create a `.env` file with the required API keys (see
   `llm_providers.py` for the expected environment variable names per
   provider).
4. Ensure a Neo4j instance is running and accessible at the URI
   configured in `config.py`.

## Data availability

`data/` and `evaluation_summary/` contain the complete query sets,
ground truth, raw per-query evaluation results, and aggregated
statistics reported in the accompanying publication, provided to
support reproducibility.

## Citation

If you use this framework or its evaluation data, please cite the
accompanying publication. Citation details will be added upon
publication. If you have any questions, please reach out to me at muhammadshoaib5308@gmail.com
