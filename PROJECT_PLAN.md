# CatalogueIQ Project Structure and Delivery Plan

## Recommended repository structure

```text
catalogueiq_capstone/
├── app.py                         # Simple Streamlit UI starter
├── data/                          # CSVs; immutable source data
├── knowledge_base/                # Markdown and HTML source documents
├── src/                           # Claude adds loaders, chunkers, index, RAG chain
├── tests/                         # Dataset tests exist; add pipeline/UI tests
├── evaluation/                    # 20-question set and RAGAS outputs
├── requirements.txt
├── README.md
├── evaluation_report.md
└── CLAUDE_BUILD_PROMPT.md
```

## Phase gates

| Phase | Deliverable | Test gate |
|---|---|---|
| 0. Data validation | Dataset inventory | `pytest -q` passes |
| 1. Ingestion | CSV/Markdown/HTML loaders and two chunkers | Inspect five representative retrieval inputs |
| 2. Retrieval | Persistent vector index, expansion, filters | Multi-product comparative results and citation tests |
| 3. Answering | Grounded chain, personas, 5-turn memory | All five capstone query types manually pass |
| 4. UI | Blue/purple/white Streamlit experience | Startup and interaction smoke test |
| 5. Evaluation | Test set/report/demo material | RAGAS baseline and improved run recorded |

The high-risk test is comparative recommendation: retrieve multiple product rows (top-k ≥ 8) before ranking under a price/specification constraint. Do not treat an embedding similarity score as a product recommendation by itself.
