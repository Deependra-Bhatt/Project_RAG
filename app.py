"""CatalogueIQ Streamlit app. Run: streamlit run app.py

Blue/purple/white, accessible, single chat area, a compact filter panel
(category + max price), an answer area with citations, and an optional
expander showing the retrieved chunks. No auth, no payments, no extra pages.
Runs with no secret in retrieval-only mode; set ANTHROPIC_API_KEY to enable
generated answers.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.chain import ConversationMemory, RagChain
from src.pipeline import get_retriever
from src.retrieval import Filters

ROOT = Path(__file__).parent
CATALOGUE = ROOT / "data" / "product_catalogue.csv"

st.set_page_config(page_title="CatalogueIQ", page_icon="\U0001f6cd\ufe0f", layout="wide")

st.markdown(
    """
<style>
  /* Global Light Mode Defaults */
  .stApp { 
      background: #f8f7ff; 
      color: #18223b; 
  }
  
  [data-testid='stSidebar'] { 
      background: #21124d; 
  }
  
  [data-testid='stSidebar'] * { 
      color: #ffffff !important; 
  }
  
  [data-testid='stSidebar'] input, 
  [data-testid='stSidebar'] select, 
  [data-testid='stSidebar'] div[data-baseweb="select"] * {
      color: #18223b !important;
      background-color: #ffffff !important;
      font-weight: 500 !important;
  }
  
  [data-testid='stSidebar'] label {
      color: #ffffff !important;
      font-weight: 600 !important;
  }
  
  [data-testid='stSidebar'] div[role="listbox"] {
      background-color: #ffffff !important;
  }
  
  /* Chat input field light mode styling */
  .stChatInput input,
  .stChatInput input:focus,
  .stChatInput input:active,
  input[data-testid*="chatInput"],
  .stChatInput [type="text"],
  [data-testid="stChatInput"] textarea,
  [data-testid="stChatInputTextArea"] {
      background-color: #ffffff !important;
      color: #18223b !important;
      font-size: 16px !important;
      font-weight: 500 !important;
      caret-color: #18223b !important;
  }
  
  .stChatInput input::placeholder,
  [data-testid="stChatInput"] textarea::placeholder {
      color: #6b7280 !important;
      opacity: 0.8 !important;
  }

  h1, h2, h3 { 
      color: #382a91; 
      font-weight: 700; 
  }
  
  .stButton > button { 
      background: linear-gradient(90deg, #2563eb, #7c3aed); 
      color: white !important; 
      border: 0; 
      font-weight: 600; 
  }
  
  .citation-pill {
      display: inline-block; 
      background: #dbeafe; 
      color: #0c4a6e;
      border-radius: 999px; 
      padding: 4px 12px; 
      margin: 2px 4px 2px 0;
      font-size: 0.82rem; 
      border: 1px solid #0284c7; 
      font-weight: 600;
  }
  
  .mode-badge {
      display: inline-block; 
      padding: 4px 12px; 
      border-radius: 6px;
      font-size: 0.78rem; 
      font-weight: 700; 
      margin-bottom: 6px;
      border: 1px solid rgba(0,0,0,0.1);
  }
  .mode-generated { background: #dbeafe; color: #0c4a6e; }
  .mode-retrieval_only { background: #e9d5ff; color: #5b21b6; }
  .mode-no_evidence { background: #fef3c7; color: #78350f; }
  .mode-out_of_scope { background: #fecaca; color: #7f1d1d; }
  
  [data-testid='stChatMessage'] {
      background-color: #ffffff !important;
      border: 1px solid #e5e7eb !important;
      border-radius: 8px !important;
      padding: 12px 16px !important;
  }
  
  [data-testid='stChatMessage'] p, [data-testid='stChatMessage'] div {
      color: #1f2937 !important;
      font-size: 15px !important;
      line-height: 1.6 !important;
  }
  
  .stMarkdown, .stMarkdown p {
      color: #18223b !important;
      font-size: 15px !important;
      font-weight: 500 !important;
  }
  
  [data-testid='stAlert'] {
      background-color: #f0f9ff !important;
      border: 1px solid #0284c7 !important;
      border-radius: 6px !important;
  }
  
  [data-testid='stAlert'] p {
      color: #0c4a6e !important;
      font-weight: 500 !important;
      font-size: 14px !important;
  }
  
  .stCaption {
      color: #4b5563 !important;
      font-weight: 500 !important;
  }

  /* Dark Mode Optimization with Explicit Text Contrast */
  @media (prefers-color-scheme: dark) {
    .stApp { 
      background: #0f1419 !important; 
      color: #f3f4f6 !important;
    }

    [data-testid='stSidebar'] { 
      background: #1a202c !important;
      border-right: 1px solid #2d3748 !important;
    }
    
    [data-testid='stSidebar'] * { 
      color: #f3f4f6 !important;
    }
    
    [data-testid='stSidebar'] label {
      color: #ffffff !important;
      font-weight: 600 !important;
    }

    [data-testid='stSidebar'] input, 
    [data-testid='stSidebar'] select, 
    [data-testid='stSidebar'] div[data-baseweb="select"] {
      background-color: #2d3748 !important;
      border: 1px solid #4a5568 !important;
      color: #f3f4f6 !important;
    }

    [data-testid='stSidebar'] div[data-baseweb="select"] * {
      color: #f3f4f6 !important;
      background-color: #2d3748 !important;
    }

    [data-testid='stSidebar'] input::placeholder {
      color: #9ca3af !important;
    }

    /* Fixed visibility for chat inputs in dark mode */
    .stChatInput input,
    .stChatInput input:focus,
    .stChatInput input:active,
    input[data-testid*="chatInput"],
    .stChatInput [type="text"],
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    .stChatInput input::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #9ca3af !important;
        opacity: 0.8 !important;
    }

    h1, h2, h3 { 
      color: #c9b3ff !important;
      font-weight: 700 !important;
    }

    [data-testid='stChatMessage'] { 
      background-color: #1a202c !important;
      border-radius: 8px !important;
      border: 1px solid #374151 !important;
    }

    [data-testid='stChatMessage'] p, 
    [data-testid='stChatMessage'] span, 
    [data-testid='stChatMessage'] div { 
      color: #f3f4f6 !important;
    }

    .stButton > button { 
      background: linear-gradient(90deg, #6366f1, #a855f7) !important;
      color: #ffffff !important;
      border: 0 !important;
      font-weight: 600 !important;
      box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    }

    .citation-pill {
      background: #1e293b !important;
      color: #c4b5fd !important;
      border: 1px solid #6366f1 !important;
      font-weight: 600 !important;
    }

    .mode-generated { 
      background: #1e3a5f !important;
      color: #93c5fd !important;
      border-color: #3b82f6 !important;
    }

    .mode-retrieval_only { 
      background: #3b2667 !important;
      color: #d8b4fe !important;
      border-color: #a855f7 !important;
    }

    .mode-no_evidence { 
      background: #542e1a !important;
      color: #fbbf24 !important;
      border-color: #f59e0b !important;
    }

    .mode-out_of_scope {
      background: #5f1f1f !important;
      color: #fca5a5 !important;
      border-color: #ef4444 !important;
    }

    .stMarkdown, .stMarkdown p, [data-testid='stMarkdownContainer'] p {
      color: #f3f4f6 !important;
      font-weight: 400 !important;
    }

    .stCaption {
      color: #9ca3af !important;
    }

    [data-testid='stAlert'] {
      background-color: #1e293b !important;
      border: 1px solid #38bdf8 !important;
    }

    [data-testid='stAlert'] p {
      color: #38bdf8 !important;
    }

    hr {
      border-color: #374151 !important;
    }
  }
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def load_products():
    return pd.read_csv(CATALOGUE)


@st.cache_resource
def load_provider():
    """Initialize the provider once and cache it. This avoids expensive
    re-initialization of the Anthropic client on every request."""
    from src.provider import AnthropicProvider
    return AnthropicProvider()


@st.cache_resource
def load_chain():
    """Load the retrieval chain with cached provider. Both the retriever and
    provider are cached at the resource level to ensure they are initialized
    exactly once per Streamlit session."""
    retriever = get_retriever()
    provider = load_provider()
    return RagChain(retriever, provider=provider)


products = load_products()
chain = load_chain()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts for rendering: persona, query, result

st.title("CatalogueIQ")
st.caption(
    "ShopSmart India product intelligence \u2014 grounded answers cite the source "
    "record or document, or say plainly when evidence is missing."
)

with st.sidebar:
    st.markdown("### Workspace")
    persona = st.selectbox("Persona", ["Shopper", "Seller", "Support Agent"])

    st.markdown("### Filters")
    category = st.selectbox("Category", ["All"] + sorted(products.category.unique().tolist()))
    max_price = int(products.price_inr.max())
    ceiling = st.slider("Maximum price (\u20b9)", 0, max_price, max_price)

    if chain.provider.available:
        st.success("Generation: connected", icon="\u2705")
    else:
        st.info("Retrieval-only mode (no API key set)", icon="\u2139\ufe0f")

    if st.button("Clear conversation"):
        st.session_state.memory = ConversationMemory()
        st.session_state.history = []
        st.rerun()

filters = Filters(
    category=None if category == "All" else category,
    max_price=float(ceiling),
)

st.subheader(f"{persona} workspace")

query = st.chat_input("Ask about products, policies, delivery or returns\u2026")

if query:
    result = chain.answer(query, persona, filters=filters, memory=st.session_state.memory, category=category)
    st.session_state.history.append({"persona": persona, "query": query, "result": result})

for turn in st.session_state.history[-5:]:
    with st.chat_message("user"):
        st.markdown(f"**{turn['persona']}:** {turn['query']}")
    with st.chat_message("assistant"):
        result = turn["result"]
        st.markdown(
            f"<span class='mode-badge mode-{result.mode}'>{result.mode.replace('_', ' ')}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(result.answer)
        if result.citations:
            st.markdown(
                "".join(f"<span class='citation-pill'>{c}</span>" for c in result.citations),
                unsafe_allow_html=True,
            )
        if result.retrieved:
            with st.expander(f"Show {len(result.retrieved)} retrieved chunks"):
                for rc in result.retrieved:
                    st.markdown(f"**{rc.citation}** &nbsp;\u00b7&nbsp; score {rc.score:.3f}")
                    st.caption(rc.chunk.text)

if not st.session_state.history:
    st.info(
        "Try: \u201cbest wireless earbuds under \u20b92000\u201d, \u201ccan I return a beauty "
        "product\u201d, or \u201cwhat images do I need for a listing\u201d. "
        "The last five turns are kept as conversation memory.",
        icon="\U0001f4ac",
    )