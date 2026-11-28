import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import build_index

if __name__ == '__main__':
    vs, embedder = build_index()
    print('backend=', embedder.backend_name)
    print('store=', vs.backend_name)
    print('chunks=', len(vs.chunks))
