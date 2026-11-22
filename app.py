"""Hosted entrypoint for the gmpe-bench explorer (Streamlit Community Cloud, Hugging Face Spaces, Docker).

    streamlit run app.py

Locally, prefer ``gmpe-bench app``. This file works with or without ``pip install``: it puts ``src/`` on
the import path, so a hosted deployment only needs the packages in ``requirements.txt``. The app reads
the published results in ``results/paper`` (or the packaged copy) and never needs the PEER flatfiles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / "src" / "gmpe_bench").is_dir():
    sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("GMPE_BENCH_ROOT", str(ROOT))

from gmpe_bench.app.streamlit_app import main

main()
