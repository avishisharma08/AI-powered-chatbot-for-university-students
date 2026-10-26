import os
import sys

# Ensure src is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rag_udemy.pages.eval_dashboard import render_dashboard

if __name__ == "__main__":
    render_dashboard()
