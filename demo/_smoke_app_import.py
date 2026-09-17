"""Smoke-import app.py to confirm no NameError / scope error.

We don't actually launch Streamlit — we just import the module so the top
level executes the def-statements and the module-level `mode = ...`
line. The previous bug was a NameError at the call to
_render_test_set_comparison, which would surface here as a name-not-bound
error at import time (Streamlit re-runs the whole module on each rerun).

We stub streamlit at the module level so it doesn't need a running browser
context, just enough to evaluate the module.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

# --- Minimal streamlit stub so app.py module-level calls don't crash on import.
fake_st = types.ModuleType("streamlit")

class _NoOp:
    def __getattr__(self, name): return _NoOp()
    def __call__(self, *a, **k): return _NoOp()

for name in (
    "set_page_config", "title", "caption", "info", "error", "success",
    "stop", "sidebar", "file_uploader", "selectbox", "button",
    "spinner", "columns", "image", "metric", "subheader", "pyplot",
    "expander", "slider", "write", "markdown", "radio", "cache_resource",
):
    setattr(fake_st, name, _NoOp())

# st.sidebar.* needs to chain
class _Sidebar:
    def __getattr__(self, name): return _NoOp()
fake_st.sidebar = _Sidebar()
fake_st.cache_resource = lambda *a, **k: (lambda f: f)

# torch CPU only — no .cuda()
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
fake_torch.device = lambda s: s
fake_torch.nn = types.SimpleNamespace(Module=type("M", (), {}))

sys.modules["streamlit"] = fake_st
sys.modules["torch"] = fake_torch

# --- Now import app.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import app  # noqa: E402

# --- Check: the symbol must exist and be callable.
assert hasattr(app, "_render_test_set_comparison"), (
    "_render_test_set_comparison not defined after import"
)
assert callable(app._render_test_set_comparison)
print("OK: _render_test_set_comparison is defined and callable at import time")
print(f"  resolves to: {app._render_test_set_comparison}")
print(f"  defined at line: {app._render_test_set_comparison.__code__.co_firstlineno}")
