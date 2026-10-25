"""Execute the model-development notebook in a fresh kernel; keep outputs gitignored."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per cell in seconds")
    args = parser.parse_args()
    for key, subdir in [("JUPYTER_DATA_DIR", "jupyter_data"),
                        ("JUPYTER_RUNTIME_DIR", "jupyter_runtime"), ("IPYTHONDIR", "ipython"),
                        ("MPLCONFIGDIR", "matplotlib")]:
        os.environ.setdefault(key, str(ROOT / "outputs" / subdir))
    # Discover the kernel installed in this virtual environment, including on Windows.
    prefix_path = str(Path(sys.prefix) / "share/jupyter")
    os.environ["JUPYTER_PATH"] = prefix_path + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    import nbformat
    from nbclient import NotebookClient

    source = ROOT / "notebooks/Retail_Shoplifting_Model_Development.ipynb"
    destination = ROOT / "outputs/model_development/Retail_Shoplifting_Model_Development_executed.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    print("Starting a fresh Python 3.11 notebook kernel...", flush=True)
    client = NotebookClient(notebook, timeout=args.timeout, startup_timeout=60, kernel_name="shoplifting-py311",
                            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False,
                            on_cell_executed=lambda **kw: print(f"Executed cell {kw['cell_index'] + 1}", flush=True))
    try:
        client.execute()
    finally:
        # Preserve diagnostics even if a cell fails; success is reported only below.
        nbformat.write(notebook, destination)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    errors = [o for c in code_cells for o in c.outputs if o.output_type == "error"]
    if errors or any(c.execution_count is None for c in code_cells):
        raise RuntimeError(f"Notebook incomplete; inspect {destination}")
    from nbconvert import HTMLExporter
    html, _ = HTMLExporter().from_notebook_node(notebook)
    html_path = destination.parent / "Retail_Shoplifting_Model_Development.html"
    html_path.write_text(html, encoding="utf-8")
    report = {"source": str(source), "executed_notebook": str(destination),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "html": str(html_path),
              "code_cells_executed": len(code_cells), "error_count": len(errors),
              "python": sys.executable}
    (destination.parent / "execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: executed {len(code_cells)} code cells from a fresh kernel, zero errors.")
    print(f"Executed notebook: {destination}")
    print(f"HTML output: {html_path}")


if __name__ == "__main__":
    main()
