"""Execute the supervised clip notebook from verified cached artifacts."""
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    for key,folder in [('JUPYTER_DATA_DIR','jupyter_data'),('JUPYTER_RUNTIME_DIR','jupyter_runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib')]:
        os.environ.setdefault(key,str(ROOT/'outputs'/folder))
    os.environ['JUPYTER_PATH']=str(Path(sys.prefix)/'share/jupyter')+os.pathsep+os.environ.get('JUPYTER_PATH','')
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    source=ROOT/'notebooks/Retail_Shoplifting_Supervised_Clips.ipynb'
    out=ROOT/'outputs/mnnit_clip_classification'
    notebook=nbformat.read(source,as_version=4)
    client=NotebookClient(notebook,timeout=120,startup_timeout=60,kernel_name='shoplifting-py311',allow_errors=False,
        resources={'metadata':{'path':str(ROOT)}},on_cell_executed=lambda **kw:print('Executed cell',kw['cell_index']+1,flush=True))
    executed=out/'Retail_Shoplifting_Supervised_Clips_executed.ipynb'
    try:client.execute()
    finally:nbformat.write(notebook,executed)
    cells=[c for c in notebook.cells if c.cell_type=='code']
    errors=[o for c in cells for o in c.outputs if o.output_type=='error']
    if errors or any(c.execution_count is None for c in cells):raise ValueError('Notebook did not finish')
    html,_=HTMLExporter().from_notebook_node(notebook)
    (out/'Retail_Shoplifting_Supervised_Clips.html').write_text(html,encoding='utf-8')
    report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),code_cells_executed=len(cells),error_count=len(errors),python=sys.executable)
    (out/'notebook_execution.json').write_text(json.dumps(report,indent=2))
    print('PASS:',len(cells),'code cells, zero errors; no retraining',flush=True)


if __name__=='__main__':main()
