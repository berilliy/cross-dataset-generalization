from pathlib import Path
import nbformat
from nbclient import NotebookClient


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    path = root / "notebook3ec3218ec0_corrected.ipynb"
    notebook = nbformat.read(path, 4)
    def progress(cell, cell_index, execute_reply):
        print("Cell", cell_index, execute_reply["content"]["status"], flush=True)
    NotebookClient(notebook, timeout=7200, kernel_name="python3",
                   resources={"metadata":{"path":str(root)}}, on_cell_executed=progress).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)

