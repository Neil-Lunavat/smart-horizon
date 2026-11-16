#!/usr/bin/env python3
"""Assemble TASK_A.ipynb from the cell definitions in nbcells_*.py.

Kept as a generator rather than editing the .ipynb by hand: the notebook is long,
and prose lives more comfortably in a .py file than inside JSON string arrays.
Run this, then execute the notebook to populate outputs.
"""
import nbformat as nbf
import nbcells_1, nbcells_2, nbcells_3

CELLS = []
def md(s): CELLS.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def co(s): CELLS.append(nbf.v4.new_code_cell(s.strip("\n")))

for mod in (nbcells_1, nbcells_2, nbcells_3):
    mod.add(md, co)

nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.1", "file_extension": ".py",
                      "mimetype": "text/x-python", "nbconvert_exporter": "python",
                      "pygments_lexer": "ipython3",
                      "codemirror_mode": {"name": "ipython", "version": 3}}}
nbf.write(nb, "TASK_A.ipynb")
print(f"wrote TASK_A.ipynb  ({len(CELLS)} cells: "
      f"{sum(c.cell_type=='markdown' for c in CELLS)} markdown, "
      f"{sum(c.cell_type=='code' for c in CELLS)} code)")
