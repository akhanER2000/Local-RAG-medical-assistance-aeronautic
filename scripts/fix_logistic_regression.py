import json

notebook_path = "notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Buscamos en cada línea de la celda
        for i, line in enumerate(cell['source']):
            if "LogisticRegression(" in line and "multi_class" in line:
                # Reemplazamos el parámetro obsoleto
                cell['source'][i] = line.replace("multi_class='multinomial', ", "")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook corregido: Parámetro 'multi_class' eliminado de LogisticRegression.")
