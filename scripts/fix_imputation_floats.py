import json

notebook_path = "notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "PASO 1: Data Quality Report" in source:
            fix_code = (
                "# Redondeo post-imputación para recuperar variables categóricas discretas\n"
                "cols_cat = ['Gen_health_state', 'Mental_health_state', 'Stress_level', 'Work_stress', 'Smoked', 'Weekly_alcohol']\n"
                "for c in cols_cat:\n"
                "    if c in df_tab_sample.columns:\n"
                "        df_tab_sample[c] = df_tab_sample[c].round().astype(int)\n\n"
            )
            # Find where to insert it safely after the imports
            if "import pandas as pd" in source:
                new_source = source.replace("import pandas as pd\n", "import pandas as pd\n\n" + fix_code)
            else:
                new_source = fix_code + source
                
            # Reconstruct source list
            cell['source'] = [line + ('\n' if i < len(new_source.split('\n')) - 1 else '') for i, line in enumerate(new_source.split('\n'))]
            # Limpieza de saltos vacíos extra
            cell['source'] = [s for s in cell['source'] if s]
            
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook corregido: Redondeo de variables categóricas aplicado.")
