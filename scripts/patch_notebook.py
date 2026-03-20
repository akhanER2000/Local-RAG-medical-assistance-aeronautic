import json

with open('notebooks/EDA_Aeronautico.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source_str = "".join(source)
        else:
            source_str = source
            
        if 'df.groupby' in source_str and 'sample' in source_str:
            # Replace the tricky pandas apply block with a 100% robust loop
            new_source = source_str.replace(
                "df_muestra = df.groupby('fuente', group_keys=False).apply(lambda x: x.sample(min(len(x), 300), random_state=42)).reset_index(drop=True)",
                "muestras = []\nfor nombre, grupo in df.groupby('fuente'):\n    muestras.append(grupo.sample(min(len(grupo), 300), random_state=42))\ndf_muestra = pd.concat(muestras).reset_index(drop=True)"
            )
            
            # Now assign it back based on format
            if isinstance(source, list):
                # We can just put the whole string as the first element and clear the rest, Jupyter handles it.
                cell['source'] = [new_source]
            else:
                cell['source'] = new_source

with open('notebooks/EDA_Aeronautico.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook parched successfully.")
