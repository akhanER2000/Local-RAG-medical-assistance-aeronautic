import json

filepath = 'notebooks/EDA_Avanzado_DualEngine_CCHS.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source_str = "".join(source)
        else:
            source_str = source
            
        # Parchear celda 2
        if "os.path.exists" in source_str and "mega_dataset_aeronautico.jsonl" in source_str:
            new_source = source_str.replace(
                'ruta_dataset = "dataset_sintetico/mega_dataset_aeronautico.jsonl" if os.path.exists("dataset_sintetico/mega_dataset_aeronautico.jsonl") else "../dataset_sintetico/mega_dataset_aeronautico.jsonl"',
                'import os\nruta_dataset = "dataset_sintetico/mega_dataset_aeronautico.jsonl" if os.path.exists("dataset_sintetico/mega_dataset_aeronautico.jsonl") else "../dataset_sintetico/mega_dataset_aeronautico.jsonl"'
            )
            if isinstance(source, list):
                cell['source'] = [new_source]
            else:
                cell['source'] = new_source
                
        # Parchear celda 6
        if "os.path.exists" in source_str and "health_dataset.csv" in source_str:
            new_source = source_str.replace(
                'ruta_csv = "datos_crudos/health_dataset.csv" if os.path.exists("datos_crudos/health_dataset.csv") else "../datos_crudos/health_dataset.csv"',
                'import os\nruta_csv = "datos_crudos/health_dataset.csv" if os.path.exists("datos_crudos/health_dataset.csv") else "../datos_crudos/health_dataset.csv"'
            )
            if isinstance(source, list):
                cell['source'] = [new_source]
            else:
                cell['source'] = new_source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook parcheado exitosamente.")
