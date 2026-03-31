import json

filepath = 'EDA_Avanzado_DualEngine_CCHS.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source_str = "".join(source)
        else:
            source_str = source
            
        if "mega_dataset_aeronautico.jsonl" in source_str and "ruta_dataset =" in source_str:
            new_source = source_str.replace(
                'import os\nruta_dataset = "dataset_sintetico/mega_dataset_aeronautico.jsonl" if os.path.exists("dataset_sintetico/mega_dataset_aeronautico.jsonl") else "../dataset_sintetico/mega_dataset_aeronautico.jsonl"',
                'ruta_dataset = "mega_dataset_aeronautico.jsonl"'
            )
            if isinstance(source, list):
                cell['source'] = [new_source]
            else:
                cell['source'] = new_source
                
        if "health_dataset.csv" in source_str and "ruta_csv =" in source_str:
            new_source = source_str.replace(
                'import os\nruta_csv = "datos_crudos/health_dataset.csv" if os.path.exists("datos_crudos/health_dataset.csv") else "../datos_crudos/health_dataset.csv"',
                'ruta_csv = "health_dataset.csv"'
            )
            if isinstance(source, list):
                cell['source'] = [new_source]
            else:
                cell['source'] = new_source

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Paths patched successfully for ROOT execution.")
