from etl.config import load_config
from etl.indexer import build_index
from run import run_experiment

cfg = load_config("rflymad_etl.yaml")
index_df = build_index(cfg)

print("总文件数:", len(index_df))
print(index_df.head())
print(index_df["domain"].value_counts())


result = run_experiment("rflymad_etl.yaml")
result
