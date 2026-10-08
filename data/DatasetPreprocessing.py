import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PREPROCESS_PARAMS = {'ENTITY_JSON': 'data/Immunotherapy/entity/entitylist.json',
 'POSITIVE_DIR': 'data/Immunotherapy/graph',
 'NEGATIVE_DIR': 'data/Immunotherapy/negative',
 'OUT_DIR': 'data/processed/Immunotherapy',
 'POS_PATTERN': 'KWgraph_all_keys_{time}.edgelist',
 'NEG_PATTERN': 'KWgraph_all_keys_{time}.edgelist',
 'TIME_POINTS': [1990, 2000, 2005, 2010, 2014, 2017, 2019, 2020, 2021, 2022, 2023, 2024],
 'SKIP_FIRST_TIME_SAMPLES': True,
 'POSITIVE_SNAPSHOT_MODE': 'cumulative',
 'MAX_DESC_TEXTS': 5,
 'DESC_RANDOM_SEED': 2026,
 'WRITE_TSV': True,
 'WRITE_ZST': True,
 'SAMPLE_SHUFFLE_SEED': 2026,
 'VALID_POS_SIZE': 5000,
 'VALID_NEG_SIZE': 5000,
 'VALID_RANDOM_SEED': 2026,
 'K_INIT': 32,
 'WEAK_ONLY_FOR_NEW_LINK': False}

SEMANTIC_PARAMS = { 'MODEL_NAME_OR_PATH': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
 'OUTPUT_FILE': 'data/entity_sem_emb.f16.npy',
 'BATCH_SIZE': 32,
 'MAX_LENGTH': 512,
 'NORMALIZE': True,
 'OUTPUT_DTYPE': 'float16',
 'DEVICE': 'auto',
 'SEED': 2026,
 'LOG_EVERY': 20}

STRUCTURAL_PARAMS = { 'SEMANTIC_EMB_FILE': 'data/entity_sem_emb.f16.npy',
 'ENTITY_FILE': 'data/entities.tsv',
 'STRUCT_EMB_DIR': 'data/struct_emb',
 'MODEL_CKPT_DIR': 'data/struct_emb/model_ckpt',
 'CONTEXT_TIME_POINTS': [1990, 2000, 2005, 2010, 2014, 2017, 2019, 2020, 2021, 2022, 2023],
 'HIDDEN_DIM': 256,
 'OUT_DIM': 256,
 'NUM_LAYERS': 2,
 'DROPOUT': 0.15,
 'INCREMENTAL_TRAINING': True,
 'RESET_OPTIMIZER_EACH_TIME': True,
 'FIRST_TIME_EPOCHS': 2000,
 'INCREMENTAL_EPOCHS': 500,
 'EPOCHS_WHEN_TRAIN_FROM_SCRATCH': 20,
 'LR': 0.001,
 'WEIGHT_DECAY': 1e-05,
 'EDGE_BATCH_SIZE': 65536,
 'NEGATIVE_RATIO': 1.0,
 'NEGATIVE_SAMPLE_NODE_SCOPE': 'active',
 'USE_FREQ_AS_POS_WEIGHT': False,
 'MAX_FREQ_WEIGHT': 10.0,
 'NORMALIZE_OUTPUT': True,
 'OUTPUT_DTYPE': 'float16',
 'DEVICE': 'auto',
 'SEED': 2026,
 'SKIP_IF_EXISTS': False,
 'LOAD_CKPT_WHEN_SKIP': True,
 'SAVE_MODEL_CHECKPOINT': True,
 'LOG_EVERY_EPOCH': 1}


def main(stage="all"):
    if stage not in {"all", "dataset", "semantic", "structure"}:
        raise ValueError(f"Unknown preprocessing stage: {stage}")
    cfg = {
        "preprocessing": dict(PREPROCESS_PARAMS),
        "semantic_embeddings": dict(SEMANTIC_PARAMS),
        "structural_embeddings": dict(STRUCTURAL_PARAMS),
    }
    for key in ("ENTITY_JSON", "POSITIVE_DIR", "NEGATIVE_DIR", "OUT_DIR"):
        cfg["preprocessing"][key] = str((REPO_ROOT / cfg["preprocessing"][key]).resolve())
    params = cfg["preprocessing"]

    graph_dirs = [Path(params["POSITIVE_DIR"]), REPO_ROOT / "data/Immunotherapy/graph_new"]
    for directory in graph_dirs:
        if all((directory / params["POS_PATTERN"].format(time=t)).is_file() for t in params["TIME_POINTS"]):
            params["POSITIVE_DIR"] = str(directory)
            break
    local_encoder = REPO_ROOT / "models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    if local_encoder.is_dir():
        cfg["semantic_embeddings"]["MODEL_NAME_OR_PATH"] = str(local_encoder)

    for section in ("semantic_embeddings", "structural_embeddings"):
        cfg[section]["PROCESSED_ROOT"] = cfg["preprocessing"]["OUT_DIR"]
    if stage in ("all", "dataset"):
        params = cfg["preprocessing"]
        paths = [Path(params["ENTITY_JSON"])]
        for year in params["TIME_POINTS"]:
            paths.append(Path(params["POSITIVE_DIR"]) / params["POS_PATTERN"].format(time=year))
        for year in params["TIME_POINTS"][1:]:
            paths.append(Path(params["NEGATIVE_DIR"]) / params["NEG_PATTERN"].format(time=year))
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing raw dataset files:\n" + "\n".join(missing))
        from data.preprocess import make_builder_from_config
        make_builder_from_config(cfg["preprocessing"]).run()
    if stage in ("all", "semantic"):
        from data.semantic_embeddings import main as build_semantic
        build_semantic(cfg["semantic_embeddings"])
        import yaml
        meta_path = Path(cfg["preprocessing"]["OUT_DIR"]) / "data/meta.yaml"
        with meta_path.open(encoding="utf-8") as handle:
            meta = yaml.safe_load(handle)
        meta["semantic_embedding"] = {
            "enabled": True,
            "file": cfg["semantic_embeddings"]["OUTPUT_FILE"],
            "dtype": cfg["semantic_embeddings"]["OUTPUT_DTYPE"],
            "row_index": "eid",
            "text_input": "name_plus_desc",
        }
        with meta_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(meta, handle, allow_unicode=True, sort_keys=False)
    if stage in ("all", "structure"):
        from data.structural_embeddings import main as build_structure
        build_structure(cfg["structural_embeddings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
