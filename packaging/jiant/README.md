### 0. Basic Setup 

Follow Steps 1-4 in [Simple Setup](../simple_setup.md)

### 1. Tokenize and Cache

```bash
python \
    nlprunners/nlpr/proj/simple/tokenize_and_cache.py \
    --task_config_path ${NLPR_BASE_DIR}/data/nlpr_data/configs/mrpc.json \
    --model_type roberta-base \
    --model_tokenizer_path ${NLPR_BASE_DIR}/models/roberta-base/tokenizer \
    --smart_truncate \
    --max_seq_length 256 \
    --chunk_size 10000 \
    --output_dir ${NLPR_BASE_DIR}/cache/roberta-base/mrpc
```

### 2. Write run config

```bash
python nlprunners/nlpr/proj/jiant/scripts/configurator.py \
    json \
    --func single_task_config \
    --path ./nlprunners/packaging/jiant/single_task_template.json \
    --output_base_path ${NLPR_BASE_DIR}/metadata/run_configs/mrpc
```

### 3. Run

```bash
python \
    nlprunners/nlpr/proj/jiant/runscript.py \
    run \
    --ZZsrc ${NLPR_BASE_DIR}/models/roberta-base/config.json \
    --ZZsrc ${NLPR_BASE_DIR}/metadata/run_configs/mrpc/zz_full.json \
    --force_overwrite \
    --do_train --do_val \
    --no_write_preds \
    --output_dir ${NLPR_BASE_DIR}/runs/roberta-base/mrpc
```