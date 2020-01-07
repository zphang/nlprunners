# Simple Setup

This README goes over the procedure for setting up a working environment for NLPRunners, other than things like setting up CUDA and Conda.

**You should run every line of bash commands in this README.** 

### 0. Choose the working directory

Set `NLPR_BASE_DIR` as our base directory where we'll setup everything.

```bash
export NLPR_BASE_DIR=/home/zphang/working/bowman/nlpr1
``` 

### 1. Directory Setup

Set up our working directory and clone some repositories.

```bash
mkdir -p ${NLPR_BASE_DIR}
mkdir -p ${NLPR_BASE_DIR}/code
mkdir -p ${NLPR_BASE_DIR}/data
mkdir -p ${NLPR_BASE_DIR}/models
mkdir -p ${NLPR_BASE_DIR}/working
cd ${NLPR_BASE_DIR}/code
git clone git@github.com:zphang/zutils.git
git clone git@github.com:zphang/nlprunners.git
git clone git@github.com:huggingface/transformers.git
```

### 2. Environment Setup

Create a conda environment, modify our `PYTHONPATH`, and install some packages.

```bash
conda create --prefix ${NLPR_BASE_DIR}/env python=3.7 -y
conda activate ${NLPR_BASE_DIR}/env
export PYTHONPATH=${NLPR_BASE_DIR}/code/transformers/src:$PYTHONPATH
export PYTHONPATH=${NLPR_BASE_DIR}/code/zutils:$PYTHONPATH
export PYTHONPATH=${NLPR_BASE_DIR}/code/nlprunners:$PYTHONPATH
conda install -y pytorch cudatoolkit=9.2 -c pytorch
conda install -y pandas jupyter notebook lxml scikit-learn
pip install tokenizers tqdm boto3 requests filelock sentencepiece regex sacremoses bs4 overrides
```

### 3. Data Setup

Download GLUE data, and convert to JSONL format.

```bash
cd ${NLPR_BASE_DIR}/code
python transformers/utils/download_glue_data.py \
	--data_dir ${NLPR_BASE_DIR}/data/raw_glue_data
python nlprunners/nlpr/preproc/preproc_glue_data.py \
	--input_base_path ${NLPR_BASE_DIR}/data/raw_glue_data \
	--output_base_path ${NLPR_BASE_DIR}/data/nlpr_data
```

### 4. Data Setup

Download the weights for a given model (in this case *RoBERTa-base*) via HuggingFace, and export. (We do this to have explicit control over the weights dict, tokenization and configuration, as opposed to using HuggingFace's setup.) 

```bash
python nlprunners/nlpr/preproc/export_model.py \
	--model_type roberta-base \
	--output_base_path ${NLPR_BASE_DIR}/models/roberta-base
```

### 5. Train!

Fine-tune *RoBERTa-Base* to MRPC. Note the use of the `--ZZsrc` flag, which uses values from `models/roberta-base/config.json` as additional command-line arguments, so we don't have to individually specify `--model_type`, `--model_path`, `--model_config_path`, and `--model_tokenizer_path`.

```bash
python nlprunners/nlpr/proj/simple/runscript.py \
	--ZZsrc ${NLPR_BASE_DIR}/models/roberta-base/config.json \
	--task_config_path ${NLPR_BASE_DIR}/data/nlpr_data/configs/mrpc.json \
	--train_batch_size 4 \
	--eval_every_steps 500 --partial_eval_number 500 \
	--do_train --do_val --do_save \
	--output_dir ${NLPR_BASE_DIR}/working/roberta_base___mrpc
```
