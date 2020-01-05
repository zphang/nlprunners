
export NLPR_BASE_DIR=/home/zphang/working/bowman/nlpr1
mkdir -p $NLPR_BASE_DIR
mkdir -p $NLPR_BASE_DIR/code
mkdir -p $NLPR_BASE_DIR/data
mkdir -p $NLPR_BASE_DIR/models
mkdir -p $NLPR_BASE_DIR/working
cd $NLPR_BASE_DIR/code
git clone git@github.com:zphang/zutils.git
git clone git@github.com:zphang/nlprunners.git
git clone git@github.com:huggingface/transformers.git
conda create --prefix $NLPR_BASE_DIR/env python=3.7 -y
conda activate $NLPR_BASE_DIR/env
export PYTHONPATH=$NLPR_BASE_DIR/code/transformers/src:$PYTHONPATH
export PYTHONPATH=$NLPR_BASE_DIR/code/zutils:$PYTHONPATH
export PYTHONPATH=$NLPR_BASE_DIR/code/nlprunners:$PYTHONPATH

# --
conda install -y pytorch cudatoolkit=9.2 -c pytorch
conda install -y pandas jupyter notebook lxml scikit-learn
pip install tokenizers tqdm boto3 requests filelock sentencepiece regex sacremoses bs4 overrides

cd $NLPR_BASE_DIR/code
python transformers/utils/download_glue_data.py \
	--data_dir $NLPR_BASE_DIR/data/raw_glue_data
python nlprunners/nlpr/preproc/preproc_glue_data.py \
	--input_base_path $NLPR_BASE_DIR/data/raw_glue_data \
	--output_base_path $NLPR_BASE_DIR/data/nlpr_data
python nlprunners/nlpr/preproc/export_model.py \
	--model_type roberta-base \
	--output_base_path $NLPR_BASE_DIR/models/roberta-base
python nlprunners/nlpr/proj/simple/runscript.py \
	--ZZsrc $NLPR_BASE_DIR/models/roberta-base/config.json \
	--task_config_path $NLPR_BASE_DIR/data/nlpr_data/configs/mrpc.json \
	--train_batch_size 4 \
	--eval_every_steps 500 --partial_eval_number 5000 \
	--do_train --do_val \
	--output_dir $NLPR_BASE_DIR/working/roberta_base___mrpc






