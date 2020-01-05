# NLP Runners

This repository contains code for various NLP fine-tuning/transfer learning experiments, including semi-supervised learning, multi-task training, and adapters. 

**This codebase is heavily WIP.** 

### Quick Setup

* For a quick environment setup, see: [Simple Setup](packaging)

### Dependencies

These are the main notable dependencies. For more, see: [Simple Setup](packaging)

* PyTorch 1.2+
* HuggingFace/Transformers (usually the latest version. Currently 2.3.0)
* My own Python utility libraries: [zutils](https://github.com/zphang/zutils)

### Overview

#### Running

* Different research projects can be found in [nlpr/proj](nlpr/proj). The basic fine-tuning version can be found in [nlpr/proj/simple](nlpr/proj/simple).
* Each proj has one or more run scripts (`runscript.py`). Run scripts are the command-line scripts for kicking off a run, but also a good entry point for reading code.
* Run scripts use a `zconf.RunConfiguration` object, which allows for easy command-line or in-session instantiation of arguments. Importantly, you can use the `--ZZsrc {path.json}` argument to specify a JSON file that provides keys/values that correspond to the attributes of the `RunConfiguration` for more convenient instantiation of a configuration/script.
* More broadly, we make heavy use of JSON files for various configuration (e.g. model configs, task configs)
* `Runner` objects contain the core logic for the training/eval loop of a project. Often, the goal of a runscript is simply to setup the `Runner` object and let the `Runner` object do all the work.

#### Tasks

* Tasks are defined in [nlpr/tasks/lib](nlpr/tasks/lib), one per file.
* Each task broadly needs to specify the following: 
    * loading data
    * tokenization (`Example.tokenize`), giving a `TokenizedExmaple`. 
    * featurization (`TokenizedExample.featurize`), giving a `DataRow`. This converts the tokenized data into a format that the model can take in (e.g. concatenating inputs, truncating sequence length, adding `[SEP]` tokens.)
    * Conversion to a `Batch` (Batch.from_data_rows`), that our dataloaders know how to split up ` 

### Guiding Principles

* Use simple data formats (dictionaries, JSON/JSONL for serialization)
* Code should be straightforward to run either on command-line or within notebooks. See: `zconf` from [zutils](https://github.com/zphang/zutils)
* Explicit is better than implicit. Use classes rather than dicts for known data structures, refrain from using `kwargs`, use keyword arguments where possible, etc
* Verbose is better than implicit. Use an IDE.
* There should be a clean separation of messy "research" code, and solid "software engineering" code.
