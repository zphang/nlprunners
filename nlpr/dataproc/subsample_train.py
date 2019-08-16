import random

import pyutils.io as io
import pyutils.datastructures as datastructures


def subsample_train(base_config_path,
                    num_samples_per_class,
                    out_config_path,
                    out_data_path,
                    out_metadata_path):
    config = io.read_json(base_config_path)
    raw_train_examples = io.read_jsonl(config["paths"]["train"])

    new_config = config.copy()
    new_config["paths"]["train"] = out_data_path

    index_label_list = [
        {"idx": i, "label": example["label"]}
        for i, example in enumerate(raw_train_examples)
    ]

    grouped = datastructures.group_by(index_label_list, lambda _: _["label"])
    sorted_keys = sorted(list(grouped.keys()))

    sub_examples = []
    metadata = {}
    for key in sorted_keys:
        key_examples = grouped[key]
        indices = [_["idx"] for _ in key_examples]
        selected_key_examples = random.choices(indices, k=num_samples_per_class)
        sub_examples += [raw_train_examples[i] for i in selected_key_examples]
        metadata[key] = selected_key_examples

    io.create_containing_folder(out_config_path)
    io.create_containing_folder(out_data_path)
    io.create_containing_folder(out_metadata_path)

    io.write_json(new_config, out_config_path)
    io.write_jsonl(sub_examples, out_data_path)
    io.write_json(metadata, out_metadata_path)
