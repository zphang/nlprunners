import os
import sacremoses

import zconf
import pyutils.io as io
import pyutils.display as display


def preprocess_wikitext103_data(input_base_path, output_base_path):
    os.makedirs(output_base_path, exist_ok=True)
    detokenizer = sacremoses.MosesDetokenizer()
    path_map = {
        #"train": "wiki.train.raw",
        "val": "wiki.valid.raw",
        "test": "wiki.test.raw",
    }

    for phase, filename in path_map.items():
        input_path = os.path.join(input_base_path, filename)
        output_path = os.path.join(output_base_path, f"{phase}.txt")
        num_lines = io.get_num_lines(input_path)
        with open(input_path) as f_in, open(output_path, "w") as f_out:
            for line in display.tqdm(f_in, total=num_lines):
                line = line.strip()
                if not line or line.startswith("= "):
                    continue
                else:
                    line = line.replace(" @.@ ", ".")
                    line = line.replace(" @,@ ", ",")
                    line = line.replace(" @,@", ",")
                    line = line.replace(" – ", "–")
                    _ = f_out.write(detokenizer.detokenize(line.split()) + "\n")


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    task_name = zconf.attr(type=str)
    input_base_path = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)


def main(args: RunConfiguration):
    if args.task_name == "mlm_wikitext103":
        preprocess_wikitext103_data(
            input_base_path=args.input_base_path,
            output_base_path=args.output_base_path,
        )


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())