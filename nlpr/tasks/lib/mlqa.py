import json
import tqdm

from dataclasses import dataclass

import nlpr.tasks.lib.templates.squad_style as squad_style_template
from nlpr.constants import PHASE


@dataclass
class Example(squad_style_template.Example):

    def tokenize(self, tokenizer):
        raise NotImplementedError("SQuaD is weird")


@dataclass
class DataRow(squad_style_template.DataRow):
    pass


@dataclass
class Batch(squad_style_template.Batch):
    pass


class MlqaTask(squad_style_template.BaseSquadStyleTask):
    Example = Example
    DataRow = DataRow
    Batch = Batch

    # Todo: hack-ish to remove title, refactor
    @classmethod
    def read_squad_examples(cls, path, set_type):
        with open(path, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        is_training = set_type == PHASE.TRAIN
        examples = []
        for entry in tqdm.tqdm(input_data, desc="Reading SQuAD Data [1]"):
            title = "-"
            for paragraph in tqdm.tqdm(entry["paragraphs"], desc="Reading SQuAD Data [2]"):
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = Example(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples
