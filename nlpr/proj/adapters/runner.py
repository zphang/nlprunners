import pandas as pd
import math

import nlpr.shared.metarunner as metarunner
import nlpr.proj.adapters.multi_adapter_modeling as multi_adapter_modeling


class AdapterMetaRunner(metarunner.MetaRunner):
    def __init__(self, *args, **kwargs):
        # TODO: Temporary hack
        self.modified_layers = kwargs.pop("modified_layers")
        super().__init__(*args, **kwargs)
        self.steps_per_log = math.ceil(self.runner.train_schedule.t_total / 1000)

    def inject_at_step(self):
        if self.train_global_state.global_step % self.steps_per_log == 0:
            formatted_weights = {
                layer_name: pd.Series(weights).to_dict()
                for layer_name, weights
                in multi_adapter_modeling.get_multi_adapter_weight_dict(self.modified_layers).items()
            }
            self.log_writer.write_entry("weights", {
                "weights": formatted_weights,
                "tgs": self.train_global_state.asdict(),
            })
