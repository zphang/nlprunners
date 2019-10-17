import pandas as pd

import nlpr.shared.metarunner as metarunner
import nlpr.proj.adapters.multi_adapter_modeling as multi_adapter_modeling


class AdapterMetarunner(metarunner.MetaRunner):
    def __init__(self, *args, **kwargs):
        # TODO: Temporary hack
        self.modified_layers = kwargs.pop("modified_layers")
        super().__init__(*args, **kwargs)

    def inject_at_step(self):
        formatted_weights = {
            layer_name: pd.Series(weights).to_dict()
            for layer_name, weights
            in multi_adapter_modeling.get_multi_adapter_weight_dict(self.modified_layers).items()
        }

        self.log_writer.write_entry("weights", {
            "weights": formatted_weights,
            "tgs": self.train_global_state.asdict(),
        })
