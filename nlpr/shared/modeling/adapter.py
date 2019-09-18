
def load_non_adapter_base_weights(model, state_dict):
    curr_state_dict = model.state_dict()
    for k, v in state_dict.items():
        if k in curr_state_dict:
            curr_state_dict[k] = v
    model.load_state_dict(curr_state_dict)
