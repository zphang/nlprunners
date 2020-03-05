"""
TODO: Remove when Tokenizers gets better
"""
import transformers as ptt


def map_tags_to_token_position(flat_stripped, indices, split_text):
    char_index = 0
    current_string = flat_stripped
    positions = [None] * len(split_text)
    for i, token in enumerate(split_text):
        found_index = current_string.find(token.lower())
        assert found_index != -1
        positions[i] = indices[char_index + found_index]
        char_index += found_index + len(token)
        current_string = flat_stripped[char_index:]
    for elem in positions:
        assert elem is not None
    return positions


def convert_mapped_tags(positions, tag_ids, length):
    labels = [None] * length
    mask = [0] * length
    for pos, tag_id in zip(positions, tag_ids):
        labels[pos] = tag_id
        mask[pos] = 1
    return labels, mask


def input_flat_strip(tokens):
    return "".join(tokens).lower()


def delegate_flat_strip(tokens, tokenizer, return_indices=False):
    if isinstance(tokenizer, ptt.BertTokenizer):
        return bert_flat_strip(tokens=tokens, return_indices=return_indices)
    elif isinstance(tokenizer, ptt.RobertaTokenizer):
        return roberta_flat_strip(tokens=tokens, return_indices=return_indices)
    elif isinstance(tokenizer, ptt.AlbertTokenizer):
        return albert_flat_strip(tokens=tokens, return_indices=return_indices)
    else:
        raise KeyError(type(tokenizer))


def bert_flat_strip(tokens, return_indices=False):
    ls = []
    count = 0
    indices = []
    for token in tokens:
        if token.startswith("##"):
            token = token.replace("##", "")
        else:
            pass
        ls.append(token)
        indices += [count] * len(token)
        count += 1
    string = "".join(ls).lower()
    if return_indices:
        return string, indices
    else:
        return string


def roberta_flat_strip(tokens, return_indices=False):
    ls = []
    count = 0
    indices = []
    for token in tokens:
        if token.startswith("Ġ"):
            token = token.replace("Ġ", "")
        else:
            pass
        ls.append(token)
        indices += [count] * len(token)
        count += 1
    string = "".join(ls).lower()
    if return_indices:
        return string, indices
    else:
        return string


def albert_flat_strip(tokens, return_indices=False):
    ls = []
    count = 0
    indices = []
    for token in tokens:
        token = token.replace('"', "``")
        if token.startswith("▁"):
            token = token[1:]
        else:
            pass
        ls.append(token)
        indices += [count] * len(token)
        count += 1
    string = "".join(ls).lower()
    if return_indices:
        return string, indices
    else:
        return string


def starts_with(ls, prefix):
    return ls[:len(prefix)] == prefix
