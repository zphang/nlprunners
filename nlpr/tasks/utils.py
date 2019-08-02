import numpy as np


def truncate_sequences(tokens_ls, max_length, truncate_end=True):
    if len(tokens_ls) == 0:
        return []
    if len(tokens_ls) == 1:
        if truncate_end:
            return [tokens_ls[0][:max_length]]
        else:
            return [tokens_ls[0][-max_length:]]
    lengths = np.array([len(tokens) for tokens in tokens_ls])
    total_length = lengths.sum()
    if total_length < max_length:
        return tokens_ls
    target_lengths = lengths
    while sum(target_lengths) > max_length:
        target_lengths[np.argmax(target_lengths)] -= 1

    return [
        tokens[:target_length] if truncate_end else tokens[-target_length:]
        for tokens, target_length in zip(tokens_ls, target_lengths)
    ]


def pad_to_max_seq_length(ls, max_seq_length, pad_idx=0, check=True):
    padding = [pad_idx] * (max_seq_length - len(ls))
    result = ls + padding

    if check:
        assert len(result) == max_seq_length
    return result


def convert_char_span_for_bert_tokens(text, bert_tokens, span_ls, check=True):
    bert_postsum = np.cumsum([
        len(token.replace("##", ""))
        for token in bert_tokens
    ])
    result_span_ls = []
    for span_start_idx, span_text in span_ls:
        before = text[:span_start_idx]
        chars_before = len(before.replace(" ", ""))
        span_chars = len("".join(span_text.split()))
        if chars_before == 0:
            start_idx = 0
        else:
            start_idx = np.argmax(bert_postsum == chars_before) + 1
        end_idx = np.argmax(bert_postsum == chars_before + span_chars) + 1  # exclusive
        result_span_ls.append([start_idx, end_idx])  # json compatibility

        if check:
            bert_chars_str = bert_tokens_to_text(bert_tokens[start_idx:end_idx])
            span_chars_str = "".join(span_text.split())
            assert bert_chars_str.lower() == span_chars_str.lower()
            assert bert_postsum[-1] == len(text.replace(" ", ""))
    return result_span_ls


def bert_tokens_to_text(bert_tokens):
    return "".join(bert_tokens).replace("##", "")


def convert_word_idx_for_bert_tokens(text, bert_tokens, word_idx_ls, check=True):
    text_tokens = text.split()
    span_ls = []
    for word_idx in word_idx_ls:
        if word_idx == 0:
            start_idx = 0
        else:
            start_idx = len(" ".join(text_tokens[:word_idx]) + " ")
        # end_idx = start_idx + len(word)
        span_ls.append([start_idx, text_tokens[word_idx]])
    return convert_char_span_for_bert_tokens(
        text=text,
        bert_tokens=bert_tokens,
        span_ls=span_ls,
        check=check,
    )
