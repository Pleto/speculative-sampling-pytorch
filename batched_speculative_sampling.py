from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
import torch

from common import Benchmark, ModelFn, model_fn, set_seed


def extend_sequence(sequence, attention_mask, next_tokens, pad_token_id):
    '''
    Extends token sequences by inserting new tokens at the first 0 position
    in the attention mask

    :param sequence: the token sequence to extend. Shape is (batch, seq_len)
    :param attention_mask: the attention mask for the sequence. Shape is (batch, seq_len)
    :param next_tokens: the tokens to insert. Shape is (batch, 1)
    :param pad_token_id: the ID of the padding token

    :return: the extended sequence. Shape is (batch, seq_len + 1)

    Example:
    ```
    sequence = torch.IntTensor([
        [ 7608,   1,  1],
        [  100, 437,  1],
        [  100, 437, 45],
        [12375,  16,  5],
    ])

    attention_mask = torch.IntTensor([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ])

    next_tokens = torch.IntTensor([
        [754],
        [754],
        [ 16],
        [121],
    ])

    output = extend_sequence(sequence, attention_mask, next_tokens, 1)
    # output should be:
    # tensor([[  7608, 754,   1,   1],
    #         [   100, 437, 754,   1],
    #         [   100, 437,  45,  16],
    #         [ 12375,  16,   5, 121]])

    ```
    '''
    batch_size, _ = attention_mask.shape
    insert_positions = attention_mask.sum(dim=1, dtype=torch.int, keepdim=True)
    
    if ((insert_positions >= sequence.size(1))).any():
        sequence = torch.cat([sequence, torch.full((batch_size, 1), pad_token_id, dtype=sequence.dtype)], dim=1)

    batch_indices = torch.arange(batch_size).unsqueeze(1)
    sequence[batch_indices, insert_positions] = next_tokens

    return sequence


def extend_attention_mask(attention_mask):
    '''
    Extends attention masks by adding a new column of 1s at the end of each
    sequence

    :param attention_mask: the attention mask to extend. Shape is (batch, seq_len)

    :return: the extended attention mask. Shape is (batch, seq_len + 1)

    Example:
    ```
    attention_mask = torch.IntTensor([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ])

    output = extend_attention_mask(attention_mask)
    # output should be:
    # tensor([[1, 1, 0, 0],
    #         [1, 1, 1, 0],
    #         [1, 1, 1, 1],
    #         [1, 1, 1, 1]])
    ```
    '''
    batch_size, _ = attention_mask.shape
    next_tokens = torch.ones(batch_size, dtype=attention_mask.dtype).unsqueeze(-1)
    return extend_sequence(attention_mask, attention_mask, next_tokens, 0)


def accept(sequence, mask, active_mask, accepted, pad_token_id):
    '''
    Merges accepted tokens into active sequences in a batch, extending the sequences and
    their corresponding masks

    :param sequence: token sequences with shape [batch_size, seq_length]
    :param mask: integer mask indicating valid tokens (1) vs padding (0) in the sequence
    :param active_mask: boolean mask of shape [batch_size] indicating which sequences in
                        the batch should receive new tokens.
    :param accepted: tensor of tokens to append to active sequences, with shape
                     [num_active, max_accepted_length], where
                     num_active == active_mask.sum()
    :param pad_token_id: token ID used for padding

    :return new_seq: updated sequences with accepted tokens appended to active sequences
    :return new_mask: updated mask reflecting the new valid token positions.

    Example:
    ```
    sequence = torch.IntTensor([
        [16, 1, 1],
        [50118, 132, 15],
        [50118, 1, 1],
        [1, 1, 1]
    ])

    mask = torch.IntTensor([
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
        [0, 0, 0]
    ])

    active_mask = torch.BoolTensor([False, True, False, True])

    # all 0 columns at the end will be trimmed
    accepted = torch.IntTensor([
        [14, 0, 0, 0],
        [15, 16, 0, 0],
    ])

    new_seq, new_mask = accept(sequence, mask, active_mask, accepted, 1)

    # new_seq should be:
    # tensor([[16, 1, 1, 1],
    #         [50118, 132, 15, 14],
    #         [50118, 1, 1, 1],
    #         [15, 16, 1, 1]])
    #
    # new mask should be:
    # tensor([[1, 0, 0, 0],
    #         [1, 1, 1, 1],
    #         [1, 0, 0, 0],
    #         [1, 1, 0, 0]
    # ])
    ```
    '''
    batch_size = sequence.size(0)
    orig_valid = mask.sum(dim=1).tolist()
    
    accepted_tokens = []
    accepted_idx = 0
    for is_active in active_mask:
        if is_active:
            filtered = accepted[accepted_idx][accepted[accepted_idx] != 0]
            accepted_tokens.append(filtered)
            accepted_idx += 1
        else:
            accepted_tokens.append(torch.tensor([], device=sequence.device))
    
    total_lengths = [orig + len(acc) for orig, acc in zip(orig_valid, accepted_tokens)]
    new_seq_len = max(total_lengths)
    
    new_seq = torch.full((batch_size, new_seq_len), pad_token_id, 
                        dtype=sequence.dtype, device=sequence.device)
    new_mask = torch.zeros_like(new_seq)
    
    for i in range(batch_size):
        orig_end = orig_valid[i]
        if orig_end > 0:
            new_seq[i, :orig_end] = sequence[i, :orig_end]
            new_mask[i, :orig_end] = 1
        
        acc_tokens = accepted_tokens[i]
        if len(acc_tokens) > 0:
            acc_end = orig_end + len(acc_tokens)
            new_seq[i, orig_end:acc_end] = acc_tokens
            new_mask[i, orig_end:acc_end] = 1
    
    return new_seq, new_mask


@torch.no_grad()
def batched_speculative_sampling(
    prefix: torch.Tensor,
    attention_mask: torch.Tensor,
    verifier: ModelFn,
    drafter: ModelFn,
    max_new_tokens: int,
    gamma: int = 4,
) -> torch.Tensor:
    '''
    :param prefix: input token IDs to start generation from. Shape is (batch, prefix_len)
    :param attention_mask: binary tensor indicating which tokens to attend to (1=visible, 0=masked).
                           Shape is (batch, prefix_len)
    :param verifier: the larger, more accurate model used to verify the drafted tokens
    :param drafter: the smaller, faster model used to draft candidate token
    :param max_new_tokens: maximum number of tokens to generate
    :param gamma: the number of tokens the drafter guesses

    :return: generated tokens including the prefix. Shape is (batch, prefix_len + max_new_tokens)
    '''
    pad_token_id = 1

    sequence = prefix
    mask = attention_mask
    
    batch_size, _ = sequence.shape

    # indices to keep track of the original order of the sequences
    indices = torch.arange(batch_size, dtype=torch.int)

    # boolean tensor to keep track for what sequences I'm still generating tokens
    active_mask = torch.ones(batch_size, dtype=torch.bool)

    # keep track of how many tokens I've generated for each sequence
    num_generated_tokens = torch.zeros(batch_size, dtype=torch.int)

    while active_mask.any():
        spec_sequence = sequence[active_mask]
        spec_mask = mask[active_mask]

        active_size = active_mask.sum()
        active_indices = torch.arange(active_size)

        spec_steps = min(max_new_tokens - num_generated_tokens.min().item(), gamma)
        if spec_steps <= 0:
            break

        draft_logits_history = []

        # pos keeps the position of the last generated token in each sequence
        pos = mask.sum(dim=1, keepdim=True).squeeze()
        if pos.numel() > 1:
            pos = pos[active_mask]
            
        for _ in range(spec_steps):
            draft_logits = drafter(spec_sequence, spec_mask)

            last = draft_logits[active_indices, pos - 1, :]
            draft_logits_history.append(last)

            next_tokens = torch.argmax(last, dim=-1, keepdim=True)

            spec_sequence = extend_sequence(spec_sequence, spec_mask, next_tokens, pad_token_id)
            spec_mask = extend_attention_mask(spec_mask)

            pos += 1

        draft_probs_history = [torch.softmax(logits, dim=-1) for logits in draft_logits_history]
        draft_probs_history = torch.stack(draft_probs_history)

        verify_logits = verifier(spec_sequence, spec_mask)
        verify_probs = torch.softmax(verify_logits, dim=-1)

        # the indices of the active sequences. these DON'T correspond to the indices of the
        # whole batch, but to the active sequences only
        active_indices = torch.arange(active_size)
        # the tokens that the verifier has accepted
        accepted_tokens = torch.zeros(active_size, gamma + 1, dtype=sequence.dtype)

        # here pos is the position of each generated token per sequence
        pos = mask.sum(dim=1, keepdim=True).squeeze()
        if pos.numel() > 1:
            pos = pos[active_mask]

        # for what sequences we have accepted the draft token
        accept_decisions = torch.ones(active_size, dtype=torch.bool)

        for i in range(spec_steps):
            if not accept_decisions.any():
                break

            draft_tokens = spec_sequence[active_indices, pos]

            def prob_of_tokens_in_distribution(tokens, distribution):
                return distribution.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

            draft_prob = prob_of_tokens_in_distribution(draft_tokens, draft_probs_history[i, active_indices, :])
            verify_prob = prob_of_tokens_in_distribution(draft_tokens, verify_probs[active_indices, pos - 1, :])

            acceptance_ratio = torch.min(torch.ones(active_size), verify_prob / draft_prob)
            random_values = torch.rand(active_size)
            accept_decisions = accept_decisions & (random_values < acceptance_ratio)

            accepted_tokens[active_indices[accept_decisions], i] = draft_tokens[accept_decisions]

            pos += 1

        accepted_tokens_len = accepted_tokens.count_nonzero(dim=1)
        need_resampling = ~accept_decisions

        if need_resampling.any():
            resampling_active_indices = active_indices[need_resampling]

            # here pos will indicate the position of the token that needs resampling
            pos = mask.sum(dim=1, keepdim=True).squeeze()
            if pos.numel() > 1:
                pos = pos[resampling_active_indices]

            _accepted_tokens_len = accepted_tokens_len[resampling_active_indices]
            pos += _accepted_tokens_len

            v_probs = verify_probs[resampling_active_indices, pos - 1, :]
            d_probs = draft_probs_history[_accepted_tokens_len - 1, resampling_active_indices]
            
            adjusted_probs = torch.clamp(v_probs - d_probs, min=0)
            sum_adjusted = torch.clamp(adjusted_probs.sum(dim=-1, keepdim=True), min=1e-12)
            adjusted_probs /= sum_adjusted

            next_tokens = torch.argmax(adjusted_probs, dim=-1, keepdim=True)
            accepted_tokens[resampling_active_indices, _accepted_tokens_len] = next_tokens.squeeze()

        if (~need_resampling).any():
            append_active_indices = active_indices[~need_resampling]
            next_tokens = torch.argmax(verify_probs[append_active_indices, -1, :], dim=-1, keepdim=True)
            _accepted_tokens_len = accepted_tokens_len[append_active_indices]
            accepted_tokens[append_active_indices, _accepted_tokens_len] = next_tokens.squeeze()

        num_generated_tokens[active_mask] += accepted_tokens_len + 1
        sequence, mask = accept(sequence, mask, active_mask, accepted_tokens, pad_token_id)

        finished_sequences = indices[num_generated_tokens >= max_new_tokens]

        if finished_sequences.numel() > 0:
            active_mask[finished_sequences] = False

    return sequence


if __name__ == '__main__':
    set_seed(42)

    verifier_id = 'facebook/opt-350m'
    drafter_id = 'facebook/opt-125m'

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(verifier_id)

    verifier_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(verifier_id)
    verifier = model_fn(verifier_model)

    drafter_model = AutoModelForCausalLM.from_pretrained(drafter_id)
    drafter = model_fn(drafter_model)

    input_texts = ["Why", "I'm not convinced of the", "I'm not convinced of the", "Who is the president of the united states?"]
    encoded_input = [tokenizer.encode(text, return_tensors='pt') for text in input_texts]
    batched_encoded_input = tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')

    max_new_tokens = 10

    with Benchmark('batched without speculative sampling'):
        verifier_model.generate(
            batched_encoded_input.input_ids,
            attention_mask=batched_encoded_input.attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )

    with Benchmark('batched with speculative sampling'):
        output = batched_speculative_sampling(
            batched_encoded_input.input_ids,
            batched_encoded_input.attention_mask,
            verifier,
            drafter,
            max_new_tokens,
        )
    
    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(output)