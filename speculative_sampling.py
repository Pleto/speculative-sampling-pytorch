import time
from typing import Callable
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
import torch
import numpy as np
import random
import os

type ModelFn = Callable[[torch.Tensor], torch.Tensor]

def model_fn(model: PreTrainedModel) -> ModelFn:
    def forward(input: torch.Tensor):
        output = model(input, use_cache=False)
        return output.logits
    
    return forward

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

@torch.no_grad()
def speculative_sampling(
    prefix: torch.Tensor,
    verifier: ModelFn,
    drafter: ModelFn,
    max_new_tokens: int,
    gamma: int = 4,
) -> torch.Tensor:
    """
    :param prefix: input token IDs to start generation from. Shape is (batch, prefix_len)
    :param tokenizer: 
    :param verifier: the larger, more accurate model used to verify the drafted tokens
    :param max_new_tokens:
    :param drafter: the smaller, faster model used to draft candidate token
    :param gamma: the number of tokens the drafter guesses

    :return: generated tokens including the prefix. Shape is (batch, seq_len)
    """

    sequence = prefix
    assert sequence.shape[0] == 1, 'Only batch_size == 1 supported'

    target_length = sequence.size(1) + max_new_tokens

    drafted_count = 0
    resample_count = 0
    accepted_count = 0

    while sequence.size(1) < target_length:
        speculated_sequence = sequence

        remaining = target_length - sequence.size(1)
        spec_steps = min(remaining, gamma)

        # we need the logits for each generated token to compare against the
        # logits of the verifier
        draft_logits_history = []

        for _ in range(spec_steps):
            drafted_count += 1
            
            draft_logits = drafter(speculated_sequence)
            # draft_logits has shape [batch_size, sequence_length, vocabulary_size].
            # To get the next token we need to select only the last position in
            # the sequence
            last = draft_logits[:, -1, :]
            draft_logits_history.append(last)
            
            # greedy decoding for now :(
            next_tokens = torch.argmax(last, dim=-1, keepdim=True)
            speculated_sequence = torch.cat([speculated_sequence, next_tokens], dim=1)
        
        draft_probs_history = [torch.softmax(logits, dim=-1) for logits in draft_logits_history]
        
        verify_logits = verifier(speculated_sequence)
        verify_probs = torch.softmax(verify_logits, dim=-1)

        accepted_tokens = []
        for i in range(spec_steps):
            pos = sequence.size(1) + i
            if pos >= speculated_sequence.size(1):
                break

            draft_token = speculated_sequence[:, pos]

            def prob_of_token_in_distribution(token, distribution):
                '''
                :param token: token id tensor with shape [batch_size]
                :param distribution: probability distribution over vocabulary with shape
                                     [batch_size, vocab_size]

                :return: tensor of shape [batch_size] containing the probability values
                         for the specified tokens
                '''

                # token.unsqueeze(-1) makes the tensor's shape [batch_size, 1] required for the
                # gather operation
                # 
                # gather(-1, token) selects the values from the prob distribution of the vocabulary
                # dimension (last) according to the indices in draft_token
                # 
                # squeeze(-1) removes the extra dimension that was added earlier, resulting in a
                # tensor of shape [batch_size]
                return distribution.gather(-1, token.unsqueeze(-1)).squeeze(-1)

            draft_prob = prob_of_token_in_distribution(draft_token, draft_probs_history[i])
            verify_prob = prob_of_token_in_distribution(draft_token, verify_probs[:, pos - 1, :])

            # verify_prob / draft_prob tells us "how much" the verifier "agrees" with the drafter's choices.
            # In other words, if the verifier assigns a higher probability to the token than the drafter
            # did (ratio > 1), we always accept it (since min(1, ratio) = 1)
            # However, if the verifier assigns a lower probability (ratio < 1), we accept it with probability
            # equal to that ratio
            acceptance_ratio = torch.min(torch.ones_like(verify_prob), verify_prob / draft_prob)

            if torch.rand(1).item() < acceptance_ratio.item():
                accepted_tokens.append(draft_token)
            else:
                break

        accepted_count += len(accepted_tokens)

        if len(accepted_tokens) < spec_steps:
            next_pos = sequence.size(1) + len(accepted_tokens)
            
            # removing tokens that the verifier assigned a lower probability than drafter
            adjusted_probs = torch.clamp(verify_probs[:, next_pos - 1, :] - draft_probs_history[len(accepted_tokens)], min=0)
            # clamping here to avoid division by zero
            sum_adjusted = torch.clamp(adjusted_probs.sum(dim=-1, keepdim=True), min=1e-12)
            # here we're dividing adjusted_probs by sum to rescale the values, to ensure they form
            # a proper probability distribution. this is required since we're subtracting the draft's
            # probabilities from the verifier's and zero out any negative values, resulting in a
            # distribution that no longer sums to 1
            adjusted_probs /= sum_adjusted
            next_token = torch.argmax(adjusted_probs, dim=-1, keepdim=True)

            resample_count += 1
        else:
            if sequence.size(1) + spec_steps >= target_length:
                next_token = None
            else:
                next_token = torch.argmax(verify_probs[:, -1, :], dim=-1, keepdim=True)

        if next_token is not None:
            accepted_tokens.append(next_token[0])

        for t in accepted_tokens:
            sequence = torch.cat([sequence, t.unsqueeze(1)], dim=1)

    print("Accepted count:", accepted_count)
    print("Drafted count:", drafted_count)
    print("Resampled count:", resample_count)
    print("Accepted / drafted ratio:", accepted_count / drafted_count)
    
    return sequence


if __name__ == '__main__':
    set_seed(42)

    verifier_id = 'facebook/opt-350m'
    drafter_id = 'facebook/opt-125m'

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(verifier_id)

    verifier = AutoModelForCausalLM.from_pretrained(verifier_id)
    verifier = model_fn(verifier)

    drafter = AutoModelForCausalLM.from_pretrained(drafter_id)
    drafter = model_fn(drafter)

    input_text = "I'm not convinced"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output = speculative_sampling(input_ids, verifier, drafter, 10)
    output = tokenizer.decode(output[0])
    print(output)