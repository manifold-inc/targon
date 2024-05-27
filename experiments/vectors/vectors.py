from tqdm import tqdm
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler


#######################################################################################################################
# Paper Section 4 Methods
#######################################################################################################################
def get_q(
    llama,
    dataset,
    text_key: str,
    n: int = 5000,
    batch_size: int = 1,
    logit_fn: Callable[[nn.Module, str], torch.Tensor] = None
):
    """
    Args:
        llama: llama model
        dataset: dataset to use for random prompt sampling
        text_key: key in dataset that corresponds to the prompts
        n (optional): the number of samples to produce logits for
        batch_size (optional): batch size for model inference
        logit_fn: method to use to compute logit vectors
    
    Returns:
        q: matrix of logit vectors with size (n, l)
    """
    q = torch.zeros(n, llama.tokenizer.n_words)

    assert batch_size == 1, "Currently, only works with batch size of 1"

    random_sampler = RandomSampler(dataset, num_samples=n, generator=torch.Generator(device="cuda"))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=random_sampler,
        generator=torch.Generator(device="cuda")
    )

    if logit_fn:
        num_logits = 0
        pbar = tqdm(total=n)
        for batch in dataloader:
            prompt = batch[text_key][0]
            # add logits to q as we go
            logits = logit_fn(llama, prompt)
            if logits.ndim > 1:
                samples = logits.size(0)
                q[num_logits:num_logits+samples] = logits.to("cpu")
                num_logits += samples
                pbar.update(samples) 
            else:
                q[num_logits] = logits.to("cpu")
                num_logits += 1
                pbar.update(1) 
            
            if num_logits == n:
                break
        pbar.close()
    else:
        direct_logits(
            llama,
            q,
            dataloader,
            text_key,
            batch_size
        )
    
    return q


@torch.inference_mode()
def direct_logits(
    llama,
    q,
    dataloader,
    text_key: str,
    batch_size: int = 1,
):
    """
    Args:
        llama: llama model
        q: empty matrix of size (n, l) 
        dataloader: dataloader that contains prompts as samples
        text_key: key in dataset that corresponds to the prompts
        batch_size (optional): batch size for model inference
    
    Returns:
        q: matrix of logit vectors with size (n, l)
    """
    for i, batch in enumerate(tqdm(dataloader)):
        prompts = batch[text_key]
        tokens = [llama.tokenizer.encode(p, bos=True, eos=False) for p in prompts]

        # pad tokens following llama implementation
        max_prompt_len = max(len(t) for t in tokens)
        prompt_tokens = torch.full(
            (batch_size, max_prompt_len),
            llama.tokenizer.pad_id,
            dtype=torch.long,
            device="cuda"
        )
        for k, t in enumerate(tokens):
            prompt_tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        # add logits to q as we go
        logits = llama.model(prompt_tokens, 0)[:, -1]
        q[i * batch_size:i * (batch_size + 1)] = logits.to("cpu")
    
    return


def h_dim_extraction(
    q,
    predict_norm: bool = False
):
    """
    Args:
        q: matrix of logit vectors with size (n, l)
        predict_norm (optional): whehther to predict which normalization layer type is used
    
    Returns:
        u: unitary matrix
        s: singular values
        s_dim: log of the absolute singular values
        count: predicted hidden dimension size
    """    
    # compute singular values and prepare them to find the multiplicative gap
    u, s, _ = torch.linalg.svd(q.T.to(torch.float64), full_matrices=False)
    s_dim = torch.log(s.abs())

    # avoid large drops in negative singular values from causing a larger h_dim to be predicted
    # do so by multiplying by the sign of the first number -> multiplicative gap remains negative
    # also the last singular value is 0 so avoid using it for argmax computation
    count = torch.argmax(
        torch.where(s_dim[:-2] >= 0, 1, -1) * (s_dim[:-2] - s_dim[1:-1])
    ).item() + 1
    
    # TODO: Test with other models
    if predict_norm:
        # Detailed in appendix B.2.2
        q = q.to(torch.float16)
        q_sub = q - q.mean(dim=0)
        del q

        s_sub = torch.linalg.svdvals(q_sub.T.to(torch.float64))
        s_sub_dim = torch.log(s_sub.abs())
        count_sub =  torch.argmax(
            torch.where(s_sub_dim[:-2] >= 0, 1, -1) * (s_sub_dim[:-2] - s_sub_dim[1:-1])
        ).item() + 1

        print("Model uses LayerNorm") if count_sub == count - 1 else print("Model uses RMSNorm")

    print(f"Hidden Dim: {count}")
    return u, s, s_dim, count


def layer_extraction(w, u, s, h_dim):
    """
    Args:
        w: model's actual weight matrix for last layer
        u: unitary matrix computed from `h_dim_extraction`
        s: singular values computed from `h_dim_extraction`
        h_dim: model's actual hidden dimension

    Returns:
        pred_w: predicted w
        g: affine transformation such that pred_w@g ~ w
    """
    w, u, s = w.to("cuda"), u[:, :h_dim].to("cuda"), s[:h_dim].to("cuda")
    pred_w = u @ torch.diag(s)
    g = torch.linalg.lstsq(pred_w, w.to(torch.float64)).solution

    return pred_w, g


#######################################################################################################################
# Paper Section 5 Methods
#######################################################################################################################
@torch.inference_mode()
def topk_logit_extraction(llama, prompt, k=5):
    """
    Incrementally construct logit vector by using logit biases to promote sets of tokens for each query.
    Subtract the bias from the output logits to find their actual values and repeat for all tokens.
    """
    n_words = llama.tokenizer.n_words
    out_logits = torch.zeros((n_words,))

    tokens = [llama.tokenizer.encode(prompt, bos=True, eos=False)]

    # pad tokens following llama implementation
    max_prompt_len = max(len(t) for t in tokens)
    prompt_tokens = torch.full(
        (1, max_prompt_len),
        llama.tokenizer.pad_id,
        dtype=torch.long,
        device="cuda"
    )
    for k, t in enumerate(tokens):
        prompt_tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    # mimic a logit api
    bias = 100
    logits = llama.model(prompt_tokens, 0)[:, -1]

    for i in range(0, n_words, k):
        # act as though we add bias internally just for demonstration
        sampled_logits = logits.clone()
        sampled_logits[:, i:i+k] += bias

        # return topk logits and their tokens
        outputs = torch.topk(sampled_logits, k)
        top_logits = outputs.values[0] - bias
        top_tokens = outputs.indices[0]
        
        out_logits[top_tokens] = top_logits.half()  # assume that bias pushes ref token to k
    
    return out_logits


def topk_logprob_ref_extraction(llama, prompt, k=5):
    """
    Incrementally construct logit vector by using logit biases to promote sets of tokens for each query
    while maintaining a reference token.
    Compute relative logit differences between reference token and other promoted tokens.
    """
    n_words = llama.tokenizer.n_words
    logits = torch.zeros((n_words,))

    prompt_tokens = [llama.tokenizer.encode(prompt, bos=True, eos=False)]

    # compute our reference token and its logprob
    out_tokens, logprobs = llama.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=1,
        temperature=0,
        logprobs=True,
        k=1
    )

    ref_token = out_tokens[0][0]

    bias = 100

    for i in range(0, n_words, k-1):
        # promote at most k-1 tokens that are not the reference token
        logitbias = {min(i+j, n_words-1):bias for j in range(k-1) if i + j != ref_token}
        out_tokens, logprobs = llama.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=1,
            temperature=0,
            logprobs=True,
            logitbias=logitbias,
            k=len(logitbias) + 1  # avoid unadjusted tokens in top-k logprobs
        )
        
        # find relative logit difference between reference and promoted tokens
        token_logits = logprobs[0]["values"][0, -1] - (logprobs[0]["values"][0, :-1] - bias)
        tokens = logprobs[0]["tokens"][0, :-1]
        logits[tokens] = token_logits.half()  # assume that bias pushes ref token to k
    
    return logits


def topk_logprob_exp_extraction(llama, prompt, k=5, exp_size=10):
    """
    Incrementally construct logit vector by using logit biases to promote sets of tokens for each query
    while maintaining a reference token.
    Unlike `topk_logprob_ref_extraction`, we generate many sets for a single prompt to reduce attack cost.
    Compute relative logit differences between reference token and other promoted tokens.
    """
    n_words = llama.tokenizer.n_words
    logits = torch.zeros((exp_size, n_words))

    prompt_tokens = [llama.tokenizer.encode(prompt, bos=True, eos=False)]

    ref_token = 0
    ref_bias = 100
    aux_bias = 70

    for i in range(1, n_words, k-1):
        # promote at most k-1 tokens that are not the reference token
        logitbias = {min(i+j, n_words-1):aux_bias for j in range(k-1)}
        logitbias.update({ref_token:ref_bias})
        _, logprobs = llama.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=exp_size,
            temperature=0,
            logprobs=True,
            logitbias=logitbias,
            k=len(logitbias)
        )
        
        # find relative logit difference between reference and promoted tokens
        vals = logprobs[0]["values"]
        toks = logprobs[0]["tokens"]

        token_logits = (vals[:, 0] - ref_bias).unsqueeze(1) - (vals[:, 1:] - aux_bias)
        tokens = toks[:, 1:]
        logits[torch.arange(exp_size).unsqueeze(-1), tokens] = token_logits.half()  # assume that bias pushes ref token to k
    
    return logits


def token_logprob_extraction(llama, prompt):
    """
    Method described in section 5.4
    Incrementally construct logit vector by using logit biases.
    Specifically, compute a token's logprob using the top token's logprob change
    when generating with a logit bias of 0 and -1 for the specific token.

    NOTE: Expensive to compute all logits with this method.
    """
    n_words = llama.tokenizer.n_words

    prompt_tokens = [llama.tokenizer.encode(prompt, bos=True, eos=False)]

    # compute our reference token and its logprob
    out_tokens, logprobs = llama.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=1,
        temperature=0,
        logprobs=True,
        k=1
    )

    top_token = out_tokens[0][0]
    top_logprob = logprobs[0]["values"][0, 0]

    # store top token's logprob when we apply a bias of -1 for all other tokens
    bias_logprobs = []
    for i in tqdm(range(n_words), desc="collecting logit bias logprobs"):
        if i == top_token:
            bias_logprobs.append(logprobs[0]["values"][0, 0])
            continue
        
        logitbias = {i:-1}
        _, logprobs = llama.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=1,
            temperature=0,
            logprobs=True,
            logitbias=logitbias,
            k=1
        )
        
        bias_logprobs.append(logprobs[0]["values"][0, 0])
    
    logprobs = (torch.exp(torch.tensor(top_logprob) - torch.tensor(bias_logprobs)) - 1) / ((1 / torch.e) - 1)

    # TODO: can we use logprobs directly in place of logits for last layer computation?        
    return logprobs


#######################################################################################################################
# Paper Section 6 Methods
#######################################################################################################################
def binary_search_extraction(llama, prompt, error=0.5):
    """
    Method described in section 6.1
    Incrementally construct logit vector by finding logit bias for each token that
    causes it to be the the top-token (output with probability of 1 when temperature is 0).
    """
    n_words = llama.tokenizer.n_words
    logits = torch.zeros((n_words,))

    prompt_tokens = [llama.tokenizer.encode(prompt, bos=True, eos=False)]

    # compute our reference token
    out_tokens, _ = llama.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=1,
        temperature=0,
        k=1
    )

    try:
        top_token = out_tokens[0][0]
    except IndexError:
        # if no token is output, then eos reached
        top_token = llama.tokenizer.eos_id

    for token in range(n_words):
        if token == top_token:
            continue

        alpha = -100
        beta = 0
        while beta - alpha > error:
            mid = (alpha + beta) / 2

            logitbias = {token:-mid}
            out_tokens, _ = llama.generate(
                prompt_tokens=prompt_tokens,
                max_gen_len=1,
                temperature=0,
                logitbias=logitbias,
                k=1
            )
            
            try:
                curr_top_token = out_tokens[0][0]
            except IndexError:
                # if no token is output, then eos reached
                curr_top_token = llama.tokenizer.eos_id

            if curr_top_token == top_token:
                beta = mid
            else:
                alpha = mid
        
        logits[token] = (beta - alpha) / 2
    
    return logits


def hyperrectangle_extraction(llama, prompt, N=10, T=10, better_queries=False):
    """
    Method described in section 6.2
    Incrementally construct logit vector by modifying logit bias for multiple tokens.
    
    Referencing: https://github.com/dpaleka/stealing-part-lm-supplementary/tree/main/optimize_logit_queries
    """
    n_words = llama.tokenizer.n_words
    logits = torch.zeros((n_words,))

    prompt_tokens = [llama.tokenizer.encode(prompt, bos=True, eos=False)]
    
    # compute our reference token
    out_tokens, _ = llama.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=1,
        temperature=0,
        k=1
    )

    try:
        top_token = out_tokens[0][0]
    except IndexError:
        # if no token is output, then eos reached
        top_token = llama.tokenizer.eos_id
    
    for start_token in range(0, n_words, N):
        alphas = dict()
        betas = dict()
        curr_tokens = []
        for token in range(start_token, start_token+N):
            if token == top_token or llama.tokenizer.eos_id:
                continue
            curr_tokens.append(token)
            alphas[token] = -100
            betas[token] = 0
        
        num_tokens = len(curr_tokens)
        constraints = torch.zeros((num_tokens * num_tokens + 1, num_tokens + 1))
        biases = torch.zeros((num_tokens * num_tokens + 1))

        # initialize contraints
        for t in range(num_tokens):
            constraints[t * num_tokens + num_tokens, t] = 1
            constraints[t * num_tokens + num_tokens, num_tokens] = -1
            biases[t * num_tokens + num_tokens] = 100

        for _ in range(T):
            logitbias = dict()
            for token in curr_tokens:
                if better_queries:
                    c = torch.exp(-torch.log(num_tokens) / (num_tokens - 1))
                    mid = -(1 - c) * alphas[token] - c * betas[token]
                else:
                    mid = -(alphas[token] + betas[token]) / 2
                logitbias[token] = mid

            out_tokens, _ = llama.generate(
                prompt_tokens=prompt_tokens,
                max_gen_len=1,
                temperature=0,
                logitbias=logitbias,
                k=N
            )
            out_tokens = out_tokens[0]

            for i, tok_i in enumerate(out_tokens):
                for j, tok_j in enumerate(out_tokens):
                    if i == j:
                        continue
                    constraints[i * num_tokens + j, i] = -1
                    constraints[i * num_tokens + j, j] = 1
                    biases[i * num_tokens + j] = logitbias[tok_j] - logitbias[tok_i]

            # base token constraint
            constraints[-1, -1] = 1
            biases[-1] = 100

            X = torch.linalg.lstsq(constraints, biases).solution

            # update alphas and betas
            for t, token in enumerate(num_tokens):
                # paper minimizes both?
                # maybe need to use bounder such as: https://github.com/dpaleka/stealing-part-lm-supplementary/blob/main/optimize_logit_queries/bounders/iterate_constraints.py
                alphas[token] = max(X[t] - X[-1], alphas[token])
                betas[token] = min(X[t] - X[-1], alphas[token])

        for token in curr_tokens:
            logits[token] = (betas[token] - alphas[token]) / 2
    
    return logits