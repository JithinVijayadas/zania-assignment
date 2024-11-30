import numpy as np
from .prompts import CONFIDENCE_SCORE_PROMPT

# Confidence Score
def get_completion_for_scoring(
    messages: list[dict[str, str]],
    model,
    logit_bias={7556: 100, 3309: 100}, #token for true and false and their weightages
    max_tokens=1,
    temperature=0,
    stop=None,
    seed=123,
    logprobs=True,
    top_logprobs=5,
) -> str:
    """
    Generate a completion for scoring using the provided model.

    Args:
        messages (list[dict[str, str]]): List of messages to be sent to the model.
        model: The language model to be used for generating the completion.
        logit_bias (dict, optional): Bias to be applied to logits. Defaults to {7556: 100, 3309: 100}.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1.
        temperature (float, optional): Sampling temperature. Defaults to 0.
        stop (str or list, optional): Sequence(s) where the generation should stop. Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to 123.
        logprobs (bool, optional): Whether to return log probabilities. Defaults to True.
        top_logprobs (int, optional): Number of top log probabilities to return. Defaults to 5.

    Returns:
        str: The generated completion.
    """
    params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }

    completion = model.invoke(messages, **params)
    return completion


def find_token_logprob(token_list, token_name):
    """
    Find the log probability of a specific token in a list of log probabilities.

    Args:
        token_list (list[dict]): List of log probabilities.
        token_name (str): The token to find the log probability for.

    Returns:
        float: The log probability of the token, or None if not found.
    """
    for logprob in token_list:
        if logprob["token"] == token_name:
            return logprob["logprob"]
    return None


def logprob_scoring(API_RESPONSE):
    """
    Calculate the confidence score based on log probabilities from the API response.

    Args:
        API_RESPONSE: The response from the language model API.

    Returns:
        float: The confidence score as a percentage.
    """
    top_logprobs =API_RESPONSE.response_metadata['logprobs']['content'][0]['top_logprobs']
    false_logprob = find_token_logprob(top_logprobs, 'false')
    true_logprob = find_token_logprob(top_logprobs, 'true')
    if true_logprob is None:
        lowest_logprob = min(top_logprobs, key=lambda x: x["logprob"])
        true_logprob = lowest_logprob["logprob"]

    if false_logprob is None:
        lowest_logprob = min(top_logprobs, key=lambda x: x["logprob"])
        false_logprob = lowest_logprob["logprob"]

    def softmax(logprobs):
        """
        Compute softmax probabilities from log probabilities.

        Args:
            logprobs (numpy.ndarray): Array of log probabilities.

        Returns:
            numpy.ndarray: Array of softmax probabilities.
        """
        probs = np.exp(logprobs - np.max(logprobs))
        return probs / np.sum(probs)

    logprobs = np.array([false_logprob, true_logprob])
    probabilities = softmax(logprobs)
    false_prob, true_prob = probabilities
    score = round(true_prob*100, 2)
    return score

def calculate_confidence_score(answer,context,model):
    """
    Calculate the confidence score for a given answer and context using the provided model.

    Args:
        answer (str): The generated answer.
        context (str): The context in which the answer was generated.
        model: The language model to be used for scoring.

    Returns:
        float: The confidence score as a percentage.
    """
    print("Inside calculate_confidence_score")
    response = get_completion_for_scoring(
            [{"role": "user", "content": CONFIDENCE_SCORE_PROMPT.format(answer=answer, context=context)}],
            model=model,
            logprobs=True,
            top_logprobs=5
        )
    confidence_score=logprob_scoring(response)
    print("Faithfulness Score(Answer+Context) ", confidence_score)
    return confidence_score