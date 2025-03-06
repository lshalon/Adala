import re
import litellm
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def normalize_litellm_model_and_provider(model_name: str, provider: str):
    """
    When using litellm.get_model_info() some models are accessed with their provider prefix
    while others are not.

    This helper function contains logic which normalizes this for supported providers
    """
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]
    provider = provider.lower()
    # TODO: move this logic to LSE, this is the last place Adala needs to be updated when adding a provider connection
    if provider == "vertexai":
        provider = "vertex_ai"
    if provider == "azureopenai":
        provider = "azure"
    if provider == "azureaifoundry":
        provider = "azure_ai"

    return model_name, provider


def normalize_canonical_model_name(model: str) -> str:
    """Strip date suffix from model name if present at the end (e.g. gpt-4-0613 -> gpt-4)"""
    # We only know that this works for models hosted on azure openai, azure foundry, and openai
    # 'gpt-4-2024-04-01'
    if re.search(r"-\d{4}-\d{2}-\d{2}$", model):
        model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    # 'gpt-4-0613'
    elif re.search(r"-\d{4}$", model):
        model = re.sub(r"-\d{4}$", "", model)
    return model


class NoModelsFoundError(ValueError):
    """Raised when a model cannot be found in litellm's model map"""

    pass


def match_model_provider_string(model: str) -> str:
    """Given a string of the form 'provider/model', return the 'provider/model' as listed in litellm's model map"""
    # NOTE: if needed, can pass api_base and api_key into this function for additional hints
    model_name, provider, _, _ = litellm.get_llm_provider(model)

    # amazingly, litellm.cost_per_token refers to a hardcoded dictionary litellm.model_cost which is case-sensitive with inconsistent casing.....
    # Example: 'azure_ai/deepseek-r1' vs 'azure_ai/Llama-3.3-70B-Instruct'
    lowercase_to_canonical_case = {
        k.lower(): k for k in litellm.models_by_provider[provider]
    }
    candidate_model_names = []
    for name in [model_name, normalize_canonical_model_name(model_name)]:
        candidate_model_names.append("/".join([provider, name.lower()]))
    # ...and Azure AI Foundry openai models are not listed there, but under Azure OpenAI
    if provider == "azure_ai":
        for model in candidate_model_names:
            candidate_model_names.append(model.replace("azure_ai/", "azure/"))
    matched_models = set(candidate_model_names) & set(lowercase_to_canonical_case)
    if len(matched_models) == 0:
        raise NoModelsFoundError(model)
    if len(matched_models) > 1:
        logger.warning(f"Multiple models found for {model}: {matched_models}")
    return lowercase_to_canonical_case[matched_models.pop()]


def get_model_info(
    provider: str, model_name: str, auth_info: Optional[dict] = None
) -> dict:
    if auth_info is None:
        auth_info = {}
    try:
        # for azure models, need to get the canonical name for the model
        if provider == "azure":
            dummy_completion = litellm.completion(
                model=f"azure/{model_name}",
                messages=[{"role": "user", "content": ""}],
                max_tokens=1,
                **auth_info,
            )
            model_name = dummy_completion.model
        model_name, provider = normalize_litellm_model_and_provider(
            model_name, provider
        )
        return litellm.get_model_info(model=model_name, custom_llm_provider=provider)
    except Exception as err:
        logger.error("Hit error when trying to get model metadata: %s", err)
        return {}


def _get_prompt_tokens(string: str, model: str, output_fields: List[str]) -> int:
    user_tokens = litellm.token_counter(model=model, text=string)
    # FIXME surprisingly difficult to get function call tokens, and doesn't add a ton of value, so hard-coding until something like litellm supports doing this for us.
    #       currently seems like we'd need to scrape the instructor logs to get the function call info, then use (at best) an openai-specific 3rd party lib to get a token estimate from that.
    system_tokens = 56 + (6 * len(output_fields))
    return user_tokens + system_tokens


def _get_completion_tokens(
    model: str,
    output_fields: Optional[List[str]],
) -> int:
    max_tokens = litellm.get_model_info(model=model).get("max_tokens", None)
    if not max_tokens:
        raise ValueError(f"Model {model} has no max tokens.")
    # extremely rough heuristic, from testing on some anecdotal examples
    n_outputs = len(output_fields) if output_fields else 1
    return min(max_tokens, 4 * n_outputs)


def _estimate_cost(
    user_prompt: str,
    model: str,
    output_fields: Optional[List[str]],
    provider: str,
):
    try:
        prompt_tokens = _get_prompt_tokens(user_prompt, model, output_fields)

        completion_tokens = _get_completion_tokens(model, output_fields)

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as e:
        # Missing model exception doesn't have a type to catch:
        # Exception("This model isn't mapped yet. model=azure_ai/deepseek-R1, custom_llm_provider=azure_ai. Add it here - https://github.com/ BerriAI/litellm/blob/main/model_prices_and_context_window.json.")
        if "model isn't mapped" in str(e):
            raise ValueError(f"Model {model} for provider {provider} not found.")
        else:
            raise e

    total_cost = prompt_cost + completion_cost

    return prompt_cost, completion_cost, total_cost
