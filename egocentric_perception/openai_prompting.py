"""Tools to generate from OpenAI prompts."""

import asyncio
import sys
import logging
import tiktoken
import os
from typing import Any

import aiolimiter
import json
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from typing import Dict, Any, List

from utils import lm_config
from utils import chat_prompt


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_to_max_len(string: str, encoding_name: str, max_len: int) -> str:
    encoding = tiktoken.encoding_for_model(encoding_name)
    encoded_string = encoding.encode(string)
    truncated_encoded_string = encoded_string[:max_len]
    truncated_string = encoding.decode(truncated_encoded_string)
    return truncated_string


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> Dict[str, Any]:
    async with limiter:
        res = None
        while res is None:
            try:
                res = await openai.Completion.acreate(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return res


async def generate_from_openai_completion(
    contexts: List[chat_prompt.ChatMessages],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 10,
) -> List[str]:
    """Generate from OpenAI Completion API.

    Args:
        contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=model_config.model,
            prompt=prompt_template.to_text_prompt(
                context=context.limit_length(context_length),
                name_replacements=model_config.name_replacements,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for context in contexts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    return [x["choices"][0]["text"] for x in responses]


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> Dict[str, Any]:
    async with limiter:
        res = None
        while res is None:
            try:
                res = await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.InvalidRequestError:
                logging.warning("OpenAI API Invalid Request: Prompt was filtered")
                return {
                    "choices": [
                        {"message": {"content": "Invalid Request: Prompt was filtered"}}
                    ]
                }
            except openai.error.APIConnectionError:
                logging.warning(
                    "OpenAI API Connection Error: Error Communicating with OpenAI"
                )
                await asyncio.sleep(10)
            except openai.error.Timeout:
                logging.warning("OpenAI APITimeout Error: OpenAI Timeout")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return res


async def generate_from_openai_chat_completion(
    contexts: List[str],
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int = 100,
) -> List[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        contexts: List of full contexts to generate from.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    contexts = [
        [{'role': 'user', 'content': context}] for context in contexts
    ]
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model_config.model,
            messages=context,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for context in contexts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore

    return_msg = []
    for x in responses:
        if x is None:
            return_msg.append('')
        else:
            return_msg.append(x["choices"][0]["message"]["content"])
    return return_msg


if __name__ == '__main__':
    model_name = 'gpt-3.5-turbo-16k-0613'
    max_input_len = (16000 - 512)
    max_output_len = 512
    config = lm_config.LMConfig(provider='openai', model=model_name)
    contexts = ['hello world' for _ in range(100)]
    gen_responses = asyncio.run(generate_from_openai_chat_completion(
        contexts,
        config,
        temperature=0.2,
        max_tokens=max_output_len,
        top_p=0.8,
    ))

