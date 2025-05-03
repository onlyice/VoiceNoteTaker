"""
This file holds the core functions of WhisperNote. It contains the following functions:
* transcribe_voice_message: This function is used to transcribe the voice message to text.
* paraphrase_text: This function is used to paraphrase the text using GPT and return the processed text.
* convert_audio_file_to_format: This function is used to convert the audio file to a specific format.
"""

import json
from typing import AsyncGenerator, Dict, Tuple

from openai import AsyncOpenAI, OpenAI
from pydub import AudioSegment

from prompts import CHOICE_TO_PROMPT, PROMPTS

client = OpenAI()
aclient = AsyncOpenAI()


def transcribe_voice_message(filename: str) -> str:
    """Invoke the Whisper ASR API to transcribe the voice message to text.

    Args:
        filename (str): filename of the voice message. Note it has to be compatible with Whisper ASR API.

    Returns:
        str: Transcribed text.
    """
    with open(filename, "rb") as file:
        response = client.audio.transcriptions.create(
            file=file, model="gpt-4o-mini-transcribe", prompt="简体中文，使用简体中文标点符号。",
            response_format="text", temperature=0, language="zh",
        )
    return response


def classify_outline_content(text: str) -> Dict[str, str]:
    """Invokes GPT-3.5 API to tell the actual content of the request.
    Args:
        text (str): the initial transcribed text to be processed.

    Returns:
    """
    parse_result = gpt_process_text(
        text, PROMPTS["outline-content-classification"], model="gpt-3.5-turbo"
    )
    try:
        parse_result = json.loads(parse_result)
    except json.decoder.JSONDecodeError:
        # Default to append
        parse_result = {"content": text, "intent": "append", "line": -1}
    return parse_result


def classify_outline_intent_mode(text: str) -> bool:
    """Invokes GPT-3.5 API to tell whether the intent of the given text is to enter the outline mode.
    Args:
        text (str): the initial transcribed text to be processed.

    Returns:
        bool: whether the intent of the given text is to enter the outline mode.
    """
    if len(text) > 30:
        # A small trick is, because the outline mode triggering word is so short, we can directly tell outline mode is not the intent when the text is too long.
        return False
    processed_text = gpt_process_text(
        text, PROMPTS["outline-intent-classification"], model="gpt-3.5-turbo"
    )
    return processed_text == "True"


def preprocess_text(text: str) -> str:
    """Invokes GPT-3.5 API to preprocess the text.
    We use certain format to parse the text, and output a json with two fields, content and tag.
    The tag is then used to determine which model to invoke.

    Args:
        text (str): the initial transcribed text to be processed.

    Returns:
        str: paraphrased text.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": PROMPTS["transcribe-and-parse"]},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    processed_text = response.choices[0].message.content.strip()
    return processed_text


def gpt_iterate_on_thoughts(text: str, target_usage: str) -> str:
    """Invokes GPT-4 API to iterate on thoughts, using the provided target usage, which is expected to be one of the keys in the CHOINCE_TO_PROMPT.

    Args:
        text (str): the inptu text.
        target_usage (str): the target usage of the text.

    Returns:
        str: processed text/thought.
    """
    if target_usage not in CHOICE_TO_PROMPT:
        raise ValueError(f"Invalid target usage: {target_usage}")
    return gpt_process_text(
        text, system_prompt=CHOICE_TO_PROMPT[target_usage], model="gpt-4"
    )


async def gpt_process_text_async(
    text: str, system_prompt: str = PROMPTS["paraphrase"], model: str = "gpt-4"
) -> AsyncGenerator[Tuple[str, str], None]:
    """Invokes GPT-4 API to process the text in stream mode.

    Args:
        text (str): the transcribed text to be paraphrased.
        system_prompt (str): the system prompt to be used. Defaults to PROMPTS['paraphrase'].
        model (str, optional): the GPT model to be used. Defaults to 'gpt-4'.

    Returns:
        str: output text.
    """
    gen = await aclient.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        stream=True,
        temperature=0,
    )

    answer = ""
    async for item in gen:
        content = item.choices[0].delta.content
        if not content:
            continue

        answer += content
        yield "not_finished", answer

    yield (
        "finished",
        answer,
    )


def gpt_process_text(
    text: str, system_prompt: str = PROMPTS["paraphrase"], model: str = "gpt-4"
) -> str:
    """Invokes GPT-4 API to process the text.

    Args:
        text (str): the transcribed text to be paraphrased.
        system_prompt (str): the system prompt to be used. Defaults to PROMPTS['paraphrase'].
        model (str, optional): the GPT model to be used. Defaults to 'gpt-4'.

    Returns:
        str: output text.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    processed_text = response.choices[0].message.content.strip()
    return processed_text


def convert_audio_file_to_format(input_file: str, output_file: str, OUTPUT_FORMAT: str):
    """Converts the audio file to a specific format.

    Args:
        input_file (str): input audio file
        output_file (str): output audio file
        OUTPUT_FORMAT (str): audio format
    """
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format=OUTPUT_FORMAT)
