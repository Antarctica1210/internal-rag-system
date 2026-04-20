import os
from pathlib import Path
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger

def load_prompt(name: str, **kwargs) -> str:
    """
    Load a prompt file and render variable placeholders.
    :param name: Prompt filename without the .prompt extension (e.g. image_summary)
    :param **kwargs: Key-value pairs for placeholder substitution
                     (e.g. root_folder="my_doc", image_content=("pre-text", "post-text"))
    :return: Final rendered prompt string
    """
    # 1. Build the prompt file path
    prompt_path = Path(os.path.join(PROJECT_ROOT, 'prompts', f'{name}.prompt'))

    # 2. Validate the file exists before attempting to read it
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path.absolute()}")

    # 3. Read the raw prompt text
    raw_prompt = prompt_path.read_text(encoding='utf-8')

    # 4. If kwargs were provided, render placeholders; otherwise return the raw text
    if kwargs:
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"Prompt rendered successfully, substituted variables: {list(kwargs.keys())}")
        return rendered_prompt
    return raw_prompt

