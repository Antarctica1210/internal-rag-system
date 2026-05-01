# app/utils/path_utils.py
from pathlib import Path
from dotenv import load_dotenv
import os
from pathlib import Path

def get_path_dir(ps: int = 0) -> Path:
    """
    pathlib.Path provides a 'parents' attribute — an ordered sequence of ancestor directories.
    Indexing it directly retrieves any ancestor level without chaining multiple .parent calls.
    This is the recommended approach for navigating up directory trees.

    Index rules: parents[N] corresponds to N+1 levels up from the current path.
      parents[0] → equivalent to .parent          (1 level up)
      parents[1] → equivalent to .parent.parent   (2 levels up)
      parents[2] → equivalent to .parent.parent.parent (3 levels up)
      ...and so on: higher index = further up the directory tree.

    :param ps: Number of levels to traverse upward (0 = immediate parent)
    :return: Path object for the directory ps+1 levels above this file
    """
    dir_path = Path(__file__).parents[ps]
    return dir_path


def get_project_root(identifier: str = ".env") -> Path:
    # Step 1: Check for PROJECT_ROOT environment variable first (used in production)
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    # Step 2: Walk up the directory tree to find and load the .env file
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            load_dotenv(dotenv_path=current_dir / identifier)
            break
        current_dir = current_dir.parent

    # Step 3: Walk up again to return the directory containing the identifier (fallback for development)
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(
        f"Project root not found: no '{identifier}' file detected, and PROJECT_ROOT env var is not set"
    )


PROJECT_ROOT = get_project_root(".env")
