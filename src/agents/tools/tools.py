import json
import math
import os
import random
from datetime import datetime

from smolagents import tool


@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression.
    Args:
        expression: The mathematical expression to evaluate (e.g. '2 + 2').
    """
    try:
        # Use safer eval with math functions available
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def read_file(file_path: str) -> str:
    """Reads a text file from the local file system.
    Args:
        file_path: Path to the file to read.
    """
    try:
        with open(file_path) as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Writes content to a text file.
    Args:
        file_path: Path to the file to write.
        content: Content to write to the file.
    """
    try:
        with open(file_path, "w") as file:
            file.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"


@tool
def list_directory(directory_path: str = ".") -> str:
    """Lists files and folders in a directory.
    Args:
        directory_path: Path to the directory to list.
    """
    try:
        items = os.listdir(directory_path)
        return json.dumps(items, indent=2)
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def word_counter(text: str) -> str:
    """Counts words in a given text.
    Args:
        text: The text to count words in.
    """
    words = text.split()
    return f"Word count: {len(words)}"


@tool
def random_number(min_val: int = 1, max_val: int = 100) -> str:
    """Generates a random number between min_val and max_val.
    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).
    """
    return str(random.randint(min_val, max_val))


@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """Summarizes text to be no longer than max_length.
    Args:
        text: Text to summarize.
        max_length: Maximum length of summary.
    """
    # This is a simple truncation - your LLM will do the real summarization
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
