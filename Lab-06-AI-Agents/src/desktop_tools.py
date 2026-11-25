import ast
import operator
import platform
import subprocess
import time

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil

from langchain_core.tools import tool
from src.config import WORKSPACE, TIMEZONE


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _calculate(expr: str):
    expr = expr.replace("×", "*").replace("÷", "/")
    node = ast.parse(expr, mode="eval")

    def walk(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            if abs(n.value) > 1e9:
                raise ValueError("Number is too large")
            return n.value

        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = walk(n.operand)
            return val if isinstance(n.op, ast.UAdd) else -val

        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            left = walk(n.left)
            right = walk(n.right)

            if isinstance(n.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent is too large")

            result = OPS[type(n.op)](left, right)

            if abs(result) > 1e12:
                raise ValueError("Result is too large")

            return result

        raise ValueError(
            "Only +, -, *, /, %, ** and numbers are permitted"
        )

    return walk(node.body)


@tool
def calculator(expression: str) -> str:
    """Calculate a numerical expression and open Windows Calculator to enter it. Example expression: 25*4."""
    value = _calculate(expression)
    message = f"{expression} = {value}. "

    try:
        import pyautogui
        import pygetwindow as gw

        subprocess.Popen(["calc.exe"])
        time.sleep(2)

        windows = list(gw.getWindowsWithTitle("Calculator"))

        if not windows:
            return (
                message
                + "Calculator launched, but window focus could "
                + "not be verified; GUI input not attempted."
            )

        windows[0].activate()
        time.sleep(0.5)

        pyautogui.write(
            expression.replace("×", "*").replace("÷", "/") + "=",
            interval=0.1
        )

        return (
            message
            + "Expression sent to Windows Calculator; "
            + "verify its screen."
        )

    except Exception as exc:
        return message + f"Calculator GUI was not verified: {exc}"


@tool
def date_time() -> str:
    """Return the current local date and time in Sri Lanka."""
    return datetime.now(
        ZoneInfo(TIMEZONE)
    ).strftime("%A, %d %B %Y, %I:%M:%S %p %Z")


@tool
def notepad_write(text: str) -> str:
    """Open Notepad displaying the user's exact text in a file within the workspace. Requires user approval."""
    path = WORKSPACE / "notepad_note.txt"

    path.write_text(text, encoding="utf-8")

    subprocess.Popen([
        "notepad.exe",
        str(path)
    ])

    return f"Opened Notepad with text saved in {path}"


def _safe_workspace(relative_path: str) -> Path:
    path = (WORKSPACE / relative_path).resolve()

    if not path.is_relative_to(WORKSPACE.resolve()):
        raise ValueError(
            "Only paths inside workspace/ are allowed"
        )

    return path


@tool
def file_search(extension: str = ".pdf", subfolder: str = ".") -> str:
    """Find files by extension inside the project's workspace folder or a subfolder. Example extension: .pdf."""
    root = _safe_workspace(subfolder)

    if not root.is_dir():
        return "Directory not found within workspace."

    suffix = extension.lower().lstrip("*.")

    if not suffix.isalnum() or len(suffix) > 12:
        return "Invalid file extension."

    files = [
        str(p.relative_to(WORKSPACE))
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == "." + suffix
    ]

    return (
        "\n".join(files[:100])
        if files
        else "No matching files found."
    )


@tool
def create_text_file(filename: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file inside workspace. Requires user approval. Example: test.txt."""
    path = _safe_workspace(filename)

    if path.suffix.lower() != ".txt":
        return "Only .txt files may be created."

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return f"Created {path}"


@tool
def system_information() -> str:
    """Show operating system, CPU details and total RAM on this computer."""
    return (
        f"OS: {platform.platform()}\n"
        f"CPU: {platform.processor() or 'Unavailable'}\n"
        f"Logical CPU cores: {psutil.cpu_count(logical=True)}\n"
        f"RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB"
    )