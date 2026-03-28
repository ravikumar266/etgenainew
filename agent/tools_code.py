"""
agent/tools_code.py
───────────────────
Code execution and debugging tools via Cloud Run.

Required .env variable:
  CLOUD_RUN_URL=https://your-service-xxxx-uc.a.run.app
"""

import requests
from langchain_core.tools import tool

from agent.config import CLOUD_RUN_URL, llm


@tool
def run_code_cloud(language: str, code: str) -> str:
    """
    Execute Python or JavaScript code on the remote Cloud Run service.
    Returns stdout output or a runtime error.
    Supported languages: python, javascript.
    """
    if not CLOUD_RUN_URL:
        return (
            "❌ CLOUD_RUN_URL is not set in the environment. "
            "Add it to your .env file and restart the server."
        )

    url = f"{CLOUD_RUN_URL.rstrip('/')}/run"
    try:
        res = requests.post(
            url,
            json={"language": language, "code": code},
            timeout=15,
        )
        res.raise_for_status()
        data = res.json()
        if data.get("error"):
            return f"❌ Runtime error:\n{data['error']}"
        return f"✅ Output:\n{data.get('output', '(no output)')}"
    except requests.exceptions.ConnectionError:
        return f"❌ Could not connect to Cloud Run at {url}. Check CLOUD_RUN_URL."
    except requests.exceptions.Timeout:
        return "❌ Code execution timed out after 15 seconds."
    except Exception as e:
        return f"❌ Request failed: {str(e)}"


@tool
def run_code_with_tests(language: str, code: str, test_input: str) -> str:
    """
    Execute code with provided test input on the Cloud Run service.
    For Python, test_input is injected into stdin so input() calls work.
    Supported languages: python, javascript.
    """
    if not CLOUD_RUN_URL:
        return (
            "❌ CLOUD_RUN_URL is not set in the environment. "
            "Add it to your .env file and restart the server."
        )

    # Inject test_input into stdin for Python
    if language.lower() == "python":
        injected = (
            "import sys as _sys, io as _io\n"
            f"_sys.stdin = _io.StringIO({repr(test_input)})\n\n"
        ) + code
    else:
        injected = code + f"\n// test_input: {test_input}"

    url = f"{CLOUD_RUN_URL.rstrip('/')}/run"
    try:
        res = requests.post(
            url,
            json={"language": language, "code": injected},
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        if data.get("error"):
            return f"❌ Runtime error:\n{data['error']}"
        return f"✅ Output:\n{data.get('output', '(no output)')}"
    except requests.exceptions.ConnectionError:
        return f"❌ Could not connect to Cloud Run at {url}. Check CLOUD_RUN_URL."
    except requests.exceptions.Timeout:
        return "❌ Code execution timed out after 20 seconds."
    except Exception as e:
        return f"❌ Failed: {str(e)}"


@tool
def debug_code(language: str, code: str, error: str) -> str:
    """
    Analyse a code error and return a corrected version of the code.
    Provide the language, the original code, and the exact error message.
    Returns only the fixed code, ready to run.
    """
    prompt = f"""You are an expert {language} debugger.

A user has code that produces the following error. Fix it.

LANGUAGE: {language}

CODE:
{code}

ERROR:
{error}

Return ONLY the corrected code — no explanation, no markdown fences,
no extra text. Just the fixed code ready to run.
"""
    try:
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        return f"❌ Debug failed: {str(e)}"