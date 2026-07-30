import json
import os
import re
import textwrap
import traceback

import cohere
from cohere import (
    Tool,
    ToolParameterDefinitionsValue,
    ToolCall,
    ToolResult,
)

co_client = None

TOOL_DEFS = [
    Tool(
        name="web_search",
        description="Search the web for information. Use this to find MOSPI datasets, reports, and data-analysis sources.",
        parameter_definitions={
            "query": ToolParameterDefinitionsValue(
                description="The search query",
                type="str",
                required=True,
            )
        },
    ),
    Tool(
        name="web_fetch",
        description="Fetch and read the text content of a web page. Use this to read data tables, reports, or API responses from MOSPI and other public data sources.",
        parameter_definitions={
            "url": ToolParameterDefinitionsValue(
                description="The full URL to fetch",
                type="str",
                required=True,
            )
        },
    ),
    Tool(
        name="python_repl",
        description="Execute Python code for data analysis, computation, parsing, or statistics. numpy and pandas are available. Assign the final value to a variable named 'result'. Use print() for debugging.",
        parameter_definitions={
            "code": ToolParameterDefinitionsValue(
                description="Python code to execute",
                type="str",
                required=True,
            )
        },
    ),
]

ALLOWED_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "format": format,
    "frozenset": frozenset, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "pow": pow, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "zip": zip, "Exception": Exception,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
}


def _init_client():
    global co_client
    if co_client is None:
        api_key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")
        co_client = cohere.Client(api_key=api_key)


def web_search(query: str) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
        if not results:
            return "No search results found."
        lines = []
        for r in results:
            title = r.get("title", "N/A")
            href = r.get("href", "N/A")
            body = r.get("body", "")
            lines.append(f"Title: {title}\nURL: {href}\nSummary: {body}")
        return "\n\n---\n\n".join(lines)
    except Exception as e:
        return f"[web_search error: {e}]"


def web_fetch(url: str) -> str:
    try:
        import httpx
        from bs4 import BeautifulSoup
        resp = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (compatible; DataBot/1.0)"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.split("\n") if line.strip()]
        return "\n".join(lines[:600])
    except Exception as e:
        return f"[web_fetch error: {e}]"


def python_repl(code: str) -> str:
    try:
        import builtins
        safe_builtins = {k: v for k, v in ALLOWED_BUILTINS.items()}
        safe_builtins["__import__"] = builtins.__import__
        safe_globals = {"__builtins__": safe_builtins, "result": None}
        exec(textwrap.dedent(code), safe_globals)
        out = safe_globals.get("result")
        if out is not None:
            return str(out)
        return "Code executed. No 'result' variable set."
    except Exception as e:
        return f"[python error: {traceback.format_exc()}]"


def _execute_tool(name: str, params: dict) -> str:
    if name == "web_search":
        return web_search(**params)
    elif name == "web_fetch":
        return web_fetch(**params)
    elif name == "python_repl":
        return python_repl(**params)
    return f"Unknown tool: {name}"


def _extract_json(text: str) -> str | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return None


def run_agent(question: str, log_url: str, logger=None) -> str:
    _init_client()

    preamble = (
        "You are a data analyst AI assistant deployed as a Telegram bot.\n"
        "Your job is to answer data-analysis questions using the available tools.\n\n"
        "## Tools\n"
        "- web_search(query): search the web for information (MOSPI datasets, reports, etc.)\n"
        "- web_fetch(url): read the text content of a web page. Prefer MOSPI (mospi.gov.in), Census India (censusindia.gov.in), data.gov.in for India data.\n"
        "- python_repl(code): execute Python code (pandas & numpy available). Use it to parse data and compute answers.\n\n"
        "## Rules\n"
        "1. Research thoroughly — prefer MOSPI, Census India, data.gov.in sources.\n"
        "2. The user's message specifies the EXACT JSON format expected for the answer.\n"
        "3. Output ONLY the JSON object the question asks for — no markdown, no extra text.\n"
        f"4. If the answer format includes \"log_url\", use this value: {log_url}\n"
        "5. Double-check your answer against the source data.\n"
        f"6. Your log URL is: {log_url}\n"
        "7. If a tool call fails with the same error twice, try a different approach."
    )

    if logger:
        logger.log({"event": "start", "question": question[:500]})

    response = co_client.chat(
        message=question,
        model="command-r-plus-08-2024",
        tools=TOOL_DEFS,
        preamble=preamble,
        temperature=0.1,
    )

    repeat_penalty = {}
    for iteration in range(15):
        step = {"iteration": iteration}

        if response.tool_calls:
            tool_results = []
            for tc in response.tool_calls:
                fn_name = tc.name
                fn_args = tc.parameters

                step["tool"] = fn_name
                step["args"] = fn_args

                result = _execute_tool(fn_name, fn_args)

                key = (fn_name, str(fn_args))
                repeat_penalty[key] = repeat_penalty.get(key, 0) + 1

                if result.startswith("[python error") and repeat_penalty.get(key, 0) >= 2:
                    result += "\n\n[SYSTEM: This same code keeps failing. Try a different approach or use web_fetch to read data instead.]"

                step["result_preview"] = str(result)[:500]
                if logger:
                    logger.log(step)

                tool_results.append(
                    ToolResult(call=tc, outputs=[{"result": result}])
                )

            response = co_client.chat(
                message="",
                model="command-r-plus-08-2024",
                tools=TOOL_DEFS,
                chat_history=response.chat_history,
                tool_results=tool_results,
                temperature=0.1,
            )
        else:
            content = response.text or ""
            step["response"] = content[:200]

            json_str = _extract_json(content)
            if json_str:
                try:
                    json.loads(json_str)
                    if logger:
                        logger.log({"event": "complete", "answer": json_str[:2000]})
                    return json_str
                except json.JSONDecodeError:
                    pass

            response = co_client.chat(
                message="You must output ONLY a valid JSON object matching the format the user originally requested. No markdown, no code fences, no extra text — just the raw JSON.",
                model="command-r-plus-08-2024",
                tools=TOOL_DEFS,
                chat_history=response.chat_history,
                temperature=0.1,
            )

    fallback = json.dumps({"error": "processing_failed", "detail": "Could not compute answer after multiple attempts"})
    if logger:
        logger.log({"event": "fallback", "answer": fallback})
    return fallback
