import json
import os
import re
import textwrap
import traceback

from cohere import ClientV2, ToolV2, ToolV2Function

co_client = None

TOOLS = [
    ToolV2(
        type="function",
        function=ToolV2Function(
            name="web_search",
            description="Search the web for information. Use this to find MOSPI datasets, reports, and data-analysis sources.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        ),
    ),
    ToolV2(
        type="function",
        function=ToolV2Function(
            name="web_fetch",
            description="Fetch and read the text content of a web page. Use this on MOSPI (mospi.gov.in), Census India (censusindia.gov.in), data.gov.in, and similar Indian government sources.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch",
                    }
                },
                "required": ["url"],
            },
        ),
    ),
    ToolV2(
        type="function",
        function=ToolV2Function(
            name="python_repl",
            description="Execute Python code for data analysis, computation, parsing, or statistics. pandas & numpy are available. Assign the final value to a variable named 'result'. Use print() for debugging output.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        ),
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
    "tuple": tuple, "type": type, "zip": zip,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "ZeroDivisionError": ZeroDivisionError,
    "Exception": Exception,
}


def _init_client():
    global co_client
    if co_client is None:
        api_key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")
        co_client = ClientV2(api_key=api_key)


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
        resp = httpx.get(
            url, timeout=30, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DataBot/1.0)"},
        )
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
        safe = {k: v for k, v in ALLOWED_BUILTINS.items()}
        safe["__import__"] = builtins.__import__
        g = {"__builtins__": safe, "result": None}
        exec(textwrap.dedent(code), g)
        out = g.get("result")
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

    messages: list = [
        {
            "role": "system",
            "content": (
                "You are a data analyst AI assistant deployed as a Telegram bot.\n"
                "Answer data-analysis questions using the available tools.\n"
                "\n"
                "## Tools\n"
                "- web_search(query): search the web (MOSPI, Census India, data.gov.in)\n"
                "- web_fetch(url): read a web page's text content\n"
                "- python_repl(code): execute Python (pandas & numpy available)\n"
                "\n"
                "## Rules\n"
                "1. Research thoroughly — prefer MOSPI, Census India, data.gov.in sources.\n"
                "2. The user's message specifies the EXACT JSON format expected.\n"
                "3. Output ONLY the JSON object — no markdown, no extra text.\n"
                f"4. If the format includes \"log_url\", use: {log_url}\n"
                "5. Double-check your answer against the source data.\n"
                f"6. Your log URL is: {log_url}"
            ),
        },
        {"role": "user", "content": question},
    ]

    if logger:
        logger.log({"event": "start", "question": question[:500]})

    for iteration in range(15):
        step = {"iteration": iteration}

        response = co_client.v2.chat(
            model="command-a-plus-05-2026",
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
        )

        msg = response.message

        if msg.tool_calls:
            tool_results = []
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                step["tool"] = fn_name
                step["args"] = fn_args

                result = _execute_tool(fn_name, fn_args)

                step["result_preview"] = str(result)[:500]
                if logger:
                    logger.log(step)

                tool_results.append((tc.id, result))

            assistant_content = None
            if msg.content:
                texts = []
                for c in msg.content:
                    if hasattr(c, "text") and c.text:
                        texts.append(c.text)
                assistant_content = "\n".join(texts) if texts else None

            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc_id, result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": str(result),
                })
        else:
            content = ""
            if msg.content:
                texts = []
                for c in msg.content:
                    if hasattr(c, "text") and c.text:
                        texts.append(c.text)
                content = "\n".join(texts)

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

            messages.append({
                "role": "user",
                "content": "You must output ONLY a valid JSON object matching the format the user originally requested. No markdown, no code fences, no extra text — just the raw JSON.",
            })

    fallback = json.dumps({"error": "processing_failed", "detail": "Could not compute answer"})
    if logger:
        logger.log({"event": "fallback", "answer": fallback})
    return fallback
