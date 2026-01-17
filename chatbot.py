import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from typing import List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

DB_PATH = os.path.join(os.path.dirname(__file__), "Chinook.db")
MODEL_NAME = "qwen2:7b"

PERSONA = (
    "You are ChinookAnalyst, a helpful data analyst for the Chinook music store. "
    "You only answer questions that can be answered by the Chinook.db database. "
    "If a question is not about the Chinook database, respond with a brief refusal "
    "and ask the user to rephrase using the database."
)

SQL_GEN_SYSTEM = (
    PERSONA
    + "\n\nYou generate a single SQLite SELECT query that answers the user's question.\n"
    + "Rules:\n"
    + "- Use only SELECT queries.\n"
    + "- Do not modify data.\n"
    + "- If the question is not about the database, output exactly: NO_SQL\n"
    + "- Prefer explicit column names and JOINs over SELECT *.\n\n"
    + "Output requirements:\n"
    + "- Return only SQL (no prose, no markdown).\n"
    + "- End the query with a semicolon.\n\n"
    + "If the question is about artists, albums, tracks, invoices, customers, genres, or sales,\n"
    + "you MUST return a SQL query (not NO_SQL).\n\n"
    + "Database schema:\n{schema}\n"
    + "\nExamples:\n"
    + "Q: Which country has the highest number of invoices?\n"
    + "A: SELECT BillingCountry, COUNT(*) AS InvoiceCount "
    + "FROM Invoice GROUP BY BillingCountry ORDER BY InvoiceCount DESC LIMIT 1;\n"
    + "Q: Which artist has the most albums?\n"
    + "A: SELECT ar.Name, COUNT(al.AlbumId) AS AlbumCount "
    + "FROM Artist ar JOIN Album al ON ar.ArtistId = al.ArtistId "
    + "GROUP BY ar.ArtistId ORDER BY AlbumCount DESC LIMIT 1;\n"
)

ANSWER_SYSTEM = (
    PERSONA
    + "\n\nAnswer the user based strictly on the SQL results provided. "
    + "If SQL result is NO_SQL, refuse and ask for a DB question. "
    + "If SQL result is NO_ROWS, say no matching data was found and ask a clarifying question."
)


class GraphState(TypedDict):
    messages: List[BaseMessage]
    sql: str
    sql_result: str


def _get_schema_description(db_path: str) -> str:
    if not os.path.exists(db_path):
        return "ERROR: Chinook.db not found."
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cur.fetchall()]
        lines = []
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            cols = [f"{col[1]} ({col[2]})" for col in cur.fetchall()]
            lines.append(f"- {table}: " + ", ".join(cols))
        return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def _extract_sql(text: str) -> str:
    candidate = _strip_code_fences(text)
    if ";" in candidate:
        candidate = candidate.split(";", 1)[0] + ";"
    return candidate.strip()


def _format_history(messages: List[BaseMessage], max_turns: int = 6) -> str:
    recent = messages[-max_turns:]
    lines = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            role = "System"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _is_select_only(query: str) -> bool:
    q = query.strip().strip(";")
    if not q.lower().startswith("select"):
        return False
    banned = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "pragma",
        "attach",
        "detach",
        "replace",
    ]
    for kw in banned:
        if re.search(rf"\\b{kw}\\b", q, flags=re.IGNORECASE):
            return False
    return True


def _heuristic_sql(question: str) -> str | None:
    q = question.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    if "how many" in q and "artist" in q:
        return "SELECT COUNT(*) AS ArtistCount FROM Artist;"
    if "how many" in q and "album" in q:
        return "SELECT COUNT(*) AS AlbumCount FROM Album;"
    if "how many" in q and "track" in q:
        return "SELECT COUNT(*) AS TrackCount FROM Track;"
    if "which artist" in q and ("most album" in q or "most albums" in q):
        return (
            "SELECT ar.Name, COUNT(al.AlbumId) AS AlbumCount "
            "FROM Artist ar JOIN Album al ON ar.ArtistId = al.ArtistId "
            "GROUP BY ar.ArtistId ORDER BY AlbumCount DESC LIMIT 1;"
        )
    if "top 5" in q and "artist" in q and "track" in q:
        return (
            "SELECT ar.Name, COUNT(t.TrackId) AS TrackCount "
            "FROM Artist ar "
            "JOIN Album al ON ar.ArtistId = al.ArtistId "
            "JOIN Track t ON al.AlbumId = t.AlbumId "
            "GROUP BY ar.ArtistId ORDER BY TrackCount DESC LIMIT 5;"
        )
    if ("most revenue" in q or "highest revenue" in q) and "album" in q:
        return (
            "SELECT al.Title, SUM(il.UnitPrice * il.Quantity) AS Revenue "
            "FROM Album al "
            "JOIN Track t ON al.AlbumId = t.AlbumId "
            "JOIN InvoiceLine il ON t.TrackId = il.TrackId "
            "GROUP BY al.AlbumId ORDER BY Revenue DESC LIMIT 1;"
        )
    if ("top 10" in q or "top ten" in q) and "track" in q and "sold" in q:
        return (
            "SELECT t.Name, SUM(il.Quantity) AS UnitsSold "
            "FROM Track t "
            "JOIN InvoiceLine il ON t.TrackId = il.TrackId "
            "GROUP BY t.TrackId ORDER BY UnitsSold DESC LIMIT 10;"
        )
    if "country" in q and "invoice" in q and ("highest" in q or "most" in q):
        return (
            "SELECT BillingCountry, COUNT(*) AS InvoiceCount "
            "FROM Invoice GROUP BY BillingCountry "
            "ORDER BY InvoiceCount DESC LIMIT 1;"
        )
    if "genre" in q and "most" in q and "track" in q:
        return (
            "SELECT g.Name, COUNT(t.TrackId) AS TrackCount "
            "FROM Genre g JOIN Track t ON g.GenreId = t.GenreId "
            "GROUP BY g.GenreId ORDER BY TrackCount DESC LIMIT 1;"
        )
    return None


@tool("run_sql_query")
def run_sql_query(query: str) -> str:
    """Execute a read-only SQL query against Chinook.db and return a text table."""
    if not _is_select_only(query):
        return "ERROR: Only SELECT queries are allowed."
    if not os.path.exists(DB_PATH):
        return "ERROR: Chinook.db not found."
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        try:
            cur.execute(query)
        except sqlite3.Error as exc:
            return f"ERROR: {exc}"
        rows = cur.fetchmany(21)
        if not rows:
            return "NO_ROWS"
        columns = rows[0].keys()
        display_rows = rows[:20]
        header = " | ".join(columns)
        separator = "-+-".join(["-" * len(col) for col in columns])
        body = []
        for row in display_rows:
            body.append(" | ".join(str(row[col]) for col in columns))
        suffix = ""
        if len(rows) > 20:
            suffix = "\n... showing first 20 rows"
        return "\n".join([header, separator] + body) + suffix


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL") or os.environ.get(
        "OLLAMA_HOST", "http://localhost:11434"
    )


def _ollama_server_check() -> str | None:
    base_url = _ollama_base_url().rstrip("/")
    url = f"{base_url}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return (
            "ERROR: Ollama server is not reachable.\n"
            "Start it with: ollama serve\n"
            f"Then ensure the model is pulled: ollama pull {MODEL_NAME}\n"
            f"If Ollama is remote, set OLLAMA_HOST or OLLAMA_BASE_URL to {base_url}"
        )

    models = {m.get("name") for m in payload.get("models", []) if m.get("name")}
    if MODEL_NAME not in models:
        return (
            f"ERROR: Ollama model '{MODEL_NAME}' not found.\n"
            f"Run: ollama pull {MODEL_NAME}"
        )
    return None


def _build_model() -> ChatOllama:
    return ChatOllama(model=MODEL_NAME, base_url=_ollama_base_url())


def sql_generate_node(state: GraphState) -> GraphState:
    model = _build_model()
    schema = _get_schema_description(DB_PATH)
    history_text = _format_history(state["messages"])
    latest_question = state["messages"][-1].content
    prompt = [
        SystemMessage(content=SQL_GEN_SYSTEM.format(schema=schema)),
        HumanMessage(
            content=(
                "Conversation so far:\n"
                f"{history_text}\n\n"
                f"User question: {latest_question}\n\n"
                "Return only SQL or NO_SQL."
            )
        ),
    ]
    response = model.invoke(prompt).content
    sql = _extract_sql(response)
    if sql.upper() != "NO_SQL" and not _is_select_only(sql):
        retry_prompt = [
            SystemMessage(content=SQL_GEN_SYSTEM.format(schema=schema)),
            HumanMessage(
                content=(
                    "Return ONLY a single SELECT statement ending with a semicolon. "
                    "No prose. No markdown. If not a DB question, return NO_SQL.\n\n"
                    f"User question: {latest_question}"
                )
            ),
        ]
        retry_response = model.invoke(retry_prompt).content
        sql = _extract_sql(retry_response)
        if sql.upper() != "NO_SQL" and not _is_select_only(sql):
            sql = "NO_SQL"
    if sql.upper() == "NO_SQL":
        heuristic = _heuristic_sql(latest_question)
        if heuristic and _is_select_only(heuristic):
            sql = heuristic
    return {"sql": sql}


def sql_execute_node(state: GraphState) -> GraphState:
    sql = (state.get("sql") or "").strip()
    if not sql or sql.upper().startswith("NO_SQL"):
        return {"sql_result": "NO_SQL"}
    return {"sql_result": run_sql_query.invoke(sql)}


def answer_node(state: GraphState) -> GraphState:
    user_question = state["messages"][-1].content
    sql = state.get("sql", "")
    sql_result = state.get("sql_result", "")
    if sql_result in {"NO_SQL", "NO_ROWS"} or sql_result.startswith("ERROR:"):
        if sql_result == "NO_ROWS":
            response = (
                "I couldn't find matching data in Chinook.db. "
                "Can you clarify your request (artist, album, track, invoice, customer)?"
            )
        elif sql_result == "NO_SQL":
            response = (
                "Please ask a question that can be answered from Chinook.db "
                "(artists, albums, tracks, invoices, customers, genres)."
            )
        else:
            response = (
                "There was an issue running the SQL query. "
                "Please try rephrasing your question."
            )
        updated_messages = state["messages"] + [AIMessage(content=response)]
        return {"messages": updated_messages}

    model = _build_model()
    prompt = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(
            content=(
                f"User question: {user_question}\n\n"
                f"SQL used:\n{sql}\n\n"
                f"SQL result:\n{sql_result}\n\n"
                "Answer the user."
            )
        ),
    ]
    response = model.invoke(prompt).content
    updated_messages = state["messages"] + [AIMessage(content=response)]
    return {"messages": updated_messages}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("sql_generate", sql_generate_node)
    graph.add_node("sql_execute", sql_execute_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("sql_generate")
    graph.add_edge("sql_generate", "sql_execute")
    graph.add_edge("sql_execute", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def main() -> None:
    if not os.path.exists(DB_PATH):
        print("ERROR: Chinook.db not found next to chatbot.py")
        return
    ollama_error = _ollama_server_check()
    if ollama_error:
        print(ollama_error)
        return
    app = build_graph()
    state: GraphState = {"messages": [], "sql": "", "sql_result": ""}
    print("Chinook chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        state["messages"].append(HumanMessage(content=user_input))
        state = app.invoke(state)
        print(state["messages"][-1].content)


if __name__ == "__main__":
    main()
