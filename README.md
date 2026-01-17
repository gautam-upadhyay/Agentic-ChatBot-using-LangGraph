# LangGraph Chinook Chatbot

Simple CLI chatbot using LangGraph + Ollama that answers questions by generating and executing SQL against `Chinook.db`.

## Requirements
- Python 3.10+
- Ollama running locally
- Ollama model: `qwen2:7b`
  - Start server: `ollama serve`
  - Pull model: `ollama pull qwen2:7b`

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python chatbot.py
```

### Remote Ollama
If Ollama is not local, set one of:
- `OLLAMA_HOST` (e.g. `http://10.0.0.5:11434`)
- `OLLAMA_BASE_URL` (same value)

## What It Includes
- **Persona**: `ChinookAnalyst` (helpful data analyst for Chinook.db)
- **Knowledge Base**: SQLite `Chinook.db`
- **Tools**: database query tool `run_sql_query`
- **Memory**: conversation history stored in graph state
- **Context**: last few turns used for follow-up questions

## LangGraph Nodes
- `sql_generate` (SQL generation node)
  - Located in `sql_generate_node()` in `chatbot.py`
  - Uses schema + conversation history to generate SQL or `NO_SQL`
- `sql_execute` (SQL execution node)
  - Located in `sql_execute_node()` in `chatbot.py`
  - Calls the `run_sql_query` tool to execute the SQL
- `answer` (final response node)
  - Located in `answer_node()` in `chatbot.py`
  - Uses the SQL results to answer the user

## SQL Tool
The database tool is defined in `chatbot.py`:
- `run_sql_query(query: str)` — read-only SELECT queries against `Chinook.db`

## Notes
- DB-only behavior: non-database questions are refused and the user is asked to rephrase.
- Result limit: first 20 rows are returned if more are available.

## Example Prompts
- "Top 5 artists by number of tracks"
- "What is the total sales by country?"
- "List customers who bought Jazz tracks"


![Alt text](images/image.png)
![Alt text](https://example.com/path/to/image.png)