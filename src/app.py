from openai import OpenAI
from config import EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL, COMPLETION_BASE_URL, GRADIO_THEME, CUSTOM_CSS

from pydantic import BaseModel
import re
import gradio as gr

class SqlCompletion(BaseModel):
    sql: str

def extract_json_from_response(response: str, response_model: BaseModel) -> str:
    """Extract JSON from response that may contain reasoning tags."""
    # Extract reasoning (if present)
    reasoning_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    # Extract JSON
    json_match = re.search(r"</think>\s*(\{.*\})", response, re.DOTALL) if reasoning else re.search(r"(\{.*\})", response, re.DOTALL)
    json_str = json_match.group(1).strip() if json_match else "{}"

    # Parse into Pydantic model
    parsed_response = response_model.model_validate_json(json_str)
    return parsed_response

def complete_sql_partial(api_key: str, model: str, partial_sql: str, schema_context: str=None) -> str:
    if schema_context is None:
        schema_context = """
        """
    # 1) Build prompt
    system_prompt = (
        "You are a SQL autocomplete assistant. "
        "Given a partial SQL query and database schema, "
        "you suggest the next tokens to complete the query. "
        "Return ONLY the completion, not the original input."
    )
    user_prompt = f"""SCHEMA:
{schema_context}

PARTIAL_SQL:
{partial_sql}

Return only the continuation of PARTIAL_SQL."""

    client = OpenAI(
        api_key=api_key, # hardcoded for now
        base_url=COMPLETION_BASE_URL,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SqlCompletion",
                "schema": SqlCompletion.model_json_schema()
            }
        }
    )
    parsed_response = extract_json_from_response(response.choices[0].message.content, SqlCompletion)
    return parsed_response.sql

def autocomplete(api_key, model, partial_sql):
    schema_context = open("schema_context.txt").read()
    return complete_sql_partial(api_key, model, partial_sql, schema_context)

with gr.Blocks(theme=GRADIO_THEME, css=CUSTOM_CSS, title="SQL Autocomplete") as demo:
    api_key = gr.Textbox(label="Fireworks API key", type="password")
    model = gr.Dropdown([LLM_MODEL], value=LLM_MODEL)
    partial_sql = gr.Textbox(lines=8, label="Partial SQL")
    output = gr.Textbox(label="Suggested completion", lines=8)

    btn = gr.Button("Autocomplete")
    btn.click(autocomplete, [api_key, model, partial_sql], output)

if __name__ == "__main__":
    demo.launch()
#     print(complete_sql_partial(
#         api_key='fw_3ZiLG2oTvDjNowNub2vRAUuG', # hardcoded for now
#         model=LLM_MODEL,
#         partial_sql="SELECT * FROM customers JOIN transactions ON",
#         schema_context="""
#         """
#     ))