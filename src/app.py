"""Gradio UI for the schema-aware SQL autocomplete demo."""

import gradio as gr

from src.config import LLM_MODEL_OPTIONS
from src.fireworks.inference import complete_sql
from src.metrics import (
    export_interactions,
    export_metrics,
    get_interactions_summary,
    get_metrics_summary,
    log_interaction,
)
from src.ui.theme import CUSTOM_CSS, GRADIO_THEME

EXAMPLE_QUERIES = [
    ["-- get customer info for user 1\nSELECT * FROM cus"],
    ["-- total balance per account type\nSELECT account_type, SUM(balance) FROM accounts GROUP BY "],
    ["-- recent transactions with customer names\nSELECT t.transaction_id, c.full_name FROM transactions t JOIN "],
]


def generate_completion(
    api_key: str,
    model: str,
    partial_sql: str,
    use_retrieval: bool = True,
    top_k: int = 3,
):
    """Generate a SQL completion without applying it.

    Returns:
        Tuple of (suggested_completion, retrieved_context, apply-buttons visibility).
    """
    if not api_key or not api_key.strip():
        return "Error: Please enter your Fireworks API key.", "", gr.update(visible=False)
    if not partial_sql or not partial_sql.strip():
        return "Error: Please enter a partial SQL query.", "", gr.update(visible=False)

    try:
        suggested_completion, retrieved_context = complete_sql(
            api_key=api_key.strip(),
            model=model,
            partial_sql=partial_sql,
            use_retrieval=use_retrieval,
            top_k=int(top_k),
        )

        # Show the apply buttons only for a valid completion (not an error)
        if suggested_completion.startswith("Error:"):
            return suggested_completion, retrieved_context or "", gr.update(visible=False)
        return suggested_completion, retrieved_context or "", gr.update(visible=True)
    except Exception as e:
        return f"Error: {e}", "", gr.update(visible=False)


def apply_completion_choice(
    partial_sql: str,
    suggested_completion: str,
    apply: bool,
):
    """Apply or reject a completion, logging the interaction either way."""
    if not partial_sql or not suggested_completion or suggested_completion.startswith("Error:"):
        return partial_sql, gr.update(visible=False)

    log_interaction(
        input_sql=partial_sql,
        suggested_completion=suggested_completion,
        applied_autocomplete=apply,
    )

    updated_sql = partial_sql + suggested_completion if apply else partial_sql
    return updated_sql, gr.update(visible=False)


def export_performance_metrics():
    """Export performance metrics to a downloadable file."""
    try:
        summary = get_metrics_summary()
        if summary.get("total_requests", 0) == 0:
            return gr.update(value=None, visible=False), "No metrics to export. Make some requests first."

        file_path = export_metrics(format="json")
        status_msg = f"✅ Metrics exported successfully! ({summary['total_requests']} requests)"
        return gr.update(value=file_path, visible=True), status_msg
    except Exception as e:
        return gr.update(value=None, visible=False), f"Error exporting metrics: {e}"


def export_interaction_logs():
    """Export interaction logs to a downloadable JSONL file."""
    try:
        summary = get_interactions_summary()
        if summary.get("total_interactions", 0) == 0:
            return gr.update(value=None, visible=False), "No interactions to export. Use autocomplete first and log some interactions."

        file_path = export_interactions(format="jsonl")
        status_msg = f"✅ Interactions exported successfully! ({summary['total_interactions']} interactions, {summary.get('apply_rate_percent', 0)}% apply rate)"
        return gr.update(value=file_path, visible=True), status_msg
    except Exception as e:
        return gr.update(value=None, visible=False), f"Error exporting interactions: {e}"


# The UI is designed for a light palette; force it so system dark mode
# doesn't mix dark Gradio components with the light custom CSS.
FORCE_LIGHT_THEME_JS = """
() => {
    const url = new URL(window.location);
    if (!url.searchParams.has('__theme')) {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(theme=GRADIO_THEME, css=CUSTOM_CSS, js=FORCE_LIGHT_THEME_JS, title="SQL Autocomplete") as demo:
    gr.Markdown("## SQL Autocomplete", elem_id="title-markdown")
    gr.Markdown("Enter your partial SQL query to get autocomplete suggestions.", elem_id="subtitle-markdown")

    api_key = gr.Textbox(label="Fireworks API key", type="password", placeholder="Enter your Fireworks API key")
    model = gr.Dropdown(LLM_MODEL_OPTIONS, value=LLM_MODEL_OPTIONS[0], label="Select a model")

    # Main SQL editor, autocomplete, and context in three columns
    with gr.Row():
        with gr.Column():
            partial_sql = gr.Textbox(
                lines=10,
                label="SQL Editor",
                placeholder="Enter your partial SQL query here...",
                elem_id="sql-textbox",
            )

        with gr.Column():
            output = gr.Textbox(
                label="Suggested Completion",
                lines=10,
                interactive=False,
            )

        with gr.Column():
            retrieved_context = gr.Textbox(
                label="Retrieved Context",
                lines=10,
                interactive=False,
                visible=True,
            )

    gr.Examples(
        examples=EXAMPLE_QUERIES,
        inputs=partial_sql,
        label="Example partial queries",
    )

    # Retrieval Parameters section
    with gr.Group():
        gr.Markdown("### Retrieval Parameters")
        with gr.Row():
            use_retrieval = gr.Checkbox(
                label="Use Schema Retrieval",
                value=True,
                info="Retrieve relevant schema information for better context",
            )
            top_k = gr.Number(
                label="Top K",
                value=5,
                minimum=1,
                maximum=20,
                step=1,
                info="Number of top schema documents to retrieve",
            )

    btn = gr.Button("Get Autocomplete", variant="primary")

    # Apply buttons (initially hidden, shown after completion is generated)
    with gr.Row(visible=False) as apply_buttons:
        apply_yes_btn = gr.Button("Yes, Apply", variant="primary", size="sm")
        apply_no_btn = gr.Button("No, Don't Apply", variant="secondary", size="sm")

    with gr.Row():
        with gr.Column():
            export_metrics_btn = gr.Button("Export Performance Metrics", variant="secondary")
            export_metrics_file = gr.File(
                label="Download Metrics",
                visible=False,
                interactive=False,
            )
            export_metrics_status = gr.Textbox(
                label="Metrics Export Status",
                interactive=False,
                visible=True,
                value="",
            )

        with gr.Column():
            export_interactions_btn = gr.Button("Export Autocomplete Interactions", variant="secondary")
            export_interactions_file = gr.File(
                label="Download Interactions",
                visible=False,
                interactive=False,
            )
            export_interactions_status = gr.Textbox(
                label="Interactions Export Status",
                interactive=False,
                visible=True,
                value="",
            )

    # Main autocomplete action - generates completion and shows apply buttons
    btn.click(
        generate_completion,
        inputs=[api_key, model, partial_sql, use_retrieval, top_k],
        outputs=[output, retrieved_context, apply_buttons],
    )

    apply_yes_btn.click(
        lambda sql, completion: apply_completion_choice(sql, completion, True),
        inputs=[partial_sql, output],
        outputs=[partial_sql, apply_buttons],
    )

    apply_no_btn.click(
        lambda sql, completion: apply_completion_choice(sql, completion, False),
        inputs=[partial_sql, output],
        outputs=[partial_sql, apply_buttons],
    )

    export_metrics_btn.click(
        export_performance_metrics,
        inputs=[],
        outputs=[export_metrics_file, export_metrics_status],
    )

    export_interactions_btn.click(
        export_interaction_logs,
        inputs=[],
        outputs=[export_interactions_file, export_interactions_status],
    )

if __name__ == "__main__":
    demo.launch()
