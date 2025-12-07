from src.config import LLM_MODEL, GRADIO_THEME, CUSTOM_CSS
import gradio as gr
from src.fireworks.inference import (
    complete_sql, 
    export_metrics, 
    get_metrics_summary, 
    log_interaction,
    export_interactions,
    get_interactions_summary
)


def generate_completion(
    api_key: str, 
    model: str, 
    partial_sql: str, 
    use_retrieval: bool = True
):
    """
    Generate SQL completion without applying it.
    
    Args:
        api_key: Fireworks API key
        model: Model name to use
        partial_sql: Partial SQL query to complete
        use_retrieval: Whether to use schema retrieval for context
    
    Returns:
        Tuple of (suggested_completion, visibility_update_for_apply_buttons)
    """
    if not partial_sql:
        return "", gr.update(visible=False)
    
    try:
        suggested_completion = complete_sql(
            api_key=api_key,
            model=model,
            partial_sql=partial_sql,
            use_retrieval=use_retrieval,
            top_k=3
        )
        
        # Show the apply buttons if we got a valid completion (not an error)
        if suggested_completion.startswith("Error:"):
            return suggested_completion, gr.update(visible=False)
        else:
            return suggested_completion, gr.update(visible=True)
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, gr.update(visible=False)


def apply_completion_choice(
    partial_sql: str,
    suggested_completion: str,
    apply: bool
):
    """
    Handle user's choice to apply or not apply the completion.
    Logs the interaction and updates the SQL if applied.
    
    Args:
        partial_sql: The original partial SQL query
        suggested_completion: The suggested completion
        apply: True if user wants to apply, False otherwise
    
    Returns:
        Updated partial SQL and visibility update for buttons
    """
    if not partial_sql or not suggested_completion or suggested_completion.startswith("Error:"):
        return partial_sql, gr.update(visible=False)
    
    # Log the interaction based on user's response
    log_interaction(
        input_sql=partial_sql,
        suggested_completion=suggested_completion,
        applied_autocomplete=apply
    )
    
    # If user chose to apply, append the completion
    if apply:
        updated_sql = partial_sql + suggested_completion
    else:
        updated_sql = partial_sql
    
    # Hide the apply buttons after user responds
    return updated_sql, gr.update(visible=False)


def export_performance_metrics():
    """
    Export performance metrics to a file.
    
    Returns:
        Tuple of (file path, status message, visibility flag)
    """
    try:
        summary = get_metrics_summary()
        if summary.get("total_requests", 0) == 0:
            return gr.update(value=None, visible=False), "No metrics to export. Make some requests first."
        
        file_path = export_metrics(format="json")
        status_msg = f"✅ Metrics exported successfully! ({summary['total_requests']} requests)"
        return gr.update(value=file_path, visible=True), status_msg
    except Exception as e:
        return gr.update(value=None, visible=False), f"Error exporting metrics: {str(e)}"


def export_interaction_logs():
    """
    Export interaction logs to a file for fine-tuning.
    
    Returns:
        Tuple of (file path, status message, visibility flag)
    """
    try:
        summary = get_interactions_summary()
        if summary.get("total_interactions", 0) == 0:
            return gr.update(value=None, visible=False), "No interactions to export. Use autocomplete first and log some interactions."
        
        file_path = export_interactions(format="json")
        status_msg = f"✅ Interactions exported successfully! ({summary['total_interactions']} interactions, {summary.get('apply_rate_percent', 0)}% apply rate)"
        return gr.update(value=file_path, visible=True), status_msg
    except Exception as e:
        return gr.update(value=None, visible=False), f"Error exporting interactions: {str(e)}"


with gr.Blocks(theme=GRADIO_THEME, css=CUSTOM_CSS, title="SQL Autocomplete") as demo:
    gr.Markdown("## SQL Autocomplete")
    gr.Markdown("Enter your partial SQL query to get autocomplete suggestions.")
    
    api_key = gr.Textbox(label="Fireworks API key", type="password", placeholder="Enter your Fireworks API key")
    model = gr.Dropdown([LLM_MODEL], value=LLM_MODEL, label="Model")
    
    with gr.Row():
        with gr.Column(scale=2):
            partial_sql = gr.Textbox(
                lines=10, 
                label="Partial SQL Query",
                placeholder="Enter your partial SQL query here...",
                elem_id="sql-textbox"
            )
            
            use_retrieval = gr.Checkbox(
                label="Use Schema Retrieval",
                value=True,
                info="Retrieve relevant schema information for better context"
            )
            
            btn = gr.Button("Get Autocomplete", variant="primary")
            
            # Apply buttons (initially hidden, shown after completion is generated)
            with gr.Row(visible=False) as apply_buttons:
                apply_yes_btn = gr.Button("Yes, Apply", variant="primary", size="sm")
                apply_no_btn = gr.Button("No, Don't Apply", variant="secondary", size="sm")
            
            with gr.Row():
                export_metrics_btn = gr.Button("Export Performance Metrics", variant="secondary")
                export_metrics_file = gr.File(
                    label="Download Metrics",
                    visible=False,
                    interactive=False
                )
            export_metrics_status = gr.Textbox(
                label="Metrics Export Status",
                interactive=False,
                visible=True,
                value=""
            )
            
            with gr.Row():
                export_interactions_btn = gr.Button("Export Autocomplete Interactions", variant="secondary")
                export_interactions_file = gr.File(
                    label="Download Interactions",
                    visible=False,
                    interactive=False
                )
            export_interactions_status = gr.Textbox(
                label="Interactions Export Status",
                interactive=False,
                visible=True,
                value=""
            )
            
        with gr.Column(scale=1):
            output = gr.Textbox(
                label="Suggested Completion", 
                lines=12,
                interactive=False,
                info="Suggested SQL completion"
            )
    
    # Main autocomplete action - generates completion and shows apply buttons
    btn.click(
        generate_completion,
        inputs=[api_key, model, partial_sql, use_retrieval],
        outputs=[output, apply_buttons]
    )
    
    # Apply Yes button - applies completion and logs interaction
    def apply_yes(partial_sql, suggested_completion):
        return apply_completion_choice(partial_sql, suggested_completion, True)
    
    def apply_no(partial_sql, suggested_completion):
        return apply_completion_choice(partial_sql, suggested_completion, False)
    
    apply_yes_btn.click(
        apply_yes,
        inputs=[partial_sql, output],
        outputs=[partial_sql, apply_buttons]
    )
    
    # Apply No button - doesn't apply but logs interaction
    apply_no_btn.click(
        apply_no,
        inputs=[partial_sql, output],
        outputs=[partial_sql, apply_buttons]
    )
    
    # Export metrics action
    export_metrics_btn.click(
        export_performance_metrics,
        inputs=[],
        outputs=[export_metrics_file, export_metrics_status]
    )
    
    # Export interactions action
    export_interactions_btn.click(
        export_interaction_logs,
        inputs=[],
        outputs=[export_interactions_file, export_interactions_status]
    )

if __name__ == "__main__":
    demo.launch()