import gradio as gr

# Fireworks AI Model Configuration
EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
# TODO: allow user to choose from a list of models
LLM_MODEL = "accounts/fireworks/models/qwen3-8b"
RERANKER_MODEL = "accounts/fireworks/models/qwen3-reranker-8b"

# API Constants
COMPLETION_BASE_URL = "https://api.fireworks.ai/inference/v1"
EMBEDDING_BASE_URL = "https://api.fireworks.ai/inference/v1/embeddings"
RERANK_BASE_URL = "https://api.fireworks.ai/inference/v1/rerank"

GRADIO_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    spacing_size=gr.themes.sizes.spacing_lg,
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    button_primary_background_fill="#6720FF",
    button_primary_background_fill_hover="#7B2FFF",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#F3F0FF",
    button_secondary_background_fill_hover="#EDE9FE",
    button_secondary_text_color="#6720FF",
    slider_color="#6720FF",
    link_text_color="#6720FF",
    link_text_color_hover="#7B2FFF",
    link_text_color_visited="#8B5CF6",
    body_background_fill="#FAFBFC",
    block_background_fill="#FFFFFF",
    input_background_fill="#FFFFFF",
    border_color_primary="#E6EAF4",
)


CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    background: linear-gradient(135deg, #FAFBFC 0%, #F3F0FF 100%);
}
.header-title {
    background: linear-gradient(135deg, #6720FF 0%, #8B5CF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5em;
}
.subtitle {
    color: #64748B;
    text-align: center;
    font-size: 1.1em;
    margin-top: 0;
}
.search-box {
    border: 2px solid #E6EAF4;
    border-radius: 10px;
    transition: all 0.2s ease;
}
.search-box:focus {
    border-color: #6720FF;
    box-shadow: 0 0 0 3px rgba(103, 32, 255, 0.1);
}
.result-card {
    background: white;
    border-radius: 12px;
    padding: 18px;
    margin: 10px 0;
    box-shadow: 0 2px 6px rgba(103, 32, 255, 0.08);
    border: 1px solid #E6EAF4;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(103, 32, 255, 0.15);
    border-color: #C4B5FD;
}
.metric-box {
    background: linear-gradient(135deg, #F3F0FF 0%, #FFFFFF 100%);
    border-left: 4px solid #6720FF;
    padding: 16px;
    margin: 8px 0;
    border-radius: 10px;
    font-size: 0.9em;
    box-shadow: 0 2px 4px rgba(103, 32, 255, 0.05);
}
.code-section {
    background: linear-gradient(135deg, #F3F0FF 0%, #FFFFFF 100%);
    border-left: 4px solid #6720FF;
    padding: 18px;
    margin: 12px 0;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9em;
}
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(103, 32, 255, 0.08);
    border-radius: 10px;
    overflow: hidden;
}
.comparison-table th {
    background: linear-gradient(135deg, #6720FF 0%, #7B2FFF 100%);
    color: white;
    padding: 14px;
    text-align: left;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.85em;
    letter-spacing: 0.5px;
}
.comparison-table td {
    padding: 14px;
    border-bottom: 1px solid #E6EAF4;
    background: white;
}
.comparison-table tr:hover td {
    background: #F8F7FF;
}
.comparison-table tr:last-child td {
    border-bottom: none;
}
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: #F3F0FF;
    border-radius: 5px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #C4B5FD 0%, #A78BFA 100%);
    border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #A78BFA 0%, #8B5CF6 100%);
}
details {
    border: 1px solid #E6EAF4;
    border-radius: 12px;
    padding: 14px;
    margin: 12px 0;
    background: white;
    transition: all 0.3s ease;
}
details[open] {
    border-color: #C4B5FD;
    box-shadow: 0 4px 16px rgba(103, 32, 255, 0.12);
}
summary {
    font-weight: 600;
    color: #6720FF;
    cursor: pointer;
    padding: 6px;
    user-select: none;
    transition: color 0.2s ease;
}
summary:hover {
    color: #7B2FFF;
}
.logo-image {
    display: flex;
    justify-content: flex-end;
    align-items: center;
}
.api-config-accordion {
    margin: 10px 0;
    padding: 0;
}
.api-config-accordion > .label-wrap {
    font-size: 0.85em;
    padding: 8px 12px;
}
/* Tab styling */
.tabs button {
    transition: all 0.2s ease;
}
.tabs button[aria-selected="true"] {
    border-bottom: 3px solid #6720FF !important;
}
/* Button enhancements */
button.primary {
    background: linear-gradient(135deg, #6720FF 0%, #7B2FFF 100%) !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(103, 32, 255, 0.25) !important;
}
"""