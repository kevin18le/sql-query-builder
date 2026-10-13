"""Central configuration for models and Fireworks AI API endpoints.

This module intentionally performs no I/O at import time so that scripts
(e.g. the schema-index builder) can import configuration before any
retrieval artifacts exist on disk.
"""

# Fireworks AI model configuration.
# These models were verified to be available on Fireworks serverless
# inference and to support json_schema structured output.
EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
LLM_MODEL = "accounts/fireworks/models/glm-5p3-flash"
LLM_MODEL_OPTIONS = [
    "accounts/fireworks/models/glm-5p3-flash",
    "accounts/fireworks/models/glm-5p3",
    "accounts/fireworks/models/minimax-m3",
]

# API endpoints
COMPLETION_BASE_URL = "https://api.fireworks.ai/inference/v1"
INFERENCE_BASE_URL = COMPLETION_BASE_URL  # OpenAI-compatible base URL
EMBEDDING_BASE_URL = "https://api.fireworks.ai/inference/v1/embeddings"
