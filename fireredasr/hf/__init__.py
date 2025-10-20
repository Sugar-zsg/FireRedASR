from .configuration_fireredasr import FireRedASRConfig
from .modeling_fireredasr import FireRedASRForSpeechToText, SpeechToTextOutput

# Optional: register with Hugging Face Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel

    # Register config under model_type "fireredasr"
    AutoConfig.register("fireredasr", FireRedASRConfig)
    # Map our config to our model implementation
    AutoModel.register(FireRedASRConfig, FireRedASRForSpeechToText)
except Exception:
    # transformers not available or API changed; ignore registration
    pass

__all__ = [
    "FireRedASRConfig",
    "FireRedASRForSpeechToText",
    "SpeechToTextOutput",
]
