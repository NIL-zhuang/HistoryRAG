from rag.server.models.chat_spec import ModelConfig
from rag.settings import Settings


def get_model_configs(model_name: str, platform_name: str = None) -> ModelConfig:
    platforms = Settings.model_settings.MODEL_PLATFORMS
    if platform_name is not None:
        platform = next(p for p in platforms if p.PLATFORM_NAME == platform_name)
        if platform is None:
            raise ValueError(f"Platform {platform_name} not found in platforms")
    else:
        platform = next(p for p in platforms if model_name in p.LLM_MODELS)
        if platform is None:
            raise ValueError(f"Model {model_name} not found in any platforms")

    return ModelConfig(
        platform_name=platform.PLATFORM_NAME,
        platform_type=platform.PLATFORM_TYPE,
        api_base_url=platform.API_BASE_URL,
        api_key=platform.API_KEY,
        model_name=model_name,
    )
