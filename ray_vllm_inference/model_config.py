from typing import Optional
import os
from ray_vllm_inference.prompt_format import ModelConfig

def load_model_config(model_id:str, config_path:Optional[str]=None) -> ModelConfig:
    if not config_path:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, 'models')
    path = os.path.join(config_path, model_id.replace('/', '--')) + '.yaml'
    with open(path, "r") as stream:
        return ModelConfig.parse_yaml(stream)
