"""Reusable configuration and constants for the RAG notebook project."""

import os
from typing import Any, Literal
from pathlib import Path
from dataclasses import dataclass, asdict, field

# Config

@dataclass
class EnvConfig:
    seed: int = 42
    is_kaggle: bool = "KAGGLE_" in "".join(os.environ.keys())
    requires_keys: bool = True

@dataclass
class DataConfig:
    dataset_repo: str = "QuotaClimat/frugalaichallenge-text-train"

@dataclass
class ModelConfig:
    base_model: str = "roberta-large"
    version: str = "v1.1"
    finetuned_model_binary: str = "climate-skeptic-frugalai-roberta-large-binary-LoRA"
    finetuned_model_multiclass: str = "climate-skeptic-frugalai-roberta-large-multiclass-LoRA"

@dataclass
class TrackingCongif:
    # set a literal with wandb, trackio or none
    tracker: Literal["tensorboard", "wandb", "trackio", "none"] = "trackio"
    project: str = "ClimateSkepticism"
    trackio_space_id: str = "trackio"

@dataclass
# NOTE: Use default_factory for mutable default values to prevent unintended shared state.
class CFG:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingCongif = field(default_factory=TrackingCongif)

    def __post_init__(self) -> None:
        """Runs automatically after the class is initialized."""

        if self.tracking.tracker == 'wandb':
            os.environ["WANDB_PROJECT"] = self.tracking.project
            print(f"✅ Set WANDB_PROJECT to {self.tracking.project}.")
            # Optional: Group runs together (useful for multi-seed or k-fold runs)
            # os.environ["WANDB_RUN_GROUP"] = "transformer-tuning" 

        if self.env.requires_keys:
            if self.env.is_kaggle:
                try:
                    from kaggle_secrets import UserSecretsClient
                    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_WRITE_TOKEN")
                    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
                    print("✅ All Kaggle Secrets loaded.")
                except Exception as e:
                    print(f"⚠️ Kaggle secrets not found or an error occurred: {e}")
            else:
                try:
                    from google.colab import userdata
                    os.environ["HF_TOKEN"] = userdata.get("HF_WRITE_TOKEN")
                    os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
                    print("✅ All Colab Userdata Secrets loaded.")
                except Exception as e:
                    print(f"⚠️ Colab secrets not found or an error occurred: {e}")
                    if "Secrets can only be fetched when running from the Colab UI." in str(e):
                        try:
                            from dotenv import load_dotenv
                            load_dotenv()
                            print("✅ Loaded .env file for local development.")
                        except Exception as e:
                            print(f"⚠️ .env file not found or an error occurred: {e}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def load_configs():
    return CFG()