"""Model and training configuration schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    d_model: int = 64
    d_x: int = 8
    d_y: int = 1
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 128  # equal spacing of y values as the vocab size
    init_range: float = 0.02
    n_ctx: int = 100  # we train only on index 0,2,4,...n_ctx - 2
    d_mlp: int = 4 * d_model
    n_heads: int = 2
    d_head: int = d_model // n_heads  # should be d_model / n_heads
    n_layers: int = 4
    y_min: float = -3.0
    y_max: float = 3.0


@dataclass
class TrainConfig:
    device: Optional[str] = None
    task_size: int = 2
    num_tasks: int = 1
    noise_var: float = 0.25
    num_examples: int = 64
    learning_rate: float = 1e-3
    training_steps: int = 1000
    batch_size: int = 1024
    eval_batch_size: int = 1024
    print_loss_interval: int = 100
    print_metrics_interval: int = 1000
    n_checkpoints: Optional[int] = 10
    logarithmic_checkpoints: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def copy_with(self, **overrides: Any) -> "TrainConfig":
        return replace(self, **overrides)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


if __name__ == "__main__":
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    print(model_cfg)
    print(train_cfg)
