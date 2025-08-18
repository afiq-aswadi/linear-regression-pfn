from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 64
    d_x: int = 8
    d_y: int = 1
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 128 #let's be lazy and just use equal spacing of y values as the vocab size.
    init_range: float = 0.02
    n_ctx: int = 100 ## we train only on index 0,2,4,...n_ctx - 2
    d_mlp: int = 4* d_model
    n_heads: int = 2
    d_head: int = d_model // n_heads #from my understanding should be d_model / n_heads
    n_layers: int = 4
    y_min: float = -3.0
    y_max: float = 3.0


if __name__ == "__main__":
    model_cfg = ModelConfig()
    print(model_cfg)