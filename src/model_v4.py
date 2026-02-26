# src/model_v4.py
from physicsnemo.models.mlp.fully_connected import FullyConnected

def build_model(num_layers: int, layer_size: int, device):
    # Input: (r,z,t) -> Outputs: [Tm, Ts]
    model = FullyConnected(
        in_features=3,
        out_features=2,
        num_layers=num_layers,
        layer_size=layer_size,
    )
    return model.to(device)