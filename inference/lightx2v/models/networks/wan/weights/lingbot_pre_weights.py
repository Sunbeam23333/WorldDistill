"""
LingBot Camera Control - Pre Weights Module.

Extends WanPreWeights with:
  - patch_embedding_wancamctrl: Linear projection for camera Plücker patches (not Conv3D!)
  - c2ws_hidden_states_layer1/2: MLP to process projected Plücker embeddings

Original lingbot-world uses nn.Linear for patch_embedding_wancamctrl:
  nn.Linear(6 * 64 * patch_size[0] * patch_size[1] * patch_size[2], dim)
"""

from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class LingBotPreWeights(WanPreWeights):
    def __init__(self, config):
        super().__init__(config)

        # Camera control Plücker patch embedding
        # Original lingbot uses nn.Linear (not Conv3D):
        #   nn.Linear(6 * 64 * patch_size_prod, dim)
        self.add_module(
            "patch_embedding_wancamctrl",
            MM_WEIGHT_REGISTER["Default"](
                "patch_embedding_wancamctrl.weight",
                "patch_embedding_wancamctrl.bias",
            ),
        )

        # Camera-to-world hidden states MLP (both layers: dim -> dim)
        self.add_module(
            "c2ws_hidden_states_layer1",
            MM_WEIGHT_REGISTER["Default"](
                "c2ws_hidden_states_layer1.weight",
                "c2ws_hidden_states_layer1.bias",
            ),
        )
        self.add_module(
            "c2ws_hidden_states_layer2",
            MM_WEIGHT_REGISTER["Default"](
                "c2ws_hidden_states_layer2.weight",
                "c2ws_hidden_states_layer2.bias",
            ),
        )
