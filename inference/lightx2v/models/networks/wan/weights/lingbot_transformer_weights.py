"""
LingBot Camera Control - Transformer Weights Module.

Extends WanTransformerWeights with per-block camera control layers:
  - cam_injector_layer1/2: MLP for camera embedding
  - cam_scale_layer: linear layer for modulation scale
  - cam_shift_layer: linear layer for modulation shift
"""

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
    WanTransformerAttentionBlock,
    WanSelfAttention,
    WanCrossAttention,
    WanFFN,
)
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class LingBotSelfAttention(WanSelfAttention):
    """Self-attention with additional camera control weights."""

    def __init__(self, block_index, block_prefix, task, mm_type, config,
                 create_cuda_buffer=False, create_cpu_buffer=False,
                 lazy_load=False, lazy_load_file=None, lora_path=None):
        super().__init__(
            block_index, block_prefix, task, mm_type, config,
            create_cuda_buffer, create_cpu_buffer,
            lazy_load, lazy_load_file, lora_path,
        )

        # Camera injector MLP
        self.add_module(
            "cam_injector_layer1",
            MM_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.cam_injector_layer1.weight",
                f"{block_prefix}.{self.block_index}.cam_injector_layer1.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
            ),
        )
        self.add_module(
            "cam_injector_layer2",
            MM_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.cam_injector_layer2.weight",
                f"{block_prefix}.{self.block_index}.cam_injector_layer2.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
            ),
        )

        # Camera scale and shift
        self.add_module(
            "cam_scale_layer",
            MM_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.cam_scale_layer.weight",
                f"{block_prefix}.{self.block_index}.cam_scale_layer.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
            ),
        )
        self.add_module(
            "cam_shift_layer",
            MM_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{self.block_index}.cam_shift_layer.weight",
                f"{block_prefix}.{self.block_index}.cam_shift_layer.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
            ),
        )


class LingBotAttentionBlock(WeightModule):
    """Attention block with LingBot camera control weights in self-attention phase."""

    def __init__(
        self,
        block_index,
        task,
        mm_type,
        config,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        block_prefix="blocks",
        lazy_load=False,
        lazy_load_path=None,
        lora_path=None,
    ):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.create_cuda_buffer = create_cuda_buffer
        self.create_cpu_buffer = create_cpu_buffer
        self.quant_method = config.get("quant_method", None)

        self.lazy_load = lazy_load
        if self.lazy_load:
            self.lazy_load_file = lazy_load_path
        else:
            self.lazy_load_file = None

        self.compute_phases = WeightModuleList(
            [
                LingBotSelfAttention(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    self.lazy_load,
                    self.lazy_load_file,
                    lora_path,
                ),
                WanCrossAttention(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    self.lazy_load,
                    self.lazy_load_file,
                    lora_path,
                ),
                WanFFN(
                    block_index,
                    block_prefix,
                    task,
                    mm_type,
                    config,
                    create_cuda_buffer,
                    create_cpu_buffer,
                    self.lazy_load,
                    self.lazy_load_file,
                    lora_path,
                ),
            ]
        )

        self.add_module("compute_phases", self.compute_phases)


class LingBotTransformerWeights(WeightModule):
    """Transformer weights with LingBot camera control per block."""

    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.blocks_num = config["num_layers"]
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        if self.mm_type != "Default":
            assert config.get("dit_quantized") is True
        if config.get("do_mm_calib", False):
            self.mm_type = "Calib"
            assert not config["cpu_offload"]
        self.lazy_load = self.config.get("lazy_load", False)
        self.blocks = WeightModuleList(
            [
                LingBotAttentionBlock(
                    block_index=i,
                    task=self.task,
                    mm_type=self.mm_type,
                    config=self.config,
                    create_cuda_buffer=False,
                    create_cpu_buffer=False,
                    block_prefix="blocks",
                    lazy_load=self.lazy_load,
                    lazy_load_path=lazy_load_path,
                )
                for i in range(self.blocks_num)
            ]
        )
        self.register_offload_buffers(config, lazy_load_path, lora_path)
        self.add_module("blocks", self.blocks)

        # non blocks weights
        self.register_parameter("norm", LN_WEIGHT_REGISTER["torch"]())
        self.add_module(
            "head",
            MM_WEIGHT_REGISTER["Default"](
                "head.head.weight",
                "head.head.bias",
                lora_prefix="diffusion_model.head",
            ),
        )
        self.register_parameter("head_modulation", TENSOR_REGISTER["Default"]("head.modulation"))

    def register_offload_buffers(self, config, lazy_load_path, lora_path):
        if config["cpu_offload"]:
            if config["offload_granularity"] == "block":
                self.offload_blocks_num = 2
                self.offload_block_cuda_buffers = WeightModuleList(
                    [
                        LingBotAttentionBlock(
                            block_index=i,
                            task=self.task,
                            mm_type=self.mm_type,
                            config=self.config,
                            create_cuda_buffer=True,
                            create_cpu_buffer=False,
                            block_prefix="blocks",
                            lazy_load=self.lazy_load,
                            lazy_load_path=lazy_load_path,
                        )
                        for i in range(self.offload_blocks_num)
                    ]
                )
                self.add_module("offload_block_cuda_buffers", self.offload_block_cuda_buffers)
                self.offload_phase_cuda_buffers = None

                if self.lazy_load:
                    self.offload_block_cpu_buffers = WeightModuleList(
                        [
                            LingBotAttentionBlock(
                                block_index=i,
                                task=self.task,
                                mm_type=self.mm_type,
                                config=self.config,
                                create_cuda_buffer=False,
                                create_cpu_buffer=True,
                                block_prefix="blocks",
                                lazy_load=self.lazy_load,
                                lazy_load_path=lazy_load_path,
                            )
                            for i in range(self.offload_blocks_num)
                        ]
                    )
                    self.add_module("offload_block_cpu_buffers", self.offload_block_cpu_buffers)
                    self.offload_phase_cpu_buffers = None

            elif config["offload_granularity"] == "phase":
                self.offload_phase_cuda_buffers = LingBotAttentionBlock(
                    block_index=0,
                    task=self.task,
                    mm_type=self.mm_type,
                    config=self.config,
                    create_cuda_buffer=True,
                    create_cpu_buffer=False,
                    block_prefix="blocks",
                    lazy_load=self.lazy_load,
                    lazy_load_path=lazy_load_path,
                ).compute_phases
                self.add_module("offload_phase_cuda_buffers", self.offload_phase_cuda_buffers)
                self.offload_block_cuda_buffers = None
                if self.lazy_load:
                    self.offload_phase_cpu_buffers = WeightModuleList(
                        [
                            LingBotAttentionBlock(
                                block_index=i,
                                task=self.task,
                                mm_type=self.mm_type,
                                config=self.config,
                                create_cuda_buffer=False,
                                create_cpu_buffer=True,
                                block_prefix="blocks",
                                lazy_load=self.lazy_load,
                                lazy_load_path=lazy_load_path,
                                lora_path=lora_path,
                            ).compute_phases
                            for i in range(2)
                        ]
                    )
                    self.add_module("offload_phase_cpu_buffers", self.offload_phase_cpu_buffers)
                    self.offload_block_cpu_buffers = None

    def non_block_weights_to_cuda(self):
        self.norm.to_cuda()
        self.head.to_cuda()
        self.head_modulation.to_cuda()

    def non_block_weights_to_cpu(self):
        self.norm.to_cpu()
        self.head.to_cpu()
        self.head_modulation.to_cpu()
