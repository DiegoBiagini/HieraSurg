from diffusers import CogVideoXTransformer3DModel
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, LabelEmbedding, CogVideoXPatchEmbed
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.embeddings import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed
from diffusers.utils import logging, USE_PEFT_BACKEND, unscale_lora_layers, scale_lora_layers
from diffusers.utils.torch_utils import is_torch_version
from diffusers.models.attention import Attention
from diffusers.models.normalization import AdaLayerNorm
from diffusers.models.attention_processor import FusedCogVideoXAttnProcessor2_0
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput 

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=downsample_factor, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=downsample_factor)
    
    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor=1):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=(1,downsample_factor,downsample_factor), padding=1, bias=False)
        #self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=downsample_factor, padding=1, bias=False)        
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if downsample_factor != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1,downsample_factor,downsample_factor), bias=False),
                #nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=downsample_factor, bias=False),                
                nn.BatchNorm3d(out_channels),
            )


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CogVideoXPatchEmbedImgFeatures(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        img_feature_dim: int = 1024,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        self.proj_f = nn.Conv2d(
            img_feature_dim, embed_dim, kernel_size=(1, 1), stride=1, bias=bias
        )        
        #self.text_proj = nn.Linear(text_embed_dim, embed_dim)

        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            #pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames, num_patches_features=42)
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames, 10,10)            
            self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)

    def _get_positional_embeddings(self, sample_height: int, sample_width: int, sample_frames: int,
                                   feature_sample_height, feature_sample_width) -> torch.Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
            output_type='pt'
        )
        pos_embedding = pos_embedding.flatten(0, 1)

        post_patch_heightf = feature_sample_height #// self.patch_size
        post_patch_widthf = feature_sample_width #// self.patch_size

        feature_pos_embedding = get_2d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_widthf, post_patch_heightf),
            interpolation_scale=self.spatial_interpolation_scale,
            output_type='pt'
        )
        #feature_pos_embedding = feature_pos_embedding.flatten(0, 1)
        
        joint_pos_embedding = torch.cat([feature_pos_embedding, pos_embedding], dim=0)
        return joint_pos_embedding
    
    def forward(self, img_features: torch.Tensor, 
                image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        #text_embeds = self.text_proj(text_embeds)

        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

        batch_f, height_f, width_f, channels_f = img_features.shape
        img_features = img_features.reshape(-1, channels_f, height_f, width_f)
        img_features = self.proj_f(img_features)
        img_features = img_features.view(batch_f, *img_features.shape[1:])
        img_features = img_features.flatten(2).transpose(1,2)  # [batch, height x width, channels]
        num_patches_features = img_features.shape[1]

        embeds = torch.cat(
            [img_features, image_embeds], dim=1
        ).contiguous()  # [batch, num_frames x height x width + height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                #pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames, num_patches_features=num_patches_features)
                pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames, height_f, width_f)
                pos_embedding = pos_embedding.to(embeds.device, dtype=embeds.dtype)
            else:
                pos_embedding = self.pos_embedding

            embeds = embeds + pos_embedding

        return embeds, num_patches_features


class CogVideoXTransformer3DModel_SegCond_I2V(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        img_feature_dim: int = 1024,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        use_segmap_posenc = False,
        img_cond = "RADIO"
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                text_embed_dim=768,
                bias=True,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_frames=sample_frames,
                temporal_compression_ratio=temporal_compression_ratio,
                max_text_seq_length=1,
                spatial_interpolation_scale=spatial_interpolation_scale,
                temporal_interpolation_scale=temporal_interpolation_scale,
                use_positional_embeddings=not use_rotary_positional_embeddings,
                use_learned_positional_embeddings=use_learned_positional_embeddings,
            )         

        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 2.1 Encoder for the mask conditioning
        self.encoder_segmap = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(patch_size, patch_size), stride=patch_size, bias=True),
        )

        self.segmap_resblocks = nn.ModuleList(
            [ResBlock3D(64, 128, downsample_factor=2),
             ResBlock3D(128, 512, downsample_factor=2),
             ResBlock3D(512, inner_dim, downsample_factor=2),
             ]
        )
        
        self.use_segmap_posenc = use_segmap_posenc
        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)
        self.inner_dim = inner_dim
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)


    def encode_segmap(self, segmap: torch.Tensor):
        batch, num_frames, height, width = segmap.shape
        # Downsample the segmap by 2x2
        segmap = torch.nn.functional.interpolate(
             segmap.view(batch * num_frames, 1, height, width),
             scale_factor=(1/2, 1/2), # Probably better not to downscale x8 here
             mode='nearest'
        )
        _, _, height, width = segmap.shape

        segmap_embeds = self.encoder_segmap(segmap)
        segmap_embeds = segmap_embeds.view(batch, num_frames, *segmap_embeds.shape[1:]).permute(0,2,1,3,4)

        # Add it here if you want before the resblocks
        # Pass segmap_encoding through ResBlock
        for resblock in self.segmap_resblocks:
            segmap_embeds = resblock(segmap_embeds)

        # Add the pos encoding at the end maybe(or before the resblocks?)
        if self.use_segmap_posenc:
            post_time_compression_frames, post_patch_height, post_patch_width = list(segmap_embeds.shape[2:])
            num_patches = post_patch_height * post_patch_width * post_time_compression_frames
            pos_embedding = get_3d_sincos_pos_embed(
                self.inner_dim,
                (post_patch_width, post_patch_height),
                post_time_compression_frames,
                1.875,
                1.0,
                output_type='pt'
            )
            pos_embedding = pos_embedding.flatten(0, 1)
            pos_embedding = pos_embedding.to(segmap_embeds.device, dtype=segmap_embeds.dtype)
            
            segmap_embeds = torch.flatten(segmap_embeds, start_dim=2).permute(0,2,1)                  
            segmap_encoding = segmap_embeds + pos_embedding      
        else:
            segmap_encoding = torch.flatten(segmap_embeds, start_dim=2).permute(0,2,1)                  
        return segmap_encoding
    
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        video_segmap: torch.Tensor, # Was encoder_hidden_states
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states= self.patch_embed(torch.zeros(size=(batch_size,1,768), device=hidden_states.device, dtype=hidden_states.dtype), hidden_states)        
        img_feature_embed_len = 1
        hidden_states = self.embedding_dropout(hidden_states)

        encoder_hidden_states = self.encode_segmap(video_segmap)
        hidden_states = hidden_states#[:, text_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, img_feature_embed_len:, ...]#  to cut out the encoder_hidden_states contribution, i.e.: [:, text_seq_length:]

        else:
            # CogVideoX-5B#
            # Or Just dont cat and pass it through the norm_final
            # Then remove the img_feature_cond contribution
            #hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, img_feature_embed_len:, ...]#  to cut out the encoder_hidden_states contribution, i.e.: [:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

class CogVideoXTransformer3DModel_SegCond_Hierarchical_I2V(CogVideoXTransformer3DModel_SegCond_I2V):
    N_CLASSES_PHASE = 7 
    N_CLASSES_TRIPLET = 101

    @register_to_config    
    def __init__(self,
                 num_attention_heads: int = 30,
                 attention_head_dim: int = 64,
                 in_channels: int = 16,
                 out_channels: Optional[int] = 16,
                 flip_sin_to_cos: bool = True,
                 freq_shift: int = 0,
                 time_embed_dim: int = 512,
                 img_feature_dim: int = 1024,
                 num_layers: int = 30,
                 dropout: float = 0.0,
                 attention_bias: bool = True,
                 sample_width: int = 90,
                 sample_height: int = 60,
                 sample_frames: int = 49,
                 patch_size: int = 2,
                 temporal_compression_ratio: int = 4,
                 activation_fn: str = "gelu-approximate",
                 timestep_activation_fn: str = "silu",
                 norm_elementwise_affine: bool = True,
                 norm_eps: float = 1e-5,
                 spatial_interpolation_scale: float = 1.875,
                 temporal_interpolation_scale: float = 1.0,
                 use_rotary_positional_embeddings: bool = False,
                 use_learned_positional_embeddings: bool = False,
                 use_segmap_posenc = False,
                 img_cond = "RADIO",  
                 use_phase_emb=True,
                 use_triplet_emb=True,
                 text_cond = "label_emb"):
        super().__init__(num_attention_heads,attention_head_dim, in_channels, out_channels, flip_sin_to_cos, freq_shift, time_embed_dim*2, img_feature_dim,
                         num_layers, dropout, attention_bias, sample_width, sample_height, sample_frames, patch_size, temporal_compression_ratio,
                         activation_fn, timestep_activation_fn, norm_elementwise_affine, norm_eps, spatial_interpolation_scale, temporal_interpolation_scale,
                         use_rotary_positional_embeddings, use_learned_positional_embeddings, use_segmap_posenc, img_cond)
        self.time_embed_dim = time_embed_dim    
        
        inner_dim = num_attention_heads * attention_head_dim        
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)
        
        self.use_phase_emb = use_phase_emb
        self.use_triplet_emb = use_triplet_emb
        self.text_cond = text_cond
        if self.use_phase_emb:
            if text_cond == 'label_emb':
                self.phase_emb = LabelEmbedding(self.N_CLASSES_PHASE, self.time_embed_dim, dropout_prob=0)
            elif text_cond == 'SurgVLP':
                self.phase_emb = nn.Sequential(
                    nn.Linear(768, self.time_embed_dim),
                    nn.ReLU()
                )                
            self.phase_condense = nn.Sequential(
                nn.Conv1d(self.time_embed_dim, self.time_embed_dim//2, kernel_size=5, padding=5//2),
                nn.AdaptiveAvgPool1d(1)  # Pooling to (1)
            )                
                #raise Exception                 
        if self.use_triplet_emb:
            if text_cond == 'label_emb':        
                self.triplet_emb = LabelEmbedding(self.N_CLASSES_TRIPLET, self.time_embed_dim, dropout_prob=0)
            elif text_cond == 'SurgVLP':
                self.triplet_emb = nn.Sequential(
                    nn.Linear(768, self.time_embed_dim),
                    nn.ReLU()
                )                    
            self.triplet_condense = nn.Sequential(
                nn.Conv1d(self.time_embed_dim, self.time_embed_dim//2, kernel_size=5, padding=5//2),
                nn.AdaptiveAvgPool1d(1)  # Pooling to (1)
            )                   
    def forward(
        self,
        hidden_states: torch.Tensor,
        video_segmap: torch.Tensor, # Was encoder_hidden_states,
        timestep: Union[int, float, torch.LongTensor],
        phase_emb: Optional[torch.Tensor] = None,
        triplet_emb: Optional[torch.Tensor] = None,        
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 1.5 other embeddings
        if self.use_phase_emb:
            if self.text_cond == "SurgVLP":
                batch_size, n_frames_emb, emb_dim = phase_emb.shape
                phase_emb = phase_emb.reshape(batch_size*n_frames_emb, -1)
            phase_emb = self.phase_emb(phase_emb)
            if self.text_cond == "SurgVLP":
                phase_emb = phase_emb.reshape(batch_size, n_frames_emb,-1)            
            phase_emb = self.phase_condense(phase_emb.permute(0,2,1)).squeeze(-1)            
            emb = torch.cat([emb,phase_emb],axis=1)
        if self.use_triplet_emb:
            if self.text_cond == "SurgVLP":
                batch_size, n_frames_emb, emb_dim = triplet_emb.shape
                triplet_emb = triplet_emb.reshape(batch_size*n_frames_emb, -1)            
            triplet_emb = self.triplet_emb(triplet_emb)
            if self.text_cond == "SurgVLP":
                triplet_emb = triplet_emb.reshape(batch_size, n_frames_emb,-1)               
            triplet_emb = self.triplet_condense(triplet_emb.permute(0,2,1)).squeeze(-1)            
            emb = torch.cat([emb,triplet_emb],axis=1)

        # 2. Patch embedding
        hidden_states= self.patch_embed(torch.zeros(size=(batch_size,1,768), device=hidden_states.device, dtype=hidden_states.dtype), hidden_states)        
        img_feature_embed_len = 1
        hidden_states = self.embedding_dropout(hidden_states)

        encoder_hidden_states = self.encode_segmap(video_segmap)
        hidden_states = hidden_states#[:, text_seq_length:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, img_feature_embed_len:, ...]#  to cut out the encoder_hidden_states contribution, i.e.: [:, text_seq_length:]

        else:
            # CogVideoX-5B#
            # Or Just dont cat and pass it through the norm_final
            # Then remove the img_feature_cond contribution
            #hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, img_feature_embed_len:, ...]#  to cut out the encoder_hidden_states contribution, i.e.: [:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
    