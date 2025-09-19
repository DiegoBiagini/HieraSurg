import cv2
import numpy as np
import torch
from einops import rearrange
import os
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from typing import NamedTuple, Tuple, Union
from dataclasses import dataclass
import torch.nn.functional as F

class AdaptorInput(NamedTuple):
    images: torch.Tensor
    summary: torch.Tensor
    features: torch.Tensor


class RadioOutput(NamedTuple):
    summary: torch.Tensor
    features: torch.Tensor

    def to(self, *args, **kwargs):
        return RadioOutput(
            self.summary.to(*args, **kwargs) if self.summary is not None else None,
            self.features.to(*args, **kwargs) if self.features is not None else None,
        )


class AdaptorBase(nn.Module):
    def forward(self, input: AdaptorInput) -> RadioOutput:
        raise NotImplementedError("Subclasses must implement this!")


norm_t = Union[Tuple[float, float, float], torch.Tensor]

class InputConditioner(nn.Module):
    def __init__(self,
                 input_scale: float,
                 norm_mean: norm_t,
                 norm_std: norm_t,
                 dtype: torch.dtype = None,
    ):
        super().__init__()

        self.dtype = dtype

        self.register_buffer("norm_mean", _to_tensor(norm_mean) / input_scale)
        self.register_buffer("norm_std", _to_tensor(norm_std) / input_scale)

    def forward(self, x: torch.Tensor):
        y = (x - self.norm_mean) / self.norm_std
        if self.dtype is not None:
            y = y.to(self.dtype)
        return y


def get_default_conditioner():
    from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

    return InputConditioner(
        input_scale=1.0,
        norm_mean=OPENAI_CLIP_MEAN,
        norm_std=OPENAI_CLIP_STD,
    )


def _to_tensor(v: norm_t):
    return torch.as_tensor(v, dtype=torch.float32).view(-1, 1, 1)

class DinoWrapper(nn.Module):
    def __init__(self, dino_model: nn.Module):
        super().__init__()
        self.inner = dino_model

    @property
    def patch_size(self):
        return self.inner.patch_size

    @property
    def vision_encoder(self):
        return self.inner

    def forward(self, *args, **kwargs):
        parts = self.inner.forward_features(*args, **kwargs)

        cls_token = parts['x_norm_clstoken']
        features = parts['x_norm_patchtokens']

        return cls_token, features


class CLIPWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module, tokenizer, adaptor_name: str, clip_mode: bool = False):
        super().__init__()
        self.inner = clip_model
        clip_model.visual.output_tokens = True
        self.tokenizer = tokenizer
        self.adaptor_name = adaptor_name

        if not clip_mode and hasattr(clip_model.visual, 'proj'):
            visual = clip_model.visual
            proj = visual.proj
            I = torch.eye(proj.shape[0], dtype=proj.dtype, device=proj.device)
            visual.proj = nn.Parameter(I)

    @property
    def patch_size(self):
        return self.inner.visual.patch_size[0]

    @property
    def vision_encoder(self):
        return self.inner.visual

    def forward(self, *args, **kwargs):
        enc = self.inner.visual(*args, **kwargs)

        if isinstance(enc, (tuple, list)):
            token, features = enc
        else:
            token, features = enc, None

        return self._wrap_output(token, features)

    def _wrap_output(self, token, features):
        op = RadioOutput(token, features)

        if self.adaptor_name:
            return {
                'backbone': op,
                self.adaptor_name: op,
            }
        return op

    def encode_image(self, image, normalize: bool = False):
        token, _ = self(image)

        if normalize:
            token = F.normalize(token, dim=-1)

        return token

    def encode_text(self, text, normalize: bool = False):
        try:
            return self.inner.encode_text(text, normalize=normalize)
        except TypeError:
            ret = self.inner.encode_text(text)
            if normalize:
                ret = F.normalize(ret, dim=-1)
            return ret


class SigLIPWrapper(CLIPWrapper):
    def forward(self, *args, **kwargs):
        features = self.inner.visual.trunk.forward_features(*args, **kwargs)
        token = self.inner.visual.trunk.attn_pool(features)
        return self._wrap_output(token, features)


class SAMWrapper(nn.Module):
    def __init__(self, sam_encoder: nn.Module):
        super().__init__()
        self.inner = sam_encoder

    @property
    def embed_dim(self):
        return self.inner.patch_embed.proj.out_channels

    @property
    def patch_size(self):
        return self.inner.patch_embed.proj.kernel_size[0]

    @property
    def vision_encoder(self):
        return self.inner

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.inner.patch_embed(x)
        if self.inner.pos_embed is not None:
            x = x + self.inner.pos_embed

        for blk in self.inner.blocks:
            x = blk(x)

        features = x.flatten(1, 2)

        summary = features.mean(dim=1)

        return summary, features


class InternViTWrapper(nn.Module):
    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.inner = model

        if tokenizer is not None:
            self.tokenizer = lambda texts: tokenizer(texts, return_tensors='pt', max_length=80,
                                                     truncation=True, padding='max_length').input_ids
        else:
            self.tokenizer = None

    @property
    def embed_dim(self):
        return 3200

    @property
    def patch_size(self):
        return self.inner.embeddings.patch_size

    @property
    def vision_encoder(self):
        return self.inner

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is not None:
            y = self.inner.encode_image(x, mode='InternVL-C')
            ret = RadioOutput(y.float(), None)
            return dict(backbone=ret, clip=ret)

        z = self.inner(x)
        y = z.last_hidden_state.float()

        summary = y[:, 0]
        features = y[:, 1:]

        return RadioOutput(summary, features)

    def encode_image(self, image, normalize: bool = False):
        token, _ = self(image)
        token = self.inner.clip_projector(token)

        if normalize:
            token = F.normalize(token, dim=-1)

        return token

    def encode_text(self, text, normalize: bool = False):
        token = self.inner.encode_text(text)

        if normalize:
            token = F.normalize(token, dim=-1)

        return token


class OpenAI_CLIP_VisionAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input_resolution = model.input_resolution
        self.output_dim = model.output_dim
        self.conv1 = model.conv1

        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre

        self.transformer = model.transformer

        self.ln_post = model.ln_post
        self.proj = model.proj

    @property
    def patch_size(self):
        return self.conv1.kernel_size

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        feats = x[:, 1:]

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, feats

@dataclass
class ModelInfo:
    model_class: str
    model_subtype: str


def predict_features_for_frame(model, image_processor, frame):
    with torch.no_grad():
        enc_type = ['backbone', 'clip']
        if len(frame.shape) == 4:  # Batch of images
            batch_size, height, width, channels = frame.shape
            out_height, out_width = height, (width + 15) // 16 * 16

            resized_frames = []
            for i in range(batch_size):
                resized_frame = cv2.resize(frame[i], (out_width, out_height))
                pad_height = out_height - resized_frame.shape[0]
                pad_width = out_width - resized_frame.shape[1]
                padded_frame = np.pad(resized_frame, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
                resized_frames.append(padded_frame)
            frame = np.stack(resized_frames)
        else:  # Single image
            out_height, out_width = frame.shape[0], (frame.shape[1] + 15) // 16 * 16

            frame = cv2.resize(frame, (out_width, out_height))
            pad_height = out_height - frame.shape[0]
            pad_width = out_width - frame.shape[1]
            frame = np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)

        pixel_values = image_processor(images=frame, return_tensors='pt', do_resize=False, do_center_crop=True, 
                                    crop_size={'height': out_height, 'width': out_width}).pixel_values.to(device=next(model.parameters()).device)

        out = model(pixel_values)
        feature_dict = {}
        for k in enc_type:
            features = out[k].features
            patch_size = 16
            n_rows, n_cols = out_height // patch_size, out_width // patch_size
            features = rearrange(features, 'b (h w) c -> b h w c', h=n_rows, w=n_cols).float()
            feature_dict[k] = features

        return feature_dict, torch.squeeze(pixel_values, dim=0)


def load_radio_model(device=None):
    from transformers import AutoModel, CLIPImageProcessor
    hf_repo = "nvidia/RADIO-L"  # For RADIO-L.
    image_processor = CLIPImageProcessor.from_pretrained(hf_repo)

    model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)

    model, preprocessor, info = _load_radio_model(model_version, vitdet_window_size=None, adaptor_names=['clip'],
                                           torchhub_repo="NVlabs/RADIO", use_local_lib=True, device=device)
    if "e-radio" in model_version:
        model.model.set_optimal_window_size((256, 448)) #where it expects a tuple of (height, width) of the input image.

    model = model.cuda()
    return model, image_processor


def _load_radio_model(version: str, adaptor_names: str = None, use_huggingface: bool = False, use_local_lib: bool = True,
               device: torch.device = None, return_spatial_features: bool = True, force_reload: bool = False,
               torchhub_repo="NVlabs/RADIO", **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if os.path.isfile(version) or 'radio' in version:
        if use_huggingface:
            from transformers import AutoModel, AutoConfig
            hf_repo = 'E-RADIO' if 'eradio' in version else 'RADIO'
            hf_repo = f"nvidia/{hf_repo}"
            config = AutoConfig.from_pretrained(
                hf_repo,
                trust_remote_code=True,
                version=version,
                adaptor_names=adaptor_names,
                **kwargs,
            )
            model: nn.Module = AutoModel.from_pretrained(hf_repo, config=config, trust_remote_code=True, **kwargs)
        elif use_local_lib:
            from RADIO.hubconf import radio_model
            model = radio_model(version=version, progress=True, adaptor_names=adaptor_names, **kwargs)
        else:
            model: nn.Module = torch.hub.load(torchhub_repo, 'radio_model', version=version, progress=True,
                                              adaptor_names=adaptor_names, return_spatial_features=return_spatial_features,
                                              force_reload=force_reload, **kwargs,
            )

        preprocessor = model.make_preprocessor_external()
        info = ModelInfo(model_class='RADIO', model_subtype=version.replace('/', '_'))
    elif version.startswith('dinov2'):
        model = torch.hub.load('facebookresearch/dinov2', version, pretrained=True, force_reload=force_reload, **kwargs)
        model = DinoWrapper(model)

        preprocessor = InputConditioner(1.0, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        info = ModelInfo(model_class='DINOv2', model_subtype=version.replace('dinov2_', ''))
    elif version.startswith('open_clip'):
        import open_clip
        _, model_arch, pretrained = version.split(',')
        model = open_clip.create_model(model_arch, pretrained, device=device)
        viz_model = model.visual

        preprocessor = InputConditioner(1.0,
            getattr(viz_model, 'image_mean', open_clip.OPENAI_DATASET_MEAN),
            getattr(viz_model, 'image_std', open_clip.OPENAI_DATASET_STD),
        )

        tokenizer = open_clip.get_tokenizer(model_arch)

        factory = CLIPWrapper
        if model_arch == 'ViT-SO400M-14-SigLIP-384':
            factory = SigLIPWrapper

        model = factory(model, tokenizer, adaptor_names, clip_mode='clip' in adaptor_names if adaptor_names else False)
        info = ModelInfo(model_class='open_clip', model_subtype=pretrained)
    elif version.startswith('openai_clip'):
        import clip as openai_clip

        _, model_name = version.split(',')
        model, preprocess = openai_clip.load(
            model_name,
            device=device,
            jit=False,
        )

        model.visual = OpenAI_CLIP_VisionAdapter(model.visual).to(device)
        norm = preprocess.transforms[-1]
        preprocessor = InputConditioner(
            input_scale=1.0,
            norm_mean=norm.mean,
            norm_std=norm.std,
            dtype=torch.float16,
        )

        model = CLIPWrapper(model, tokenizer=openai_clip.tokenize, adaptor_name=adaptor_names, clip_mode='clip' in adaptor_names if adaptor_names else False)
        info = ModelInfo(model_class='openai_clip', model_subtype=model_name)
    elif version.startswith('sam'):
        from segment_anything.build_sam import sam_model_registry, ImageEncoderViT, Sam
        
        _, chk_path = version.split(',')
        fname = os.path.basename(chk_path)
        prefix = 'sam_vit_'
        assert fname.startswith(prefix) and fname[len(prefix)] in ('h', 'l', 'b'), "Invalid checkpoint file"
        model_name = fname[4:9]
        model = sam_model_registry[model_name](checkpoint=chk_path)

        preprocessor = InputConditioner(
            input_scale=255.0,
            norm_mean=model.pixel_mean,
            norm_std=model.pixel_std,
        )

        img_encoder = model.image_encoder
        model = SAMWrapper(img_encoder)
        info = ModelInfo(model_class='SAM', model_subtype=model_name)
    elif version.startswith('InternV'):
        from transformers import AutoModel, AutoConfig, CLIPImageProcessor, AutoTokenizer

        hfhub_name = f'OpenGVLab/{version}'
        model = AutoModel.from_pretrained(
            hfhub_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)

        if version.startswith('InternVL'):
            tokenizer = AutoTokenizer.from_pretrained(hfhub_name, use_fast=False, add_eos_token=True, trust_remote_code=True)
            tokenizer.pad_token_id = 0
        else:
            tokenizer = None

        preprocessor = CLIPImageProcessor.from_pretrained(hfhub_name)

        preprocessor = InputConditioner(1.0,
            norm_mean=preprocessor.image_mean,
            norm_std=preprocessor.image_std,
            dtype=torch.bfloat16,
        )

        model = InternViTWrapper(model, tokenizer)
        info = ModelInfo(model_class='InternViT', model_subtype=version[10:])
    else:
        raise ValueError(f'Unsupported model version: {version}')

    if device is not None:
        model.to(device=device)

    return model, preprocessor, info

def get_img_emb(input_tensor, main_model):
    if main_model['name'] == "RADIO":
        # Radio
        radio_model, radio_preprocessor = main_model['model'], main_model['preprocessor']
        # Convert init_image from torch tensor to cv2 format
        init_image_cv2 = (input_tensor.permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)
        # Check if init_image_cv2 is a batch of images
        if init_image_cv2.ndim == 4:  # Batch of images
            init_image_cv2 = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in init_image_cv2])
        else:  # Single image
            init_image_cv2 = cv2.cvtColor(init_image_cv2, cv2.COLOR_RGB2BGR)
        init_img_features, _ = predict_features_for_frame(radio_model, radio_preprocessor, init_image_cv2)
        return init_img_features['backbone']
    elif main_model['name'] == "SurgVLP":
        # Surgvlp
        from torchvision import transforms
        surgvlp_model, preprocess = main_model['model'], main_model['preprocessor']

        # Create a new Compose object with the first 2 transforms
        preprocess = transforms.Compose(preprocess.transforms[:2] + [preprocess.transforms[-1]])
        with torch.no_grad():
            image = preprocess((input_tensor+1)/2)
            global_emb, local_emb = surgvlp_model.backbone_img.resnet_forward(image, extract_features=True)

        #image_embeddings = output_dict['img_emb']
        #image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        return local_emb.permute(0,2,3,1)
    elif main_model['name'] == "SurgVLP_global":
        # Surgvlp
        from torchvision import transforms
        surgvlp_model, preprocess = main_model['model'], main_model['preprocessor']

        # Create a new Compose object with the first 2 transforms
        preprocess = transforms.Compose(preprocess.transforms[:2] + [preprocess.transforms[-1]])
        with torch.no_grad():
            image = preprocess((input_tensor+1)/2)    
            output_dict = surgvlp_model(image, mode="video")

        image_embeddings = output_dict['img_emb']
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        return image_embeddings.unsqueeze(1)
    return None