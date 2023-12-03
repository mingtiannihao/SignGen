# adding conditional group-norm as per https://arxiv.org/pdf/2105.05233.pdf

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
import logging

from einops import rearrange, reduce, repeat
from . import layers, layerspp
from .layerspp import Transformer_v2,SpatialTransformer,TemporalTransformer

from .. import get_sigmas
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANppGN
get_act = layers.get_act
default_initializer = layers.default_init


class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()

    self.training = True
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', get_sigmas(config))
    self.is3d = (config.model.arch in ["unetmore3d", "unetmorepseudo3d"])
    self.pseudo3d = (config.model.arch == "unetmorepseudo3d")
    if self.is3d:
      from . import layers3d
    self.use_motion = config.model.use_motion # motion
    self.use_depthmap = config.model.use_depthmap # deepth
    self.use_pose = config.model.use_pose  # pose
    self.concat_dim = concat_dim = config.model.concat_dim
    self.context_dim =  config.model.context_dim

    misc_dropout = 0.5 # mask
    self.p_all_zero = 0.1
    self.p_all_keep = 0.1
    self.beta = 0.1

    self.channels = channels = config.data.channels
    self.num_frames = num_frames = config.data.num_frames
    self.num_frames_cond = num_frames_cond = config.data.num_frames_cond + getattr(config.data, "num_frames_future", 0)
    self.n_frames = num_frames + num_frames_cond
    self.num_frames_pred = config.sampling.num_frames_pred

    self.nf = nf = config.model.ngf*self.n_frames if self.is3d else config.model.ngf # We must prevent problems by multiplying by n_frames
    self.numf = numf = config.model.ngf*self.num_frames if self.is3d else config.model.ngf # We must prevent problems by multiplying by n_frames
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = getattr(config.model, 'dropout', 0.0)
    resamp_with_conv = True
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = getattr(config.model, 'time_conditional', True)  # noise-conditional
    self.cond_emb = getattr(config.model, 'cond_emb', False)
    fir = True
    fir_kernel = [1, 3, 3, 1]
    self.skip_rescale = skip_rescale = True
    self.resblock_type = resblock_type = 'biggan'
    self.embedding_type = embedding_type = 'positional'

    self.pre_image = nn.Sequential(nn.Conv2d(num_frames + concat_dim, num_frames, 1, padding=0))
    init_scale = 0.0
    disabled_sa = False
    assert embedding_type in ['fourier', 'positional']

    modules = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=16
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    temb_dim = None

    if hasattr(config.model, "adapter_transformer_layers"):
        adapter_transformer_layers = config.model.adapter_transformer_layers
    else:
        adapter_transformer_layers = 1

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      temb_dim = nf * 4

      if self.cond_emb:
        modules.append(torch.nn.Embedding(num_embeddings=2, embedding_dim=nf // 2)) # makes it 8 times smaller (16 if ngf=32) since it should be small because there are only two possible values:
        temb_dim += nf // 2

    if self.pseudo3d:
      conv3x3 = functools.partial(layers3d.ddpm_conv3x3_pseudo3d, n_frames=self.n_frames, act=self.act) # Activation here as per https://arxiv.org/abs/1809.04096
      conv3x3_last = functools.partial(layers3d.ddpm_conv3x3_pseudo3d, n_frames=self.num_frames, act=self.act)
    elif self.is3d:
      conv3x3 = functools.partial(layers3d.ddpm_conv3x3_3d, n_frames=self.n_frames)
      conv3x3_last = functools.partial(layers3d.ddpm_conv3x3_3d, n_frames=self.num_frames)
    else:
      conv3x3 = layerspp.conv3x3
      conv3x3_last = layerspp.conv3x3

    if self.is3d:
      AttnBlockDown = functools.partial(layers3d.AttnBlockpp3d,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        n_head_channels=config.model.n_head_channels,
                                        n_frames = self.n_frames,
                                        act=None) # No activation here as per https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L131
      AttnBlockUp = functools.partial(layers3d.AttnBlockpp3d,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      n_head_channels=config.model.n_head_channels,
                                      n_frames = self.num_frames,
                                      act=None) # No activation here as per https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py#L131
    else:
      AttnBlockDown = AttnBlockUp = functools.partial(layerspp.AttnBlockpp,
                                                      init_scale=init_scale,
                                                      skip_rescale=skip_rescale, n_head_channels=config.model.n_head_channels)
      CrossAttnBlock = functools.partial(layerspp.CrossAttnBlock,
                                                      init_scale=init_scale,
                                                      skip_rescale=skip_rescale, n_head_channels=config.model.n_head_channels)
      ImageTextsAttention = functools.partial(layerspp.ImageTextsAttention,
                                                      init_scale=init_scale,
                                                      skip_rescale=skip_rescale, n_head_channels=config.model.n_head_channels)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if resblock_type == 'ddpm':
      ResnetBlockDown = functools.partial(ResnetBlockDDPM,
                                          act=act,
                                          dropout=dropout,
                                          init_scale=init_scale,
                                          skip_rescale=skip_rescale,
                                          temb_dim=temb_dim,
                                          is3d = self.is3d,
                                          n_frames = self.n_frames,
                                          pseudo3d = self.pseudo3d,
                                          act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096
      ResnetBlockUp = functools.partial(ResnetBlockDDPM,
                                        act=act,
                                        dropout=dropout,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        temb_dim=temb_dim,
                                        is3d = self.is3d,
                                        n_frames = self.num_frames,
                                        pseudo3d = self.pseudo3d,
                                        act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096

    elif resblock_type == 'biggan':
      ResnetBlockDown = functools.partial(ResnetBlockBigGAN,
                                          act=act,
                                          dropout=dropout,
                                          fir=fir,
                                          fir_kernel=fir_kernel,
                                          init_scale=init_scale,
                                          skip_rescale=skip_rescale,
                                          temb_dim=temb_dim,
                                          is3d = self.is3d,
                                          n_frames = self.n_frames,
                                          pseudo3d = self.pseudo3d,
                                          act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096
      ResnetBlockUp = functools.partial(ResnetBlockBigGAN,
                                        act=act,
                                        dropout=dropout,
                                        fir=fir,
                                        fir_kernel=fir_kernel,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        temb_dim=temb_dim,
                                        is3d = self.is3d,
                                        n_frames = self.num_frames,
                                        pseudo3d = self.pseudo3d,
                                        act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    self.conv_in = torch.nn.Conv2d(self.channels * self.num_frames + self.num_frames, self.channels * self.num_frames,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    ### Condition Dropout
    # self.misc_dropout = DropPath(misc_dropout)

      ### depth embedding
    if self.use_depthmap:
        self.depth_embedding = nn.Sequential(
            nn.Conv2d(1, concat_dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((128, 128)),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=1, padding=1))

        self.depth_embedding_after = Transformer_v2(heads=2, dim=concat_dim, dim_head_k=concat_dim,
                                                      dim_head_v=concat_dim, dropout_atte=0.05, mlp_dim=concat_dim/2,
                                                      dropout_ffn=0.05, depth=adapter_transformer_layers)
    if self.use_pose:
        self.pose_embedding = nn.Sequential(
            nn.Conv2d(1, concat_dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((128, 128)),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=1, padding=1))

        self.pose_embedding_after = Transformer_v2(heads=2, dim=concat_dim, dim_head_k=concat_dim,
                                                      dim_head_v=concat_dim, dropout_atte=0.05, mlp_dim=concat_dim/2,
                                                      dropout_ffn=0.05, depth=adapter_transformer_layers)


    if  self.use_motion:
        self.motion_embedding = nn.Sequential(
            nn.Conv2d(2, concat_dim * 4, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((128, 128)),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=1, padding=1))
        self.motion_embedding_after = Transformer_v2(heads=2, dim=concat_dim, dim_head_k=concat_dim,
                                                     dim_head_v=concat_dim, dropout_atte=0.05, mlp_dim=concat_dim/2,
                                                     dropout_ffn=0.05, depth=adapter_transformer_layers)

    self.pre_image = nn.Sequential(nn.Conv2d(channels + concat_dim, channels, 1, padding=0))


    # Downsampling block
    modules.append(conv3x3(channels * self.n_frames, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlockDown(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlockDown(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlockDown(down=True, in_ch=in_ch))

        hs_c.append(in_ch)

    # Middle Block
    in_ch = hs_c[-1]
    modules.append(ResnetBlockDown(in_ch=in_ch))
    modules.append(AttnBlockDown(channels=in_ch))
    modules.append(ImageTextsAttention(channels=in_ch))
    if self.is3d:
      # Converter
      modules.append(layerspp.conv1x1(self.n_frames, self.num_frames))
      in_ch =  int(in_ch * self.num_frames / self.n_frames)
    modules.append(ResnetBlockUp(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = numf * ch_mult[i_level]
        if self.is3d: # 1x1 self.num_frames + self.num_frames_cond -> self.num_frames
          modules.append(layerspp.conv1x1(self.n_frames, self.num_frames))
          in_ch_old = int(hs_c.pop() * self.num_frames / self.n_frames)
        else:
          in_ch_old = hs_c.pop()
        modules.append(ResnetBlockUp(in_ch=in_ch + in_ch_old,
                                     out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlockUp(channels=in_ch))
        modules.append(ImageTextsAttention(channels=in_ch))

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlockUp(in_ch=in_ch, up=True))

    assert not hs_c

    modules.append(layerspp.get_act_norm(act=act, act_emb=act, norm='group', ch=in_ch, is3d=self.is3d, n_frames=self.num_frames))
    modules.append(conv3x3_last(in_ch, channels*self.num_frames, init_scale=init_scale))
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond, cond=None, cond_mask=None,texts =None,deepth_video = None,motion = None,is_video_gen = None,pose_video = None):


    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0
    beta = self.beta
    self.batch = x.shape[0]

    x_0 = x
    x = rearrange(x, 'b (f c) h w -> b f c h w ', c=3)
    batch, f, c, h, w = x.shape
    x = rearrange(x, ' b f c h w -> b c f h w ')

    # all-zero and all-keep masks
    zero = torch.zeros(batch, dtype=torch.bool).to(x.device)
    keep = torch.zeros(batch, dtype=torch.bool).to(x.device)
    if self.training:
        nzero = (torch.rand(batch) < self.p_all_zero).sum()
        nkeep = (torch.rand(batch) < self.p_all_keep).sum()
        index = torch.randperm(batch)
        zero[index[0:nzero]] = True
        keep[index[nzero:nzero + nkeep]] = True
    assert not (zero & keep).any()
    # misc_dropout = functools.partial(self.misc_dropout, zero=zero, keep=keep)

    concat = x.new_zeros(batch, self.concat_dim/2, f, h, w)
    if deepth_video is not None:
        if cond is not None:
            n = self.config.data.num_frames_cond
            dv = deepth_video[:, n:n+self.num_frames,:,:].cpu().numpy()
            deepth_video = torch.tensor(dv).to(torch.float32).to(self.config.device)

            deepth_video = deepth_video.unsqueeze(2) # b f c h w
            # print(deepth_video.shape, 'd_o')
            deepth_video = rearrange(deepth_video, 'b f c h w -> (b f) c h w')
            # print(deepth_video.shape,'d')
            deepth_video = self.depth_embedding(deepth_video)
            h = deepth_video.shape[2]
            depth = self.depth_embedding_after(rearrange(deepth_video, '(b f) c h w -> (b h w) f c', b=batch))

            #
            depth = rearrange(depth, '(b h w) f c -> b c f h w', b=batch, h=h)
            concat = concat + depth
    if pose_video is not None:
        if cond is not None:
            n = self.config.data.num_frames_cond
            dv = deepth_video[:, n:n+self.num_frames,:,:].cpu().numpy()
            pose_video = torch.tensor(dv).to(torch.float32).to(self.config.device)

            pose_video = pose_video.unsqueeze(2) # b f c h w
            pose_video = rearrange(pose_video, 'b f c h w -> (b f) c h w')
            pose_video = self.depth_embedding(pose_video)
            h = pose_video.shape[2]
            pose_video = self.pose_embedding_after(rearrange(pose_video, '(b f) c h w -> (b h w) f c', b=batch))

            #
            pose_video = rearrange(pose_video, '(b h w) f c -> b c f h w', b=batch, h=h)
            concat = concat + pose_video

    if motion is not None:
        n = self.config.data.num_frames_cond
        dv = motion[:, n:n+self.num_frames, :, :].cpu().numpy()
        motion = torch.tensor(dv).to(torch.float32).to(self.config.device)
        motion = rearrange(motion, 'b f c h w -> (b f) c h w')
        motion = self.motion_embedding(motion.to(torch.float32))
        h = motion.shape[2]
        motion = self.motion_embedding_after(rearrange(motion, '(b f) c h w -> (b h w) f c', b = batch))
        motion = rearrange(motion, '(b h w) f c -> b c f h w', b = batch, h = h)
        concat =torch.cat([concat,motion])



    x = torch.cat([x, concat], dim=1)
    x = rearrange(x, 'b c f h w -> (b f) c h w')
    x = self.pre_image(x)
    x = rearrange(x, '(b f) c h w -> b c f h w', b=batch)

    x = rearrange(x, 'b c f h w -> b (f c) h w')
    x = x_0+beta*x
    if cond is not None:
        x = torch.cat([x, cond], dim=1) # B, (num_frames+num_frames_cond)*C, H, W


    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1
    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
      used_sigmas = self.sigmas[time_cond.long()]
      temb = layers.get_timestep_embedding(timesteps, self.nf)
    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb)) # b x k
      m_idx += 1
      if self.cond_emb:
        if cond_mask is None:
          cond_mask = torch.ones(x.shape[0], device=x.device, dtype=torch.int32)
        temb = torch.cat([temb, modules[m_idx](cond_mask)], dim=1) # b x (k/8 + k)
        m_idx += 1
    else:
      temb = None

    # Downsampling block
    input_pyramid = None

    x = x.contiguous()

    hs = [modules[m_idx](x)]
    m_idx += 1

    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
            h = modules[m_idx](h)
            m_idx += 1
            # h = rearrange(h, 'b (f c) h w -> b c f h w ', c=3)
            # h = rearrange(h, 'b c f h w -> (b f) c h w')
            # h = modules[m_idx](h)
            # m_idx += 1
            # h = rearrange(h, '(b f) c h w -> b c f h w', b=self.batch)
            # h = modules[m_idx](h)
            # h = rearrange(h, 'b c f h w -> b (f c) h w')
            # m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb)
          m_idx += 1

        hs.append(h)

    # Middle Block
    # ResBlock
    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    # AttnBlock
    h = modules[m_idx](h)
    m_idx += 1

    # h = rearrange(h, 'b (f c) h w -> b c f h w ', c=3)
    # h = rearrange(h, 'b c f h w -> (b f) c h w')
    # h = modules[m_idx](h)
    # m_idx += 1
    # h = rearrange(h, '(b f) c h w -> b c f h w', b=self.batch)
    # h = modules[m_idx](h)
    # h = rearrange(h, 'b c f h w -> b (f c) h w')
    # m_idx += 1


    c = texts.to(torch.float32)
    h_pre = h
    h = h_pre + modules[m_idx](h,c)*beta
    m_idx += 1

    # Converter
    if self.is3d: # downscale time-dim, we decided to do it here, but could also have been done earlier or at the end
      # B, C*(num_frames+num_cond), H, W -> B, C, (num_frames+num_cond), H, W -----conv1x1-----> B, C, num_frames, H, W -> B, C*num_frames, H, W
      B, CN, H, W = h.shape
      h = h.reshape(-1, self.n_frames, H, W)
      h = modules[m_idx](h)
      m_idx += 1
      h = h.reshape(B, -1, H, W)

    # ResBlock
    h = modules[m_idx](h, temb)
    m_idx += 1

    pyramid = None
    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        if self.is3d:
          # Get h and h_olda
          B, CN, H, W = h.shape
          h = h.reshape(B, -1, self.num_frames, H, W)
          prev = hs.pop().reshape(-1, self.n_frames, H, W)
          # B, C*Nhs, H, W -> B, C, Nhs, H, W -----conv1x1-----> B, C, Nh, H, W -> B, C*Nh, H, W
          prev = modules[m_idx](prev).reshape(B, -1, self.num_frames, H, W)
          m_idx += 1
          # Concatenate
          h_comb = torch.cat([h, prev], dim=1) # B, C, N, H, W
          h_comb = h_comb.reshape(B, -1, H, W)
        else:
          prev = hs.pop()
          h_comb = torch.cat([h, prev], dim=1)
          # 2 384 128 128
        h = modules[m_idx](h_comb, temb)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h,c)
        m_idx +=1

        # h = rearrange(h, 'b (f c) h w -> b c f h w ', c=3)
        # h = rearrange(h, 'b c f h w -> (b f) c h w')
        # h = modules[m_idx](h)
        # m_idx += 1
        # h = rearrange(h, '(b f) c h w -> b c f h w', b=self.batch)
        # h = rearrange(h, 'b (f c) h w -> b c f h w ', c=3)
        # h = modules[m_idx](h)
        # h = rearrange(h, 'b c f h w -> b (f c) h w')
        # m_idx += 1

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb)
          m_idx += 1

    assert not hs
    # GroupNorm
    h = modules[m_idx](h)
    m_idx += 1

    # conv3x3_last
    h = modules[m_idx](h)
    m_idx += 1

    assert m_idx == len(modules)

    if getattr(self.config.model, 'output_all_frames', False) and cond is not None: # we only keep the non-cond images (but we could use them eventually)
      _, h = torch.split(h, [self.num_frames_cond*self.config.data.channels,self.num_frames*self.config.data.channels], dim=1)

    if self.is3d: # B, C*N, H, W -> B, N*C, H, W subtle but important difference!
      B, CN, H, W = h.shape
      NC = CN
      h = h.reshape(B, self.channels, self.num_frames, H, W).permute(0, 2, 1, 3, 4).reshape(B, NC, H, W)

    return h

class UNetMore_DDPM(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.version = getattr(config.model, 'version', 'DDPM').upper()
    assert self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM", f"models/unet : version is not DDPM or DDIM! Given: {self.version}"

    self.config = config

    if getattr(config.model, 'spade', False):
      self.unet = SPADE_NCSNpp(config)
    else:
      self.unet = NCSNpp(config)

    self.schedule = getattr(config.model, 'sigma_dist', 'linear')
    if self.schedule == 'linear':
      self.register_buffer('betas', get_sigmas(config))
      self.register_buffer('alphas', torch.cumprod(1 - self.betas.flip(0), 0).flip(0))
      self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
    elif self.schedule == 'cosine':
      self.register_buffer('alphas', get_sigmas(config))
      self.register_buffer('alphas_prev', torch.cat([self.alphas[1:], torch.tensor([1.0]).to(self.alphas)]))
      self.register_buffer('betas', 1 - self.alphas/self.alphas_prev)
    self.gamma = getattr(config.model, 'gamma', False)
    if self.gamma:
        self.theta_0 = 0.001
        self.register_buffer('k', self.betas/(self.alphas*(self.theta_0 ** 2))) # large to small, doesn't match paper, match code instead
        self.register_buffer('k_cum', torch.cumsum(self.k.flip(0), 0).flip(0)) # flip for small-to-large, then flip back
        self.register_buffer('theta_t', torch.sqrt(self.alphas)*self.theta_0)

    self.noise_in_cond = getattr(config.model, 'noise_in_cond', False)

  def forward(self, x, y, cond=None, cond_mask=None,texts = None,deepth_video =None,motion = None,is_video_gen=None,pose_video = pose_video):

    if self.noise_in_cond and cond is not None: # We add noise to cond
      alphas = self.alphas
      # if labels is None:
      #     labels = torch.randint(0, len(alphas), (cond.shape[0],), device=cond.device)
      labels = y
      used_alphas = alphas[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:])))
      if self.gamma:
        used_k = self.k_cum[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
        used_theta = self.theta_t[labels].reshape(cond.shape[0], *([1] * len(cond.shape[1:]))).repeat(1, *cond.shape[1:])
        z = torch.distributions.gamma.Gamma(used_k, 1 / used_theta).sample()
        z = (z - used_k*used_theta)/(1 - used_alphas).sqrt()
      else:
        z = torch.randn_like(cond)
      cond = used_alphas.sqrt() * cond + (1 - used_alphas).sqrt() * z



    return self.unet(x, y, cond, cond_mask=cond_mask,texts=texts,deepth_video=deepth_video,motion = motion,is_video_gen = is_video_gen,pose_video = pose_video)


def p_mean_variance(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None):
    r"""Distribution of p(x_{t-1} | x_t).
    """
    # predict distribution
    if guide_scale is None:
        out = model(xt, self._scale_timesteps(t), **model_kwargs)
    else:
        # classifier-free guidance
        # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
        assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
        y_out = model(xt, self._scale_timesteps(t), **model_kwargs[0])
        u_out = model(xt, self._scale_timesteps(t), **model_kwargs[1])
        dim = y_out.size(1) if self.var_type.startswith('fixed') else y_out.size(1) // 2
        out = torch.cat([
            u_out[:, :dim] + guide_scale * (y_out[:, :dim] - u_out[:, :dim]),
            y_out[:, dim:]], dim=1)  # guide_scale=9.0

    # compute variance
    if self.var_type == 'learned':
        out, log_var = out.chunk(2, dim=1)
        var = torch.exp(log_var)
    elif self.var_type == 'learned_range':
        out, fraction = out.chunk(2, dim=1)
        min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
        max_log_var = _i(torch.log(self.betas), t, xt)
        fraction = (fraction + 1) / 2.0
        log_var = fraction * max_log_var + (1 - fraction) * min_log_var
        var = torch.exp(log_var)
    elif self.var_type == 'fixed_large':
        var = _i(torch.cat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
        log_var = torch.log(var)
    elif self.var_type == 'fixed_small':
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)

    # compute mean and x0
    if self.mean_type == 'x_{t-1}':
        mu = out  # x_{t-1}
        x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
             _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
    elif self.mean_type == 'x0':
        x0 = out
        mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
    elif self.mean_type == 'eps':
        x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
             _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
        mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)

    # restrict the range of x0
    if percentile is not None:
        assert percentile > 0 and percentile <= 1  # e.g., 0.995
        s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1).clamp_(1.0).view(-1, 1, 1, 1)
        x0 = torch.min(s, torch.max(-s, x0)) / s
    elif clamp is not None:
        x0 = x0.clamp(-clamp, clamp)
    return mu, var, log_var, x0