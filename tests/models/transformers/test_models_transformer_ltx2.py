# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import unittest

import torch

from diffusers import LTX2VideoTransformer3DModel

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import ModelTesterMixin, TorchCompileTesterMixin


enable_full_determinism()


class LTX2TransformerTests(ModelTesterMixin, unittest.TestCase):
    model_class = LTX2VideoTransformer3DModel
    main_input_name = "hidden_states"
    uses_custom_attn_processor = True

    @property
    def dummy_input(self):
        # Common
        batch_size = 2

        # Video
        num_frames = 2
        num_channels = 4
        height = 16
        width = 16

        # Audio
        audio_num_frames = 9
        audio_num_channels = 2
        num_mel_bins = 2

        # Text
        embedding_dim = 16
        sequence_length = 16

        hidden_states = torch.randn((batch_size, num_frames * height * width, num_channels)).to(torch_device)
        audio_hidden_states = torch.randn((batch_size, audio_num_frames, audio_num_channels * num_mel_bins)).to(
            torch_device
        )
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        audio_encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        encoder_attention_mask = torch.ones((batch_size, sequence_length)).bool().to(torch_device)
        timestep = torch.rand((batch_size,)).to(torch_device) * 1000

        return {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "audio_num_frames": audio_num_frames,
            "fps": 25.0,
        }

    @property
    def input_shape(self):
        return (512, 4)

    @property
    def output_shape(self):
        return (512, 4)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 4,
            "out_channels": 4,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "audio_cross_attention_dim": 8,
            "num_layers": 2,
            "qk_norm": "rms_norm_across_heads",
            "caption_channels": 16,
            "rope_double_precision": False,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"LTX2VideoTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)

    # def test_ltx2_consistency(self, seed=0, dtype=torch.float32):
    #     torch.manual_seed(seed)
    #     init_dict, _ = self.prepare_init_args_and_inputs_for_common()

    #     # Calculate dummy inputs in a custom manner to ensure compatibility with original code
    #     batch_size = 2
    #     num_frames = 9
    #     latent_frames = 2
    #     text_embedding_dim = 16
    #     text_seq_len = 16
    #     fps = 25.0
    #     sampling_rate = 16000.0
    #     hop_length = 160.0

    #     sigma = torch.rand((1,), generator=torch.manual_seed(seed), dtype=dtype, device="cpu") * 1000
    #     timestep = (sigma * torch.ones((batch_size,), dtype=dtype, device="cpu")).to(device=torch_device)

    #     num_channels = 4
    #     latent_height = 4
    #     latent_width = 4
    #     hidden_states = torch.randn(
    #         (batch_size, num_channels, latent_frames, latent_height, latent_width),
    #         generator=torch.manual_seed(seed),
    #         dtype=dtype,
    #         device="cpu",
    #     )
    #     # Patchify video latents (with patch_size (1, 1, 1))
    #     hidden_states = hidden_states.reshape(batch_size, -1, latent_frames, 1, latent_height, 1, latent_width, 1)
    #     hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    #     encoder_hidden_states = torch.randn(
    #         (batch_size, text_seq_len, text_embedding_dim),
    #         generator=torch.manual_seed(seed),
    #         dtype=dtype,
    #         device="cpu",
    #     )

    #     audio_num_channels = 2
    #     num_mel_bins = 2
    #     latent_length = int((sampling_rate / hop_length / 4) * (num_frames / fps))
    #     audio_hidden_states = torch.randn(
    #         (batch_size, audio_num_channels, latent_length, num_mel_bins),
    #         generator=torch.manual_seed(seed),
    #         dtype=dtype,
    #         device="cpu",
    #     )
    #     # Patchify audio latents
    #     audio_hidden_states = audio_hidden_states.transpose(1, 2).flatten(2, 3)
    #     audio_encoder_hidden_states = torch.randn(
    #         (batch_size, text_seq_len, text_embedding_dim),
    #         generator=torch.manual_seed(seed),
    #         dtype=dtype,
    #         device="cpu",
    #     )

    #     inputs_dict = {
    #         "hidden_states": hidden_states.to(device=torch_device),
    #         "audio_hidden_states": audio_hidden_states.to(device=torch_device),
    #         "encoder_hidden_states": encoder_hidden_states.to(device=torch_device),
    #         "audio_encoder_hidden_states": audio_encoder_hidden_states.to(device=torch_device),
    #         "timestep": timestep,
    #         "num_frames": latent_frames,
    #         "height": latent_height,
    #         "width": latent_width,
    #         "audio_num_frames": num_frames,
    #         "fps": 25.0,
    #     }

    #     model = self.model_class.from_pretrained(
    #         "diffusers-internal-dev/dummy-ltx2",
    #         subfolder="transformer",
    #         device_map="cpu",
    #     )
    #     # torch.manual_seed(seed)
    #     # model = self.model_class(**init_dict)
    #     model.to(torch_device)
    #     model.eval()

    #     with attention_backend("native"):
    #         with torch.no_grad():
    #             output = model(**inputs_dict)

    #             video_output, audio_output = output.to_tuple()

    #     self.assertIsNotNone(video_output)
    #     self.assertIsNotNone(audio_output)

    #     # input & output have to have the same shape
    #     video_expected_shape = (batch_size, latent_frames * latent_height * latent_width, num_channels)
    #     self.assertEqual(video_output.shape, video_expected_shape, "Video input and output shapes do not match")
    #     audio_expected_shape = (batch_size, latent_length, audio_num_channels * num_mel_bins)
    #     self.assertEqual(audio_output.shape, audio_expected_shape, "Audio input and output shapes do not match")

    #     # Check against expected slice
    #     # fmt: off
    #     video_expected_slice = torch.tensor([0.4783, 1.6954, -1.2092, 0.1762, 0.7801, 1.2025, -1.4525, -0.2721, 0.3354, 1.9144, -1.5546, 0.0831, 0.4391, 1.7012, -1.7373, -0.2676])
    #     audio_expected_slice = torch.tensor([-0.4236, 0.4750, 0.3901, -0.4339, -0.2782, 0.4357, 0.4526, -0.3927, -0.0980, 0.4870, 0.3964, -0.3169, -0.3974, 0.4408, 0.3809, -0.4692])
    #     # fmt: on

    #     video_output_flat = video_output.cpu().flatten().float()
    #     video_generated_slice = torch.cat([video_output_flat[:8], video_output_flat[-8:]])
    #     self.assertTrue(torch.allclose(video_generated_slice, video_expected_slice, atol=1e-4))

    #     audio_output_flat = audio_output.cpu().flatten().float()
    #     audio_generated_slice = torch.cat([audio_output_flat[:8], audio_output_flat[-8:]])
    #     self.assertTrue(torch.allclose(audio_generated_slice, audio_expected_slice, atol=1e-4))


class LTX2VideoOnlyTransformerTests(unittest.TestCase):
    """Tests for LTX2VideoTransformer3DModel in video-only mode (with_audio=False)."""

    def _get_init_dict(self):
        return {
            "in_channels": 4,
            "out_channels": 4,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "audio_cross_attention_dim": 8,
            "num_layers": 2,
            "qk_norm": "rms_norm_across_heads",
            "caption_channels": 16,
            "rope_double_precision": False,
            "with_audio": False,
        }

    def _get_video_inputs(
        self, batch_size=2, num_frames=2, height=16, width=16, num_channels=4, seq_len=16, emb_dim=16
    ):
        hidden_states = torch.randn((batch_size, num_frames * height * width, num_channels)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, seq_len, emb_dim)).to(torch_device)
        encoder_attention_mask = torch.ones((batch_size, seq_len)).bool().to(torch_device)
        timestep = torch.rand((batch_size,)).to(torch_device) * 1000
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": 25.0,
        }

    def test_video_only_no_audio_parameters(self):
        """When with_audio=False, no parameters should have 'audio_' in their name."""
        model = LTX2VideoTransformer3DModel(**self._get_init_dict()).to(torch_device)
        audio_params = [name for name, _ in model.named_parameters() if "audio_" in name]
        self.assertEqual(
            audio_params,
            [],
            f"Expected no audio parameters, but found: {audio_params}",
        )

    def test_video_only_no_audio_attributes(self):
        """When with_audio=False, audio-specific module attributes should not exist."""
        model = LTX2VideoTransformer3DModel(**self._get_init_dict()).to(torch_device)
        audio_attrs = [
            "audio_proj_in",
            "audio_caption_projection",
            "audio_time_embed",
            "audio_rope",
            "cross_attn_rope",
            "cross_attn_audio_rope",
            "audio_norm_out",
            "audio_proj_out",
            "audio_scale_shift_table",
            "av_cross_attn_video_scale_shift",
            "av_cross_attn_audio_scale_shift",
            "av_cross_attn_video_a2v_gate",
            "av_cross_attn_audio_v2a_gate",
        ]
        for attr in audio_attrs:
            self.assertFalse(
                hasattr(model, attr),
                f"Expected model to NOT have attribute '{attr}' in video-only mode, but it does.",
            )

    def test_video_only_forward_output_shape(self):
        """forward() with only video inputs should return the correct output shape."""
        init_dict = self._get_init_dict()
        num_channels = init_dict["in_channels"]
        model = LTX2VideoTransformer3DModel(**init_dict).to(torch_device)
        model.eval()

        batch_size = 2
        num_frames, height, width = 2, 16, 16
        inputs = self._get_video_inputs(
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            num_channels=num_channels,
        )
        with torch.no_grad():
            output = model(**inputs)

        expected_shape = (batch_size, num_frames * height * width, num_channels)
        self.assertEqual(output.sample.shape, expected_shape)
        self.assertIsNone(output.audio_sample)

    def test_video_only_forward_tuple_output(self):
        """forward() with return_dict=False should return a tuple with only the video output."""
        init_dict = self._get_init_dict()
        num_channels = init_dict["in_channels"]
        model = LTX2VideoTransformer3DModel(**init_dict).to(torch_device)
        model.eval()

        num_frames, height, width = 2, 16, 16
        inputs = self._get_video_inputs(num_frames=num_frames, height=height, width=width, num_channels=num_channels)
        inputs["return_dict"] = False
        with torch.no_grad():
            output = model(**inputs)

        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 1)

    def test_with_audio_true_unchanged(self):
        """with_audio=True (default) should still work and produce both video and audio outputs."""
        init_dict = self._get_init_dict()
        init_dict["with_audio"] = True
        num_channels = init_dict["in_channels"]
        model = LTX2VideoTransformer3DModel(**init_dict).to(torch_device)
        model.eval()

        batch_size = 2
        num_frames, height, width = 2, 16, 16
        audio_num_frames = 9
        audio_num_channels, num_mel_bins = 2, 2
        seq_len, emb_dim = 16, 16

        inputs = {
            "hidden_states": torch.randn((batch_size, num_frames * height * width, num_channels)).to(torch_device),
            "audio_hidden_states": torch.randn(
                (batch_size, audio_num_frames, audio_num_channels * num_mel_bins)
            ).to(torch_device),
            "encoder_hidden_states": torch.randn((batch_size, seq_len, emb_dim)).to(torch_device),
            "audio_encoder_hidden_states": torch.randn((batch_size, seq_len, emb_dim)).to(torch_device),
            "timestep": torch.rand((batch_size,)).to(torch_device) * 1000,
            "encoder_attention_mask": torch.ones((batch_size, seq_len)).bool().to(torch_device),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "audio_num_frames": audio_num_frames,
            "fps": 25.0,
        }
        with torch.no_grad():
            output = model(**inputs)

        self.assertIsNotNone(output.sample)
        self.assertIsNotNone(output.audio_sample)
        expected_video_shape = (batch_size, num_frames * height * width, num_channels)
        self.assertEqual(output.sample.shape, expected_video_shape)

    def test_video_only_config_registered(self):
        """with_audio flag should be stored in the model config."""
        model = LTX2VideoTransformer3DModel(**self._get_init_dict())
        self.assertFalse(model.config.with_audio)

        av_init_dict = self._get_init_dict()
        av_init_dict["with_audio"] = True
        av_model = LTX2VideoTransformer3DModel(**av_init_dict)
        self.assertTrue(av_model.config.with_audio)


class LTX2TransformerCompileTests(TorchCompileTesterMixin, unittest.TestCase):
    model_class = LTX2VideoTransformer3DModel

    def prepare_init_args_and_inputs_for_common(self):
        return LTX2TransformerTests().prepare_init_args_and_inputs_for_common()
