# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'DiT-XL-2-256x256.pt'} 


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, 'pretrained_models')

    from diffusers import AutoencoderKL
    vae_dir = f"pretrained_models/sd-vae-ft-ema"
    if not os.path.exists(os.path.join(vae_dir, "config.json")):
        try:
            print(f">>> Downloading VAE model: stabilityai/sd-vae-ft-ema")
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
            os.makedirs(vae_dir, exist_ok=True)
            vae.save_pretrained(vae_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download VAE model 'stabilityai/sd-vae-ft-ema'. "
                f"Please pre-download manually if running on a cluster. Error: {e}"
            )
    else:
        print(f">>> VAE model already exists at: {vae_dir}")

    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    # Download all DiT checkpoints
    for model in pretrained_models:
        download_model(model)
    print('Done.')
