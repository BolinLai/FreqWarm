# Toward Diffusible High-Dimensional Latent Spaces: A Frequency Perspective

### CVPR 2026

### [Project Page](https://bolinlai.github.io/projects/FreqWarm/) | [Paper](https://arxiv.org/pdf/2511.22249)

#### [Bolin Lai](https://bolinlai.github.io/), [Xudong Wang](https://people.eecs.berkeley.edu/~xdwang/), [Saketh Rambhatla](https://rssaketh.github.io/), [James M. Rehg](https://rehg.org/), [Zsolt Kira](https://faculty.cc.gatech.edu/~zk15/), [Rohit Girdhar](https://rohitgirdhar.github.io/), [Ishan Misra](https://imisra.github.io/)

<p align="center">
  <img src="https://bolinlai.github.io/projects/FreqWarm/figures/method.png" height="300"/>
  <img src="https://bolinlai.github.io/projects/FreqWarm/figures/teaser.png" height="300"/>
</p>

This repository provides the core code snippet for our FreqWarm method. The full codebase cannot be released due to legal restrictions, but this snippet demonstrates the key idea.


## Usage

```python
import torch
from frequency_filter import apply_frequency_filter

# Low-pass filter the images before encoding
filtered_images = apply_frequency_filter(
    images,                # (B, C, H, W), normalized to [-1, 1]
    keep="low",            # keep low frequencies, discard high
    cutoff=0.20,           # normalized cutoff frequency
    filter_type="ideal",   # "ideal", "butterworth", or "gaussian"
)

# Encode filtered images into latent space
latents = encoder(filtered_images)  # (B, latent_C, H/f, W/f)
```

## Running the Demo

```bash
python frequency_filter.py
```

## BibTex

If you find our paper helpful to your work, please cite with this BibTex.

```BibTex
@article{lai2025toward,
    title={Toward Diffusible High-Dimensional Latent Spaces: A Frequency Perspective},
    author={Lai, Bolin and Wang, Xudong and Rambhatla, Saketh and Rehg, James M and Kira, Zsolt and Girdhar, Rohit and Misra, Ishan},
    journal={arXiv preprint arXiv:2511.22249},
    year={2025}}
```
