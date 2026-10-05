# NeBLa V6

Verified NeBLa V6 recovery package.

## Verified dataset

- Volume shape: 128 x 256 x 256
- Rays: 256
- Points per ray: 200
- Beta: 0.007450414773728917
- Geometry version: nebla-paper-geometry/2
- Index version: nebla-paper-index/1
- Cases: ToothFairy3F_001 through ToothFairy3F_010

## Generation model

- Encoder features: 64, 128, 256, 512
- Encoder output dimension: 128
- Multiresolution embedding: 7
- MLP depth: 8
- MLP width: 128
- MLP skip: 4
- Refinement feature maps: 64, 128, 256, 512
- Refinement groups: 8
- Chunk pixels: 2048
- Total parameters: 47,517,540

## Training configuration

- Epochs: 300
- Learning rate: 1e-4
- Patience: 30
- Projection loss weight: 10
- Perceptual loss weight: 1
- AMP: enabled

## Repository structure

- `CODE/NeBLa_Paper_Pipeline/` — source implementation
- `DATA/nebla_paper_10/` — verified dataset artifacts
- `V6_MANIFEST_SHA256.txt` — SHA-256 manifest from the recovered package

Runtime caches and generated Python bytecode are excluded from this GitHub-ready package.
