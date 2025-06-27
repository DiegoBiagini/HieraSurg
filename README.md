# HieraSurg
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-B31B1B.svg)](https://arxiv.org/abs/2506.21207)  
Early Accepted at MICCAI 2025

HieraSurg is a video diffusion model that is able to generate realistic .
It achieves this by decoupling the generation process in two semantic levels, first Hierasurg-Semantic2Map generates the evolution of a surgical scene in panoptic-segmentation-space, given surgical information like phase and interaction triplet.
Once a temporal set of segmentation maps is available HieraSurg-Map2Vid is able to bring them to video space to visualize the actual evolution of the surgical scene.

This repository contains the code used to train the VDMs as well as the procedure to obtain panoptic segmentation maps from Cholec80.


## Table of Contents
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Dataset](#dataset)  
- [File Structure](#file-structure)  
- [Citation](#citation)  
- [License](#license)  

## Features

- Labeling procedure for CholecT45/Cholec80
- Training code for HieraSurg (S2M and M2V)
- Inference with/without GT segmentation maps

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/USERNAME/REPOSITORY_NAME.git
   cd REPOSITORY_NAME
   ```
2. Create a virtual environment (optional but recommended)  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install required packages  
   ```bash
   pip install -r requirements.txt
   ```  

## Usage

### Training

```bash
python scripts/train.py \
  --config configs/train.yaml \
  --data_dir /path/to/data \
  --output_dir /path/to/output
```


### Inference

```bash
python scripts/infer.py \
  --model /path/to/best_model.pth \
  --input sample_input.txt \
  --output sample_output.txt
```

## Dataset

All the data used is from Cholec80 TODO link and CholecT45 TODO.
Refer to the repositories to download them.

### Automatic Labeling Pipeline


Not all videos of Cholec80 were processed and used, data splits can be found in txt 


## File Structure

```text
REPOSITORY_NAME/
├── configs/              # YAML configuration files
├── data/                 # Dataset download scripts or samples
├── docs/                 # Paper PDF and supplementary materials
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Training, evaluation, inference scripts
├── src/                  # Source code modules
│   ├── __init__.py
│   └── main.py
├── tests/                # Unit and integration tests
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Citation
If you find this work useful, please cite our paper:
```bibtex
@misc{biagini2025hierasurghierarchyawarediffusionmodel,
      title={HieraSurg: Hierarchy-Aware Diffusion Model for Surgical Video Generation}, 
      author={Diego Biagini and Nassir Navab and Azade Farshad},
      year={2025},
      eprint={2506.21287},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.21287}, 
}
```
## License

This code may be used for **non-commercial scientific research purposes** as defined by [Creative Commons 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). 