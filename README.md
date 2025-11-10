# ParaMorph
ParaMorph: Enhancing Unsupervised Deformable Image Registration via Transformer-ConvNet Fusion

ParaMorph is an unsupervised volumetric medical image registration network that integrates a parallel Transformer-ConvNet architecture for multi-scale feature fusion, enhancing both local feature extraction and global contextual understanding.

🖥️ Experimental Setup
Hardware Configuration
All experiments were conducted on a high-performance computing workstation equipped with an NVIDIA GeForce RTX 3080 Laptop GPU (16GB VRAM), an Intel Core i7-11800H processor, and 32GB of RAM, ensuring efficient handling of large-scale 3D medical image data.

Software Environment
Our implementation was developed on Ubuntu 20.04/Windows 11 using Python 3.8+ and PyTorch 2.0+ with CUDA 11.8. Key dependencies include torch, torchvision, nibabel, numpy, scikit-image, and opencv-python.

Code Implementation
Train.py serves as the main entry point for unsupervised end-to-end model training, handling loss function optimization and logging. Infer.py is used to load trained models for image registration inference, generating registered images and deformation fields while computing evaluation metrics such as Dice and ASSD.

🏗️Method Overview
The core of our proposed ParaMorph is a parallel Transformer-ConvNet architecture designed for multi-scale feature fusion. The following diagram illustrates the overall framework:

As illustrated in the diagram above, our framework consists of three key components:
![Fig1(1)](https://github.com/user-attachments/assets/c27ec28b-41b2-49fc-981c-4a85c78dbb50)

Parallel Encoder: The input image pair is processed simultaneously by two branches:

A Transformer Branch based on 3D Swin Transformer blocks to capture long-range spatial dependencies and global contextual information.

A ConvNet Branch built with our enhanced residual Transformer-style convblocks (NeXt-B and NeXt-D) for extracting hierarchical local features.

Multi-Scale Feature Fusion (PMFE): Feature maps from both branches are fused at multiple scales. This allows the model to integrate fine-grained local details with a comprehensive global understanding, which is crucial for accurately predicting large and complex deformations.

Decoder with Gap Filling: The fused multi-scale features are passed to a ConvNet-based decoder. Enhanced convblocks are also employed in the long skip connections for "gap filling," ensuring that high-resolution spatial information is effectively propagated to the final output, resulting in a dense, high-quality deformation field.

📊 Datasets
The model was trained and evaluated on three volumetric MRI datasets:

HPH-PM (Internal Prostate): 110 T2-weighted scans (100 train, 10 test) with expert-annotated prostate masks, characterized by large deformations.

Prostate158 (Public Prostate): 158 multi-parametric MRI scans (80/20 split) used to validate generalization on public data.

LPBA40 (Public Brain): 40 brain MRI scans (30 train, 10 test) used to assess cross-organ generalization.

🔄 Data Preprocessing
All volumetric data underwent a consistent preprocessing pipeline: images were first affinely aligned, then resampled to a uniform voxel spacing and resized to a fixed dimension of 160×160×96. Intensity values were normalized to the [0, 1] range using min-max normalization. During training, data augmentation including random rotation, scaling, elastic deformation, and intensity perturbation was applied to improve model robustness.



