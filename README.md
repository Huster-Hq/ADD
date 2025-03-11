<div align="center">
<h2>Holistic White-light Polyp Classification via Alignment-free Dense Distillation of Auxiliary Optical Chromoendoscopy</h2>
</div>

## 1. Overview
This work implements a novel framework for **enhancing holistic WLI polyp classification** through cross-domain **(NBI->WLI)** knowledge distillation, **without requiring any additinal labeling cost**. The core innovation is Alignment-free Dense Distillation (ADD) module, which establishes dense distillation pathways between misaligned cross-domain features guided by learned affinities. Additionally, we capture the semantic relations to ensure distillation is restricted to semantically consistent regions. Extensive experiments demonstrate that our method achieves the state-of-the-art performance in WLI image classification on both the public CPC-Paired and our in-house datasets.
<p align="center">
<img src="https://github.com/Huster-Hq/ADD/blob/main/imgs/method.png" alt="Image" width="600px">
<p>


## 2. Visulization of Results
### 2.1 ROC Curve:
#### In-house Dataset
<p align="center">
<img src="https://github.com/Huster-Hq/ADD/blob/main/imgs/average_ROC_private.png" alt="Image" width="600px">
<p>

#### Public Dataset (CPC-Paried)
<p align="center">
<img src="https://github.com/Huster-Hq/ADD/blob/main/imgs/average_ROC_CPC.png" alt="Image" width="600px">
<p>


### 2.2 CAM Maps:
<p align="center">
<img src="https://github.com/Huster-Hq/ADD/blob/main/imgs/CAM_visualization.png" alt="Image" width="600px">
<p>


## 3. Getting Started
### 3.1 Recommended environment:
- Python 3.8+
- PyTorch 2.1+ 
- TorchVision corresponding to the PyTorch version
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Install other dependent packages:
```
cd ADD
pip install -r requirements.txt
```

### 3.2 Data preparation
- Downloading the [CPC-Paired dataset](https://github.com/WeijieMax/CPC-Trans) (public WLI-NBI paired polyp classification dataset). The file paths should be arranged as follows:
```
ADD
├── dataset
├── ├── White_light
├── ├── ├── adenomas
├── ├── ├── ├── ├── 01-1.png
├── ├── ├── ├── ├── 02-1.png
├── ├── ├── ├── ├── ......
├── ├── ├── hyperplastic_lesions
├── ├── ├── ├── ├── 011-1.png
├── ├── ├── ├── ├── 011-2.png
├── ├── ├── ├── ├── ......
├── ├── NBI
├── ├── ├── adenomas
├── ├── ├── ├── ├── 01-1.png
├── ├── ├── ├── ├── 02-1.png
├── ├── ├── ├── ├── ......
├── ├── ├── hyperplastic_lesions
├── ├── ├── ├── ├── 011-1.png
├── ├── ├── ├── ├── 011-2.png
├── ├── ├── ├── ├── ......
```

- Note that the details of dataset splitation in the 5-fold experiment can be downloaded in [here](https://drive.google.com/drive/folders/1UkLZxZDGyKH3P3TIAra-tORzBEuwx-3E?usp=drive_link). You need to download these `.txt` files and put them into a newly created folder `split` and the file paths should be arranged as follows:
```
ADD
├── split
├── ├── xxx.txt
├── ├── ......
```


### 3.3 Training:
Stage 1: pre-traning the NBI classifier:
```
python train_teacher.py
```
Stage 2: training the WLI classifier:
```
python train.py
```

### 3.4 Testing and Evaluation:
```
python test.py
```
You can also directly download the `well-trained model` from [Google Drive](https://drive.google.com/file/d/1Qi7tvsnm4bTTKYPPuLCPE12OQjeZ0SC1/view?usp=drive_link), and predict the results by `test.py`.
