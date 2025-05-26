<div align="center">
<h1>ADD</h1>
<h3>Holistic White-light Polyp Classification via Alignment-free Dense Distillation of Auxiliary Optical Chromoendoscopy</h3>
<br>
<a href="https://scholar.google.com/citations?user=rU2JxLIAAAAJ&hl=en">Qiang Hu</a><sup><span>1, &#42</span></sup>, Qimei Wang</a><sup><span>1, &#42</span></sup>, Jia Chen</a><sup><span>2</span></sup>, Xuantao Ji</a><sup><span>2</span></sup>, <a href="http://faculty.hust.edu.cn/liqiang15/zh_CN/index.htm">Qiang Li</a><sup><span>1, &#8224;</span></sup>, <a href="https://scholar.google.com/citations?user=LwQcmgYAAAAJ&hl=en">Zhiwei Wang</a><sup><span>1, &#8224;</span></sup>
</br>

<sup>1</sup>  WNLO, HUST, <sup>2</sup>  UIH
<br>
(<span>&#42;</span>: equal contribution, <span>&#8224;</span>: corresponding author)
</div>

## 1. Overview
This work implements a novel framework for **enhancing holistic WLI polyp classification** through cross-domain **(NBI->WLI)** knowledge distillation, **without requiring any additinal labeling cost**. The core innovation is Alignment-free Dense Distillation (ADD) module, which establishes dense distillation pathways between misaligned cross-domain features guided by learned affinities. Additionally, we capture the semantic relations to ensure distillation is restricted to semantically consistent regions. Extensive experiments demonstrate that our method achieves the state-of-the-art performance in WLI image classification on both the public CPC-Paired and our in-house datasets.
<p align="center">
<img src="https://github.com/Huster-Hq/ADD/blob/main/imgs/method.png" alt="Image" width="600px">
<p>

## 2. Checkpoints
| Model | CPC-Paired (AUC) | In-house (AUC) | Weights (5-folds) |
| :---- | :------: | :------: | :------: |
| Ours | 0.936 | 0.826 | [ckpts](https://drive.google.com/drive/folders/18k-cnyyQ8rO_OAzg5RXmnk4PyhDqrVcg?usp=drive_link) |
| CIC variant | 0.801 | 0.603 | [ckpts](https://drive.google.com/drive/folders/1uppDyiT8EijUbRKGXlHQikJ5QOj3_zdf?usp=drive_link) |
| w/o ADD & SRG | 0.857 | 0.683 | [ckpts](https://drive.google.com/drive/folders/1OwAjYQY7p1MxkZ6pyZJXmp5Nd0wmqmrU?usp=drive_link) |
| w/o SRG | 0.925 | 0.775 | [ckpts](https://drive.google.com/drive/folders/1hSYyHicQRs9DCwMcnP0uMJngs9-7Y_xw?usp=drive_link) |
| w/o Bi-A | 0.918 | 0.762 | [ckpts](https://drive.google.com/drive/folders/1O0iNXcGgI1ZJYi4Xhi-MuzzG6h3mpL-a?usp=drive_link) |
| w/o PSR | 0.928 | 0.786 | [ckpts](https://drive.google.com/drive/folders/1hSYyHicQRs9DCwMcnP0uMJngs9-7Y_xw?usp=drive_link) |

## 3. Visulization of Results
### 3.1 ROC Curve:
#### In-house Dataset
<p align="center">
<img src="https://github.com/Huster-Hq/ADD/blob/main/imgs/average_ROC_private.png" alt="Image" width="600px">
<p>

#### Public Dataset (CPC-Paried)
<p align="center">
<img src="https://github.com/Huster-Hq/ADD/blob/main/imgs/average_ROC_CPC.png" alt="Image" width="600px">
<p>


### 3.2 CAM Maps:
<p align="center">
<img src="https://github.com/Huster-Hq/ADD/blob/main/imgs/CAM_visualization.png" alt="Image" width="600px">
<p>


## 4. Getting Started
### 4.1 Recommended Environment:
- Python 3.8+
- PyTorch 2.1+ 
- TorchVision corresponding to the PyTorch version
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Install other dependent packages:
```
cd ADD
pip install -r requirements.txt
```

### 4.2 Data Preparation
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


### 4.3 Training:
Stage 1: pre-traning the NBI classifier:
```
python train_teacher.py
```
Stage 2: training the WLI classifier:
```
python train.py
```

### 4.4 Testing and Evaluation:
```
python test.py
```
You can also directly download the `well-trained model` from [Google Drive](https://drive.google.com/drive/folders/18k-cnyyQ8rO_OAzg5RXmnk4PyhDqrVcg?usp=drive_link), and predict the results by `test.py`.
