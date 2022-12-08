# HerosNet: Hyperspectral Explicable Reconstruction and Optimal Sampling Deep Network for Snapshot Compressive Imaging (CVPR 2022)
Xuanyu Zhang, Yongbing Zhang, Ruiqin Xiong, Qilin Sun, [Jian Zhang](http://jianzhang.tech/) [[PDF](https://arxiv.org/abs/2112.06238)]

## Introduction
Hyperspectral imaging is an essential imaging modality for a wide range of applications, especially in remote sensing, agriculture, and medicine. Inspired by existing
hyperspectral cameras that are either slow, expensive, or bulky, reconstructing hyperspectral images (HSIs) from a low-budget snapshot measurement has drawn wide attention. By mapping a truncated numerical optimization algorithm into a network with a fixed number of phases, recent deep unfolding networks (DUNs) for spectral snapshot compressive sensing (SCI) have achieved remarkable success. However, DUNs are far from reaching the scope of industrial applications limited by the lack of cross-phase feature interaction and adaptive parameter adjustment. In this paper, we propose a novel Hyperspectral Explicable Reconstruction and Optimal Sampling deep Network for SCI, dubbed HerosNet, which includes several phases under the ISTA-unfolding framework. Each phase can flexibly simulate the sensing matrix and contextually adjust the step size in the gradient descent step, and hierarchically fuse and interact the hidden states of previous phases to effectively recover current HSI frames in the proximal mapping step. Simultaneously, a hardware-friendly optimal binary mask is learned end-to-end to further improve the reconstruction performance. Finally, our HerosNet is validated to outperform the state-of-the-art methods on both simulation and real datasets by large margins.

## Overall architecture
![image](https://github.com/jianzhangcs/HerosNet/blob/main/Figs/net.png)
Figure 1. Illustration of the proposed HerosNet framework.

## Contents
1. [Test](#Test)
2. [Train](#Train)
3. [Results](#Results)
4. [Citation](#Citation)
5. [Acknowledgement](#Acknowledgement)

## Test
1. Prepare test data.

    The original test dataset is in './test'.

2. Download pretrained model.

    The pretrained model can be downloaded in [Google Drive](https://drive.google.com/file/d/17kxbNfRHJpLtFfIsdUnQPxVJAR8CrUpy/view?usp=share_link).

3. Run the test scripts. 
    ```bash
    python Test.py
    ```
4. Check the results in './Results'.

## Train
1. Prepare Training data.

    Training data is the same with [DGSM](https://github.com/TaoHuang95/DGSMP).

2. Run the train scripts. 
    ```bash
    python Train.py
    ```
## Results
![image](https://github.com/jianzhangcs/HerosNet/blob/main/Figs/results.png)

## Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
@inproceedings{zhang2022heros,
 title = {HerosNet: Hyperspectral Explicable Reconstruction and Optimal Sampling Deep Network for Snapshot Compressive Imaging},
 author = {Xuanyu Zhang, Yongbing Zhang, Ruiqin Xiong, Qilin Sun, Jian Zhang},
 booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2022}
}
```

## Acknowledgement
We thank the authors of [DGSM](https://github.com/TaoHuang95/DGSMP) for sharing their codes and data.
