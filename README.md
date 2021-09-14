# S3Net

# Dependencies

pip install requirement.txt


# Training 

For example: Standford Cars dataset (1-shot)

python mytrain_cars.py  --nExemplars 1 --gpu-devices 0


# Testing 

For example: Standford Cars dataset (1-shot)

python test_car.py --nExemplars  --gpu-devices 0  --resume ./result/car/CAM/5-shot-seed5-conv4_myspp_globalcos_few_loss/best_model.pth.tar



PyTorch code for the ICME 2021 paper **Selective, Structural, Subtle: Trilinear Spatial-Awareness for Few-Shot Fine-Grained Visual Recognition**.


@inproceedings{wu2021selective,
  title={Selective, Structural, Subtle: Trilinear Spatial-Awareness for Few-Shot Fine-Grained Visual Recognition},
  author={Wu, Heng and Zhao, Yifan and Li, Jia},
  booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}



# 计算机视觉

**论文总结**

# 技术类

## Self-Supervised Learning

- 2021-ICCV-【code】-Instance Similarity Learning for Unsupervised Feature Representation
- 2021-CVPR-【code】-A Large-Scale Study on Unsupervised Spatiotemporal Representation Learning
- 2021-CVPR-An Empirical Study of Training Self-Supervised Vision Transformers
- 2021-CVPR-【code】-【oral】-Exploring Simple Siamese Representation Learning
- 2021-CVPR-【oral】-Dense Contrastive Learning for Self-Supervised Visual Pre-Training

    作者：Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, Lei Li

    单位：The University of Adelaide, Australia 2Tongji University, China 3ByteDance AI Lab

    论文链接：[https://arxiv.org/pdf/2011.09157.pdf](https://arxiv.org/pdf/2011.09157.pdf)

- 2019-CVPR-Unsupervised Feature Learning via Non-Parametric Instance Discrimination

    链接：[https://arxiv.org/pdf/1805.01978.pdf](https://arxiv.org/pdf/1805.01978.pdf)

- 2021-Arxiv-**VICReg**: **Variance-Invariance-Covariance Regularization for Self-Supervised Learning**

    作者：Adrien Bardes, Jean Ponce, **Yann LeCun**

    单位：Facebook AI Research

     论文链接：[https://arxiv.org/pdf/2105.04906.pdf](https://arxiv.org/pdf/2105.04906.pdf)

- 2021-Arxiv-【code】-**Self-Supervised Learning with Swin Transformers**

    作者：Zhenda Xie∗ †13 Yutong Lin∗†23 Zhuliang Yao†13 Zheng Zhang3 Qi Dai3 Yue Cao3 Han Hu

    单位：Tsinghua University 2Xi’an Jiaotong University

    论文链接：[https://arxiv.org/pdf/2105.04553.pdf](https://arxiv.org/pdf/2105.04553.pdf)

    代码：[https://github.com/SwinTransformer/Transformer-SSL](https://github.com/SwinTransformer/Transformer-SSL)

- 2021-ICCV-Improve Unsupervised Pretraining for Few-label Transfer

## Few-Shot Learning

- 2021-Arxiv-Learning to Affiliate: Mutual Centralized Learning
for Few-shot Classification

    链接：[https://arxiv.org/pdf/2106.05517.pdf](https://arxiv.org/pdf/2106.05517.pdf)

- 2021-Arxiv-FEDI: Few-shot learning based on Earth Mover's Distance algorithm combined with deep residual network to **identify diabetic retinopathy**

- 2021-ICCV-【code】-A Unified Objective for Novel Class Discovery
- 2021-ICCV-Relational Embedding for Few-Shot Classification
- 
- 2021-CVPR-ECKPN: Explicit Class Knowledge Propagation Network for Transductive Few-shot Learning
- 2021-CVPR-Few-Shot Classification with Feature Map Reconstruction Networks

    链接：[https://arxiv.org/pdf/2012.01506.pdf](https://arxiv.org/pdf/2012.01506.pdf)

- **2021-CVPR-【code】-Prototype Completion with Primitive Knowledge for Few-Shot Learning**

    链接：[https://arxiv.org/pdf/2009.04960.pdf](https://arxiv.org/pdf/2009.04960.pdf)

- 2021-ICLR-EPT: INSTANCE-LEVEL AND EPISODE-LEVEL PRETEXT TASKS FOR FEW-SHOT LEARNING

    [https://openreview.net/forum?id=xzqLpqRzxLq](https://openreview.net/forum?id=xzqLpqRzxLq)

- 2021-CVPR-【code】-Exploring Complementary Strengths of Invariant and Equivariant Representation for few-shot learning
- 2021-CVPR-Reinforced Attention for Few-Shot Learning and Beyond
- 2021-CVPR-Learning Dynamic Alignment via Meta-Filter for Few-Shot Learning
- 2021-CVPR-Rethinking Class Relations: Absolute-relative Supervised and Unsupervised Few-shot Learning
- 2020-CVPR-Few-shot learning via embedding adaptation with set-to-set functions
- 2020-CVPR-DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover’s Distance and Structured Classifiers
- 2020-Nips-【code】-Crosstransformers: spatially-aware few-shot transfer
- 2020-AAAI-Multi-Stage Self-Supervised Learning for Graph Convolutional Networks on Graphs with Few Labeled Nodes
- 2020-AAAI-Collaborative graph convolutional networks: Unsupervised learning meets **semi-supervised learning**
- 2020-WACV-Charting the Right Manifold: Manifold Mixup for Few-shot Learning

**期刊**

- 2021-TCSVT-Multi-Scale Metric Learning for Few-Shot Learning

    链接：[https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9097252](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9097252)

- 2020-TCSVT-Few-Shot Visual Classification Using Image Pairs With Binary Transformation（**是一个短文**）

    链接：[https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8730301](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8730301)

- 2020-TMM-Low-rank pairwise alignment bilinear network for few-shot fine-grained image classification.
- 2021-IEEE_SPL-Multi-Dimensional Edge Features Graph Neural Network on Few-Shot Image Classification
- 2020-IEEE_SPL-Layer-Wise Adaptive Updating for Few-Shot Image Classification

- 2021-Arxiv-Few-Shot Learning by Integrating Spatial and Frequency Representation

    作者：Xiangyu Chen†, Guanghui Wang‡

    单位：University of Kansas, Ryerson University

    链接：[https://arxiv.org/pdf/2105.05348.pdf](https://arxiv.org/pdf/2105.05348.pdf)

- 2021-Arxiv-Subspace Representation Learning for Few-shot Image Classification

    作者：Ting-Yao Hu, Zhi-Qi Cheng, Alexander G. Hauptmann

### Fine-grained Image Classification

- 2021-Arxiv-NDPNet: A novel non-linear data projection network for few shot fine-grained image classification

### Unsupervised Leraning

- 2021-IJCAI-Few-Shot Learning with Part Discovery and Augmentation from Unlabeled Images

    链接：[https://arxiv.org/pdf/2105.11874.pdf](https://arxiv.org/pdf/2105.11874.pdf)

- 2021-PR-Unsupervised meta-learning for few-shot learning

### Semantic Segmantation

Weakly supervised semantic segmantic

- 2020-Arxiv-Weakly Supervised Few-shot Object Segmentation using Co-Attention with Visual and Semantic Embeddings
- 2019-ICCV-Weakly Supervised One-Shot Segmentation

    [https://openaccess.thecvf.com/content_ICCVW_2019/papers/MDALC/Raza_Weakly_Supervised_One_Shot_Segmentation_ICCVW_2019_paper.pdf](https://openaccess.thecvf.com/content_ICCVW_2019/papers/MDALC/Raza_Weakly_Supervised_One_Shot_Segmentation_ICCVW_2019_paper.pdf)

- 163

Semantic segmantic

- 2021-CVPR-【code】-Incremental Few-Shot Instance Segmentation

    作者：Xiangyu Yue;* Zangwei Zheng;* Shanghang Zhang Yang Gao Trevor Darrell1 Kurt Keutzer AlbertoSangiovanni Vincentelli

    单位：UC Berkeley 2Nanjing University 3Tsinghua University

    代码：[https://github.com/zhengzangw/PCS-FUDA](https://github.com/zhengzangw/PCS-FUDA)

- 2021-CVPR—【code】Adaptive Prototype Learning and Allocation for Few-Shot Segmentation

wx418d7e6cc6ec6109

### Domain Adaption

- 2021-Arxiv-【code】Few-Shot Domain Adaptation with Polymorphic
Transformers
- 2021-ACM MM -Revisiting Mid-Level Patterns for Cross-Domain Few-Shot Recognition
- 2021**-CVPR**-Few-shot Image Generation via Cross-domain Correspondence
- 2021-CVPR-【code】-Prototypical Cross-domain Self-supervised Learning for Few-shot Unsupervised Domain Adaptation (PCS)

    作者：Dan Andrei Ganea, Bas Boom, Ronald Poppe

    单位：Utrecht University，Cyclomedia Technology，Utrecht University

    链接：[https://arxiv.org/pdf/2105.05312.pd](https://arxiv.org/pdf/2105.05312.pdf)

    代码：[https://github.com/zhengzangw/PCS-FUDA](https://github.com/zhengzangw/PCS-FUDA)

- **2019-CVPR-GCAN: Graph Convolutional Adversarial Network for Unsupervised Domain Adaptation**

- 2020-**ICLR-**【code】-Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation
- 2021**-WACV-**Domain-Adaptive Few-Shot Learning
- 2020-**ECCV**-A Broader Study of Cross-Domain Few-Shot Learning
- 2020-**PMLR**-Few-shot Domain Adaptation by Causal Mechanism Transfer
- 2019-ACMTURC-Few-Shot Domain Adaptation for Semantic Segmentation
- 2017-Nips-【code】-Few-Shot Adversarial Domain Adaptation

- 2021-Arxiv-Cross-Domain Few-Shot Learning by Representation Fusion

    [https://arxiv.org/pdf/2010.06498.pdf](https://arxiv.org/pdf/2010.06498.pdf)

- 2020-Arxiv-Cross-Domain Few-Shot Learning with Meta Fine-Tuning

    [https://arxiv.org/pdf/2005.10544v1.pdf](https://arxiv.org/pdf/2005.10544v1.pdf)

- [2020-Arxiv-Cross-domain self-supervised learning for domain adaptation with few source labels](https://arxiv.org/pdf/2003.08264.pdf)（（提出一个新的setting，源域有少标签））

    [https://arxiv.org/pdf/2003.08264.pdf](https://arxiv.org/pdf/2003.08264.pdf)

### Incremental Learning

- 2021-ICCV-【code未公布】Generalized and Incremental Few-Shot Learning by Explicit Learning and Calibration without Forgetting
- 2021-CVPR-【code】Few-Shot Incremental Learning with Continually Evolved Classifiers
- 2020-ICLR-INCREMENTAL FEW-SHOT LEARNING VIA VECTOR
QUANTIZATION IN DEEP EMBEDDED SPACE

### Long-tail Visual Recognition

- 2021-AAAI-One-shot learning for long-tail visual relation detection

### Video Classification

- 2021-IJCAI-Learning Implicit Temporal Alignment for Few-shot Video Classification

    作者：Songyang Zhang, Jiale Zhou, Xuming He

    单位：ShanghaiTech University，University of Chinese Academy of Sciences etal

    论文链接：[https://arxiv.org/pdf/2105.04823.pdf](https://arxiv.org/pdf/2105.04823.pdf)

### Nosiely label

- 2021-Meta learning to classify intent and slot labels with noisy few shot examples
- 2021-WACV-RNNP: A Robust Few-Shot Learning Approach
- 2019-AAAI-Hybrid attention-based prototypical networks for noisy few-shot relation classification

### Weakly Supervised

- Weakly-supervised Object Localization for Few-shot Learning and Fine-grained Few-shot Learning
- 2019-Weakly-supervised Compositional Feature Aggregation for Few-shot Recognition

**期刊**

- 2020-KBS-Heterogeneous graph neural networks for noisy few-shot relation classification

### Medical Image

2020-Medical IMage Analysis-Discriminative ensemble learning for few-shot chest x-ray diagnosis

### Others

- 2021-Arxiv-UVStyle-Net: Unsupervised Few-shot Learning of 3D Style Similarity Measure for B-Reps

    作者：Peter Meltzer, Hooman Shayani, Amir Khasahmadi, Pradeep Kumar Jayaraman, Aditya Sanghi, Joseph Lambourne

    单位：Autodesk AI Lab

- 2021-Arxiv-Few-Shot Learning for Image Classification of Common Flora

    作者：Joshua Ball

    单位：Edward E. Whitacre Jr. College of Engineering

    数据集：[https://www.kaggle.com/alxmamaev/flowers-recognition](https://www.kaggle.com/alxmamaev/flowers-recognition)

- 2019-TMM-Adaptation-Oriented Feature Projection for One-Shot Action Recognition

## Zero-Shot Learning

- 2021-TMM-A Novel Perspective to Zero-Shot Learning: Towards an Alignment of Manifold Structures via Semantic Feature Expansion
- 2019-TMM-Deep0Tag: Deep Multiple Instance Learning for Zero-Shot Image Tagging.
- 2019-TMM-Hierarchical Prototype Learning for Zero-Shot Recognition
- 2019-TMM-CI-GNN: Building a Category-Instance Graph for Zero-Shot Video Classification.

## Open-Set Recognition

- 2019-CVPR-Classification-Reconstruction Learning for Open-Set Recognition

    [https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoshihashi_Classification-Reconstruction_Learning_for_Open-Set_Recognition_CVPR_2019_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoshihashi_Classification-Reconstruction_Learning_for_Open-Set_Recognition_CVPR_2019_paper.pdf)

## Image Retrieval

- 2021-CVPR【code】-Compatibility-aware Heterogeneous Visual Search

    作者：Rahul Duggal* Hao Zhou Shuo Yang Yuanjun Xiong

    单位：Wei Xia† Zhuowen Tu Stefano Soatto

    论文链接：[https://arxiv.org/pdf/2105.06047.pdf](https://arxiv.org/pdf/2105.06047.pdf)

- 2017-TIP-【code】-Selective Convolutional Descriptor Aggregation for Fine-Grained Image Retrieval

## Semi-supervised Learing

**互信息：即随机变量 X 所能带来的对随机变量 Y 不确定度的减少程度.**

- 2019-Nips-MixMatch: A Holistic Approach to Semi-Supervised Learning
- 2013-ICML-Pseudo-Label: The simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks
- 2005-Nips-Semi-supervised learning by extropy Minimization

## Domain Adaptation

- 2021-CVPR-Cross-Domain Gradient Discrepancy Minimization
for Unsupervised Domain Adaptation
- 2019-ICCV-Semi-supervised Domain Adaptation via Minimax Entropy

### Contrastive domian adapation（提出一个新的setting，源域和目标域都没标签）

2021-CVPR-Contrastive Domain Adaptation

### Unversal domain adaptation

- 2021-CVPR-【code】OV ANet: One-vs-All Network for Universal Domain Adaptation
- 2021-CVPR-Domain Consensus Clustering for Universal Domain Adaptation
- 2021-ICML-Implicit Class-Conditioned Domain Alignment
for Unsupervised Domain Adaptation
- 2021-Arxiv-【Micrsoft】-ToAlign: Task-oriented Alignment for Unsupervised Domain Adaptation
- 2020-AAAI-Unsupervised Domain Adaptation via Discriminative Manifold Embedding and Alignment
- 2019-CVPR-Universal Domain Adaptation
- 2018-Nips-【mingsheng Long】Conditional adversarial domain adaptation.
- 2015-JMLR-Domain-Adversarial Training of Neural Network (DANN)

    ![%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%20442347b358c24180b1a411b3f1fbca21/Untitled.png](%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%20442347b358c24180b1a411b3f1fbca21/Untitled.png)

### Semi-supevised  domain adaptation(few-shot domain adapation)

- 2021-CVPR-Learning Invariant Representations and Risks
for Semi-supervised Domain Adaptation
- 2020-CVPR-Workshop-Classification-aware Semi-supervised Domain Adaptation
- 2020-IJCAI-Bidirectional Adversarial Training for Semi-Supervised Domain Adaptation
- 2020-Arxiv-Online Meta-Learning for Multi-Source and
Semi-Supervised Domain Adaptation

### Domain Generalization

- 2021-IJCAI-Domain Generalization under Conditional and Label Shifts via Variational Bayesian Inference

### Incremental Learning

- 2020-ECCV-Class-Incremental Domain Adaptation

- 2020-ICLR-PROGRESSIVE MEMORY BANKS FOR INCREMENTAL
DOMAIN ADAPTATION
- 2018-Arxiv-Incremental Adversarial Domain Adaptation for Continually Changing Environments

### Review

- 2020-IEEE-A Comprehensive Survey on Transfer Learning

## Metric Learning

- 2021-ICCV-【code】-Deep Relational Metric Learning
- 2021-CVPR-【code】-Deep Compositional Metric Learning
- 2021-AAAI-Semi-Supervised Metric Learning: A Deep Resurrection

    作者：Ujjal Kr Dutta,1,2 Mehrtash Harandi,3 C Chandra Sekhar2

    单位：1 Data Sciences, Myntra, India, 2Dept. of Computer Science and Eng., Indian Institute of Technology Madras, India

    论文链接：[https://arxiv.org/pdf/2105.05061.pdf](https://arxiv.org/pdf/2105.05061.pdf)

## Vision Transformer

- 2021-CVPR-High-Resolution Complex Scene Synthesis with Transformers (workshop)

    作者：Manuel Jahn, Robin Rombach, Björn Ommer

    单位：IWR, HCI, Heidelberg University

    链接: [https://arxiv.org/pdf/2105.06458.pdf](https://arxiv.org/pdf/2105.06458.pdf)

- 2021-Arxiv-Episodic Transformer for Vision-and-Language Navigation

    作者：Alexander Pashevich, Cordelia Schmid, Chen Sun

    单位：Inria, Google Researcn, Brown University

    链接：[https://arxiv.org/pdf/2105.06453.pdf](https://arxiv.org/pdf/2105.06453.pdf)

- 2021-Arxiv-Rethinking the Design Principles of Robust Vision Transformer

    作者：Xiaofeng Mao, Gege Qi, Yuefeng Chen, Xiaodan Li, Shaokai Ye, Yuan He, Hui Xue

    单位：Alibaba Group

    论文链接：[https://arxiv.org/pdf/2105.07926.pdf](https://arxiv.org/pdf/2105.07926.pdf)

    代码：https://github.com/vtddggg/Robust-Vision-Transformer.

## Attention

- 2021-ICCV-【code】-Causal Attention for Unbiased Visual Recognition
- 2021-ICCV-【code】-Counterfactual Attention Learning for Fine-Grained Visual Categorization and Re-identification

## GAN

## Novel Class Discovery

- 2021-ICCV-【code】-A Unified Objective for Novel Class Discovery
- 2021-CVPR-Neighborhood Contrastive Learning for Novel Class Discovery
- 2009-General graph-based semi-supervised learning with novel class discovery

## Self-Teaching

- 2021-Arxiv-Adversarial Learning and Self-Teaching Techniques for Domain Adaptation in Semantic Segmentation
- 2021-TPAMI-Self-Teaching Video Object Segmentation
- 2018-BMVC-Adversarial learning for semi-supervised semantic segmentation
- 2017-CVPR-Deep self-taught learning for weakly supervised object localization
- 2017-TGRS-Self-Taught Feature Learning for Hyperspectral
Image Classification

## Transformer

### Domain adaptation

- **2021-Arxiv-TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation**
- 2021-Arxiv-Few-Shot Domain Adaptation with Polymorphic
Transformers

### Image Recognition

- 2021-Arxiv-Transformer with Peak Suppression and Knowledge Guidance for Fine-grained Image Recognition
- 2021-Arxiv-TransFG: A Transformer Architecture for Fine-grained Recognition
- 2021-CVPR-MIST: Multiple Instance Spatial Transformer

## Incremental Learning

- 2021-Arxiv-【code】-Class-Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation

## CNN

- 2021-CVPR-【code】-Diverse Branch Block: Building a Convolution as an Inception-like Unit **(DBBNET)**

# 应用类

## Image Recognition

### fuzzy labels

- 2021-Arxiv-S2C2 - An orthogonal method for Semi-Supervised Learning on **fuzzy labels**

    链接：[https://arxiv-download.xixiaoyao.cn/pdf/2106.16209.pdf](https://arxiv-download.xixiaoyao.cn/pdf/2106.16209.pdf)

- 2020-Arxiv-Beyond Cats and Dogs: Semi-supervised Classification of fuzzy labels with overclustering

### Cartoon Face Recognition

- 2021-Arxiv-S2C2 -**Graph Jigsaw** Learning for Cartoon Face
Recognition

    链接：[https://arxiv-download.xixiaoyao.cn/pdf/2106.16209.pdf](https://arxiv-download.xixiaoyao.cn/pdf/2106.16209.pdf)

## Semantic Segmentation

- 2021-CVPR-【code】-Copy of Self-supervised Augmentation Consistency for Adapting Semantic Segmentation

    链接: [https://arxiv.org/pdf/2105.00097.pdf](https://arxiv.org/pdf/2105.00097.pdf)
    代码:  [https://github.com/visinf/da-sac](https://github.com/visinf/da-sac)

- 2021-CVPR-A2-FPN: Attention Aggregation based Feature Pyramid Network for Instance Segmentation

    作者：Miao Hu, Yali Li, Lu Fang, Shengjin Wang

    链接: [https://arxiv.org/pdf/2105.03186.pdf](https://arxiv.org/pdf/2105.03186.pdf)

- 2021-Arxiv-Attentional Prototype Inference for Few-Shot Semantic Segmentation

### Weakly-Supervised Few-Shot Sementic Segmentation

- 2021-WACV-Weakly-supervised Object Representation Learning for Few-shot Semantic Segmentation
- 2020-IJCAI-Weakly Supervised Few-shot Object Segmentation using Co-Attention with Visual and Semantic Embeddings
- 2019-ICCV-Worshop-Weakly Supervised One-Shot Segmentation

    [https://openaccess.thecvf.com/content_ICCVW_2019/papers/MDALC/Raza_Weakly_Supervised_One_Shot_Segmentation_ICCVW_2019_paper.pdf](https://openaccess.thecvf.com/content_ICCVW_2019/papers/MDALC/Raza_Weakly_Supervised_One_Shot_Segmentation_ICCVW_2019_paper.pdf)

### **Noisy Image Classification**

- 2020-Arxiv-Attention-Aware Noisy Label Learning for Image Classification

## Objection Detection

- 2021-Arxiv-Hallucination Improves Few-Shot Object Detection

    作者：Weilin Zhang, Yu-Xiong Wang

    单位：University of Illinois at Urbana-Champaign

    论文链接：[https://arxiv.org/pdf/2105.01294.pdf](https://arxiv.org/pdf/2105.01294.pdf)

## Face Recognition

## Salient Object Detection

## Long-tail Image Recognition

## Image Enhancement

### Image Noising

### Image Inpainting

## Image-to-Image Translation

---

- 2021-CVPR-TransferI2I: Transfer Learning for Image-to-Image Translation from Small Datasets

    作者：Yaxing Wang, Hector Laria Mantecon, Joost van de WeijerLaura Lopez-Fuentes, Bogdan Raducanu
    单位：Computer Vision Center, Universitat Autonoma de Barcelona, Spain，Universitat de les Illes Balears, Spain
    链接：[https://arxiv.org/pdf/2105.06219.pdf](https://arxiv.org/pdf/2105.06219.pdf)

- 2019-ICCV-【code】-Few-Shot Unsupervised Image-to-Image Translation

    作者：Ming-Yu Liu1, Xun Huang1,2, Arun Mallya1, Tero Karras1 Timo Aila1, Jaakko Lehtinen1,3, Jan Kautz1

    单位：1NVIDIA, 2Cornell University, 3Aalto University

## Pose Estimation

- 2021-CVPR-When Human Pose Estimation Meets Robustness: Adversarial Algorithms and Benchmarks

    作者：Jiahang Wang, Sheng Jin, Wentao Liu, Weizhong Liu, Chen Qian, Ping Luo

    单位：Huazhong University of Science and Technology，The University of Hong Kong，SenseTime Research 4 SenseTime Research and [Tetras.AI](http://tetras.ai/)

## Image Captioning

- 2021-WACV-Self-Distillation for Few-Shot Image Captioning

## Fine-Grained Image Classification

[Fine-Grained Paper Reading](https://www.notion.so/Fine-Grained-Paper-Reading-a05016693fcb483c810ee41ab2145f98)

- 2021-AAAI-Dynamic Position-aware Network for Fine-grained Image Recognition

### Attention

- 2018-ECCV- Multi-attention multi-class constraint for fine-grained image recognition

### Domain Adaption

- 2020-CVPR-Multi-Modal Domain Adaptation for Fine-Grained Action Recognition
- 2020-WACV-An Adversarial Domain Adaptation Network for Cross-Domain Fine-Grained Recognition

    链接：[https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_An_Adversarial_Domain_Adaptation_Network_for_Cross-Domain_Fine-Grained_Recognition_WACV_2020_paper.pdf](https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_An_Adversarial_Domain_Adaptation_Network_for_Cross-Domain_Fine-Grained_Recognition_WACV_2020_paper.pdf)

### Weakly supervised learning

- 2021-Arxiv-【code】Multi-branch and Multi-scale Attention Learning for Fine-Grained Visual Categorization

    链接：[https://arxiv.org/pdf/2003.09150.pdf](https://arxiv.org/pdf/2003.09150.pdf)

- 2021-TIP-AP-CNN: weakly supervised attention pyramid convolutional neural network for fine-grained visual classification

    链接：[https://arxiv.org/pdf/2002.03353](https://arxiv.org/pdf/2002.03353)

- 2020-CVPR-Weakly Supervised Fine-Grained Image Classification via Guassian Mixture Model Oriented Discriminative Learning
- 2020-AAAI-Graph-Propagation Based Correlation Learning for Weakly Supervised Fine-Grained Image Classification

    链接：[https://ojs.aaai.org/index.php/AAAI/article/view/6912](https://ojs.aaai.org/index.php/AAAI/article/view/6912)

- 2020-Acess-Classification of financial tickets using weakly supervised fine-grained networks

    链接：[https://ieeexplore.ieee.org/document/9133548?denied=](https://ieeexplore.ieee.org/document/9133548?denied=)

- 2020-Signal Processing-Progressive learning for weakly supervised fine-grained classification
- 2019-CVPR-Weakly supervised complementary parts models for fine-grained image classification from the bottom up

    链接：[https://arxiv.org/pdf/1906.04833.pdf](https://arxiv.org/pdf/1906.04833.pdf)

- 2019-Arxiv-See better before looking closer: Weakly supervised data augmentation network for fine-grained visual classification

    链接：[https://arxiv.org/pdf/1901.09891.pdf](https://arxiv.org/pdf/1901.09891.pdf)

- 2019-Arxiv-Leveraging just a few keywords for fine-grained aspect detection through weakly supervised co-training
- 2019-ICCV-Cross-X learning for fine-grained visual categorization

    链接：[https://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Cross-X_Learning_for_Fine-Grained_Visual_Categorization_ICCV_2019_paper.pdf](https://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Cross-X_Learning_for_Fine-Grained_Visual_Categorization_ICCV_2019_paper.pdf)

- 2019-TMM-Part-aware fine-grained object categorization using weakly supervised part detection network

    链接：[https://arxiv.org/pdf/1806.06198](https://arxiv.org/pdf/1806.06198)

- 2019-ACMMM-Weakly supervised fine-grained image classification via correlation-guided discriminative learning
- 2018-CVPR-Webly Supervised Learning Meets **Zero-shot Learning**: A Hybrid Approach for Fine-grained Classification

    链接：[https://openaccess.thecvf.com/content_cvpr_2018/papers/Niu_Webly_Supervised_Learning_CVPR_2018_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Niu_Webly_Supervised_Learning_CVPR_2018_paper.pdf)

- 2018-IJCAI-] Centralized Ranking Loss with Weakly Supervised Localization for Fine-Grained Object Retrieval.

    链接：[https://www.researchgate.net/profile/Xiawu-Zheng/publication/326202341_Centralized_Ranking_Loss_with_Weakly_Supervised_Localization_for_Fine-Grained_Object_Retrieval/links/5e68a2bf299bf1744f72db2b/Centralized-Ranking-Loss-with-Weakly-Supervised-Localization-for-Fine-Grained-Object-Retrieval.pdf](https://www.researchgate.net/profile/Xiawu-Zheng/publication/326202341_Centralized_Ranking_Loss_with_Weakly_Supervised_Localization_for_Fine-Grained_Object_Retrieval/links/5e68a2bf299bf1744f72db2b/Centralized-Ranking-Loss-with-Weakly-Supervised-Localization-for-Fine-Grained-Object-Retrieval.pdf)

- 2018-TCSCT-Fast fine-grained image classification via weakly supervised discriminative localization

    链接：[https://arxiv.org/pdf/1710.01168.pdf](https://arxiv.org/pdf/1710.01168.pdf)

- 2018-TOMM-User-click-data-based fine-grained image recognition via weakly supervised metric learning
- 2017-AAAI-Weakly Supervised Learning of Part Selection Model with Spatial Constraints for Fine-Grained Image Classification

    链接：[https://ojs.aaai.org/index.php/AAAI/article/view/11223](https://ojs.aaai.org/index.php/AAAI/article/view/11223)

- 2016-TIP-Weakly supervised fine-grained categorization with part-based image representation
- 2016-TIP-Friend or foe: Fine-grained categorization with weak supervision
- 2015-ICCV-Augmenting strong supervision using web data for fine-grained categorization
- 2015-Weakly supervised fine-grained image categorization

### Similarity Measure

- 2014-Learning Fine-grained Image Similarity with Deep Ranking

    链接：[https://arxiv.org/pdf/1404.4661.pdf](https://arxiv.org/pdf/1404.4661.pdf)

- 2017-Tip-Selective Convolutional Descriptor Aggregationfor Fine-Grained Image Retrieval

### Unsupervised Learning

- 2012-Nips-Unsupervised Template Learning for Fine-Grained Object Recognition

## Vehicle Re-Id

2021-CVPR-Connecting Language and Vision for Natural Language-Based Vehicle Retrieval

2021-CVPR-Refining Pseudo Labels with Clustering Consensus over Generations for Unsupervised Object Re-identification

## 理论分析

- 2020-Arxiv-Debiased Contrastive Learning
- 2021-Nips-Supervised Contrastive Learning
- 2019-ICML-A convergence theory for deep learning via over parameterization
- 2018-Arxiv-Generalization Error in Deep Learning
- 2017-Nips-An overview of gradient descent optimization
algorithms

# 未来研究方向

吸收一切可以吸收的方向

- Few-shot+一切分类任务
- Few-Shot+Recogntion

    Few-shot + Long-tail visual recognition

    Few-shot + nosiely label

- Few-shot+ image-to-image translation
- Few-shot+Unsupervised Domain Adapation(2021 CVPR和2019 NIps作为参考)
- Few-shot+semantic segmentation

    Few-shot+semantic segmentation+domain adaptation

- Zero-Shot Learning
- Fine-grained+Domain Adapation
- Few-shot+Fine-grained+Domain Adapation
- Weakly supervised + fine-grained + few-shot
- 

[Trick总结](https://www.notion.so/Trick-072ca94c244844cc8b4458124604d67f)
