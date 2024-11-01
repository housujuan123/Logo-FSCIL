# Forward Compatible Multimodal Framework for Few-Shot Logo Incremental Classification (FAM-Logo)
Logo classification has gained increasing attention for its various applications. However, the rapid emergence of new companies and product logo categories presents challenges to existing logo classification models. Often, only a limited number of examples are available for these new categories. A robust classification model should incrementally learn and recognize new logo classes while maintaining its ability to discriminate between existing ones. To address this, we formulate the task of Logo Few-Shot Class-Incremental Learning (Logo-FSCIL). Specifically, we adopt a forward-compatible embedding space rservation strategy and develop a prompt-based multimodal alignment framework to mitigate catastrophic forgetting. Furthermore, taking into account the unique attributes of logos, we devise an enhancement alignment strategy guided by textual information. Experimental results on three logo datasets demonstrate that our method significantly outperforms state-of-the-art FSCIL benchmarks. Extensive experiments on three generic datasets further validate the generalization capability of our approach. .
## Datasets
We provide the source code on three benchmark datasets, i.e., CIFAR100, CUB200 and miniImageNet. Please follow the guidelines in CEC to prepare them.
Datasets resource are available: https://pan.baidu.com/s/1bR4vO2yfutwWXc1Povap5A?pwd=k648
## Code Structures
There are four parts in the code.

*models: It contains the backbone network and training protocols for the experiment.
## Get Started

