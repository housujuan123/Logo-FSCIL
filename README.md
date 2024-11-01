# Forward Compatible Multimodal Framework for Few-Shot Logo Incremental Classification (FAM-Logo)
Logo classification has gained increasing attention for its various applications. However, the rapid emergence of new companies and product logo categories presents challenges to existing logo classification models. Often, only a limited number of examples are available for these new categories. A robust classification model should incrementally learn and recognize new logo classes while maintaining its ability to discriminate between existing ones. To address this, we formulate the task of Logo Few-Shot Class-Incremental Learning (Logo-FSCIL). Specifically, we adopt a forward-compatible embedding space rservation strategy and develop a prompt-based multimodal alignment framework to mitigate catastrophic forgetting. Furthermore, taking into account the unique attributes of logos, we devise an enhancement alignment strategy guided by textual information. Experimental results on three logo datasets demonstrate that our method significantly outperforms state-of-the-art FSCIL benchmarks. Extensive experiments on three generic datasets further validate the generalization capability of our approach. .
## Datasets
We provide the source code on three logo datasets, i.e., foodLogo200, miniLogo2k and LogoInc32. Please follow the guidelines in CEC to prepare them.
Datasets resource are available: https://pan.baidu.com/s/1bR4vO2yfutwWXc1Povap5A?pwd=k648
## Code Structures
There are four parts in the code.

*models: It contains the backbone network and training protocols for the experiment. 

*data: Images and splits for the data sets.

*dataloader: Dataloader of different datasets.

*ft4base: The codes of fintuning CLIP for three datasets. 
## Get Started
*Please execute the finetuning file in ft4base according to the dataset.

*Train FAM

```bash python train.py -project FAM_food -dataset foodlogo -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.25 -lr_base 0.005 -lr_new 0.1 -decay 0.0005 -epochs_base 200 -schedule Milestone -milestones 50 100 150 -gpu '0,1' -temperature 16 -dataroot YOURDATAROOT -batch_size_base 64 -balance 0.01 -loss_iter 0

```bash python train.py -project FAM_mini2k -dataset mini2k -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.25 -lr_base 0.005 -lr_new 0.1 -decay 0.0005 -epochs_base 180 -schedule Milestone -milestones 50 100 150 -gpu '0,1' -temperature 16 -dataroot YOURDATAROOT -batch_size_base 64 -balance 0.01 -loss_iter 0

```bash python train.py -project FAM_inc32 -dataset inc32 -base_mode "ft_cos" -new_mode "avg_cos" -gamma 0.1 -lr_base 0.01 -lr_new 0.1 -decay 0.0005 -epochs_base 150 -schedule Cosine -gpu 0,1 -temperature 16 -batch_size_base 128 -balance 0.001 -loss_iter 0 -alpha 0.5



