# SPRT-TANDEM-PyTorch
This repository contains the official PyTorch implementation of __SPRT-TANDEM__ ([ICASSP2023](https://arxiv.org/abs/2302.09810), [ICML2021](http://proceedings.mlr.press/v139/miyagawa21a.html), and [ICLR2021](https://openreview.net/forum?id=Rhsu5qD36cL)). __SPRT-TANDEM__ is a neuroscience-inspired sequential density ratio estimation algorithm that estimates log-likelihood ratios of two or more hypotheses for fast and accurate sequential data classification. For intuitive understanding, also see [SPRT-TANDEM tutorial](https://github.com/Akinori-F-Ebihara/SPRT-TANDEM_tutorial).

## Citation
___Please cite the orignal paper(s) if you use the whole or a part of our codes.___

```
# ICASSP2023
@inproceedings{saturation_problem,
  title =     {Toward Asymptotic Optimality: Sequential Unsupervised Regression of Density Ratio for Early Classification},
  author =    {Akinori F Ebihara and Taiki Miyagawa and Kazuyuki Sakurai and Hitoshi Imaoka},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing},
  year =      {2023},
}

# ICML2021
@inproceedings{MSPRT-TANDEM,
  title = 	  {The Power of Log-Sum-Exp: Sequential Density Ratio Matrix Estimation for Speed-Accuracy Optimization},
  author =    {Miyagawa, Taiki and Ebihara, Akinori F},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	  {7792--7804},
  year = 	  {2021},
  url = 	  {http://proceedings.mlr.press/v139/miyagawa21a.html}
}

# ICLR2021
@inproceedings{SPRT-TANDEM,
  title={Sequential Density Ratio Estimation for Simultaneous Optimization of Speed and Accuracy},
  author={Akinori F Ebihara and Taiki Miyagawa and Kazuyuki Sakurai and Hitoshi Imaoka},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=Rhsu5qD36cL}
}
```
