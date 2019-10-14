# Personalized Regression

The goal of *Personalized Regression* is to push the limits of machine learning for heterogeneous samples. Instead of assuming that a single model is appropriate for all samples, we go to the other extreme and fit different model parameters for each samples. At this extreme, we are required to devise ways to share statistical power between models. After estimating these sample-specific models, we have a matrix of model parameters which we may analyze.

![Personalized Regression Example](https://github.com/blengerich/Personalized_Regression/blob/master/personalized_regression_figure1.png)


# Using this code

This repository includes code for both personalized logistic and personalized linear regression, but can be extended to personalize any predictive model. The main file is `distance_matching.py`, which is designed to take in a black-box predictive model and the corresponding subgradient updates as Python functions. Examples of these functions for linear and logistic regression are included in the file `functions.py`. In addition, `DistanceMatching` objects require feature-specific distance metrics for covariates as Python functions. Examples of these distance metrics are also provided in `functions.py`.


## NeurIPS 2019 Paper
The most recent paper on this project is our 2019 NeurIPS ["Learning Sample-Specific Models with Low-Rank Personalized Regression"](). A snapshot of code relating to that paper is available at: [github.com/blengerich/Personalized_Regression_NeurIPS19](https://github.com/blengerich/Personalized_Regression_NeurIPS19).


## ISMB 2018 Paper
These ideas were first developed in our 2018 ISMB Paper: ["Personalized Regression Enables Sample-Specific Pan-Cancer Analysis."](https://academic.oup.com/bioinformatics/article/34/13/i178/5045771).
A snapshot of the code relating to that paper is available at: [github.com/blengerich/Personalized_Regression_ISMB18](https://github.com/blengerich/Personalized_Regression_ISMB18).

For more information about these experiments, please see the [slides for our ISMB 2018 presentation](//www.cs.cmu.edu/~blengeri/downloads/slides/personalized_regression_ismb_2018.pdf).


## Citing

If you use the code or machine learning ideas in this repository, please cite the most recent paper:
```
@inproceedings{lengerich2019learning,
  title = {Learning Sample-Specific Models with Low-Rank Personalized Regression},
  author = {Lengerich, Benjamin J and Aragam, Bryon and Xing, Eric P},
  booktitle={Advances in Neural Information Processing Systems},
  pages={},
  year={2019}
}
```


If you are specifically interested in the used of personalized regression for cancer analysis, please cite the ISMB 2018 paper:
```
@article{lengerich2018personalized,
author = {Lengerich, Benjamin J and Aragam, Bryon and Xing, Eric P},
title = {Personalized regression enables sample-specific pan-cancer analysis},
journal = {Bioinformatics},
volume = {34},
number = {13},
pages = {i178-i186},
year = {2018},
doi = {10.1093/bioinformatics/bty250},
URL = {http://dx.doi.org/10.1093/bioinformatics/bty250},
eprint = {/oup/backfile/content_public/journal/bioinformatics/34/13/10.1093_bioinformatics_bty250/1/bty250.pdf}
}
```

## Contact
Please contact blengeri@cs.cmu.edu, naragam@cs.cmu.edu or epxing@cs.cmu.edu with any questions. Pull requests are always welcome.
