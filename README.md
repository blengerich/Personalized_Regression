# Personalized Regression

Code repository for the paper ["Personalized Regression Enables Sample-Specific Pan-Cancer Analysis."](https://academic.oup.com/bioinformatics/article/34/13/i178/5045771).

The goal of *Personalized Regression* is to perform retrospective analysis by estimating simple models that each apply to a single sample. After estimating these sample-specific models, we have a matrix of model parameters which we may analyze as we wish. In this paper, we analyzed logistic regression models trained on a set of transcriptomic profiles of cancer patients. From these data, we were able to identify several patterns which are overlooked by traditional analyses. For more information about these experiments, please see the [slides for our ISMB 2018 presentation](//www.cs.cmu.edu/~blengeri/downloads/slides/personalized_regression_ismb_2018.pdf).

[An updated version of the slides](https://www.cs.cmu.edu/~blengeri/downloads/slides/personalized_regression_psu.pdf), with more experiments, are also available.

# Using this code

This repository includes code for both personalized logistic and personalized linear regression, but can be extended to personalize any predictive model. The main file is `distance_matching.py`, which is designed to take in a black-box predictive model and the corresponding subgradient updates as Python functions. Examples of these functions for linear and logistic regression are included in the file `functions.py`. In addition, `DistanceMatching` objects require feature-specific distance metrics for covariates as Python functions. Examples of these distance metrics are also provided in `functions.py`.

## Citing

If you use the code, data, or ideas in this repository, please cite:

```
@article{doi:10.1093/bioinformatics/bty250,
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
