# title: Dynamic Spatial-Temporal Embedding via Neural Conditional Random Field for Multivariate Time Series Forecasting 

This folder concludes the source code of our ASTCRF model. 

## Structure:
* configs: including parameter configurations for four used datasets.
  
* data: including five bechmark datasets: PEMS04, PEMS08, Solar-Energy, Electricity and Exchange-Rate datasets used in our experiments. PEMS04 and PEMS08 are released by and available at  [ASTGCN](https://github.com/Davidham3/ASTGCN/tree/master/data). Solar-Energy and Exchange-Rate are available at [Multivariate Time Series Data Sets](https://github.com/laiguokun/multivariate-time-series-data).

* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our ASTCRF model.

* results: saved forecasting results.


## Requirements

Python 3.8.5, Pytorch 1.7.0, Numpy 1.21.2, Pandas 1.3.2, matplotlib 3.3.4, argparse and configparser

To replicate the results, you can run the codes in the "model" folder directly, by setting 'DATASET=' in 'Run.py' as corresponding dataset ('PEMS04', 'PEMS08', 'solar', 'electricity', 'exchange').

## Acknowledgments
This codebase is heavily borrowed from [AGCRN](https://github.com/LeiBAI/AGCRN), we would like to thank [LEI BAI](http://leibai.site/) for his pytorch implementation of [AGCRN](https://arxiv.org/pdf/2007.02842.pdf).



