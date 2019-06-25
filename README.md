# Addressable market challange

The solution was developed in a juputer notebook with the following structure:

- Load libs and modules
- Load raw data sets
- Split actual and addressable customers
- EDA
    - Descriptive statistics analsysis
    - Customer type size analysis
    - Outlier analysis
    - Univariate distribution analysis
- Model building
    - Customer segmentation
    - Decision-tree classifier
    - Customer scorer (pearson similiarity)
- Generate Deliverables
    - Deliverable 1
    - Deliverable 2
    - Sanity Check
- Improvements (TODO)


The notebook source artifacts:

- Addressable Market.ipynb (source)
- Addressable Market.html (please, open in a chrome web browser)

## Raw Data 

- customer_CRM_2019-05-17.csv
- Neoway_database_2019-05-17.csv

## Python modules

- config.py
- utils.py
- viz.py
- clustering.py
- classifier.py

## Deliverables

- Addressable customers' ids: addressable_ids.csv
- Training dataset: training_ids.csv
- Validation dataset: testing_ids.csv
- Addressable Market in the format id/score, ordered by score: addressable_ranking.csv

## Environment

Powered by IBM Watson Studio using the following hadware and software config:

- Environment Default Spark Python 3.6 XS
- Creator IBM
- Hardware configuration (Driver) 1 vCPU and 4 GB RAM
- Hardware configuration (Executor) 1 vCPU and 4 GB RAM
- Number of executors 2
- Spark version 2.3
- Software version Python 3.6

Link: https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/89a5651d-87f7-440d-b6b4-7b831cefcad4/view?projectid=8d5fdbe5-355b-4c65-bbf1-1d1abba0077f&context=wdp


## Author
Fernando Felix do Nascimento Junior
