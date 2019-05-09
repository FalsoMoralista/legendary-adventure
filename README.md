# :sparkles: Legendary Adventure :sparkles:
> Simple Neural Network projects using the [Keras](https://keras.io/) library. 
<p align="center">
  <img  src="https://media.giphy.com/media/oj2GhTqAIoNIk/giphy.gif" alt="Bilbo Adventure" style="width: 480px; height: 196px; left: 0px; top: 0px; opacity: 0;">
</p>

## Table of contents
- [Description](#description)
- [Data](#data)  
- [Experiments](#experiments)
- [FAQ](#faq)

## Description
 This repository was initially created to **aggregate the content** about **neural networks** learned by me. Later on i decided to try to organize it by **documentating & anexing** the **resources** that **i am using to learn** so that **others** that wants to **get started** can **easily** do it **independely** *starting through here*.

## Data
<p align="center"><img src="https://i.pinimg.com/564x/0b/ac/ed/0baced7191ba1bd1cc196bdeb2fee285.jpg" alt="Hobbit House"></p>

  - All the data that we will be using in the experiments will be placed in the [data](/data) section. Description and additional resources are shown below.
    - The [breast-cancer](data/breast) diagnostic dataset, containing features from digitalized cell images and its respectively classification. See original [here](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
    - The [iris](data/iris) dataset, containing data from 150 plants of 3 different classes. Original [here](https://archive.ics.uci.edu/ml/datasets/Iris).
    - The [autos](data/autos) dataset, containing  data from 370.000 used cars from Ebay Kleinanzeigen. Original can be found [here](https://www.kaggle.com/orgesleka/used-cars-database)
    - The [barcelona](data/barcelona-data-sets) datasets, containing: Administration, Urban environment, Population, Territory, Economy and Business datasets from the city of Barcelona. Original [link](https://www.kaggle.com/xvivancos/barcelona-data-sets).

## Experiments
  The experiments are divided according to the class of the problems: [binary-classification](/binary-classification), [multiclass](/multiclass) and [regression](/regression).
  - **Binary Classification**
    For the task of binary classification, the problem will be classifying elements of a given set into two groups as you can see from the examples below.
    1. **Breast Cancer**
    
        In this example we use the data from the Breast Cancer dataset to build a Deep Feed Forward (DFF) neural network to determine whether a tumor is malignant or benign.
        
        For that we start off by importing the dataset using the [Pandas](https://github.com/pandas-dev/pandas) library as seen below:
        ```python
            import pandas as pd
            predictors = pd.read_csv('entradas-breast.csv')
            diagnosis = pd.read_csv('saidas-breast.csv')
        ```
        ***predictors*** are the predictor attributes, that is, the **features** that we will use **to train** our algorithm to **classify** the tumor.
        
        ***diagnosis*** contains the respective **diagnosis for each cell** from our dataset. With that we can **evaluate** the eficacy of our model when **comparing** the **outputed values** from our model with the **expected values**, during the training.
        
## FAQ
- **legendary-adventure**

   <img src="https://i.imgur.com/aW6QDKg.png" alt="legendary-adventure" width="500" height="30">
   
   - I just wanted to say how cool I found this Github feature I came across when starting this repository. 
   
