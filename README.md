# :sparkles: Legendary Adventure :sparkles:
> Simple Neural Network projects using the [Keras](https://keras.io/) library. 
<p align="center">
  <img  src="https://media.giphy.com/media/oj2GhTqAIoNIk/giphy.gif" alt="Bilbo Adventure" style="width: 480px; height: 196px; left: 0px; top: 0px; opacity: 0;">
</p>


## Description
In this repository you can find a set of examples of how to build small projects with different neural network architectures.

## Table of contents
- [Data](#data)  
- [Experiments](#experiments)
- [FAQ](#faq)
- [About](#about)

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
    
        In this example we use the data from the Breast Cancer dataset to build a **Feed Forward** (FF) neural network to **determine whether a tumor is malignant or benign**.
        
        For that we start off by importing the dataset using the [Pandas](https://github.com/pandas-dev/pandas) library as seen below:
        ```python
            import pandas as pd
            predictors = pd.read_csv('inputs-breast.csv')
            diagnosis = pd.read_csv('outputs-breast.csv')
        ```
        ***predictors*** are the predictor attributes, that is, the **features** that we will use **to train** our algorithm to **classify** the tumor.
        
        ***diagnosis*** contains the respective **diagnosis for each cell** from our dataset. With that we can **evaluate** the eficacy of our model when **comparing** the **outputed values** from our model with the **expected values**, during the training.
        
        Then we need to split our dataset between **testing and training**:
        ```python
        from sklearn.model_selection import train_test_split
        # Splits the dataset between testing and training where the training collection represents 75% of the set 
        predictors_training, predictors_test, training_diagnosis, testing_diagnosis = train_test_split(predictors, diagnosis, test_size=0.25) 
        ```
        Then we will **build our neural network** as follows:
        ```python
        import keras
        from keras.models import Sequential
        from keras.layers import Dense

        classifier = Sequential()
        # Builds the first layer of our neural network
        classifier.add(Dense(units=20,
                     activation='relu',
                     kernel_initializer='random_uniform',
                     input_dim=30,
                     use_bias=True))
        ```
        First we initialize our classifier adding up the first layer. By *Dense* we mean that our layer will be fully connected.
        Our model will have 20 neurons with 30 inputs (30 features) using the random uniform function to initialize the weights.
        You can find more information about the documentation on the [Keras](https://keras.io/) website.
        
        Then we will **add a hidden layer**:
        ```python
        # Builds a hidden layer with the configuration below
        classifier.add(Dense(units=20,
                     activation='relu',
                     kernel_initializer='random_uniform',
                     use_bias=True)) 
        ```
        Configure our optimizer and **add the output layer**:        
        ```python
        # Builds up a custom optimizer with the setup below
        optimizer = keras.optimizers.Adadelta(lr=1.0,rho=0.95, epsilon=None, decay=0.001)
        # Adds an output layer
        classifier.add(Dense(units=1, activation='sigmoid'))
        ```
        The **optimizer** is the function responsable to **reduce the error**, you can check on the Keras documentation for other options and its parameters. I've tested my own there so i think **you should try it yourself too**.
        
        In the output layer we will have **only an output**, since we will **classify as a binary** value. For that we use the activation function: **sigmoid** to output values between 0 and 1. 
        
        Then we will **compile our model**, adding up a loss function and our optimizer.
        Note that the loss function will be used to output a value that indicates how our model fits, in this case, we will aim to minimize it.  
        ```python
        # Compiles the network adding up the loss function & the custom optimizer
        classifier.compile(optimizer=optimizer,loss='binary_crossentropy')
        ```
        After this we are ready to **run our model** by fitting data into it through the following step: 
        ```python
        # Fits up the data to the network
        classifier.fit(predictors_training, training_diagnosis, batch_size=4, epochs=100)
        ```
        Where ***batch_size*** is the amount of samples from your data that will be used to train your model (*i tested increasing it to 30 and it reduced the training time tremendously*) and ***epochs*** is the number of times the model is trained over the entire dataset.
        
        Finally you can **evaluate your model** by:
        ```python
        # Evaluation:
        predictions = classifier.predict(predictors_test)
        predictions = (predictions > 0.5) # format to boolean values
        from sklearn.metrics import accuracy_score
        precision = accuracy_score(testing_diagnosis, predictions) # This evaluates the precision of our model (over the training data)
        print(precision)
        ```
        
## FAQ
- **legendary-adventure**

   <img src="https://i.imgur.com/aW6QDKg.png" alt="legendary-adventure" width="500" height="30">
   
   - I just wanted to say how cool I found this Github feature I came across when starting this repository. 
   
## About

This repository was initially created to **aggregate the content** about **neural networks** learned by me. Later on i decided to try to organize it by **documentating & anexing** the **resources** that **i am using to learn** so that **others** that wants to **get started** can **easily** do it **independely** *starting through here*.
