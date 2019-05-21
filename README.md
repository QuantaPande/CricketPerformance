# CricketPerformance
An analysis of different time series techniques to predict the performance of a cricket team in the next match

1. Dataset:
    1. Data of cricket matches between four (4) teams, India, Australia, South Africa, and England from the year 1976-2018.
    2. Training Data: 1976-2010
    3. Testing Data: 2011-2018
    4. Features: Runs scored by either team, balls used by either team, wickets taken by either team, Venue of the match and, of course, result
    5. We calculated the performance of the team based on a weighted formula:
    ![weight_equation](https://user-images.githubusercontent.com/28982129/58114886-a50a8700-7bad-11e9-9f02-cab913ea63c6.png)

2. Models used:
    1. Windowed Linear Regression: Since the data was not stationary, we could not use traditional ANOVA techniques. Hence, we use a windowed regression model with a window size of 5. We then train two different models, one with shuffled block of 5 data points each and one wit the unshuffled blocks. We obtain an MSE value of 5.13 (on an output between 1-12). For the shuffled data, we obtain an MSE of around 3.9
    2. MLP: Using the same window size, we train an MLP model. The number of layers, the number of hidden neurons, and the learning rate are the hyperparameters. The best model is 2 hidden layers, with 50 neurons each and a learning rate of 2e(-4). We obtain a MSE of 3.93 with the unshuffled dataset
    3. Random Forest: Using the same window size, we train a random forest model. The number of trees is the hyperparameter. The best model has 570 trees. The MSE on the random forest model was 3.97 on the unshuffled dataset.
    4. LSTM: Using the original dataset, we develop an single layer LSTM model with 17 hidden neurons. the MSE on the LSTM model is 2.42.
  
The Data used shows significant variance in the time average of the performance. The performance formula is not time dependent and hence, earlier matches, which were often low-scoring affairs, are marked lower. LSTM works best in identifying this long term creep in performance and hence, performs best.
