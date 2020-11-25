# Predicting Fantasy Football Points using Machine Learning

## Summary
In an effort to gain both experience applying machine learning techniques and a statistical advantage over my opponents in my Fantasy Football league, I've created this repository containing an analysis of fantasy football points made per player and attempts at predicting future points earned through a neural network model.

### Data Collection
Fantasy Football points can be scored differently depending on the league format and provider; I will be using Half-PPR scoring in my predictions. Currently, only data on players with the running back (RB), wide receiver (WR), and tight end (TE) position designations will be explored. 
Player data is web-scraped from the website https://www.pro-football-reference.com/ and organized per player into Pandas DataFrames. Along with the per-game player data, 
defensive data of the opposing team is collected for each game and averaged by games played in that season, up to that game week. Additionally, relevant college stats are collected and averaged per total games played. Unfortunately, not all necessary data is available from every players college career due to various reasons. College stats 
are collected differently and in less depth than NFL stats. Also, few players did not participate in a college football program and instead were signed "off the street". 

### Data Organization
The data collection process occurs for each player individually. Once a specific player is selected, all data is scraped and organized into a Pandas DataFrame. The columns
of the DataFrame include stats like game week, date, opponent, rushing yards, receiving yards, targets, touchdowns, and fantasy points earned. I will be attempting to predict fantasy points for a player's next available game using the stats from their previous games, their averages from across the season, and the defensive stats of the opponent. As such, each row contains information on a game date and data from previous games going into that game. For example, for a player's first game in their rookie year, the game information is from that first game, such as opponent, and all player stats such as rushing yards are from the player's average over the course of their college career. Another example, another row could contain the information of a game is during week 13, that has the player's stats from the previous game and their average over the course of the season, along with the defensive stats of the opponent.

### Model
The model currently consists of a 5-layer deep neural network with dropout layers between the hidden layers. The final output layer returns a single value, the predicted 
number of fantasy points earned for a player. Each player's data is trained on their own individual model. Although the amount of data varies for each player, the average number of games played is less than 100. With such limited data, a deeper model is not needed, and did not improve accuracy. The data is split into training and testing data, with the testing data being games played in the 2020-2021 season. Each hidden layer utilizes ReLU as its activation function and has sizes of 128, 256, 256, 256, and 1, respectively. 

## Case Study
In this case study, I explored the data of two players, Frank Gore and Travis Kelce. I chose these two players as they play in different positions on the offense, have a
different number of games played, and have generally done well in their career.

### Travis Kelce
Travis Kelce was drafted in the 3rd round (63rd overall) of the 2013 NFL Draft by the Kansas City Chiefs. He plays as a tight-end and has become one of the best players at the position.

Kelce's model's training data consists of 96 games worth of stats. 

![Kelce Fantasy Points](/images/logo.png)

## Conclusions
In conclusion, predicting fantasy football points can be difficult despite trends in player performance. Player performance can vary despite favorable match-ups and recent positive performances. However, the model predicts points within a reasonable range as it tries to reduce the error between actual points and predicted points from the data. 
Looking forward, I would like to continue improving the performance of the model through hypertuning and adjusting the architecture. I would also look to restructure the data 
used to train the model by creating a dataset that includes all player performances rather than individual player statistics. This would allow the model to train on more data 
and potentially create a more accurate model. 
