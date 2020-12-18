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
In this case study, I explored the data of one player, Travis Kelce. I chose this player as he has generally done well in his career, and has scored a wide range of points 
throughout his career.

### Travis Kelce
Travis Kelce was drafted in the 3rd round (63rd overall) of the 2013 NFL Draft by the Kansas City Chiefs. He plays as a tight-end and has become one of the best players at the position.

Kelce's model's training data consists of 96 games worth of stats. The following graph shows his fantasy points scored for each game played in his career. 

![Kelce Fantasy Points](/Images/TK-FP.png)

Kelce's performance over the years range from 0 to upwards of 30 points in a single game. His career average is 12.2 points per game, giving him a generous floor and a very 
high ceiling in terms of fantasy points, especially at the TE position. In the 2020/2021 season, he is averaging 16.3 points per game over 10 games played so far. His performances have been steadily increasing since his rookie year, making him one of the most consistent performers at the position.

![Kelce Histogram](/Images/TK-Hist.png)

As show in the histogram, Kelce averages between 7-12 points for most games, and has roughly 6 times more games scoring more than 12 points than he does scoring less than 7 points.

![Kelce Bivariate](/Images/TK-BV.png)

Now I'd like to take a closer look at his performances against the varying performance of the defense he is facing. Typically, players are believed to have a better matchup and score more points when the opposing defense has allowed more yards to the respective position of the player. Kelce has shown to do well regardless of how well the opposing defense has played. The graph shows most of the defenses he faces average between 200-300 passing yards allowed, which range from medium to low level caliber, as a high caliber defense would allow around less than 150 passing yards per game. In these matchups, Kelce scores upwards of 10 points as expressed by the darker blue shades on the graph. 

After training the model on Kelce's data, the model predicted the following scores for each game played so far in the 2020/2021 season.

![Kelce Preds](/Images/TK-Preds.png)

You can see that the model predicted more conservatively, which is prefered as it shows the model is not swayed by outlier performances which Kelce has shown to be capable of. 

![Kelce PredsError](/Images/TK-PredError.png)

This graph shows the prediction error by points. As you can see, out of 10 games in the test data, 5 predicted scores were within +/- 4 points of the actual score, and three of those games were within less than +/- 1 point within the actual score. This is very encouraging as 50% of the games were predicted within reason, while the other 50% of games Kelce scored 6+ points than what was predicted. 


## Conclusions
In conclusion, predicting fantasy football points can be difficult despite trends in player performance. Player performance can vary despite favorable match-ups and recent positive performances. However, the model predicts points within a reasonable range as it tries to reduce the error between actual points and predicted points from the data. 
Looking forward, I would like to continue improving the performance of the model through hypertuning and adjusting the architecture. I would also look to restructure the data 
used to train the model by creating a dataset that includes all player performances rather than individual player statistics. This would allow the model to train on more data 
and potentially create a more accurate model. Overall, creating a model that accurately predicts the random-like data that comes from sports is valuable and useful for solving other problems in the world with similar data. 
