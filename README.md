# video_game_rank_forecast
League Of Legends is one of the most popular MOBA games of today. Everyone is pushing to climb the ranks and be the best. Building a regression based model will enable me to forecast final placement, and a classification model will also enable me to analyze key factors for victory.


The API is publically accessible, so I will begin by pulling in data for individual players there, and go through cleaning. 

Cleaning will require lots of intentional categorical encoding, outlier cleaning and some standardization. 

I will also set up some engineered features with extra details, depending on the model, to assist with the task. 

For each of the tasks (Binary Classificaiton and Regression) I will test several models.

Logistic regression and Decision Tree would be interesting to test for the binary task, and then move into random forest and a neural net (Maybe MLP).

For the regression task a SVR, random forest and XGBoost would be interesting to test. 

By the completion of my investigation my goal is to have 
  (A). A richer understanding of which in game behaviors most heavily impact individual victories
  (B). Have a model that could predict accurately the rank of an individual at the end of the season (without victory signal), and see which factors most highly impact that decision. 
