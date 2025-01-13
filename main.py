import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets (csv files)
games = pd.read_csv("game.csv")
team_info = pd.read_csv("team_info.csv")
game_team_stats = pd.read_csv("game_teams_stats.csv")

# Merge game_team_stats with team_info to associate team names
game_team_stats = game_team_stats.merge(team_info[['team_id', 'teamName']], on='team_id', how='left')

# Rolling Averages function
# cols: Original Stats
# new_cols: Corresponding names for the rolling average columns
def rolling_averages(group, cols, new_cols):
    # Calculate rolling averages (3-game window, excluding the current game)
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    for col, new_col in zip(cols, new_cols):
        group[new_col] = rolling_stats[col]
    return group

# Columns for which rolling averages will be computed
# From game_teams_stats.csv
cols = ["goals", "shots", "hits", "pim", "powerPlayOpportunities", "powerPlayGoals",
        "faceOffWinPercentage", "giveaways", "takeaways", "blocked"]
new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages function to each team group
# Data in chronological order (sort by game_id)
game_team_stats = game_team_stats.groupby("teamName", group_keys=False).apply(
    lambda x: rolling_averages(x.sort_values("game_id"), cols, new_cols)
)

# Reset index to ensure it is sequential
game_team_stats.reset_index(drop=True, inplace=True)

# Merge rolling stats back into the `games` DataFrame for home and away teams
# Link rolling averages to the main games DataFrame for both home & away teams
games = games.merge(
    game_team_stats.add_suffix('_home'),
    how='left',
    left_on=['game_id', 'home_team_id'],
    right_on=['game_id_home', 'team_id_home']
)
games = games.merge(
    game_team_stats.add_suffix('_away'),
    how='left',
    left_on=['game_id', 'away_team_id'],
    right_on=['game_id_away', 'team_id_away']
)

# Convert date to datetime and sort by date
games["date"] = pd.to_datetime(games["date_time_GMT"])
games.sort_values(by='date', inplace=True)

# Create hour and day_code features
games["hour"] = games["date"].dt.hour
games["day_code"] = games["date"].dt.dayofweek

# Create target variable (1 for home win, 0 for away win)
games['target'] = (games['outcome'].str.contains('home win')).astype(int)

# Predictors
games['home_win_rate'] = games.groupby('home_team_id')['target'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
games['away_win_rate'] = games.groupby('away_team_id')['target'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
games['home_goal_diff'] = games['goals_rolling_home'] - games['goals_rolling_away']
games['away_goal_diff'] = games['goals_rolling_away'] - games['goals_rolling_home']
games['home_venue_win_rate'] = games.groupby('home_team_id')['target'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

# Define predictors
# These are the features used to train the Random Forest
predictors = [
    'hour', 'day_code', 
    'goals_rolling_home', 'shots_rolling_home', 'hits_rolling_home',
    'goals_rolling_away', 'shots_rolling_away', 'hits_rolling_away',
    'home_win_rate', 'away_win_rate', 'home_goal_diff', 'away_goal_diff', 'home_venue_win_rate'
]

# Check if predictors exist in the DataFrame
missing_columns = [col for col in predictors if col not in games.columns]
if missing_columns:
    print(f"Missing columns in predictors: {missing_columns}")
    predictors = [col for col in predictors if col in games.columns]

# Filter data for machine learning algorithm
# dropna (remove rows with missing values in predictors or target)
games = games.dropna(subset=predictors + ['target'])

# Train a new Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, min_samples_split=5, min_samples_leaf=2, random_state=1, class_weight='balanced')

# Time-series split for cross-validation
# Ensures training data is always from earlier games than the test data
# This mimicks real-world prediction scenarios
tscv = TimeSeriesSplit(n_splits=5)
accuracy_scores = []

for train_index, test_index in tscv.split(games):
    train = games.iloc[train_index]
    test = games.iloc[test_index]
    
    # Train the model
    rf.fit(train[predictors], train['target'])
    
    # Predictions
    preds = rf.predict(test[predictors])
    
    # Evaluate accuracy
    acc = accuracy_score(test['target'], preds)
    accuracy_scores.append(acc)

# Combine actual and predicted values for analysis
combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))

# True Positives (TP): Correctly predicted home wins.
# True Negatives (TN): Correctly predicted away wins.
# False Positives (FP): Predicted home wins when the away team won.
# False Negatives (FN): Predicted away wins when the home team won.

# Display confusion matrix and cross-tabulation
conf_matrix = confusion_matrix(combined["actual"], combined["prediction"])
print("Confusion Matrix:")
print(conf_matrix)
print("Cross-Validation Results:")
print(pd.crosstab(index=combined["actual"], columns=combined['prediction']))

# Print accuracy scores for each fold and the average accuracy
print("Cross-validation Accuracy Scores:", accuracy_scores)
print("Average Accuracy:", sum(accuracy_scores) / len(accuracy_scores))
