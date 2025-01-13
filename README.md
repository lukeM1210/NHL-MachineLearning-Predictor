# NHL-MachineLearning-Predictor
This project uses machine learning, specifically a Random Forest Classifier, to predict the winner of NHL games. By analyzing historical game data, team statistics, and performance trends, the model predicts whether the home team will win or lose a given matchup.

Features:

Data Engineering:
Computes rolling averages for key statistics (e.g., goals, shots, hits) over recent games to capture team performance trends.
Calculates win rates, goal differentials, and venue-specific performance metrics for deeper insights.

Model:
A Random Forest Classifier aggregates predictions from multiple decision trees to ensure accurate results.
Handles class imbalance using weighted training.

Validation:
Uses time-series cross-validation to simulate real-world prediction scenarios where only past data is available for training.

Datasets:
Game Data: Information about game outcomes, scores, and venues.
Team Data: Metadata about teams, including names and IDs.
Team Stats: Game-by-game performance statistics for each team.

Technologies:
Python
Pandas
Scikit-learn

Results:
Achieved an average prediction accuracy of ~75%
Visualized model performance using confusion matrices and cross-tabulations

How It Works:

Preprocesses and merges game, team, and stats datasets.
Engineers features such as rolling averages, win rates, and goal differentials.
Trains a Random Forest Classifier to predict game outcomes based on historical patterns.
Evaluates model performance through accuracy scores and confusion matrices.
