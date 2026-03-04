🎬 Netflix Churn Prediction

Predicting user churn from behavioral viewing patterns using machine learning.

This project analyzes Netflix-style watch history data to identify behavioral signals that indicate when users are likely to churn (stop watching content). Using session-level viewing logs, we engineer engagement-based features and train machine learning models to detect churn risk.

📊 Project Overview

Streaming platforms rely heavily on sustained user engagement. When viewing activity declines, users are more likely to cancel subscriptions or stop using the platform.

This project builds a behavior-based churn prediction pipeline using Netflix-style viewing data.

Main goals:

Identify behavioral patterns associated with churn

Build user-level features from session-level viewing logs

Train machine learning models to predict churn risk

Visualize key engagement signals that influence retention

📁 Dataset

Synthetic Netflix-style dataset simulating a real streaming platform.

Dataset characteristics:

105,000+ viewing sessions

10,000 users

1,000+ movies and shows

Time period: 2024–2025

Key columns used:

user_id
watch_date
watch_duration_minutes
completion_rate
device_type
genre_primary
content_type
is_netflix_original

The dataset was cleaned and merged to produce a unified viewing dataset.

⚙️ Data Processing

The raw watch logs were processed using:

watch_processing.py

Main steps:

Remove duplicate records

Clean invalid session data

Convert timestamps

Merge movie metadata with watch history

Produce a unified viewing dataset

Output dataset:

watch_joined.csv
🧩 Feature Engineering

Session-level viewing logs were aggregated into user-level behavioral features.

Feature categories include:

Recency
recency_days

Number of days since the user's last viewing session.

Viewing Frequency
sessions_7d
sessions_14d
sessions_30d

Number of viewing sessions within recent time windows.

Viewing Volume
watch_minutes_7d
watch_minutes_14d
watch_minutes_30d

Total viewing time in recent periods.

Engagement Metrics
avg_completion_rate
completion_ratio

Measures how frequently users finish the content they start.

Content & Device Preferences

User preferences derived from viewing behavior:

genre_share_*
device_share_*

Examples:

genre_share_comedy
genre_share_action
device_share_mobile
device_share_tv
Netflix Original Engagement
original_share

Proportion of viewing sessions involving Netflix Original content.

🎯 Churn Definition

Churn is defined as behavioral inactivity.

A user is labeled as churned if they stop watching content for a given period.

Two churn horizons were created:

Horizon	Meaning
14 days	short-term churn detection
30 days	longer-term churn prediction

Generated datasets:

user_features_churn_full_h14.csv
user_features_churn_full_h30.csv
🤖 Modeling

Two machine learning models were trained:

Logistic Regression

Interpretable baseline model.

Random Forest

Tree-based ensemble model capable of capturing nonlinear relationships.

Evaluation metrics used:

ROC-AUC
F1 Score
Precision
Recall
📈 Visualizations

Several plots were generated to analyze behavioral signals related to churn.

Churn Distribution
churn_distribution_h14_h30.png

Shows the proportion of churned vs retained users.

Feature Importance
feature_importance_rf_h14.png
feature_importance_rf_h30.png

Highlights the most important features for predicting churn.

Recency vs Churn
recency_vs_churn_h14_h30.png

Shows how churn probability increases with inactivity.

Device Type vs Churn
device_vs_churn_h14.png
device_vs_churn_h30.png

Analyzes retention differences across viewing devices.

Netflix Original Viewing vs Churn
original_vs_churn_h14.png
original_vs_churn_h30.png

Examines how viewing Netflix Original content relates to churn risk.

Netflix Original Share vs Churn
original_share_deciles_vs_churn_h14.png
original_share_deciles_vs_churn_h30.png

Users grouped by Netflix Original viewing share to observe retention trends.

📂 Repository Structure
Netflix-churn-analysis
│
├ watch_processing.py
├ make_churn_dataset_full.py
├ add_genre_device_features_windowed.py
├ train_compare_h14_h30.py
├ churn_model.py
├ make_plots.py
├ make_extra_plots.py
│
├ churn_distribution_h14_h30.png
├ feature_importance_rf_h14.png
├ feature_importance_rf_h30.png
├ recency_vs_churn_h14_h30.png
├ device_vs_churn_h14.png
├ original_vs_churn_h14.png
│
└ README.md
🧠 Key Insights

Preliminary findings from the analysis:

Recency (days since last watch) is the strongest predictor of churn.

Users with declining completion rates are more likely to churn.

Viewing Netflix Original content is associated with lower churn probability.

Device usage patterns show measurable differences in retention behavior.

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
🎯 Project Goal

Demonstrate an end-to-end churn prediction pipeline:

Data Processing
→ Feature Engineering
→ Model Training
→ Evaluation
→ Visualization

This project simulates how streaming platforms can identify churn risk early and design targeted retention strategies.

License

MIT License
