# -emotion-recognition-
Machine Learning · Natural Language Processing · Affective Computing

**📖 Overview**

This project aims to predict the emotional category of songs based on their lyrics using traditional and ensemble machine learning models.
By combining lyrical text data with valence–arousal–dominance (VAD) embeddings derived from the MuSe and Genius datasets, the project explores how linguistic and emotional cues can be used to classify songs into emotion categories such as Happy/Fun, Sad/Anxious, Romantic/Loving, Calm/Reflective, and Aggressive/Enigmatic.

The work demonstrates the integration of text-based features (TF–IDF) and emotion embeddings within the framework of Russell’s Circumplex Model of Affect to improve prediction accuracy.

**🎯 Project Goals**

- Building a robust emotion recognition system using song lyrics and emotion embeddings.

- Comparing different machine learning algorithms for multi-class emotion classification.

- Contributing to applications such as mood-based playlist generation and music recommendation systems.

**🛠️ Tools Used**

. Python (pandas, numpy, scikit-learn, xgboost)

. NLP: TF–IDF Vectorization

. Visualization: matplotlib, seaborn

. Model Evaluation: GridSearchCV, confusion matrix, F1-score

. _Dataset Sources:_ MuSe Dataset, Genius API


**🧠 Methodology
_1. Data Sources**_

. MuSe Musical Sentiment Dataset:
Contains valence, arousal, and dominance scores for ~90k songs from Last.fm tags.

. Genius Song Lyrics Dataset:
Includes song lyrics, genres, artists, and metadata scraped from Genius.

. Combined Dataset:
Merged and cleaned to create a dataset of ~29,000 songs, later refined to ~19,800 samples after balancing.

_**2. Data Preprocessing**_

. Lyrics cleaning: lowercasing, stopword removal, and removal of non-lyrical text ([chorus], [intro], etc.)

. TF–IDF vectorization with n-grams (1,2)

. Emotion label encoding (6 → 5 broad emotion classes)

. Handling class imbalance by omitting multi-tag samples

_**3. Feature Engineering**_

Combined textual TF–IDF vectors with numeric emotion embeddings:

valence_tags

arousal_tags

dominance_tags

number_of_emotion_tags

_**4. Model Development**_

Models trained and evaluated:

Multinomial Naive Bayes

XGBoost Classifier

Support Vector Machine

Random Forest Classifier (final model)

Hyperparameter tuning was conducted via Grid Search and Randomized Search CV.

**📊 Results**
Model	Features Used	Accuracy
Multinomial Naive Bayes	Lyrics only	45%
XGBoost	Lyrics only	53%
SVM	Lyrics + embeddings	48%
Random Forest (Final)	Lyrics + VAD embeddings	78.5%

Evaluation Metrics (Final Model):

Accuracy: 78.5%

Precision: 79% (macro average)

Recall: 79% (macro average)

F1-Score: 79% (macro average)

The Random Forest Classifier with optimized hyperparameters and integrated VAD features achieved the best overall performance.

**📈 Key Insights**

Incorporating valence–arousal–dominance features significantly improved accuracy (+12%).

Removing overlapping categories reduced class imbalance and increased model robustness.

Traditional ML models (Random Forest, XGBoost) performed competitively compared to transformer-based models on medium datasets.

**💡 Future Work**

- Implementing BERT or XLNet embeddings for deeper semantic understanding of lyrics.

- Exploring multi-modal approaches that combine audio and lyrical features.

- Deploying the model as a web app (e.g., Streamlit) for interactive emotion prediction.

