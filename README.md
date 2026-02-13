# Olympic Athletes: Medal Prediction

## Project Overview

This project builds a machine learning classification model to predict whether Olympic athletes will win medals based on historical data from Athens 1896 to Rio 2016.

**Dataset:** athlete_events.csv (271,116 rows × 15 columns)

**Objective:** Can we predict who will win a medal?

## Repository Structure

olympic-medal-prediction/
│
├── README.md                          # This file
├── olympic_medal_prediction.ipynb     # Main Jupyter notebook with full analysis
├── requirements.txt                    # Python dependencies
├── data/
│   └── athlete_events.csv             # Dataset (not included - download separately)
├── presentation/
│   └── Olympic_Medal_Prediction_Presentation.pptx
└── results/
    ├── figures/                        # Generated visualizations
    └── model_performance.csv           # Model comparison results

## Dataset Description

Each row represents an individual athlete competing in a specific Olympic event.

**Features:**
- `ID`: Unique athlete identifier
- `Name`: Athlete name
- `Sex`: M or F
- `Age`: Integer
- `Height`: Centimeters
- `Weight`: Kilograms
- `Team`: Team name
- `NOC`: National Olympic Committee (3-letter code)
- `Games`: Year and season
- `Year`: Integer
- `Season`: Summer or Winter
- `City`: Host city
- `Sport`: Sport name
- `Event`: Event name
- `Medal`: Gold, Silver, Bronze, or NA (**target variable**)

## Installation and Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation

1. Clone this repository:
```bash
git clone https://github.com/SwaroopManjunath/olympic-medal-prediction.git
cd olympic-medal-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Place `athlete_events.csv` in the `data/` directory
   - Dataset available at: [Kaggle - 120 Years of Olympic History](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results)

### Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## Methodology

### 1. Exploratory Data Analysis
- **Target variable analysis**: Highly imbalanced (~90% no medal)
- **Missing values**: Age (9.5%), Height (22%), Weight (22%)
- **Feature distributions**: Age, height, weight by sport and medal status
- **Temporal trends**: Participation growth, medal rates over time
- **Sport-specific patterns**: Different physical requirements and medal rates

### 2. Feature Engineering

Created 10+ new features:
- **Age groups**: Categorical age brackets
- **Experience**: Number of previous Olympic participations
- **Previous medals**: Historical medal count per athlete
- **NOC/Sport/Event medal rates**: Historical success rates
- **Team size**: Number of athletes per team in event
- **BMI & ratios**: Body composition metrics
- **Age vs sport average**: Relative age positioning

### 3. Data Preprocessing

- **Missing value imputation**: 
  - Age: Median by sport
  - Height/Weight: Median by sport and sex
- **Feature encoding**: One-hot encoding for low-cardinality features
- **Feature scaling**: StandardScaler normalization
- **Class imbalance**: SMOTE oversampling (90:10 → 50:50 in training)

### 4. Model Training

Tested 5 classification algorithms:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

**Evaluation metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

### 5. Model Selection & Tuning

- Best model selected based on F1-score (balances precision/recall)
- Hyperparameter tuning using GridSearchCV
- Cross-validation to prevent overfitting

## Results

### Model Performance

|        Model      | Accuracy | Precision |  Recall  | F1-Score |  AUC-ROC |
|-------------------|----------|-----------|----------|----------|----------|
|   Random Forest   | 0.893811 | 0.678230  | 0.525826 | 0.592383 | 0.882728 |
|      XGBoost      | 0.878061 | 0.623417  | 0.426920 | 0.506788 | 0.852183 |
| Gradient Boosting | 0.801472 | 0.388713  | 0.616313 | 0.476741 | 0.822001 |


### Top Predictive Features

1. **Event medal rate** - Historical medal distribution in specific events
2. **Sport medal rate** - Overall competitiveness of the sport
3. **Previous medals** - Athlete's historical success
4. **Experience** - Number of previous Olympic participations
5. **NOC medal rate** - Country's historical performance
6. **Age vs sport average** - Optimal age for each sport
7. **Team size** - Dynamics of team vs individual events
8. **BMI** - Sport-specific physical requirements

### Key Insights

**What predicts medal success?**
- Historical performance is the strongest indicator
- Experience matters, but with diminishing returns
- Physical attributes are highly sport-specific
- Country context provides significant signal
- Optimal age varies dramatically by sport

**Model strengths:**
- Good at identifying likely non-medal winners (high specificity)
- Captures sport-specific patterns effectively
- Robust to missing data through imputation strategy

**Model limitations:**
- Cannot predict for completely new athletes
- Better at excluding non-winners than identifying winners
- Doesn't capture intangible factors (motivation, injuries, form)

## Project Limitations

### Data Limitations
1. **Missing information**: ~22% missing physical measurements
2. **No current state**: No data on recent form, injuries, training
3. **No competition context**: Strength of field not captured
4. **Historical bias**: Past performance doesn't guarantee future results

### Model Limitations
1. **Independence assumption**: Treats each event separately
2. **Cold start problem**: Cannot predict for new athletes
3. **Feature limitations**: High-cardinality features partially captured
4. **Temporal changes**: Olympics evolve over time

### Conceptual Limitations
1. **Unobservable factors**: Motivation, luck, strategy not measured
2. **Event heterogeneity**: Team vs individual dynamics differ
3. **External factors**: Political boycotts, rule changes
4. **Generalization**: Model trained on 1896-2016 may not apply to 2020+

## Future Improvements

1. **Enhanced features:**
   - World rankings and recent competition results
   - Athlete age curves by sport
   - Head-to-head records
   - Qualifying times/scores

2. **Advanced modeling:**
   - Sport-specific models
   - Hierarchical models (sport → event → athlete)
   - Time-series analysis for form trends
   - Deep learning for sequential patterns

3. **Additional data:**
   - Training data (volume, intensity)
   - Coaching quality
   - Funding and resources
   - Weather and venue conditions

4. **Real-time integration:**
   - Live odds and predictions
   - Injury reports
   - News sentiment analysis

## Presentation

A 10-minute presentation covering:
1. Problem formulation
2. Key EDA insights
3. Feature engineering approach
4. Model selection and performance
5. Most important predictors
6. Main limitations
7. Conclusions

See `presentation/Olympic_Medal_Prediction_Presentation.pptx`

## Usage

### Running the Analysis

```bash
# Start Jupyter Notebook
jupyter notebook

# Open olympic_medal_prediction.ipynb
# Run all cells (Cell → Run All)
```

### Making Predictions

```python
# Load trained model (after running notebook)
import joblib

model = joblib.load('results/best_model.pkl')
scaler = joblib.load('results/scaler.pkl')

# Prepare new data (same features as training)
new_athlete_data = ...  # Your feature vector

# Scale features
new_athlete_scaled = scaler.transform(new_athlete_data)

# Predict
medal_probability = model.predict_proba(new_athlete_scaled)[:, 1]
medal_prediction = model.predict(new_athlete_scaled)

print(f"Medal probability: {medal_probability[0]:.2%}")
print(f"Prediction: {'Medal' if medal_prediction[0] == 1 else 'No Medal'}")
```

## Academic Integrity

This is an individual project. All code and analysis is original work. Conceptual discussions with peers were conducted in accordance with academic integrity guidelines.

## License

This project is for educational purposes only.

## Author

Swaroop Kodagahalli Manjunathaswamy  
Machine Learning Course  
February 2026

## Acknowledgments

- Dataset: [120 Years of Olympic History on Kaggle](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results)
- Olympic data originally from [Sports Reference](https://www.sports-reference.com/)

## Contact

For questions or feedback:
- Email: Swaroop.KM@stud.srh-university.de
- GitHub: [@SwaroopManjunath](https://github.com/SwaroopManjunath)

---
