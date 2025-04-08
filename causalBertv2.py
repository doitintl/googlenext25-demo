import pandas as pd
from causalnlp import CausalInferenceModel
from lightgbm import LGBMRegressor

# Load your data with columns for seasonality, marketing_spend, website_traffic, and sales
df = pd.read_csv('marketing_sales_daily_data.csv')

# 1. Impact of seasonality on marketing spend
model_seasonality_marketing = CausalInferenceModel(
    df,
    method='t-learner',
    treatment_col='is_high_season',  # Binary treatment (0/1) 
    outcome_col='marketing_spend',
    include_cols=['month', 'weekday']
)
model_seasonality_marketing.fit()

# 2. Impact of marketing spend on website traffic
model_marketing_traffic = CausalInferenceModel(
    df,
    method='t-learner',
    treatment_col='high_marketing_spend',  # Binary treatment (0/1)
    outcome_col='website_traffic',
    include_cols=['month', 'weekday', 'is_high_season']
)
model_marketing_traffic.fit()

# 3. Impact of website traffic on sales
model_traffic_sales = CausalInferenceModel(
    df,
    method='t-learner',
    treatment_col='high_website_traffic',  # Binary treatment (0/1)
    outcome_col='sales',
    include_cols=['month', 'weekday', 'is_high_season', 'high_marketing_spend']
)
model_traffic_sales.fit()

# Average Treatment Effect (ATE)
seasonality_marketing_effect = model_seasonality_marketing.estimate_ate()
print(f"Effect of high season on marketing spend: {seasonality_marketing_effect['ate']}")

# Conditional Average Treatment Effect (CATE)
holiday_effect = model_seasonality_marketing.estimate_ate(df['month'].isin([11, 12]))
print(f"Effect of high season during holidays: {holiday_effect['ate']}")

model_with_text = CausalInferenceModel(
    df,
    method='t-learner',
    treatment_col='high_marketing_spend',
    outcome_col='sales',
    text_col='campaign_description',  # Text data as a controlled-for variable
    include_cols=['month', 'seasonality']
)
model_with_text.fit()

# Interpret the model to see feature importance
feature_importance = model_with_text.interpret(plot=False)
print(feature_importance)  # Show top 10 features
