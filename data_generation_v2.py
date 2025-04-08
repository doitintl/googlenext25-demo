import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import random
from scipy.stats import pearsonr

# Set seed for reproducibility
np.random.seed(42)

# Generate a dataset with causal relationships
NUM_RECORDS = 730  # 2 years of daily data
start_date = datetime(2023, 1, 1)
data = []

# Define seasonality patterns (higher in summer and December)
def get_seasonality_factor(date):
    month = date.month - 1  # 0-11
    # Higher in summer (May-Aug) and December
    if month >= 4 and month <= 7:
        return 0.7 + (random.random() * 0.3)
    if month == 11:  # December
        return 0.7 + (random.random() * 0.3)
    return 0.3 + (random.random() * 0.4)  # Lower rest of year

# Function to determine if it's high season
def is_high_season(date):
    month = date.month - 1
    # High season: May-August and December
    return 1 if (month >= 4 and month <= 7) or month == 11 else 0

# Define a realistic pattern where marketing affects traffic with some delay and noise
def calculate_traffic(marketing_spend, seasonality_factor, date):
    # Base traffic depending on seasonality
    base_traffic = 1000 + (seasonality_factor * 1500)
    
    # Marketing impact with diminishing returns
    marketing_impact = 3 * math.sqrt(marketing_spend)
    
    # Weekend boost
    day = date.weekday()
    weekend_boost = 300 if day >= 5 else 0  # 5, 6 = Saturday, Sunday
    
    # Random variation (noise)
    noise = random.random() * 400 - 200
    
    return round(base_traffic + marketing_impact + weekend_boost + noise)

# Define how traffic and seasonality affect sales
def calculate_sales(traffic, seasonality_factor, date):
    # Conversion rate varies with seasonality
    conversion_rate = 0.02 + (seasonality_factor * 0.015)
    
    # Each visitor spends on average between $30-50 depending on seasonality
    avg_purchase = 30 + (seasonality_factor * 20)
    
    # Random variation (some days have higher/lower average purchases)
    purchase_noise = random.random() * 10 - 5
    
    # Calculate sales based on traffic, conversion rate, and average purchase
    return round(traffic * conversion_rate * (avg_purchase + purchase_noise))

# Generate campaign descriptions
campaigns = [
    "Standard Display Ads",
    "Summer Sale Promotion",
    "Back to School Campaign",
    "Holiday Season Special",
    "Social Media Boost",
    "Email Newsletter Campaign",
    "Product Launch",
    "Influencer Partnership",
    "Brand Awareness",
    "Customer Retention Program"
]

# Generate data
for i in range(NUM_RECORDS):
    current_date = start_date + timedelta(days=i)
    
    # Format date
    date_string = current_date.strftime('%Y-%m-%d')
    
    # Determine seasonality
    seasonality_factor = get_seasonality_factor(current_date)
    high_season = is_high_season(current_date)
    
    # Determine marketing spend based on season and some strategy
    # Companies often spend more in high season
    if high_season:
        marketing_spend = 5000 + (random.random() * 3000)
    else:
        marketing_spend = 2000 + (random.random() * 2000)
    
    # Add some campaign spikes regardless of season (product launches, etc.)
    if random.random() < 0.1:
        marketing_spend += 2000 + (random.random() * 4000)
    
    # Round marketing spend
    marketing_spend = round(marketing_spend)
    
    # Calculate website traffic
    website_traffic = calculate_traffic(marketing_spend, seasonality_factor, current_date)
    
    # Calculate sales
    sales = calculate_sales(website_traffic, seasonality_factor, current_date)
    
    # Placeholders for binary indicators (will set after calculating thresholds)
    is_high_marketing = None
    is_high_traffic = None
    high_marketing_spend = None
    high_website_traffic = None
    
    # Select a campaign description
    campaign_index = math.floor(random.random() * len(campaigns))
    campaign_description = campaigns[campaign_index]
    
    # Generate record
    data.append({
        'date': date_string,
        'is_high_season': high_season,
        'seasonality_factor': round(seasonality_factor, 2),
        'marketing_spend': marketing_spend,
        'is_high_marketing': is_high_marketing,  # Will set after calculating median
        'high_marketing_spend': high_marketing_spend,  # Will set after calculating threshold
        'website_traffic': website_traffic,
        'is_high_traffic': is_high_traffic,  # Will set after calculating median
        'high_website_traffic': high_website_traffic,  # Will set after calculating threshold
        'sales': sales,
        'campaign_description': campaign_description,
        'month': current_date.month,  # 1-12
        'weekday': current_date.weekday()  # 0-6
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate thresholds for binary indicators
median_marketing = df['marketing_spend'].median()
median_traffic = df['website_traffic'].median()
marketing_high_threshold = df['marketing_spend'].quantile(0.75)
traffic_high_threshold = df['website_traffic'].quantile(0.75)

# Update records with binary indicators
df['is_high_marketing'] = (df['marketing_spend'] >= median_marketing).astype(int)
df['is_high_traffic'] = (df['website_traffic'] >= median_traffic).astype(int)
df['high_marketing_spend'] = (df['marketing_spend'] >= marketing_high_threshold).astype(int)
df['high_website_traffic'] = (df['website_traffic'] >= traffic_high_threshold).astype(int)

# Display thresholds
print(f"Median marketing spend: ${median_marketing} (threshold for is_high_marketing)")
print(f"75th percentile marketing spend: ${marketing_high_threshold} (threshold for high_marketing_spend)")
print(f"Median website traffic: {median_traffic} (threshold for is_high_traffic)")
print(f"75th percentile website traffic: {traffic_high_threshold} (threshold for high_website_traffic)")

# Calculate summary statistics
print("\nDataset Summary Statistics:")
print(f"Total Records: {len(df)}")
print(f"Average Daily Sales: ${round(df['sales'].mean())}")
print(f"Average Website Traffic: {round(df['website_traffic'].mean())} visits")
print(f"Average Marketing Spend: ${round(df['marketing_spend'].mean())}")
print(f"Average Sales in High Season: ${round(df[df['is_high_season'] == 1]['sales'].mean())}")
print(f"Average Sales in Low Season: ${round(df[df['is_high_season'] == 0]['sales'].mean())}")
print(f"Average Sales with High Marketing (above median): ${round(df[df['is_high_marketing'] == 1]['sales'].mean())}")
print(f"Average Sales with Low Marketing (below median): ${round(df[df['is_high_marketing'] == 0]['sales'].mean())}")
print(f"Average Sales with Very High Marketing (top 25%): ${round(df[df['high_marketing_spend'] == 1]['sales'].mean())}")
print(f"Average Sales with High Website Traffic (above median): ${round(df[df['is_high_traffic'] == 1]['sales'].mean())}")
print(f"Average Sales with Very High Website Traffic (top 25%): ${round(df[df['high_website_traffic'] == 1]['sales'].mean())}")

# Calculate correlations
marketing_sales_corr = df['marketing_spend'].corr(df['sales'])
traffic_sales_corr = df['website_traffic'].corr(df['sales'])
marketing_traffic_corr = df['marketing_spend'].corr(df['website_traffic'])

print(f"\nCorrelation between Marketing Spend and Sales: {marketing_sales_corr:.3f}")
print(f"Correlation between Website Traffic and Sales: {traffic_sales_corr:.3f}")
print(f"Correlation between Marketing Spend and Traffic: {marketing_traffic_corr:.3f}")

# Display a sample of the data including our binary columns
print("\nSample of records with binary indicators:")
cols_to_display = ['date', 'marketing_spend', 'is_high_marketing', 'high_marketing_spend', 
                   'website_traffic', 'is_high_traffic', 'high_website_traffic', 'sales', 'is_high_season']
print(df[cols_to_display].head(10).to_string())

# Create aggregated monthly data
monthly_data = df.copy()
monthly_data['month_year'] = pd.to_datetime(monthly_data['date']).dt.strftime('%Y-%m')

monthly_aggregated = monthly_data.groupby('month_year').agg(
    avg_sales=('sales', 'mean'),
    avg_traffic=('website_traffic', 'mean'),
    avg_marketing=('marketing_spend', 'mean'),
    avg_seasonality=('seasonality_factor', 'mean'),
    high_marketing_pct=('is_high_marketing', 'mean'),
    very_high_marketing_pct=('high_marketing_spend', 'mean'),
    high_traffic_pct=('is_high_traffic', 'mean'),
    very_high_traffic_pct=('high_website_traffic', 'mean')
).reset_index()

monthly_aggregated = monthly_aggregated.round({
    'avg_sales': 0, 
    'avg_traffic': 0, 
    'avg_marketing': 0, 
    'avg_seasonality': 2,
    'high_marketing_pct': 2,
    'very_high_marketing_pct': 2,
    'high_traffic_pct': 2,
    'very_high_traffic_pct': 2
})

print("\nMonthly Aggregated Data (first 5 months):")
print(monthly_aggregated.head().to_string())

# Save data to CSV files for use in the presentation
df.to_csv('marketing_sales_daily_data.csv', index=False)
monthly_aggregated.to_csv('marketing_sales_monthly_data.csv', index=False)

print("\nCSV files saved with all requested binary indicator columns")

# Create prediction models for demonstration
def predict_sales_correlation(marketing_spend):
    # Simple linear regression coefficients
    m = 0.2  # slope
    b = 1000  # intercept
    return m * marketing_spend + b

def predict_sales_causal(marketing_spend, seasonality_factor):
    # Causal model accounting for seasonality
    base_effect = 800
    marketing_effect = 0.15 * marketing_spend
    seasonality_effect = 3000 * seasonality_factor
    return base_effect + marketing_effect + seasonality_effect

# Compare predictions for demo scenarios
print("\nComparison of Correlation vs. Causal Predictions:")

# Scenario 1: High marketing in low season
scenario1_marketing = 8000
scenario1_seasonality = 0.3
print(f"Scenario 1: High Marketing (${scenario1_marketing}) in Low Season ({scenario1_seasonality})")
print(f"Correlation Model Prediction: ${round(predict_sales_correlation(scenario1_marketing))}")
print(f"Causal Model Prediction: ${round(predict_sales_causal(scenario1_marketing, scenario1_seasonality))}")

# Scenario 2: Low marketing in high season
scenario2_marketing = 3000
scenario2_seasonality = 0.8
print(f"\nScenario 2: Low Marketing (${scenario2_marketing}) in High Season ({scenario2_seasonality})")
print(f"Correlation Model Prediction: ${round(predict_sales_correlation(scenario2_marketing))}")
print(f"Causal Model Prediction: ${round(predict_sales_causal(scenario2_marketing, scenario2_seasonality))}")

# Create visualizations for the presentation
plt.figure(figsize=(12, 12))

# Plot 1: Sales and Marketing Trends
ax1 = plt.subplot(411)
ax1.plot(monthly_aggregated['month_year'], monthly_aggregated['avg_sales'], 'b-', label='Avg. Sales')
ax1.set_ylabel('Sales ($)', color='b')
ax1.tick_params('y', colors='b')
ax1.set_title('Monthly Sales and Marketing Trends')
plt.xticks(rotation=45)

# Twin axis for marketing spend
ax2 = ax1.twinx()
ax2.plot(monthly_aggregated['month_year'], monthly_aggregated['avg_marketing'], 'r-', label='Avg. Marketing Spend')
ax2.set_ylabel('Marketing Spend ($)', color='r')
ax2.tick_params('y', colors='r')

# Plot 2: Seasonality Factor
ax3 = plt.subplot(412, sharex=ax1)
ax3.bar(monthly_aggregated['month_year'], monthly_aggregated['avg_seasonality'], alpha=0.7, color='g', label='Seasonality Factor')
ax3.set_ylabel('Seasonality Factor')
ax3.set_title('Seasonality Factor by Month')
ax3.set_ylim(0, 1.0)
plt.xticks(rotation=45)

# Plot 3: High Marketing Indicators - Fixed error here by using proper color formatting
ax4 = plt.subplot(413, sharex=ax1)
ax4.plot(monthly_aggregated['month_year'], monthly_aggregated['high_marketing_pct'] * 100, 'purple', label='% Days with High Marketing')
ax4.plot(monthly_aggregated['month_year'], monthly_aggregated['very_high_marketing_pct'] * 100, 'orange', label='% Days with Very High Marketing')
ax4.set_ylabel('Percentage')
ax4.set_title('Marketing Spend Indicators by Month')
ax4.set_ylim(0, 100)
ax4.legend()
plt.xticks(rotation=45)

# Plot 4: High Traffic Indicators - Fixed error here by using proper color formatting
ax5 = plt.subplot(414, sharex=ax1)
ax5.plot(monthly_aggregated['month_year'], monthly_aggregated['high_traffic_pct'] * 100, 'c', label='% Days with High Traffic')
ax5.plot(monthly_aggregated['month_year'], monthly_aggregated['very_high_traffic_pct'] * 100, 'm', label='% Days with Very High Traffic')
ax5.set_ylabel('Percentage')
ax5.set_title('Website Traffic Indicators by Month')
ax5.set_ylim(0, 100)
ax5.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('monthly_trends_all_indicators.png')

# Create a scatter plot matrix to show relationships
plt.figure(figsize=(16, 12))

# Marketing spend vs Sales, colored by seasonality
plt.subplot(221)
scatter = plt.scatter(df['marketing_spend'], df['sales'], 
                     c=df['seasonality_factor'], cmap='viridis', 
                     alpha=0.6)
plt.colorbar(scatter, label='Seasonality Factor')
plt.title(f'Marketing vs Sales\nCorrelation: {marketing_sales_corr:.3f}')
plt.xlabel('Marketing Spend ($)')
plt.ylabel('Sales ($)')

# Website traffic vs Sales, colored by seasonality
plt.subplot(222)
scatter = plt.scatter(df['website_traffic'], df['sales'], 
                     c=df['seasonality_factor'], cmap='viridis', 
                     alpha=0.6)
plt.colorbar(scatter, label='Seasonality Factor')
plt.title(f'Website Traffic vs Sales\nCorrelation: {traffic_sales_corr:.3f}')
plt.xlabel('Website Traffic (visits)')
plt.ylabel('Sales ($)')

# Marketing vs Traffic, colored by seasonality
plt.subplot(223)
scatter = plt.scatter(df['marketing_spend'], df['website_traffic'], 
                     c=df['seasonality_factor'], cmap='viridis', 
                     alpha=0.6)
plt.colorbar(scatter, label='Seasonality Factor')
plt.title(f'Marketing vs Traffic\nCorrelation: {marketing_traffic_corr:.3f}')
plt.xlabel('Marketing Spend ($)')
plt.ylabel('Website Traffic (visits)')

# Sales by Season and Marketing Level
plt.subplot(224)
df_grouped = df.groupby(['is_high_season', 'high_marketing_spend'])['sales'].mean().reset_index()
df_pivot = df_grouped.pivot(index='is_high_season', columns='high_marketing_spend', values='sales')
df_pivot.plot(kind='bar', ax=plt.gca())
plt.title('Average Sales by Season and Marketing Level')
plt.xlabel('High Season (1=Yes, 0=No)')
plt.ylabel('Average Sales ($)')
plt.legend(['Low Marketing', 'High Marketing'])

plt.tight_layout()
plt.savefig('correlation_plots_with_seasonality.png')

print("\nData generation, analysis, and visualization complete. All files have been saved.")