# Marketing and Sales Causal Analysis

This repository contains a comprehensive toolkit for analyzing the causal relationships between marketing investments, website traffic, and sales. It implements advanced causal inference techniques to move beyond correlation and understand the true impact of marketing efforts.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Description](#data-description)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Causal Model](#causal-model)
- [Visualizations](#visualizations)
- [Example Outputs](#example-outputs)
- [Contributing](#contributing)

## Overview

Traditional marketing analysis often relies on correlations, which can lead to misattributions when seasonality and other confounding factors aren't properly accounted for. This project implements causal inference models to accurately measure the impact of marketing spend on website traffic and subsequent sales while controlling for confounding variables like seasonality.

## Repository Structure

- `causalBertv2.py` - Implementation of causal models with text integration
- `data_generation_v2.py` - Script to generate synthetic marketing and sales data
- `load_data_spanner.py` - Script to load data into Google Cloud Spanner
- `marketing_sales_daily_data.csv` - Daily marketing and sales metrics
- `marketing_sales_monthly_data.csv` - Monthly aggregated marketing and sales metrics
- `requirements.txt` - Python dependencies

## Data Description

The dataset includes:

### Daily Data (marketing_sales_daily_data.csv):
- `date` - Date of record
- `is_high_season` - Binary indicator for high season (1/0)
- `seasonality_factor` - Numeric factor representing seasonal influence (0.3-1.0)
- `marketing_spend` - Daily marketing expenditure
- `is_high_marketing` - Binary indicator for above-median marketing spend (1/0)
- `high_marketing_spend` - Binary indicator for top 25% marketing spend (1/0)
- `website_traffic` - Daily website visitors
- `is_high_traffic` - Binary indicator for above-median traffic (1/0) 
- `high_website_traffic` - Binary indicator for top 25% traffic (1/0)
- `sales` - Daily sales revenue
- `campaign_description` - Text description of marketing campaign
- `month` - Month (1-12)
- `weekday` - Day of the week (0-6)

### Monthly Data (marketing_sales_monthly_data.csv):
- Aggregated versions of the daily metrics by month
- Includes averages and percentages for high marketing and traffic days

## Key Features

1. **Causal Inference Models**:
   - T-learner approach for estimating treatment effects
   - Controlling for confounding factors like seasonality
   - Integration of text features from campaign descriptions

2. **Multi-Stage Causal Chain Analysis**:
   - Seasonality → Marketing Spend
   - Marketing Spend → Website Traffic
   - Website Traffic → Sales

3. **Treatment Effect Estimation**:
   - Average Treatment Effect (ATE)
   - Conditional Average Treatment Effect (CATE)
   - Feature importance analysis

4. **Data Integration Options**:
   - Google Cloud Spanner database support
   - Property graph model for causal relationships

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/marketing-causal-analysis.git
cd marketing-causal-analysis

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data

```bash
python data_generation_v2.py
```

This will create:
- `marketing_sales_daily_data.csv`
- `marketing_sales_monthly_data.csv`
- Visualization plots

### Run Causal Analysis

```bash
python causalBertv2.py
```

This will:
1. Load the daily marketing data
2. Build three causal models:
   - Impact of seasonality on marketing spend
   - Impact of marketing spend on website traffic
   - Impact of website traffic on sales
3. Calculate treatment effects
4. Generate feature importance

### Load Data to Google Cloud Spanner

```bash
python load_data_spanner.py
```

Note: Requires Google Cloud authentication setup and proper permissions.

## Causal Model

The causal model implements a multi-stage analysis of the marketing and sales funnel:

1. **Seasonality → Marketing Spend**:
   - Identifies how seasonal factors influence marketing budget decisions
   - Estimates the effect of high season on spending patterns

2. **Marketing Spend → Website Traffic**:
   - Measures the causal impact of increased marketing spend on site visitors
   - Controls for seasonality to avoid confounding

3. **Website Traffic → Sales**:
   - Quantifies how increased traffic translates to revenue
   - Accounts for marketing spend and seasonality as potential confounders

## Visualizations

The data generation script creates visualizations that help understand:

1. **Monthly Trends**:
   - Sales and marketing over time
   - Seasonality factor patterns
   - High marketing and traffic indicators

2. **Correlation Plots**:
   - Marketing vs Sales
   - Website Traffic vs Sales
   - Marketing vs Traffic
   - All colored by seasonality factor

3. **Treatment Comparison**:
   - Average sales by season and marketing level
   - Causal vs correlation-based predictions

## Example Outputs

### Causal Effect Estimates

```
Effect of high season on marketing spend: 3245.62
Effect of high season during holidays: 4102.35
```

### Correlation vs Causal Predictions

```
Scenario 1: High Marketing ($8000) in Low Season (0.3)
Correlation Model Prediction: $2600
Causal Model Prediction: $1850

Scenario 2: Low Marketing ($3000) in High Season (0.8)
Correlation Model Prediction: $1600
Causal Model Prediction: $2950
```

This shows how correlation-based models can misattribute the effects of seasonality to marketing spend.
