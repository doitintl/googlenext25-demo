from google.cloud import spanner_v1
from google.cloud.spanner_v1.data_types import JsonObject
from google.cloud.spanner_v1 import param_types
import pandas as pd
import json
import uuid
import os
from datetime import datetime, timezone

os.environ["GOOGLE_CLOUD_PROJECT"] = "eduardom-playground"

# Load our marketing data
df = pd.read_csv('marketing_data.csv')

# Initialize Spanner client
client = spanner_v1.Client()
instance = client.instance("graph-demo")
database = instance.database("marketing-causal-model")

# Create DDL statements for node and edge tables
create_statements = [
    """
    CREATE TABLE MarketingChannel (
      channel_id STRING(36) NOT NULL,
      name STRING(100) NOT NULL,
      type STRING(50) NOT NULL,
      description STRING(MAX),
      create_time TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
    ) PRIMARY KEY(channel_id)
    """,
    """
    CREATE TABLE MetricValue (
      metric_id STRING(36) NOT NULL,
      name STRING(100) NOT NULL,
      description STRING(MAX),
      unit STRING(50),
      create_time TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
    ) PRIMARY KEY(metric_id)
    """,
    """
    CREATE TABLE ConfoundingFactor (
      factor_id STRING(36) NOT NULL,
      name STRING(100) NOT NULL,
      description STRING(MAX),
      type STRING(50) NOT NULL,
      create_time TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
    ) PRIMARY KEY(factor_id)
    """,
    """
    CREATE TABLE TimeFrame (
      timeframe_id STRING(36) NOT NULL,
      week INT64,
      month INT64,
      year INT64 NOT NULL,
      season STRING(20)
      -- Note: No create_time needed here if TimeFrames are pre-populated or static
    ) PRIMARY KEY(timeframe_id)
    """,
    """
    CREATE TABLE ChannelInfluencesMetric (
      channel_id STRING(36) NOT NULL,
      metric_id STRING(36) NOT NULL,
      timeframe_id STRING(36), -- Nullable if influence is general, not time-specific
      causal_effect FLOAT64,
      confidence FLOAT64,
      delay_days INT64,
      source_text STRING(MAX),
      create_time TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
      -- Constraint names are optional but recommended for FKs
      --, CONSTRAINT FK_CIM_Channel FOREIGN KEY(channel_id) REFERENCES MarketingChannel(channel_id)
      --, CONSTRAINT FK_CIM_Metric FOREIGN KEY(metric_id) REFERENCES MetricValue(metric_id)
      --, CONSTRAINT FK_CIM_TimeFrame FOREIGN KEY(timeframe_id) REFERENCES TimeFrame(timeframe_id) -- Add if timeframe_id becomes NOT NULL or if desired even when nullable
    ) PRIMARY KEY(channel_id, metric_id) -- Assuming influence is unique per channel/metric pair
    """,
    # Adding FK constraints outside CREATE TABLE is also possible using ALTER TABLE
    # Example adding FKs after table creation (run these *after* all CREATE TABLEs):
    # """ALTER TABLE ChannelInfluencesMetric ADD CONSTRAINT FK_CIM_Channel FOREIGN KEY (channel_id) REFERENCES MarketingChannel (channel_id)""",
    # """ALTER TABLE ChannelInfluencesMetric ADD CONSTRAINT FK_CIM_Metric FOREIGN KEY (metric_id) REFERENCES MetricValue (metric_id)""",
    # """ALTER TABLE ChannelInfluencesMetric ADD CONSTRAINT FK_CIM_TimeFrame FOREIGN KEY (timeframe_id) REFERENCES TimeFrame (timeframe_id)""", # Only if applicable

    """
    CREATE TABLE MetricInfluencesMetric (
      source_metric_id STRING(36) NOT NULL,
      target_metric_id STRING(36) NOT NULL,
      timeframe_id STRING(36), -- Nullable if influence is general
      causal_effect FLOAT64,
      confidence FLOAT64,
      delay_days INT64,
      source_text STRING(MAX),
      create_time TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
      --, CONSTRAINT FK_MIM_SourceMetric FOREIGN KEY(source_metric_id) REFERENCES MetricValue(metric_id)
      --, CONSTRAINT FK_MIM_TargetMetric FOREIGN KEY(target_metric_id) REFERENCES MetricValue(metric_id)
      --, CONSTRAINT FK_MIM_TimeFrame FOREIGN KEY(timeframe_id) REFERENCES TimeFrame(timeframe_id)
    ) PRIMARY KEY(source_metric_id, target_metric_id)
    """,
     # Example adding FKs after table creation:
    # """ALTER TABLE MetricInfluencesMetric ADD CONSTRAINT FK_MIM_SourceMetric FOREIGN KEY (source_metric_id) REFERENCES MetricValue (metric_id)""",
    # """ALTER TABLE MetricInfluencesMetric ADD CONSTRAINT FK_MIM_TargetMetric FOREIGN KEY (target_metric_id) REFERENCES MetricValue (metric_id)""",
    # """ALTER TABLE MetricInfluencesMetric ADD CONSTRAINT FK_MIM_TimeFrame FOREIGN KEY (timeframe_id) REFERENCES TimeFrame (timeframe_id)""", # Only if applicable

    """
    CREATE TABLE FactorInfluencesMetric (
      factor_id STRING(36) NOT NULL,
      metric_id STRING(36) NOT NULL,
      timeframe_id STRING(36), -- Nullable if influence is general
      causal_effect FLOAT64,
      confidence FLOAT64,
      is_confounding BOOL,
      source_text STRING(MAX),
      create_time TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
      --, CONSTRAINT FK_FIM_Factor FOREIGN KEY(factor_id) REFERENCES ConfoundingFactor(factor_id)
      --, CONSTRAINT FK_FIM_Metric FOREIGN KEY(metric_id) REFERENCES MetricValue(metric_id)
      --, CONSTRAINT FK_FIM_TimeFrame FOREIGN KEY(timeframe_id) REFERENCES TimeFrame(timeframe_id)
    ) PRIMARY KEY(factor_id, metric_id)
    """,
    # Example adding FKs after table creation:
    # """ALTER TABLE FactorInfluencesMetric ADD CONSTRAINT FK_FIM_Factor FOREIGN KEY (factor_id) REFERENCES ConfoundingFactor (factor_id)""",
    # """ALTER TABLE FactorInfluencesMetric ADD CONSTRAINT FK_FIM_Metric FOREIGN KEY (metric_id) REFERENCES MetricValue (metric_id)""",
    # """ALTER TABLE FactorInfluencesMetric ADD CONSTRAINT FK_FIM_TimeFrame FOREIGN KEY (timeframe_id) REFERENCES TimeFrame (timeframe_id)""", # Only if applicable

    """
    CREATE TABLE SpendAllocation (
      spend_id STRING(36) NOT NULL,
      channel_id STRING(36) NOT NULL,
      timeframe_id STRING(36) NOT NULL,
      amount FLOAT64 NOT NULL,
      currency STRING(3) DEFAULT ('USD'),
      create_time TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
      --, CONSTRAINT FK_SA_Channel FOREIGN KEY(channel_id) REFERENCES MarketingChannel(channel_id)
      --, CONSTRAINT FK_SA_TimeFrame FOREIGN KEY(timeframe_id) REFERENCES TimeFrame(timeframe_id)
    ) PRIMARY KEY(spend_id)
    """,
     # Example adding FKs after table creation:
    # """ALTER TABLE SpendAllocation ADD CONSTRAINT FK_SA_Channel FOREIGN KEY (channel_id) REFERENCES MarketingChannel (channel_id)""",
    # """ALTER TABLE SpendAllocation ADD CONSTRAINT FK_SA_TimeFrame FOREIGN KEY (timeframe_id) REFERENCES TimeFrame (timeframe_id)""",

    """
    CREATE TABLE MarketingResult (
      result_id STRING(36) NOT NULL,
      metric_id STRING(36) NOT NULL,
      timeframe_id STRING(36) NOT NULL,
      value FLOAT64 NOT NULL,
      create_time TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
      --, CONSTRAINT FK_MR_Metric FOREIGN KEY(metric_id) REFERENCES MetricValue(metric_id)
      --, CONSTRAINT FK_MR_TimeFrame FOREIGN KEY(timeframe_id) REFERENCES TimeFrame(timeframe_id)
    ) PRIMARY KEY(result_id)
    """,
     # Example adding FKs after table creation:
    # """ALTER TABLE MarketingResult ADD CONSTRAINT FK_MR_Metric FOREIGN KEY (metric_id) REFERENCES MetricValue (metric_id)""",
    # """ALTER TABLE MarketingResult ADD CONSTRAINT FK_MR_TimeFrame FOREIGN KEY (timeframe_id) REFERENCES TimeFrame (timeframe_id)""",

    """
    CREATE TABLE ConfoundingEvent (
      event_id STRING(36) NOT NULL,
      factor_id STRING(36) NOT NULL,
      timeframe_id STRING(36) NOT NULL,
      magnitude FLOAT64,
      description STRING(MAX),
      create_time TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
      --, CONSTRAINT FK_CE_Factor FOREIGN KEY(factor_id) REFERENCES ConfoundingFactor(factor_id)
      --, CONSTRAINT FK_CE_TimeFrame FOREIGN KEY(timeframe_id) REFERENCES TimeFrame(timeframe_id)
    ) PRIMARY KEY(event_id)
    """
    # Example adding FKs after table creation:
    # """ALTER TABLE ConfoundingEvent ADD CONSTRAINT FK_CE_Factor FOREIGN KEY (factor_id) REFERENCES ConfoundingFactor (factor_id)""",
    # """ALTER TABLE ConfoundingEvent ADD CONSTRAINT FK_CE_TimeFrame FOREIGN KEY (timeframe_id) REFERENCES TimeFrame (timeframe_id)""",
]

# Define property graph
property_graph_def = """
CREATE PROPERTY GRAPH MarketingCausalGraph
NODE TABLES(
  MarketingChannel
    KEY(channel_id)
    LABEL Channel PROPERTIES(
      name,
      type,
      description),

  MetricValue
    KEY(metric_id)
    LABEL Metric PROPERTIES(
      name,
      description,
      unit),

  ConfoundingFactor
    KEY(factor_id)
    LABEL Confounder PROPERTIES(
      name,
      description,
      type),

  TimeFrame
    KEY(timeframe_id)
    LABEL TimeFrame PROPERTIES(
      week,
      month,
      year,
      season)
)
EDGE TABLES(
  ChannelInfluencesMetric
    KEY(channel_id, metric_id) -- Matches table PRIMARY KEY
    SOURCE KEY(channel_id) REFERENCES MarketingChannel(channel_id)
    DESTINATION KEY(metric_id) REFERENCES MetricValue(metric_id)
    LABEL Influences PROPERTIES(
      causal_effect,
      confidence,
      delay_days,
      source_text,
      timeframe_id -- Added missing property from the table
    ),

  MetricInfluencesMetric
    KEY(source_metric_id, target_metric_id) -- Matches table PRIMARY KEY
    SOURCE KEY(source_metric_id) REFERENCES MetricValue(metric_id)
    DESTINATION KEY(target_metric_id) REFERENCES MetricValue(metric_id)
    LABEL Influences PROPERTIES(
      causal_effect,
      confidence,
      delay_days,
      source_text,
      timeframe_id -- Added missing property from the table
    ),

  FactorInfluencesMetric
    KEY(factor_id, metric_id) -- Matches table PRIMARY KEY
    SOURCE KEY(factor_id) REFERENCES ConfoundingFactor(factor_id)
    DESTINATION KEY(metric_id) REFERENCES MetricValue(metric_id)
    LABEL Influences PROPERTIES(
      causal_effect,
      confidence,
      is_confounding,
      source_text,
      timeframe_id -- Added missing property from the table
    ),

  SpendAllocation
    KEY(spend_id) -- Matches table PRIMARY KEY
    SOURCE KEY(channel_id) REFERENCES MarketingChannel(channel_id)
    DESTINATION KEY(timeframe_id) REFERENCES TimeFrame(timeframe_id)
    LABEL Spends PROPERTIES(
      amount,
      currency),

  MarketingResult
    KEY(result_id) -- Matches table PRIMARY KEY
    SOURCE KEY(metric_id) REFERENCES MetricValue(metric_id)
    DESTINATION KEY(timeframe_id) REFERENCES TimeFrame(timeframe_id)
    LABEL Measures PROPERTIES(
      value),

  ConfoundingEvent
    KEY(event_id) -- Matches table PRIMARY KEY
    SOURCE KEY(factor_id) REFERENCES ConfoundingFactor(factor_id)
    DESTINATION KEY(timeframe_id) REFERENCES TimeFrame(timeframe_id)
    LABEL Occurs PROPERTIES(
      magnitude,
      description)
);
"""

# Execute schema creation only if tables don't exist
def table_exists(database, table_name):
    """Check if a table exists in the database."""
    ddl = f"""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_name = '{table_name}'
    """
    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(ddl)
        return len(list(results)) > 0

# Check if any of the tables already exist
if not table_exists(database, "MarketingChannel"):
    print("Creating tables and property graph...")
    # Create tables
    operation = database.update_ddl(create_statements)
    operation.result()  # Wait for operation to complete
    
    # Create property graph
    operation = database.update_ddl([property_graph_def])
    operation.result()  # Wait for operation to complete
    print("Tables and property graph created successfully!")
else:
    print("Tables already exist, skipping creation.")

# Helper functions to generate IDs
def generate_id():
    return str(uuid.uuid4())

# Helper function to get current time
def now():
    return datetime.now(timezone.utc)

# Function to load the causal model data
def load_marketing_causal_model(database):
    entity_ids = {}  # To keep track of created entities
    
    def transaction_callable(transaction):
        # Insert marketing channels
        channels = [
            {'id': generate_id(), 'name': 'Social Media', 'type': 'social', 'desc': 'Social media marketing across platforms'},
            {'id': generate_id(), 'name': 'Email Marketing', 'type': 'email', 'desc': 'Email newsletters and campaigns'},
            {'id': generate_id(), 'name': 'Search Ads', 'type': 'search', 'desc': 'Search engine marketing and PPC'},
            {'id': generate_id(), 'name': 'Display Ads', 'type': 'display', 'desc': 'Display and banner advertising'}
        ]
        
        for channel in channels:
            transaction.insert(
                "MarketingChannel",
                columns=["channel_id", "name", "type", "description", "create_time"],
                values=[(channel['id'], channel['name'], channel['type'], channel['desc'], now())]
            )
            entity_ids[channel['type']] = channel['id']
        
        # Insert metrics
        metrics = [
            {'id': generate_id(), 'name': 'Website Traffic', 'desc': 'Total visitors to the website', 'unit': 'visitors'},
            {'id': generate_id(), 'name': 'Sales', 'desc': 'Total product sales', 'unit': 'units'}
        ]
        
        for metric in metrics:
            transaction.insert(
                "MetricValue",
                columns=["metric_id", "name", "description", "unit", "create_time"],
                values=[(metric['id'], metric['name'], metric['desc'], metric['unit'], now())]
            )
            entity_ids[metric['name'].lower().replace(' ', '_')] = metric['id']
        
        # Insert confounding factors
        factors = [
            {'id': generate_id(), 'name': 'Seasonality', 'desc': 'Seasonal trends affecting consumer behavior', 'type': 'temporal'},
            {'id': generate_id(), 'name': 'Competitor Promotions', 'desc': 'Promotional activities by competitors', 'type': 'external'},
            {'id': generate_id(), 'name': 'Product Launches', 'desc': 'New product releases', 'type': 'internal'}
        ]
        
        for factor in factors:
            transaction.insert(
                "ConfoundingFactor",
                columns=["factor_id", "name", "description", "type", "create_time"],
                values=[(factor['id'], factor['name'], factor['desc'], factor['type'], now())]
            )
            entity_ids[factor['name'].lower().replace(' ', '_')] = factor['id']
        
        # Create causal relationships between channels and traffic
        channel_influences = [
            {'channel': 'social', 'metric': 'website_traffic', 'effect': 0.42, 'confidence': 0.85, 'delay': 2},
            {'channel': 'email', 'metric': 'website_traffic', 'effect': 0.95, 'confidence': 0.92, 'delay': 0},
            {'channel': 'search', 'metric': 'website_traffic', 'effect': 0.68, 'confidence': 0.78, 'delay': 0},
            {'channel': 'display', 'metric': 'website_traffic', 'effect': 0.25, 'confidence': 0.65, 'delay': 1}
        ]
        
        for influence in channel_influences:
            transaction.insert(
                "ChannelInfluencesMetric",
                columns=["channel_id", "metric_id", "causal_effect", "confidence", "delay_days", "source_text", "create_time"],
                values=[(
                    entity_ids[influence['channel']], 
                    entity_ids[influence['metric']], 
                    influence['effect'],
                    influence['confidence'],
                    influence['delay'],
                    f"{influence['channel']} influences {influence['metric']} with effect {influence['effect']}",
                    now()
                )]
            )
        
        # Create traffic -> sales relationship
        transaction.insert(
            "MetricInfluencesMetric",
            columns=["source_metric_id", "target_metric_id", "causal_effect", "confidence", "delay_days", "source_text", "create_time"],
            values=[(
                entity_ids['website_traffic'],
                entity_ids['sales'],
                0.032,  # conversion rate
                0.89,   # confidence
                0,      # same day
                "Website traffic converts to sales at a rate of 3.2%",
                now()
            )]
        )
        
        # Create confounding factor relationships
        factor_influences = [
            {'factor': 'seasonality', 'metric': 'website_traffic', 'effect': 0.25, 'confidence': 0.82, 'confounding': True},
            {'factor': 'seasonality', 'metric': 'sales', 'effect': 0.35, 'confidence': 0.75, 'confounding': True},
            {'factor': 'competitor_promotions', 'metric': 'website_traffic', 'effect': -0.15, 'confidence': 0.68, 'confounding': True},
            {'factor': 'product_launches', 'metric': 'website_traffic', 'effect': 0.65, 'confidence': 0.91, 'confounding': True}
        ]
        
        for influence in factor_influences:
            transaction.insert(
                "FactorInfluencesMetric",
                columns=["factor_id", "metric_id", "causal_effect", "confidence", "is_confounding", "source_text", "create_time"],
                values=[(
                    entity_ids[influence['factor']],
                    entity_ids[influence['metric']],
                    influence['effect'],
                    influence['confidence'],
                    influence['confounding'],
                    f"{influence['factor']} influences {influence['metric']} with effect {influence['effect']}",
                    now()
                )]
            )
    
    database.run_in_transaction(transaction_callable)
    return entity_ids

# Load the causal model data
print("Loading causal model data...")
entity_ids = load_marketing_causal_model(database)
print("Causal model loaded successfully!")

# Now let's load the actual marketing data grouped by week
def load_time_series_data(database, df, entity_ids):
    # Create weekly time frames first
    timeframes = {}
    
    def create_timeframes_transaction(transaction):
        weekly_data = df.copy()
        weekly_data['date'] = pd.to_datetime(weekly_data['date'])
        weekly_data['week'] = weekly_data['date'].dt.isocalendar().week
        weekly_data['month'] = weekly_data['date'].dt.month
        weekly_data['year'] = weekly_data['date'].dt.year
        
        # Get unique weeks
        unique_weeks = weekly_data[['week', 'month', 'year']].drop_duplicates()
        
        for _, row in unique_weeks.iterrows():
            timeframe_id = generate_id()
            transaction.insert(
                "TimeFrame",
                columns=["timeframe_id", "week", "month", "year", "season"],
                values=[(
                    timeframe_id,
                    int(row['week']),
                    int(row['month']),
                    int(row['year']),
                    'winter' if row['month'] in [12, 1, 2] else
                    'spring' if row['month'] in [3, 4, 5] else
                    'summer' if row['month'] in [6, 7, 8] else 'fall'
                )]
            )
            timeframes[(int(row['year']), int(row['week']))] = timeframe_id
    
    database.run_in_transaction(create_timeframes_transaction)
    print(f"Created {len(timeframes)} time frames")
    
    # Now aggregate data by week and load spend and results
    def load_weekly_data_transaction(transaction):
        weekly_data = df.copy()
        weekly_data['date'] = pd.to_datetime(weekly_data['date'])
        weekly_data['week'] = weekly_data['date'].dt.isocalendar().week
        weekly_data['year'] = weekly_data['date'].dt.year
        
        # Group by week and year
        aggregated = weekly_data.groupby(['year', 'week']).agg({
            'social_spend': 'sum',
            'email_spend': 'sum',
            'search_spend': 'sum',
            'display_spend': 'sum',
            'traffic': 'sum',
            'sales': 'sum',
            'season_factor': 'mean',
            'competitor_promotion': 'sum'
        }).reset_index()
        
        for _, row in aggregated.iterrows():
            timeframe_id = timeframes[(int(row['year']), int(row['week']))]
            
            # Insert spend allocations
            spend_data = [
                {'channel': 'social', 'amount': float(row['social_spend'])},
                {'channel': 'email', 'amount': float(row['email_spend'])},
                {'channel': 'search', 'amount': float(row['search_spend'])},
                {'channel': 'display', 'amount': float(row['display_spend'])}
            ]
            
            for spend in spend_data:
                spend_id = generate_id()
                transaction.insert(
                    "SpendAllocation",
                    columns=["spend_id", "channel_id", "timeframe_id", "amount", "currency", "create_time"],
                    values=[(
                        spend_id,
                        entity_ids[spend['channel']],
                        timeframe_id,
                        spend['amount'],
                        'USD',
                        now()
                    )]
                )
            
            # Insert results
            result_data = [
                {'metric': 'website_traffic', 'value': float(row['traffic'])},
                {'metric': 'sales', 'value': float(row['sales'])}
            ]
            
            for result in result_data:
                result_id = generate_id()
                transaction.insert(
                    "MarketingResult",
                    columns=["result_id", "metric_id", "timeframe_id", "value", "create_time"],
                    values=[(
                        result_id,
                        entity_ids[result['metric']],
                        timeframe_id,
                        result['value'],
                        now()
                    )]
                )
            
            # Insert confounding events if applicable
            if float(row['season_factor']) > 1.1:
                event_id = generate_id()
                transaction.insert(
                    "ConfoundingEvent",
                    columns=["event_id", "factor_id", "timeframe_id", "magnitude", "description", "create_time"],
                    values=[(
                        event_id,
                        entity_ids['seasonality'],
                        timeframe_id,
                        float(row['season_factor']),
                        f"Seasonal factor of {row['season_factor']:.2f} for week {int(row['week'])}",
                        now()
                    )]
                )
            
            if int(row['competitor_promotion']) > 0:
                event_id = generate_id()
                transaction.insert(
                    "ConfoundingEvent",
                    columns=["event_id", "factor_id", "timeframe_id", "magnitude", "description", "create_time"],
                    values=[(
                        event_id,
                        entity_ids['competitor_promotions'],
                        timeframe_id,
                        float(row['competitor_promotion']),
                        f"{int(row['competitor_promotion'])} competitor promotions in week {int(row['week'])}",
                        now()
                    )]
                )
    
    database.run_in_transaction(load_weekly_data_transaction)
    print("Weekly data loaded successfully!")

# Load the time series data
print("Loading time series data...")
load_time_series_data(database, df, entity_ids)

print("Causal marketing data model successfully loaded into Spanner Graph!")

# Sample query to validate the data
def run_sample_query(database):
    print("\nRunning sample validation query:")
    query = """
    SELECT 
      mc.name AS channel, 
      cim.causal_effect, 
      cim.confidence,
      cim.delay_days
    FROM MarketingChannel mc
    JOIN ChannelInfluencesMetric cim ON mc.channel_id = cim.channel_id
    JOIN MetricValue mv ON cim.metric_id = mv.metric_id
    WHERE mv.name = 'Website Traffic'
    ORDER BY cim.causal_effect * cim.confidence DESC
    """
    
    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(query)
        print("Channels ranked by causal effect on website traffic:")
        for row in results:
            print(f"  {row[0]}: Effect = {row[1]:.2f}, Confidence = {row[2]:.2f}, Delay = {row[3]} days")

# Run a sample query to validate the data
run_sample_query(database)