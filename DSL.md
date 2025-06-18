# Domain-Specific Language Specification

**Natural Language Machine Learning Through Declarative SQL Extensions**

The DeclarativeML DSL extends SQL with machine learning primitives that read like English while compiling to optimized database operations. The language prioritizes readability and expressiveness over brevity, making ML workflows accessible to non-programmers while maintaining full power for experts.

## Design Principles

1. **Natural Language First**: Statements should read like specifications, not code
2. **Declarative Intent**: Describe what you want, not how to achieve it
3. **SQL Compatibility**: Leverage existing SQL knowledge and tooling
4. **Composable Operations**: Complex workflows built from simple, reusable components
5. **Type Safety**: Strong typing for ML primitives (models, features, datasets)

## Core Language Constructs

### Model Training

```sql
-- Basic training syntax
TRAIN MODEL model_name
  USING algorithm_spec
  FROM data_source
  PREDICT target_column
  WITH FEATURES (feature_list)
  [training_options];

-- Full example with all options
TRAIN MODEL customer_lifetime_value
  USING gradient_boosting(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
  )
  FROM customers c
  JOIN transactions t ON c.customer_id = t.customer_id
  WHERE c.signup_date > '2023-01-01'
  PREDICT lifetime_value
  WITH FEATURES (
    age,
    signup_channel,
    avg_monthly_spend,
    support_ticket_count,
    DERIVED feature_engineering.recency_frequency_monetary(t.*)
  )
  BALANCE CLASSES BY oversampling
  SPLIT DATA training=0.7, validation=0.2, test=0.1
  VALIDATE USING cross_validation(folds=5)
  OPTIMIZE FOR rmse
  STOP WHEN rmse < 100 OR epochs > 200
  SAVE CHECKPOINTS EVERY 10 epochs;
```

### Model Deployment

```sql
-- Deploy trained models
DEPLOY MODEL customer_lifetime_value
  TO ENDPOINT real_time_scoring
  WITH CONFIGURATION {
    'max_latency': '50ms',
    'throughput': '1000 rps',
    'scaling': 'auto'
  }
  MONITOR drift_detection
  VERSION CONTROL enabled;

-- Batch inference
PREDICT USING MODEL customer_lifetime_value
  FROM new_customers
  STORE RESULTS IN customer_predictions
  WITH BATCH_SIZE 1000;
```

### Feature Engineering

```sql
-- Define reusable feature transformations
CREATE FEATURE TRANSFORMER recency_frequency_monetary AS
  SELECT customer_id,
    DAYS_BETWEEN(MAX(transaction_date), CURRENT_DATE) as recency,
    COUNT(*) as frequency,
    AVG(amount) as monetary_avg,
    SUM(amount) as monetary_total
  FROM transactions
  GROUP BY customer_id;

-- Apply transformations inline
WITH FEATURES (
  age,
  signup_channel,
  TRANSFORM standard_scaler(income),
  TRANSFORM one_hot_encoder(product_category),
  TRANSFORM time_series.lag(revenue, periods=[1,7,30])
);
```

### Event-Driven Workflows

```sql
-- Define reactive workflows
WHEN MODEL fraud_detector ACCURACY > 0.95
  AND MODEL fraud_detector PRECISION > 0.90
  THEN DEPLOY TO production_endpoint
  AND ARCHIVE previous_version
  AND NOTIFY security_team('New fraud model deployed');

-- Complex conditional logic
WHEN EVENT 'data.drift_detected'
  WHERE payload->>'model_name' = 'recommendation_engine'
  AND payload->>'drift_severity' > 0.3
  THEN BEGIN
    RETRAIN MODEL recommendation_engine
      USING LATEST 90 days OF data;
    
    IF retraining_accuracy < previous_accuracy * 0.95 THEN
      ROLLBACK TO previous_version
      AND ALERT ml_team('Retraining failed - manual review needed');
    END IF;
  END;
```

### Agent Definitions

```sql
-- Create autonomous monitoring agents
CREATE AGENT performance_monitor
  FOR MODEL fraud_detector
  CHECK METRICS EVERY 1 hour
  WHEN accuracy DROPS BELOW 0.85
    OR precision DROPS BELOW 0.80  
    OR data_drift EXCEEDS 0.2
  THEN TRIGGER retraining_workflow
  AND NOTIFY on_call_engineer;

-- Hyperparameter optimization agents
CREATE AGENT hyperparameter_tuner
  FOR MODEL recommendation_engine
  OPTIMIZE USING bayesian_optimization
  SEARCH SPACE {
    'learning_rate': log_uniform(0.001, 0.1),
    'embedding_dim': choice([64, 128, 256, 512]),
    'dropout_rate': uniform(0.1, 0.5)
  }
  MAXIMIZE recall
  WITH BUDGET 50 trials
  PARALLEL EXECUTION 4 workers;
```

## Data Types and Primitives

### ML-Specific Data Types

```sql
-- Model type with metadata
CREATE TYPE ml_model AS (
  name VARCHAR(100),
  algorithm VARCHAR(50),
  version INTEGER,
  features TEXT[],
  target_column VARCHAR(100),
  performance_metrics JSONB,
  training_config JSONB,
  created_at TIMESTAMP
);

-- Feature vector with efficient storage
CREATE TYPE feature_vector AS (
  feature_names TEXT[],
  values FLOAT[],
  sparse_indices INTEGER[],  -- for sparse features
  metadata JSONB
);

-- Training dataset with lineage tracking
CREATE TYPE ml_dataset AS (
  name VARCHAR(100),
  source_query TEXT,
  feature_columns TEXT[],
  target_column VARCHAR(100),
  row_count BIGINT,
  created_from TEXT[],  -- source tables/views
  transformations JSONB,
  data_hash VARCHAR(64)  -- for change detection
);
```

### Built-in Functions

```sql
-- Model evaluation functions
SELECT evaluate_model(
  model_name := 'fraud_detector',
  test_data := 'fraud_test_set',
  metrics := ARRAY['accuracy', 'precision', 'recall', 'f1', 'auc']
);

-- Feature importance analysis
SELECT feature_importance(
  model_name := 'customer_churn',
  method := 'shap',
  sample_size := 1000
);

-- Model comparison
SELECT compare_models(
  models := ARRAY['model_v1', 'model_v2', 'model_v3'],
  test_data := 'holdout_set',
  primary_metric := 'accuracy'
);
```

## DSL Grammar and Parsing

### Core Grammar Rules

```bnf
<train_statement> ::= "TRAIN MODEL" <model_name>
                     "USING" <algorithm_spec>
                     "FROM" <data_source>
                     "PREDICT" <target_spec>
                     "WITH FEATURES" <feature_spec>
                     <training_options>*

<algorithm_spec> ::= <algorithm_name> ["(" <parameter_list> ")"]

<feature_spec> ::= "(" <feature_list> ")"

<feature_list> ::= <feature_item> ["," <feature_list>]

<feature_item> ::= <column_name>
                 | "TRANSFORM" <transformer_name> "(" <column_name> ")"
                 | "DERIVED" <function_call>

<training_options> ::= <balance_clause>
                     | <split_clause> 
                     | <validation_clause>
                     | <optimization_clause>
                     | <stopping_clause>
                     | <checkpoint_clause>
```

### Compilation Strategy

The DSL compiler transforms natural language statements into optimized SQL:

```sql
-- DSL Input
TRAIN MODEL fraud_detector
  USING logistic_regression(regularization=0.01)
  FROM transactions
  PREDICT is_fraudulent
  WITH FEATURES (amount, merchant_type, time_of_day);

-- Compiled SQL Output  
SELECT ml_train_model(
  model_name := 'fraud_detector',
  algorithm := 'logistic_regression',
  algorithm_params := '{"regularization": 0.01}',
  training_data := $$ 
    SELECT amount, merchant_type, time_of_day, is_fraudulent 
    FROM transactions 
  $$,
  target_column := 'is_fraudulent',
  feature_columns := ARRAY['amount', 'merchant_type', 'time_of_day']
);
```

## Advanced Features

### Time Series Operations

```sql
-- Time series forecasting with natural syntax
TRAIN MODEL sales_forecast
  USING prophet(
    seasonality=['yearly', 'weekly'],
    holidays=us_holidays
  )
  FROM daily_sales
  PREDICT revenue
  WITH TIME COLUMN date
  AND FEATURES (
    marketing_spend,
    weather_temperature,
    SEASONAL weekday,
    TREND linear
  )
  FORECAST 30 days AHEAD
  WITH CONFIDENCE INTERVALS 0.8, 0.95;
```

### Multi-Model Workflows

```sql
-- Ensemble learning with multiple models
CREATE ENSEMBLE recommendation_system AS
  COMBINE MODELS (
    collaborative_filtering WEIGHT 0.4,
    content_based WEIGHT 0.3,
    popularity_baseline WEIGHT 0.3
  )
  USING weighted_average
  VALIDATE ON holdout_set;

-- A/B testing multiple models
DEPLOY MODELS (model_a, model_b)
  TO ENDPOINT recommendations
  WITH TRAFFIC SPLIT (50%, 50%)
  TRACK METRICS (click_through_rate, conversion_rate)
  AUTO PROMOTE best_performer AFTER 1000 samples;
```

### Explainability and Monitoring

```sql
-- Built-in model explanation
EXPLAIN PREDICTION fraud_detector
  FOR TRANSACTION transaction_id = 12345
  USING shap_values
  SHOW TOP 5 features;

-- Continuous monitoring setup
MONITOR MODEL customer_churn
  FOR drift_detection ON features (age, tenure, monthly_spend)
  AND performance_degradation ON accuracy, f1_score  
  CHECK FREQUENCY daily
  ALERT WHEN drift_score > 0.3 OR accuracy < 0.85;
```

## Error Handling and Validation

### Compile-Time Validation

```sql
-- Type checking for features and targets
TRAIN MODEL invalid_example
  FROM customers
  PREDICT customer_name  -- ERROR: target must be numeric/categorical
  WITH FEATURES (non_existent_column);  -- ERROR: column doesn't exist

-- Algorithm parameter validation
TRAIN MODEL another_invalid
  USING linear_regression(invalid_param=true);  -- ERROR: unknown parameter
```

### Runtime Error Recovery

```sql
-- Automatic retry with degraded performance
TRAIN MODEL robust_example
  USING neural_network(layers=[1000, 500, 100])
  FROM large_dataset
  WITH FALLBACK (
    -- If OOM, try smaller network
    ON memory_error RETRY WITH layers=[100, 50, 10],
    -- If convergence fails, try different optimizer  
    ON convergence_failure RETRY WITH optimizer=adam
  )
  MAX RETRIES 3;
```

## Integration with Standard SQL

The DSL seamlessly integrates with existing SQL:

```sql
-- Use standard SQL for complex data preparation
WITH customer_features AS (
  SELECT customer_id,
         AVG(order_amount) as avg_order,
         COUNT(*) as order_frequency,
         MAX(order_date) as last_order
  FROM orders o
  JOIN customers c USING (customer_id)
  WHERE c.status = 'active'
  GROUP BY customer_id
),
enriched_features AS (
  SELECT *,
         CASE WHEN last_order > CURRENT_DATE - INTERVAL '30 days' 
              THEN 'recent' ELSE 'dormant' END as recency_segment
  FROM customer_features
)
-- Then use DSL for ML operations
TRAIN MODEL customer_segmentation
  USING kmeans(n_clusters=5)
  FROM enriched_features
  WITH FEATURES (avg_order, order_frequency, recency_segment)
  PREDICT cluster_assignment;
```

---

*The DSL bridges the gap between business intent and technical implementation, making machine learning as accessible as writing a database query.*
