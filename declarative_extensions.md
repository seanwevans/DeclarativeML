# PostgreSQL Extension Architecture

**Database-Native Machine Learning Through High-Performance Extensions**

DeclarativeML extends PostgreSQL with a comprehensive suite of ML-focused extensions that provide native data types, algorithms, and coordination primitives. All ML operations execute within the database engine, with performance-critical kernels implemented in C/Assembly/CUDA.

## Extension Architecture Overview

### Core Extension Categories

**ML Primitives Extension (`declarative_ml_core`)**
- Custom data types for tensors, models, and feature vectors
- Basic linear algebra operations and statistical functions
- Memory management for large ML objects
- Serialization/deserialization for model persistence

**Algorithm Extensions (`declarative_algorithms`)**
- Supervised learning: regression, classification, ensemble methods
- Unsupervised learning: clustering, dimensionality reduction
- Deep learning: neural networks, gradient computation
- Time series: forecasting, anomaly detection

**Coordination Extension (`declarative_coordination`)**
- Pub/sub messaging system for agent communication
- Distributed training coordination primitives
- Event handling and workflow orchestration
- Cross-instance synchronization

**DSL Compiler Extension (`declarative_dsl`)**
- Natural language SQL parsing and transformation
- Query plan optimization for ML workloads
- Custom function resolution and type checking
- Error handling and debugging support

## Data Type Extensions

### ML-Native Data Types

```c
// Core tensor type for efficient ML computations
typedef struct DeclarativeTensor {
    int32 dimensions;           // Number of dimensions
    int32 *shape;              // Shape array [dim1, dim2, ...]
    float8 *data;              // Flattened data array
    bool is_sparse;            // Sparse representation flag
    int32 *sparse_indices;     // Sparse indices (if applicable)
    TensorDType dtype;         // Data type (float32, float64, int32, etc.)
    int32 ref_count;           // Reference counting for memory management
} DeclarativeTensor;

// Model metadata and weights container
typedef struct DeclarativeModel {
    char *name;                // Model identifier
    char *algorithm;           // Algorithm type
    int32 version;             // Model version
    DeclarativeTensor *weights; // Model parameters
    char *feature_names;       // JSON array of feature names
    char *hyperparameters;     // JSON configuration
    char *training_metadata;   // Training history and metrics
    timestamp created_at;      // Creation timestamp
} DeclarativeModel;

// Feature vector with metadata
typedef struct DeclarativeFeatures {
    char **feature_names;      // Feature name array
    DeclarativeTensor *values; // Feature values
    char *metadata;            // JSON metadata
    bool is_normalized;        // Normalization flag
} DeclarativeFeatures;
```

### Type Input/Output Functions

```c
// Tensor I/O functions
Datum tensor_in(PG_FUNCTION_ARGS);
Datum tensor_out(PG_FUNCTION_ARGS);
Datum tensor_send(PG_FUNCTION_ARGS);
Datum tensor_recv(PG_FUNCTION_ARGS);

// Model serialization
Datum model_in(PG_FUNCTION_ARGS);  
Datum model_out(PG_FUNCTION_ARGS);
Datum model_serialize(PG_FUNCTION_ARGS);
Datum model_deserialize(PG_FUNCTION_ARGS);

// Feature vector operations
Datum features_create(PG_FUNCTION_ARGS);
Datum features_normalize(PG_FUNCTION_ARGS);
Datum features_select(PG_FUNCTION_ARGS);
```

## Algorithm Implementation

### Performance-Critical Kernels

```c
// Matrix multiplication with BLAS/CUDA acceleration
DeclarativeTensor* tensor_matmul(DeclarativeTensor *a, DeclarativeTensor *b) {
    // Check for CUDA availability
    if (cuda_available() && tensor_size(a) > CUDA_THRESHOLD) {
        return cuda_matmul(a, b);
    }
    
    // Fallback to optimized BLAS
    if (blas_available()) {
        return blas_matmul(a, b);
    }
    
    // Pure C implementation
    return cpu_matmul(a, b);
}

// Gradient computation for backpropagation
DeclarativeTensor* compute_gradient(DeclarativeModel *model, 
                                   DeclarativeTensor *input,
                                   DeclarativeTensor *target) {
    // Automatic differentiation implementation
    ComputationGraph *graph = build_computation_graph(model, input);
    return backward_pass(graph, target);
}

// Optimized loss functions
float8 cross_entropy_loss(DeclarativeTensor *predictions, 
                         DeclarativeTensor *targets) {
    // Numerically stable cross-entropy implementation
    // Uses log-sum-exp trick to prevent overflow
    return stable_cross_entropy(predictions, targets);
}
```

### Algorithm Integration

```c
// Training function that integrates with DSL
Datum ml_train_model(PG_FUNCTION_ARGS) {
    char *model_name = PG_GETARG_CSTRING(0);
    char *algorithm = PG_GETARG_CSTRING(1);
    char *training_query = PG_GETARG_CSTRING(2);
    ArrayType *feature_columns = PG_GETARG_ARRAYTYPE_P(3);
    char *target_column = PG_GETARG_CSTRING(4);
    
    // Execute training query to get data
    SPITupleTable *training_data = execute_training_query(training_query);
    
    // Convert to tensor format
    DeclarativeTensor *X = extract_features(training_data, feature_columns);
    DeclarativeTensor *y = extract_target(training_data, target_column);
    
    // Initialize model based on algorithm
    DeclarativeModel *model = create_model(algorithm, X->shape[1]);
    
    // Training loop
    TrainingConfig config = parse_training_config(algorithm_params);
    for (int epoch = 0; epoch < config.max_epochs; epoch++) {
        // Forward pass
        DeclarativeTensor *predictions = forward_pass(model, X);
        
        // Compute loss
        float8 loss = compute_loss(predictions, y, config.loss_function);
        
        // Backward pass and parameter update
        DeclarativeTensor *gradients = compute_gradient(model, X, y);
        update_parameters(model, gradients, config.learning_rate);
        
        // Check convergence
        if (check_convergence(loss, config.tolerance)) break;
        
        // Emit training progress event
        emit_training_event(model_name, epoch, loss);
    }
    
    // Store trained model
    store_model(model_name, model);
    
    PG_RETURN_BOOL(true);
}
```

## Coordination and Messaging

### Pub/Sub Implementation

```c
// Event publishing system
typedef struct DeclarativeEvent {
    char *event_type;          // Event category (e.g., 'model.trained')
    char *payload;             // JSON payload
    timestamp created_at;      // Event timestamp
    char *source_instance;     // Publishing instance ID
} DeclarativeEvent;

// Publish event to coordination layer
Datum publish_event(PG_FUNCTION_ARGS) {
    char *event_type = PG_GETARG_CSTRING(0);
    char *payload = PG_GETARG_CSTRING(1);
    
    DeclarativeEvent *event = create_event(event_type, payload);
    
    // Store in local event log
    store_local_event(event);
    
    // Propagate to CockroachDB coordination layer
    propagate_to_coordination_layer(event);
    
    // Notify local subscribers
    notify_local_subscribers(event);
    
    PG_RETURN_BOOL(true);
}

// Event subscription and handling
Datum subscribe_to_events(PG_FUNCTION_ARGS) {
    char *event_pattern = PG_GETARG_CSTRING(0);
    char *handler_function = PG_GETARG_CSTRING(1);
    
    // Register subscription in system catalog
    register_subscription(event_pattern, handler_function);
    
    // Set up trigger for matching events
    create_event_trigger(event_pattern, handler_function);
    
    PG_RETURN_BOOL(true);
}
```

### Cross-Instance Coordination

```c
// Distributed training coordination
typedef struct DistributedTrainingState {
    char *training_id;         // Unique training session ID
    int32 num_workers;         // Number of participating instances
    int32 current_epoch;       // Current training epoch
    DeclarativeTensor *global_gradients; // Aggregated gradients
    bool *worker_ready;        // Per-worker readiness flags
} DistributedTrainingState;

// Gradient aggregation across instances
Datum aggregate_gradients(PG_FUNCTION_ARGS) {
    char *training_id = PG_GETARG_CSTRING(0);
    DeclarativeTensor *local_gradients = PG_GETARG_TENSOR(1);
    
    DistributedTrainingState *state = get_training_state(training_id);
    
    // Add local gradients to global accumulator
    tensor_add_inplace(state->global_gradients, local_gradients);
    
    // Mark this worker as ready
    int worker_id = get_worker_id();
    state->worker_ready[worker_id] = true;
    
    // Check if all workers are ready
    if (all_workers_ready(state)) {
        // Average gradients
        tensor_scale(state->global_gradients, 1.0 / state->num_workers);
        
        // Broadcast averaged gradients to all workers
        broadcast_gradients(training_id, state->global_gradients);
        
        // Reset for next iteration
        reset_training_state(state);
    }
    
    PG_RETURN_TENSOR(state->global_gradients);
}
```

## DSL Integration

### Custom Function Resolution

```c
// DSL function registry
typedef struct DSLFunction {
    char *name;                // Function name in DSL
    char *sql_function;        // Corresponding SQL function
    int num_args;              // Number of arguments
    Oid *arg_types;           // Argument types
    bool is_aggregate;         // Aggregate function flag
} DSLFunction;

// Register DSL functions at extension load time
void register_dsl_functions(void) {
    register_dsl_function("TRAIN MODEL", "ml_train_model", 5, 
                         (Oid[]){TEXTOID, TEXTOID, TEXTOID, TEXTARRAYOID, TEXTOID},
                         false);
    
    register_dsl_function("PREDICT USING", "ml_predict", 3,
                         (Oid[]){TEXTOID, TEXTOID, TEXTOID},
                         false);
    
    register_dsl_function("EVALUATE MODEL", "ml_evaluate_model", 4,
                         (Oid[]){TEXTOID, TEXTOID, TEXTARRAYOID, JSONBOID},
                         false);
}

// DSL query transformation
Datum transform_dsl_query(PG_FUNCTION_ARGS) {
    char *dsl_query = PG_GETARG_CSTRING(0);
    
    // Parse DSL syntax tree
    DSLParseTree *tree = parse_dsl_query(dsl_query);
    
    // Transform to SQL
    StringInfo sql_query = transform_to_sql(tree);
    
    // Optimize for ML workloads
    sql_query = optimize_ml_query(sql_query);
    
    PG_RETURN_CSTRING(sql_query->data);
}
```

## Memory Management

### Large Object Handling

```c
// Custom memory context for ML objects
MemoryContext MLMemoryContext = NULL;

// Initialize ML memory management
void init_ml_memory(void) {
    MLMemoryContext = AllocSetContextCreate(TopMemoryContext,
                                           "ML Operations Context",
                                           ALLOCSET_DEFAULT_SIZES);
}

// Smart pointer system for tensors
typedef struct TensorRef {
    DeclarativeTensor *tensor;
    int32 ref_count;
    bool is_pinned;           // Prevent garbage collection
} TensorRef;

// Reference counting for automatic memory management
DeclarativeTensor* tensor_addref(DeclarativeTensor *tensor) {
    TensorRef *ref = get_tensor_ref(tensor);
    ref->ref_count++;
    return tensor;
}

void tensor_release(DeclarativeTensor *tensor) {
    TensorRef *ref = get_tensor_ref(tensor);
    ref->ref_count--;
    
    if (ref->ref_count == 0 && !ref->is_pinned) {
        // Free tensor memory
        if (tensor->data) pfree(tensor->data);
        if (tensor->sparse_indices) pfree(tensor->sparse_indices);
        if (tensor->shape) pfree(tensor->shape);
        pfree(tensor);
        pfree(ref);
    }
}
```

## Extension Installation and Configuration

### Installation SQL

```sql
-- Create extension with all components
CREATE EXTENSION declarative_ml_core CASCADE;
CREATE EXTENSION declarative_algorithms CASCADE;  
CREATE EXTENSION declarative_coordination CASCADE;
CREATE EXTENSION declarative_dsl CASCADE;

-- Configure extension parameters
SET declarative.max_tensor_size = '1GB';
SET declarative.cuda_enabled = true;
SET declarative.coordination_endpoint = 'cockroachdb://coordination-cluster:26257';
SET declarative.instance_id = 'postgres-node-1';

-- Initialize coordination tables
SELECT init_coordination_system();

-- Register this instance with coordination layer
SELECT register_postgres_instance(
    instance_id := 'postgres-node-1',
    capabilities := ARRAY['training', 'inference', 'feature_engineering'],
    max_memory := '16GB',
    gpu_available := true
);
```

### Extension Dependencies

```c
// Extension control file dependencies
#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

// Required shared libraries
void _PG_init(void) {
    // Load CUDA runtime if available
    if (cuda_runtime_available()) {
        load_cuda_functions();
    }
    
    // Initialize BLAS
    if (blas_available()) {
        init_blas_threading();
    }
    
    // Set up memory contexts
    init_ml_memory();
    
    // Register background workers for coordination
    register_coordination_workers();
    
    // Initialize event system
    init_event_system();
}
```

## Performance Optimization

### Query Plan Integration

```c
// Custom planner hook for ML operations
PlannedStmt* ml_planner_hook(Query *parse, int cursorOptions, ParamListInfo boundParams) {
    PlannedStmt *result;
    
    // Check if query contains ML operations
    if (contains_ml_operations(parse)) {
        // Apply ML-specific optimizations
        parse = optimize_ml_query_tree(parse);
        
        // Consider data locality for distributed operations
        if (is_distributed_ml_query(parse)) {
            parse = optimize_data_locality(parse);
        }
    }
    
    // Call standard planner
    if (prev_planner_hook) {
        result = prev_planner_hook(parse, cursorOptions, boundParams);
    } else {
        result = standard_planner(parse, cursorOptions, boundParams);
    }
    
    return result;
}
```

---

*Extensions transform PostgreSQL into a native machine learning platform while maintaining full compatibility with existing database operations.*