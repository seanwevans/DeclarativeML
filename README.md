# DeclarativeML

**The Database that Learns: Native Machine Learning Through Declarative SQL**

DeclarativeML is a ground-up redesign of machine learning infrastructure for the post-Moore's Law era. When datasets become too large to move and compute becomes the bottleneck, ML workloads will shift from "move data to compute" back to "move compute to data." We're building the database-native ML platform for that future.

## Vision

Traditional ML frameworks require moving massive datasets into memory, managing distributed training across clusters, and coordinating complex pipelines across multiple systems. As datasets grow exponentially while compute plateaus, this approach becomes untenable.

DeclarativeML brings machine learning directly into the database layer using a natural language DSL that extends SQL. Train models, deploy inference, and manage ML workflows using declarative statements that read like English but execute with database-level performance and reliability.

## Architecture

**Two-Tier Distributed Design:**
- **PostgreSQL Layer**: Handles blocking/synchronous operations (training iterations, local state, feature computation)
- **CockroachDB Layer**: Manages non-blocking coordination (model checkpoints, metadata, global state)
- **Pub/Sub Coordination**: Event-driven architecture with worker pools for model assembly and deployment
- **Autonomous Agents**: Database-native processes that handle convergence detection, hyperparameter tuning, and system optimization

### Interactive Architecture Visualization

Open [`visualizations/database_that_learns.html`](visualizations/database_that_learns.html) in a modern browser to explore how a
`TRAIN MODEL ...` statement travels through DeclarativeML's synchronous PostgreSQL layer, distributed CockroachDB coordination,
event-driven worker pools, and autonomous agents. The interactive cards animate the control-plane flow, plot simulated training
metrics over time, and show a hyper-parameter tuning loop progressing toward the best configuration.

## Core Principles

1. **Database-Native Everything**: All ML operations, coordination, and state management happen within the database layer
2. **Declarative DSL**: Express ML workflows in natural language that compiles to optimized database operations
3. **Performance Boundaries**: Only performance-critical kernels (matrix multiplication, CUDA operations) execute outside the database
4. **Event-Driven Coordination**: Pub/sub messaging with worker pools eliminates synchronization bottlenecks
5. **Autonomous Operation**: Database agents handle optimization, monitoring, and lifecycle management

## Example DSL Syntax

```sql
-- Train a model with natural language syntax
TRAIN MODEL fraud_detector
  USING neural_network(layers=[128, 64, 32])
  FROM transactions
  PREDICT is_fraudulent
  WITH FEATURES (amount, merchant_category, time_of_day, user_history)
  BALANCE CLASSES BY oversampling
  VALIDATE USING cross_validation(folds=5)
  OPTIMIZE FOR recall
  STOP WHEN recall > 0.90 OR epochs > 100;

-- Deploy and monitor automatically
WHEN MODEL fraud_detector CONVERGED
  DEPLOY TO real_time_scoring
  NOTIFY ops_team
  SCHEDULE retraining IN 30 days;

-- Create autonomous monitoring agents
CREATE AGENT overfitting_monitor
  CHECK MODEL fraud_detector EVERY 10 epochs
  WHEN validation_loss INCREASES FOR 3 consecutive_checks
  THEN stop_training AND rollback_to_best_checkpoint;

-- GPU compute kernels
COMPUTE add_vectors
  FROM table(foo, bar)
  INTO column(baz)
  USING vector_add BLOCK 256 GRID auto;
```

A kernel name is mandatory in the `USING` clause for all `COMPUTE` statements.

## Status

🚧 **Early Development** - Building core architecture and DSL compiler

**Current Focus:**
- Database schema design for ML primitives
- DSL parser and SQL compilation
- PostgreSQL extension framework
- Pub/sub messaging system implementation

## Getting Started


DeclarativeML is still in the conceptual and design stage. There are no
published packages or binaries yet, but the following environment is planned for
our first prototypes.

### Prerequisites

- **PostgreSQL 14+** with extension development headers (`pg_config` must be in
  your `PATH`)
- **CockroachDB** for distributed coordination (optional for local
  experimentation)
- **Rust** toolchain for building the DSL compiler
- **Python 3.9+** for running the CLI and tests

### Building and Running

The project has no runnable code today. Once the initial implementation lands
you will be able to build the PostgreSQL extensions and the DSL compiler using
`cargo build` and then load the generated libraries into your database instance.
Detailed setup instructions will be added as the repository evolves.

In the meantime, feel free to read through the design documents and open issues
to discuss ideas or questions.

DeclarativeML is under active development. The core components are evolving, but you can explore the design docs below.


## Contributing

We welcome community contributions and feedback. While the codebase is under
heavy development, the best way to participate is by opening issues to discuss
proposed features or design changes.

When code becomes available:

1. Fork the repository and create a topic branch.
2. Commit your changes with clear messages.
3. Open a pull request against `main`.
4. Include tests and documentation whenever possible.

### Running Tests

Install the Python dependencies with `pip install -r requirements.txt`. After
that, install the package in editable mode using `pip install -e .` (see
[`pyproject.toml`](pyproject.toml) for package details). Then run `pytest` from
the repository root to verify the test suite passes.

### Linting and Formatting

This project uses [pre-commit](https://pre-commit.com/) to run Black, isort and
Flake8. After installing the dependencies, install the git hook:

```bash
pre-commit install
```

Run all checks manually with:

```bash
pre-commit run --all-files
```

### CLI Usage

The repository includes a simple command line interface for compiling DSL files
into SQL. You can provide a file path or pipe DSL text via standard input:

```bash
# From a file
python -m dsl.cli path/to/model.dsl

# From stdin
echo "TRAIN MODEL example USING decision_tree FROM data PREDICT y WITH FEATURES(x)" | \
    python -m dsl.cli
```

See `AGENTS.md` for more on our autonomous approach to managing the project.

## Architecture Documents

- [`AGENTS.md`](AGENTS.md) - Autonomous agent design and coordination patterns
- [`DSL.md`](DSL.md) - Domain-specific language specification
- [`EXTENSIONS.md`](EXTENSIONS.md) - PostgreSQL extension architecture
- [`DISTRIBUTED.md`](DISTRIBUTED.md) - Two-tier coordination design

---

*"The future of machine learning is declarative, distributed, and database-native."*

## Project Repository

This is **DeclarativeML** - transforming how we think about machine learning infrastructure.
