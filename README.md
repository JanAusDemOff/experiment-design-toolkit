# Experiment Design Toolkit

A comprehensive toolkit for designing, implementing, analyzing, and documenting A/B tests and other experiments in digital products. This toolkit provides a structured approach to experimentation, from hypothesis formation to decision-making.

## Features

- **Sample Size Calculator**: Determine required sample sizes for different types of experiments
- **Experiment Design Templates**: Standard templates for different experiment types
- **Randomization Tools**: Algorithms for proper user assignment with minimal bias
- **Statistical Analysis Library**: Pre-built functions for analyzing experiment results
- **Experiment Documentation**: Standardized formats for documenting experiment results
- **Bayesian Analysis**: Advanced methods for interpreting experiment outcomes beyond frequentist approaches

## Installation

```bash
pip install experiment-design-toolkit
```

## Quick Start

```python
import experiment_toolkit as etk

# Calculate required sample size
calculator = etk.SampleSizeCalculator(
    baseline_conversion_rate=0.05,
    minimum_detectable_effect=0.10,  # 10% relative change
    power=0.8,
    significance_level=0.05
)
sample_size = calculator.calculate()
print(f"Required sample size per variant: {sample_size}")

# Create an experiment design
experiment = etk.ExperimentDesign(
    name="new_checkout_flow",
    hypothesis="The new checkout flow will increase conversion rate by reducing friction",
    metrics=[
        etk.Metric("conversion_rate", "primary"),
        etk.Metric("average_order_value", "secondary")
    ],
    variants=["control", "treatment"],
    segment="all_users"
)

# Generate experiment documentation template
experiment.generate_documentation("experiment_plan.md")

# Analyze experiment results
results = etk.ExperimentResults(
    experiment=experiment,
    data="experiment_results.csv"
)
analysis = results.analyze()
print(f"P-value: {analysis.p_value}")
print(f"Confidence interval: {analysis.confidence_interval}")
print(f"Recommendation: {analysis.get_recommendation()}")
```

## Key Components

### Sample Size Calculation

The toolkit provides comprehensive sample size calculators for various metrics:

- **Conversion Rate**: Binary outcomes like purchases or signups
- **Continuous Metrics**: Revenue, time spent, number of actions
- **Count Data**: Event frequencies
- **Ratio Metrics**: Composite metrics like revenue per user

### Experiment Design

Structured approach to experiment design with:

- **Hypothesis Formation**: Templates for clear, testable hypotheses
- **Metric Selection**: Primary and secondary metrics with definitions
- **Variant Design**: Guidelines for effective variant creation
- **Targeting and Segmentation**: Tools for defining experiment audience
- **Randomization Strategies**: Methods for different experiment types

### Statistical Analysis

Robust analytical capabilities including:

- **Frequentist Analysis**: T-tests, Z-tests, ANOVA, etc.
- **Bayesian Analysis**: Posterior probability distributions
- **Sequential Analysis**: Early stopping with appropriate adjustments
- **Multiple Testing Correction**: FWER and FDR control methods
- **Segmentation Analysis**: Identify heterogeneous treatment effects

### Visualization

Ready-to-use visualization components:

- **Results Dashboard**: Overview of experiment results
- **Time Series Analysis**: Metric trends throughout the experiment
- **Distribution Comparisons**: Visual comparison of metric distributions
- **Statistical Significance Visualization**: Clear visual indicators of confidence intervals
- **Bayesian Posterior Visualization**: Probability distribution plots

## Use Cases

### Pre-Experiment Planning

Plan experiments with appropriate parameters:

```python
# Create experiment plan
plan = etk.ExperimentPlan(
    metrics=["conversion_rate", "revenue_per_user"],
    baseline_values={
        "conversion_rate": 0.15,
        "revenue_per_user": 25.50
    },
    expected_changes={
        "conversion_rate": 0.10,  # 10% relative increase
        "revenue_per_user": 0.05   # 5% relative increase
    }
)

# Calculate duration
duration = plan.calculate_duration(
    daily_users_per_variant=5000,
    power=0.8
)
print(f"Experiment should run for {duration} days")
```

### Experiment Analysis

Comprehensive analysis of experiment results:

```python
# Load experiment data
data = etk.load_data("experiment_data.csv")

# Run analysis
analyzer = etk.ExperimentAnalyzer(
    data=data,
    metric_column="value",
    variant_column="variant",
    segment_column="user_segment"
)

results = analyzer.analyze()

# Generate report
report = etk.ExperimentReport(results)
report.generate("experiment_results.html")
```

### Decision Making Framework

Structured approach to making decisions based on experiment outcomes:

```python
# Define decision criteria
decision_framework = etk.DecisionFramework(
    statistical_significance=True,
    primary_metric_improved=True,
    no_negative_impact_on_secondary=True,
    implementation_cost="low"
)

# Apply framework to results
decision = decision_framework.apply(results)
print(f"Decision: {decision.recommendation}")
print(f"Confidence: {decision.confidence}")
print(f"Reasoning: {decision.reasoning}")
```

## Documentation

Complete documentation is available at [docs.experiment-toolkit.io](https://docs.experiment-toolkit.io).

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.