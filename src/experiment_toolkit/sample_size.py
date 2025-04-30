#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample Size Calculator Module

This module provides tools for calculating the required sample size for
different types of experiments, ensuring statistical validity.
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


class MetricType(Enum):
    """Types of metrics that can be measured in experiments."""
    BINARY = "binary"  # Conversion rates, success/failure
    CONTINUOUS = "continuous"  # Revenue, time spent
    COUNT = "count"  # Number of actions/events
    RATIO = "ratio"  # Composite metrics like revenue per user


class VariantAllocation(Enum):
    """Allocation strategies for experiment variants."""
    EQUAL = "equal"  # Equal distribution across variants
    CUSTOM = "custom"  # Custom allocation percentages
    MULTI_ARM_BANDIT = "multi_arm_bandit"  # Dynamic allocation


class SampleSizeCalculator:
    """
    Calculator for determining required sample sizes in experiments.
    
    This class provides methods to calculate the required sample size for
    different metric types, ensuring experiments have sufficient statistical power.
    """
    
    def __init__(self, 
                 baseline_value: float,
                 minimum_detectable_effect: float,
                 metric_type: Union[MetricType, str] = MetricType.BINARY,
                 power: float = 0.8,
                 significance_level: float = 0.05,
                 variance: Optional[float] = None,
                 variant_count: int = 2,
                 allocation_ratio: Optional[List[float]] = None):
        """
        Initialize the sample size calculator.
        
        Args:
            baseline_value: The current value of the metric (e.g., conversion rate)
            minimum_detectable_effect: The smallest relative change you want to detect
                (e.g., 0.1 for a 10% relative change)
            metric_type: Type of metric being measured (binary, continuous, etc.)
            power: Desired statistical power (1 - β), typically 0.8 or 0.9
            significance_level: Significance level (α), typically 0.05
            variance: For continuous metrics, the variance of the metric
                (if None, will be estimated for binary metrics)
            variant_count: Number of variants including control
            allocation_ratio: Custom allocation ratio between variants
                (must sum to 1, defaults to equal allocation)
        """
        self.baseline_value = baseline_value
        self.minimum_detectable_effect = minimum_detectable_effect
        
        # Handle string input for metric type
        if isinstance(metric_type, str):
            self.metric_type = MetricType(metric_type)
        else:
            self.metric_type = metric_type
            
        self.power = power
        self.significance_level = significance_level
        self.variance = variance
        self.variant_count = variant_count
        
        # Validate and set allocation ratio
        if allocation_ratio is None:
            # Equal allocation by default
            self.allocation_ratio = [1.0 / variant_count] * variant_count
        else:
            if len(allocation_ratio) != variant_count:
                raise ValueError(f"Allocation ratio must have {variant_count} values")
            if abs(sum(allocation_ratio) - 1.0) > 1e-10:
                raise ValueError("Allocation ratio must sum to 1")
            self.allocation_ratio = allocation_ratio
    
    def calculate(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Calculate the required sample size based on the provided parameters.
        
        Returns:
            Dictionary containing sample size information including:
            - total_sample_size: Total number of users needed
            - per_variant: Users needed per variant
            - absolute_effect: The absolute change in the metric
            - relative_effect: The relative change as a proportion
            - parameters: The input parameters used for calculation
        """
        if self.metric_type == MetricType.BINARY:
            sample_size = self._calculate_binary()
        elif self.metric_type == MetricType.CONTINUOUS:
            if self.variance is None:
                raise ValueError("Variance must be provided for continuous metrics")
            sample_size = self._calculate_continuous()
        elif self.metric_type == MetricType.COUNT:
            sample_size = self._calculate_count()
        elif self.metric_type == MetricType.RATIO:
            if self.variance is None:
                raise ValueError("Variance must be provided for ratio metrics")
            sample_size = self._calculate_ratio()
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
        
        # Calculate per-variant sample sizes based on allocation ratio
        per_variant = {f"variant_{i}": int(math.ceil(sample_size * ratio)) 
                      for i, ratio in enumerate(self.allocation_ratio)}
        
        # Calculate absolute effect
        absolute_effect = self.baseline_value * self.minimum_detectable_effect
        
        return {
            "total_sample_size": sample_size,
            "per_variant": per_variant,
            "absolute_effect": absolute_effect,
            "relative_effect": self.minimum_detectable_effect,
            "parameters": {
                "baseline_value": self.baseline_value,
                "minimum_detectable_effect": self.minimum_detectable_effect,
                "metric_type": self.metric_type.value,
                "power": self.power,
                "significance_level": self.significance_level,
                "variant_count": self.variant_count,
                "allocation_ratio": self.allocation_ratio
            }
        }
    
    def _calculate_binary(self) -> int:
        """
        Calculate sample size for binary metrics (e.g., conversion rates).
        
        Uses the two-proportion z-test formula.
        
        Returns:
            Total sample size required
        """
        # Get z-scores for significance level and power
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        z_beta = stats.norm.ppf(self.power)
        
        # Calculate baseline and expected rates
        p1 = self.baseline_value
        p2 = p1 * (1 + self.minimum_detectable_effect)
        
        # Ensure p2 is valid (between 0 and 1)
        p2 = min(max(p2, 0), 1)
        
        # Calculate pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Calculate standardized effect
        effect = abs(p1 - p2)
        
        # Calculate sample size per variant
        numerator = (z_alpha + z_beta)**2 * 2 * p_pooled * (1 - p_pooled)
        denominator = effect**2
        
        sample_size_per_variant = numerator / denominator
        
        # Calculate total sample size
        total_sample_size = sample_size_per_variant * self.variant_count
        
        # Apply correction for unequal allocation
        if len(set(self.allocation_ratio)) > 1:
            # Calculate the effective sample size reduction factor
            weighted_sum = sum(ratio**2 for ratio in self.allocation_ratio)
            correction_factor = self.variant_count * weighted_sum
            total_sample_size *= correction_factor
        
        return int(math.ceil(total_sample_size))
    
    def _calculate_continuous(self) -> int:
        """
        Calculate sample size for continuous metrics (e.g., revenue).
        
        Uses the two-sample t-test formula.
        
        Returns:
            Total sample size required
        """
        # Get z-scores for significance level and power
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        z_beta = stats.norm.ppf(self.power)
        
        # Calculate expected value for treatment
        mean1 = self.baseline_value
        mean2 = mean1 * (1 + self.minimum_detectable_effect)
        
        # Calculate standardized effect
        effect = abs(mean1 - mean2)
        variance = self.variance
        
        # Calculate sample size per variant
        numerator = 2 * (z_alpha + z_beta)**2 * variance
        denominator = effect**2
        
        sample_size_per_variant = numerator / denominator
        
        # Calculate total sample size
        total_sample_size = sample_size_per_variant * self.variant_count
        
        # Apply correction for unequal allocation
        if len(set(self.allocation_ratio)) > 1:
            # Calculate the effective sample size reduction factor
            weighted_sum = sum(ratio**2 for ratio in self.allocation_ratio)
            correction_factor = self.variant_count * weighted_sum
            total_sample_size *= correction_factor
        
        return int(math.ceil(total_sample_size))
    
    def _calculate_count(self) -> int:
        """
        Calculate sample size for count metrics (e.g., number of purchases).
        
        Uses the Poisson distribution approximation.
        
        Returns:
            Total sample size required
        """
        # Get z-scores for significance level and power
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        z_beta = stats.norm.ppf(self.power)
        
        # Calculate baseline and expected rates
        lambda1 = self.baseline_value
        lambda2 = lambda1 * (1 + self.minimum_detectable_effect)
        
        # Calculate sample size per variant
        numerator = (z_alpha + z_beta)**2 * (lambda1 + lambda2)
        denominator = (lambda1 - lambda2)**2
        
        sample_size_per_variant = numerator / denominator
        
        # Calculate total sample size
        total_sample_size = sample_size_per_variant * self.variant_count
        
        # Apply correction for unequal allocation
        if len(set(self.allocation_ratio)) > 1:
            # Calculate the effective sample size reduction factor
            weighted_sum = sum(ratio**2 for ratio in self.allocation_ratio)
            correction_factor = self.variant_count * weighted_sum
            total_sample_size *= correction_factor
        
        return int(math.ceil(total_sample_size))
    
    def _calculate_ratio(self) -> int:
        """
        Calculate sample size for ratio metrics (e.g., revenue per user).
        
        Uses the delta method approximation.
        
        Returns:
            Total sample size required
        """
        # Get z-scores for significance level and power
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        z_beta = stats.norm.ppf(self.power)
        
        # Calculate baseline and expected values
        ratio1 = self.baseline_value
        ratio2 = ratio1 * (1 + self.minimum_detectable_effect)
        
        # Calculate standardized effect
        effect = abs(ratio1 - ratio2)
        variance = self.variance
        
        # Calculate sample size per variant
        numerator = 2 * (z_alpha + z_beta)**2 * variance
        denominator = effect**2
        
        sample_size_per_variant = numerator / denominator
        
        # Calculate total sample size
        total_sample_size = sample_size_per_variant * self.variant_count
        
        # Apply correction for unequal allocation
        if len(set(self.allocation_ratio)) > 1:
            # Calculate the effective sample size reduction factor
            weighted_sum = sum(ratio**2 for ratio in self.allocation_ratio)
            correction_factor = self.variant_count * weighted_sum
            total_sample_size *= correction_factor
        
        return int(math.ceil(total_sample_size))


class ExperimentDurationCalculator:
    """
    Calculator for determining the expected duration of an experiment.
    
    This class uses traffic estimates and required sample size to estimate
    how long an experiment will need to run.
    """
    
    def __init__(self, 
                 sample_size_result: Dict[str, Union[int, float, Dict]],
                 daily_traffic: int,
                 expected_inclusion_rate: float = 1.0,
                 seasonal_factors: Optional[Dict[str, float]] = None):
        """
        Initialize the duration calculator.
        
        Args:
            sample_size_result: Output from SampleSizeCalculator.calculate()
            daily_traffic: Total daily traffic/users
            expected_inclusion_rate: Proportion of traffic to be included in the experiment
            seasonal_factors: Dictionary mapping date ranges to traffic multipliers to account
                for seasonal variations in traffic
        """
        self.sample_size = sample_size_result["total_sample_size"]
        self.per_variant = sample_size_result["per_variant"]
        self.daily_traffic = daily_traffic
        self.inclusion_rate = expected_inclusion_rate
        self.seasonal_factors = seasonal_factors or {}
    
    def calculate(self) -> Dict[str, Union[int, float, Dict, str]]:
        """
        Calculate the expected experiment duration.
        
        Returns:
            Dictionary containing duration information:
            - days: Number of days required
            - weeks: Number of weeks required
            - traffic_per_day: Expected traffic per day
            - confidence: Qualitative assessment of the estimate reliability
        """
        # Calculate effective daily traffic
        effective_daily_traffic = self.daily_traffic * self.inclusion_rate
        
        # Calculate duration without seasonality
        base_duration_days = self.sample_size / effective_daily_traffic
        
        # Apply seasonal adjustments if provided
        adjusted_duration = self._apply_seasonal_adjustments(base_duration_days)
        
        # Round up to whole days
        duration_days = math.ceil(adjusted_duration)
        duration_weeks = math.ceil(duration_days / 7)
        
        # Assess confidence in the estimate
        if self.seasonal_factors:
            confidence = "Medium - Seasonal factors considered"
        else:
            confidence = "Low - No seasonal adjustments applied"
            
        if duration_days > 30:
            confidence_level = "Low - Long duration increases risk"
        else:
            confidence_level = "Medium"
        
        return {
            "days": duration_days,
            "weeks": duration_weeks,
            "traffic_per_day": effective_daily_traffic,
            "confidence": confidence_level,
            "expected_completion_date": f"Current date + {duration_days} days",
            "variant_completion": {
                variant: f"{math.ceil(size / effective_daily_traffic * len(self.per_variant))} days"
                for variant, size in self.per_variant.items()
            }
        }
    
    def _apply_seasonal_adjustments(self, base_duration: float) -> float:
        """
        Apply seasonal traffic adjustments to refine duration estimate.
        
        Args:
            base_duration: Initial duration estimate in days
            
        Returns:
            Adjusted duration in days
        """
        if not self.seasonal_factors:
            return base_duration
        
        # More sophisticated seasonal adjustment would go here
        # For now, we'll use a simple average adjustment
        avg_factor = sum(self.seasonal_factors.values()) / len(self.seasonal_factors)
        return base_duration / avg_factor


class PowerAnalysis:
    """
    Performs power analysis for experiments.
    
    This class provides methods to calculate statistical power for a given
    sample size and effect size, or to determine the minimum detectable effect
    for a given sample size and desired power.
    """
    
    def __init__(self, 
                 sample_size: int,
                 baseline_value: float,
                 metric_type: Union[MetricType, str] = MetricType.BINARY,
                 significance_level: float = 0.05,
                 variance: Optional[float] = None,
                 variant_count: int = 2,
                 allocation_ratio: Optional[List[float]] = None):
        """
        Initialize the power analysis calculator.
        
        Args:
            sample_size: Total sample size available
            baseline_value: The current value of the metric (e.g., conversion rate)
            metric_type: Type of metric being measured (binary, continuous, etc.)
            significance_level: Significance level (α), typically 0.05
            variance: For continuous metrics, the variance of the metric
                (if None, will be estimated for binary metrics)
            variant_count: Number of variants including control
            allocation_ratio: Custom allocation ratio between variants
                (must sum to 1, defaults to equal allocation)
        """
        self.sample_size = sample_size
        self.baseline_value = baseline_value
        
        # Handle string input for metric type
        if isinstance(metric_type, str):
            self.metric_type = MetricType(metric_type)
        else:
            self.metric_type = metric_type
            
        self.significance_level = significance_level
        self.variance = variance
        self.variant_count = variant_count
        
        # Validate and set allocation ratio
        if allocation_ratio is None:
            # Equal allocation by default
            self.allocation_ratio = [1.0 / variant_count] * variant_count
        else:
            if len(allocation_ratio) != variant_count:
                raise ValueError(f"Allocation ratio must have {variant_count} values")
            if abs(sum(allocation_ratio) - 1.0) > 1e-10:
                raise ValueError("Allocation ratio must sum to 1")
            self.allocation_ratio = allocation_ratio
    
    def calculate_power(self, minimum_detectable_effect: float) -> float:
        """
        Calculate the statistical power for a given effect size.
        
        Args:
            minimum_detectable_effect: The relative change to detect
            
        Returns:
            Statistical power (between 0 and 1)
        """
        if self.metric_type == MetricType.BINARY:
            return self._calculate_power_binary(minimum_detectable_effect)
        elif self.metric_type == MetricType.CONTINUOUS:
            if self.variance is None:
                raise ValueError("Variance must be provided for continuous metrics")
            return self._calculate_power_continuous(minimum_detectable_effect)
        elif self.metric_type == MetricType.COUNT:
            return self._calculate_power_count(minimum_detectable_effect)
        elif self.metric_type == MetricType.RATIO:
            if self.variance is None:
                raise ValueError("Variance must be provided for ratio metrics")
            return self._calculate_power_continuous(minimum_detectable_effect)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
    
    def calculate_mde(self, desired_power: float = 0.8) -> float:
        """
        Calculate the minimum detectable effect for a given power.
        
        Args:
            desired_power: The desired statistical power, typically 0.8 or 0.9
            
        Returns:
            Minimum detectable effect as a relative change
        """
        # Use binary search to find the MDE that gives the desired power
        low = 0.001  # 0.1% effect
        high = 1.0   # 100% effect
        
        while high - low > 0.001:  # Precision to 0.1%
            mid = (high + low) / 2
            power = self.calculate_power(mid)
            
            if power < desired_power:
                low = mid
            else:
                high = mid
        
        return (high + low) / 2
    
    def power_curve(self, effect_range: Optional[List[float]] = None, 
                  points: int = 20) -> Dict[str, List[float]]:
        """
        Generate data for a power curve.
        
        Args:
            effect_range: List of effect sizes to calculate power for
                (defaults to 20 points from 1% to 50%)
            points: Number of points to calculate if effect_range not provided
            
        Returns:
            Dictionary with 'effects' and 'power' lists
        """
        if effect_range is None:
            effect_range = np.linspace(0.01, 0.5, points)
        
        powers = [self.calculate_power(effect) for effect in effect_range]
        
        return {
            "effects": list(effect_range),
            "power": powers
        }
    
    def _calculate_power_binary(self, minimum_detectable_effect: float) -> float:
        """
        Calculate power for binary metrics.
        
        Args:
            minimum_detectable_effect: The relative change to detect
            
        Returns:
            Statistical power (between 0 and 1)
        """
        # Get z-score for significance level
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        
        # Calculate baseline and expected rates
        p1 = self.baseline_value
        p2 = p1 * (1 + minimum_detectable_effect)
        
        # Ensure p2 is valid (between 0 and 1)
        p2 = min(max(p2, 0), 1)
        
        # Calculate pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Calculate standardized effect
        effect = abs(p1 - p2)
        
        # Calculate sample size per variant (assuming equal allocation)
        n_per_variant = self.sample_size / self.variant_count
        
        # Apply correction for unequal allocation
        if len(set(self.allocation_ratio)) > 1:
            # Calculate the effective sample size reduction factor
            weighted_sum = sum(ratio**2 for ratio in self.allocation_ratio)
            correction_factor = self.variant_count * weighted_sum
            n_per_variant /= correction_factor
        
        # Calculate standard error
        se = math.sqrt(2 * p_pooled * (1 - p_pooled) / n_per_variant)
        
        # Calculate z-beta (z-score corresponding to power)
        z_beta = effect / se - z_alpha
        
        # Convert z-beta to power
        power = stats.norm.cdf(z_beta)
        
        return power
    
    def _calculate_power_continuous(self, minimum_detectable_effect: float) -> float:
        """
        Calculate power for continuous metrics.
        
        Args:
            minimum_detectable_effect: The relative change to detect
            
        Returns:
            Statistical power (between 0 and 1)
        """
        # Get z-score for significance level
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        
        # Calculate baseline and expected values
        mean1 = self.baseline_value
        mean2 = mean1 * (1 + minimum_detectable_effect)
        
        # Calculate standardized effect
        effect = abs(mean1 - mean2)
        variance = self.variance
        
        # Calculate sample size per variant (assuming equal allocation)
        n_per_variant = self.sample_size / self.variant_count
        
        # Apply correction for unequal allocation
        if len(set(self.allocation_ratio)) > 1:
            # Calculate the effective sample size reduction factor
            weighted_sum = sum(ratio**2 for ratio in self.allocation_ratio)
            correction_factor = self.variant_count * weighted_sum
            n_per_variant /= correction_factor
        
        # Calculate standard error
        se = math.sqrt(2 * variance / n_per_variant)
        
        # Calculate z-beta (z-score corresponding to power)
        z_beta = effect / se - z_alpha
        
        # Convert z-beta to power
        power = stats.norm.cdf(z_beta)
        
        return power
    
    def _calculate_power_count(self, minimum_detectable_effect: float) -> float:
        """
        Calculate power for count metrics.
        
        Args:
            minimum_detectable_effect: The relative change to detect
            
        Returns:
            Statistical power (between 0 and 1)
        """
        # Get z-score for significance level
        z_alpha = stats.norm.ppf(1 - self.significance_level / 2)
        
        # Calculate baseline and expected rates
        lambda1 = self.baseline_value
        lambda2 = lambda1 * (1 + minimum_detectable_effect)
        
        # Calculate standardized effect
        effect = abs(lambda1 - lambda2)
        
        # Calculate sample size per variant (assuming equal allocation)
        n_per_variant = self.sample_size / self.variant_count
        
        # Apply correction for unequal allocation
        if len(set(self.allocation_ratio)) > 1:
            # Calculate the effective sample size reduction factor
            weighted_sum = sum(ratio**2 for ratio in self.allocation_ratio)
            correction_factor = self.variant_count * weighted_sum
            n_per_variant /= correction_factor
        
        # Calculate standard error
        se = math.sqrt((lambda1 + lambda2) / n_per_variant)
        
        # Calculate z-beta (z-score corresponding to power)
        z_beta = effect / se - z_alpha
        
        # Convert z-beta to power
        power = stats.norm.cdf(z_beta)
        
        return power


# Usage example
if __name__ == "__main__":
    # Example 1: Calculate sample size for a conversion rate experiment
    calculator = SampleSizeCalculator(
        baseline_value=0.05,  # 5% baseline conversion rate
        minimum_detectable_effect=0.10,  # Detect a 10% relative change
        metric_type=MetricType.BINARY,
        power=0.8,
        significance_level=0.05
    )
    
    result = calculator.calculate()
    print(f"Required sample size: {result['total_sample_size']} total users")
    print(f"Per variant: {result['per_variant']}")
    
    # Example 2: Calculate experiment duration
    duration_calc = ExperimentDurationCalculator(
        sample_size_result=result,
        daily_traffic=5000,  # 5000 users per day
        expected_inclusion_rate=0.5  # Include 50% of traffic in the experiment
    )
    
    duration = duration_calc.calculate()
    print(f"Experiment duration: {duration['days']} days ({duration['weeks']} weeks)")
    
    # Example 3: Calculate power for a given sample size and effect
    power_calc = PowerAnalysis(
        sample_size=10000,  # Total sample size available
        baseline_value=0.05,  # 5% baseline conversion rate
        metric_type=MetricType.BINARY,
        significance_level=0.05
    )
    
    power = power_calc.calculate_power(0.15)  # 15% relative effect
    print(f"Statistical power: {power:.2f}")
    
    # Example 4: Calculate minimum detectable effect for a given sample size and power
    mde = power_calc.calculate_mde(desired_power=0.8)
    print(f"Minimum detectable effect: {mde:.2%} relative change")