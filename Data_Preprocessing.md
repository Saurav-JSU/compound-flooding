# Data Preprocessing Methodology

This document outlines the preprocessing steps and theoretical principles applied to the GESLA sea level and ERA5 precipitation datasets for compound flood analysis.

## 1. Quality Control Framework

### 1.1 Quality Tiering System

We implement a four-tier quality classification system:

- **Tier A (Excellent)**: Complete data with realistic values; minimal missing data (<5%); sufficient joint exceedances
- **Tier B (Good)**: Minor issues that can be corrected; 5-10% missing data; fewer joint exceedances
- **Tier C (Problematic)**: Major issues but potentially usable with corrections; 10-25% missing data
- **Tier D (Unusable)**: Severe data quality issues; >25% missing data; unrealistic values; no joint exceedances

### 1.2 Quality Metrics

Key metrics used to assess data quality:
- Percentage of missing sea level data
- 99th percentile value of sea level (to detect unrealistic values)
- Number of joint exceedances of high thresholds
- Conditional Probability Ratio (CPR) of joint exceedances

## 2. Sea Level Data Preprocessing

### 2.1 Missing Value Identification

- We use the station-specific null value (typically -99.9999) as indicated in metadata
- Valid sea level measurements are separated from missing values for all analyses
- Percentiles and statistics are calculated only on valid measurements

### 2.2 Short Gap Filling Principle

For minor gaps in sea level data, we apply a controlled linear interpolation approach:

1. **Identification**: Locate sequences of 1-2 consecutive hours of missing data
2. **Eligibility**: Only gaps that are surrounded by valid measurements are filled
3. **Linear Interpolation**: For a gap between times t₁ and t₂, with values h₁ and h₂:
   ```
   h(t) = h₁ + (h₂ - h₁) × (t - t₁)/(t₂ - t₁)
   ```
4. **Theoretical Basis**: For short timescales (1-2 hours), sea level often changes approximately linearly when not at the peak of a storm or tide cycle

This approach is consistent with common practices in oceanographic studies and follows the principle of minimal intervention - we only fill gaps where we have high confidence in the interpolated values.

### 2.3 Outlier Detection and Correction

Two main correction methods for unrealistic sea level values:

1. **Unit Conversion Error Correction**:
   - Detection: Identify stations where max sea level values are orders of magnitude (>50×) higher than the median
   - Correction: Apply division factor (10 or 100) based on the magnitude of discrepancy
   - Theoretical Basis: Most unit errors come from conversion between cm, m, or feet

2. **Physical Plausibility Capping**:
   - Regional Maximum: 10m for oceanic stations, 5m for Great Lakes stations
   - Values exceeding these thresholds are capped
   - Theoretical Basis: Even extreme events like tsunamis rarely exceed these limits in the historical record

## 3. Precipitation Data Processing

### 3.1 Unit Standardization

- ERA5 total precipitation is provided in meters and converted to millimeters (×1000)
- Ground precipitation data unit handling:
  - If ground precipitation values are significantly larger than ERA5 (>10×), assume ground data is already in mm
  - Otherwise, convert ground precipitation to mm (×1000)

### 3.2 Merging Precipitation Sources

**Hierarchical Source Selection Principle**:
1. Ground precipitation data (from weather stations) is prioritized where available
2. ERA5 reanalysis data is used as a backup when ground data is missing or zero
3. The consolidated 'precipitation_mm' column combines the best available source

Theoretical basis: Ground measurements generally provide more accurate point estimates, while ERA5 offers comprehensive spatial coverage with lower point accuracy.

### 3.3 Extreme Precipitation Correction

- Physical cap of 500mm per hour applied to unrealistic values
- Theoretical basis: World record for 1-hour rainfall is approximately 305mm (Holt, Missouri, 1947)

## 4. Joint Analysis Preprocessing

### 4.1 Threshold Selection

- 95th percentile thresholds used for preliminary joint exceedance analysis
- Thresholds calculated only on valid data points
- Both sea level and precipitation thresholds are station-specific to account for regional differences

### 4.2 Conditional Probability Ratio (CPR)

CPR is calculated as:
```
CPR = P(SL > SL₉₅ AND Precip > P₉₅) / [P(SL > SL₉₅) × P(Precip > P₉₅)]
```

Where:
- P(SL > SL₉₅) is the probability of sea level exceeding its 95th percentile
- P(Precip > P₉₅) is the probability of precipitation exceeding its 95th percentile

Theoretical interpretation:
- CPR = 1: Sea level and precipitation exceedances are statistically independent
- CPR > 1: Higher probability of joint occurrence than expected by chance (positive dependency)
- CPR < 1: Lower probability of joint occurrence than expected by chance (negative dependency)

## 5. Corrective Methodology

Our approach to improving data quality follows these principles:

1. **Non-destructive enhancement**: Original patterns in the data are preserved while correcting obvious errors
2. **Minimal intervention**: Corrections are applied only where strong evidence of data issues exists
3. **Transparency**: All corrections are documented and can be traced back to original data
4. **Physical plausibility**: Corrections ensure data remains within physically realistic bounds
5. **Statistical consistency**: Corrected data maintains the statistical properties of the original time series

This methodology ensures a robust dataset for Tier 1 analysis while maximizing the number of usable stations.