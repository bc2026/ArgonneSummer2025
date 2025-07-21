# Algorithm Formalization: Region Detection, Merging, and Outlier Removal

## Overview

This document formalizes the three-stage algorithm used for time-series data analysis:

1. **Stage 1: Region Detection** - Uses left derivatives to identify change points
2. **Stage 2: Region Merging** - Merges regions with similar response characteristics  
3. **Stage 3: Outlier Removal** - Removes outliers within each final region

---

## Stage 1: Region Detection Algorithm

### Purpose
Detect regions of interest in time-series data by identifying significant changes using left derivative analysis.

### Input
- Time series data: `DataFrame` with columns `['Time', response_column]`
- First region boundary: `first_region_end_index` (user-defined)
- Response column name: `response`

### Algorithm Steps

#### Step 1.1: Data Preprocessing
```
1. Filter data to maximum response value:
   df_c = df[['Time', response]]
   max_idx = df_c[response].idxmax()
   df_c = df_c[:max_idx + 1]

2. Normalize time by subtracting minimum:
   df_c['Time'] = df_c['Time'] - df_c['Time'].min()
   first_region_end_normalized = first_region_end_index - df['Time'].min()
```

#### Step 1.2: Left Derivative Calculation
For each point i in the time series:

```
Left_Derivative[i] = {
    NaN,                           if i = 0
    (Y[i] - Y[i-1])/(X[i] - X[i-1]), if i > 0 and X[i] ≠ X[i-1]
    ±∞,                            if i > 0 and X[i] = X[i-1]
}
```

**Mathematical Formula:**
```
∂Y/∂X|_{left}(t_i) = lim_{h→0-} [Y(t_i) - Y(t_i + h)] / [t_i - (t_i + h)]
```

#### Step 1.3: Shift Point Detection
Identify significant changes where:
```
mask = (Left_Derivative > Y) AND (Time > first_region_end_normalized)
shifts = Time[mask]
```

#### Step 1.4: Region Construction
```
regions = []
regions.append((0.0, first_region_end_normalized))  // First region

For each shift_point in shifts:
    regions.append((shift_point, shift_point + interval_size))

// Add final region from last shift to end
if regions.length > 0:
    last_end = regions[-1][1]
    max_time = df_c['Time'].max()
    regions.append((last_end, max_time))
```

### Output
List of detected regions: `[(t0_1, tf_1), (t0_2, tf_2), ..., (t0_n, tf_n)]`

---

## Stage 2: Region Merging Algorithm

### Purpose
Merge adjacent regions that have statistically similar response characteristics to reduce over-segmentation.

### Input
- Original DataFrame: `df`
- Detected regions: `regions = [(t0_1, tf_1), (t0_2, tf_2), ..., (t0_n, tf_n)]`
- Percentage threshold: `p` (e.g., 5 for 5%)
- Response column: `response`

### Algorithm Steps

#### Step 2.1: Size-Based Filtering
```
total_time_range = regions[-1][1] - regions[0][0]
min_region_size = total_time_range × 0.02  // 2% of total range

For each region in regions:
    if region_size < min_region_size AND previous_region_exists:
        merge region with previous_region
```

#### Step 2.2: Similarity-Based Merging

**Mergeability Test Function:**
```
is_mergeable(X1, X2, p):
    if X1 == X2:
        return True
    
    dist = |X2 - X1|
    norm_reference = max(|X1|, |X2|) if min(|X1|, |X2|) = 0 else |X1|
    
    return (dist / norm_reference) ≤ (p / 100)
```

**Greedy Merging Algorithm:**
```
merged_regions = []
curr_region = regions[0]
curr_median = median(df[curr_region][response])

For i = 1 to length(regions):
    next_region = regions[i]
    next_median = median(df[next_region][response])
    
    if NOT is_mergeable(curr_median, next_median, p):
        merged_regions.append(curr_region)
        curr_region = next_region
        curr_median = next_median
    else:
        // Extend current region to include next region
        curr_region = (curr_region[0], next_region[1])
        curr_median = median(df[curr_region][response])

merged_regions.append(curr_region)  // Add final region
```

### Mathematical Foundation
The mergeability criterion is based on relative percentage difference:

```
Relative_Difference = |median_2 - median_1| / |median_1| × 100%

Mergeable ⟺ Relative_Difference ≤ p%
```

### Output
List of merged regions: `[(t0_1', tf_1'), (t0_2', tf_2'), ..., (t0_m', tf_m')]` where `m ≤ n`

---

## Stage 3: Outlier Removal Algorithm

### Purpose
Remove statistical outliers within each merged region using the k-standard deviation rule.

### Input
- Original DataFrame: `Y`
- Merged regions: `merged_regions`
- Response column: `response_column_name`
- Standard deviation multiplier: `k = 3` (configurable)

### Algorithm Steps

#### Step 3.1: Region-wise Outlier Detection
For each region `R = (t0, tf)` in merged_regions:

```
1. Extract region data:
   region_data = Y[(Y['Time'] ≥ t0) AND (Y['Time'] ≤ tf)]

2. Calculate statistics:
   μ = mean(region_data[response_column_name])
   σ = std(region_data[response_column_name])

3. Define outlier bounds:
   lower_bound = μ - k × σ
   upper_bound = μ + k × σ

4. Filter outliers:
   clean_data = region_data[(region_data[response] ≥ lower_bound) AND 
                           (region_data[response] ≤ upper_bound)]
```

### Mathematical Foundation

**Three-Sigma Rule (k=3):**
For normally distributed data, approximately 99.7% of values lie within 3 standard deviations of the mean.

```
P(μ - 3σ ≤ X ≤ μ + 3σ) ≈ 0.997

Outlier ⟺ X < (μ - 3σ) OR X > (μ + 3σ)
```

**General k-Sigma Rule:**
```
Outlier ⟺ |X - μ| > k × σ

Where:
- X = observed value
- μ = sample mean
- σ = sample standard deviation
- k = threshold multiplier (typically 3)
```

### Output
Cleaned DataFrame with outliers removed from each region.

---

## Algorithm Configuration Parameters

| Parameter | Description | Default Value | Range |
|-----------|-------------|---------------|-------|
| `k` | Standard deviation multiplier for outlier detection | 3 | [2, 4] |
| `p` | Percentage threshold for region merging | 5% | [1%, 50%] |
| `min_region_size_ratio` | Minimum region size as fraction of total time | 0.02 | [0.01, 0.1] |
| `first_region_end_index` | User-defined boundary for first region | User input | Data-dependent |

---

## Computational Complexity

- **Stage 1 (Region Detection):** O(n) where n = number of data points
- **Stage 2 (Region Merging):** O(m²) where m = number of detected regions
- **Stage 3 (Outlier Removal):** O(n) where n = number of data points

**Overall Complexity:** O(n + m²) ≈ O(n) for typical cases where m << n

---

## Implementation Notes

### Edge Cases Handled
1. **Empty regions:** Return NaN median, mark as non-mergeable
2. **Zero denominators:** Handle division by zero in derivative calculation
3. **Single-point regions:** Treated as valid but non-mergeable
4. **Identical medians:** Automatically mergeable (return True)

### Robustness Features
1. **Input validation:** Check for numeric data types and required columns
2. **Boundary handling:** Ensure regions don't exceed data boundaries
3. **Logging:** Comprehensive debug information for each stage
4. **Error recovery:** Graceful handling of malformed data

---

## Memory Considerations

### User Preference Integration
Based on the existing memory [[memory:3232847]], the algorithm maintains compatibility with left derivative detection while providing the option for alternative methods:

```python
# Current implementation uses left derivatives as preferred
if use_left_derivatives:  # Default: True per user preference
    shifts = detect_shifts_left_derivative(data)
else:
    shifts = detect_shifts_step_change(data)  # Alternative method
```

This ensures the formalized algorithm respects the user's preference for left derivative-based edge detection. 