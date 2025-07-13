# =============================================================================
# OPTIMIZED P-PARAMETER SELECTION WITH TARGET REGIONS
# Replace your current merge_close_points call with this cell
# =============================================================================

print("🎯 Starting Constrained P-Parameter Optimization")
print(f"Target: {target_regions} regions (tolerance: ±{tolerance})")

# Option 1: Full optimization with plots and analysis (RECOMMENDED)
merged_points, optimal_p, optimization_result = replace_merge_close_points_with_optimization(
    df_c=df_c, 
    detected_intervals=detected_intervals, 
    response=response, 
    target_regions=target_regions,
    tolerance=tolerance,
    figure_dir=figure_dir
)

# The rest of your notebook processing remains the same
# merged_points now contains the optimally merged regions

print(f"\n📋 OPTIMIZATION SUMMARY:")
print(f"   Optimal p value: {optimal_p}%")
print(f"   Number of regions: {len(merged_points)}")
print(f"   Separation score: {optimization_result['separation_score']:.3f}")

# Continue with your existing notebook code...
# No need to modify the code below this point 