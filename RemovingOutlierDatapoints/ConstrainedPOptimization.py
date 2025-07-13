import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.OutlierUtils import merge_close_points, get_region
import os

def find_optimal_p_for_target_regions(df_c, detected_intervals, response, target_regions, tolerance=1, p_range=None, verbose=True):
    """
    Find optimal p value that gives approximately the target number of regions
    while maximizing separation score.
    
    Args:
        df_c: DataFrame with time series data
        detected_intervals: Initial detected intervals before merging
        response: Response column name
        target_regions: Expected/desired number of regions
        tolerance: Allow ±tolerance regions from target (default: ±1)
        p_range: Range of p values to test
        verbose: Print progress information
    
    Returns:
        Dictionary with optimal p value and metrics
    """
    if p_range is None:
        p_range = np.arange(10, 201, 5)  # Finer grid search: 10%, 15%, 20%, ..., 200%
    
    if verbose:
        print(f"🎯 Constrained P-Parameter Optimization")
        print("=" * 60)
        print(f"Target regions: {target_regions} ± {tolerance}")
        print(f"Testing p values: {p_range[0]}% to {p_range[-1]}% (step: {p_range[1]-p_range[0]}%)")
        print("=" * 60)
    
    # Track all results
    all_results = []
    # Track only results within target range
    target_results = []
    
    for p_val in p_range:
        try:
            merged_points = merge_close_points(df=df_c, regions=detected_intervals, p=p_val, response=response)
            num_regions = len(merged_points)
            
            # Calculate separation metrics
            within_var = 0
            between_var = 0
            region_medians = []
            
            for start, end in merged_points:
                region_data = get_region(df_c, start, end)
                if not region_data.empty:
                    region_median = np.median(region_data[response])
                    region_medians.append(region_median)
                    within_var += np.var(region_data[response])
            
            if len(region_medians) > 1:
                between_var = np.var(region_medians)
            
            separation_score = between_var / (within_var + 1e-10) if within_var > 0 else 0
            
            result = {
                'p_value': p_val,
                'num_regions': num_regions,
                'within_variance': within_var,
                'between_variance': between_var,
                'separation_score': separation_score,
                'distance_from_target': abs(num_regions - target_regions),
                'merged_points': merged_points
            }
            
            all_results.append(result)
            
            # Check if within target range
            if abs(num_regions - target_regions) <= tolerance:
                target_results.append(result)
                if verbose:
                    print(f"✓ p = {p_val:3d}%: {num_regions} regions, separation = {separation_score:.3f}")
            else:
                if verbose and p_val % 20 == 0:  # Print every 20% to avoid spam
                    print(f"  p = {p_val:3d}%: {num_regions} regions, separation = {separation_score:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"✗ p = {p_val}%: Error - {e}")
            continue
    
    # Find optimal p value
    if target_results:
        # Among results with target number of regions, pick highest separation score
        best_result = max(target_results, key=lambda x: x['separation_score'])
        if verbose:
            print(f"\n🎯 OPTIMAL RESULT (within target range):")
            print(f"   p = {best_result['p_value']}%")
            print(f"   Regions: {best_result['num_regions']} (target: {target_regions})")
            print(f"   Separation score: {best_result['separation_score']:.3f}")
        
    else:
        # No exact matches, find closest to target
        best_result = min(all_results, key=lambda x: x['distance_from_target'])
        if verbose:
            print(f"\n⚠️  NO EXACT MATCH FOUND")
            print(f"   Closest result: p = {best_result['p_value']}%")
            print(f"   Regions: {best_result['num_regions']} (target: {target_regions})")
            print(f"   Separation score: {best_result['separation_score']:.3f}")
            
            # Also show the best separation score overall
            best_separation = max(all_results, key=lambda x: x['separation_score'])
            if best_separation != best_result:
                print(f"\n   Best separation (different p): p = {best_separation['p_value']}%")
                print(f"   Regions: {best_separation['num_regions']}, separation = {best_separation['separation_score']:.3f}")
    
    return best_result, target_results, all_results

def plot_constrained_optimization_results(all_results, target_results, target_regions, best_result, figure_dir='./output'):
    """
    Plot optimization results with target region constraint highlighted
    """
    os.makedirs(figure_dir, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data for plotting
    p_vals = [r['p_value'] for r in all_results]
    num_regions = [r['num_regions'] for r in all_results]
    separation_scores = [r['separation_score'] for r in all_results]
    
    # Plot 1: Number of regions vs p (with target highlighted)
    ax1.plot(p_vals, num_regions, 'b-o', alpha=0.7, markersize=4, label='All results')
    
    # Highlight target region range
    ax1.axhspan(target_regions - 0.5, target_regions + 0.5, alpha=0.2, color='green', label='Target ± 0.5')
    
    # Highlight target results
    if target_results:
        target_p_vals = [r['p_value'] for r in target_results]
        target_num_regions = [r['num_regions'] for r in target_results]
        ax1.plot(target_p_vals, target_num_regions, 'go', markersize=8, label='Within target')
    
    # Mark optimal
    ax1.plot(best_result['p_value'], best_result['num_regions'], 'r*', markersize=15, label='Optimal')
    
    ax1.set_xlabel('p value (%)')
    ax1.set_ylabel('Number of Regions')
    ax1.set_title(f'Number of Regions vs p Value (Target: {target_regions})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Separation score vs p (with target results highlighted)
    ax2.plot(p_vals, separation_scores, 'b-o', alpha=0.7, markersize=4, label='All results')
    
    if target_results:
        target_separation = [r['separation_score'] for r in target_results]
        ax2.plot(target_p_vals, target_separation, 'go', markersize=8, label='Within target')
    
    ax2.plot(best_result['p_value'], best_result['separation_score'], 'r*', markersize=15, label='Optimal')
    ax2.set_xlabel('p value (%)')
    ax2.set_ylabel('Separation Score')
    ax2.set_title('Separation Score vs p Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance from target vs p
    distances = [r['distance_from_target'] for r in all_results]
    ax3.plot(p_vals, distances, 'purple', marker='o', alpha=0.7, markersize=4)
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Perfect match')
    ax3.plot(best_result['p_value'], best_result['distance_from_target'], 'r*', markersize=15, label='Optimal')
    ax3.set_xlabel('p value (%)')
    ax3.set_ylabel('Distance from Target')
    ax3.set_title(f'Distance from Target ({target_regions} regions)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined score (separation score - penalty for distance from target)
    max_separation = max(separation_scores) if separation_scores else 1
    penalty_weight = max_separation / 10  # Adjust this to change penalty strength
    
    combined_scores = []
    for r in all_results:
        penalty = penalty_weight * r['distance_from_target']
        combined_score = r['separation_score'] - penalty
        combined_scores.append(combined_score)
    
    ax4.plot(p_vals, combined_scores, 'orange', marker='o', alpha=0.7, markersize=4, label='Combined Score')
    
    if combined_scores:
        best_combined_idx = np.argmax(combined_scores)
        best_combined_p = p_vals[best_combined_idx]
        ax4.plot(best_combined_p, combined_scores[best_combined_idx], 'r*', markersize=15, label='Best Combined')
        ax4.plot(best_result['p_value'], combined_scores[all_results.index(best_result)], 'g*', markersize=12, label='Selected')
    
    ax4.set_xlabel('p value (%)')
    ax4.set_ylabel('Combined Score')
    ax4.set_title('Combined Score (Separation - Distance Penalty)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{figure_dir}/constrained_p_optimization.png', bbox_inches='tight', dpi=300)
    plt.show()

def analyze_optimization_results(all_results, target_regions):
    """
    Provide detailed analysis of optimization results
    """
    print("\n" + "="*60)
    print("OPTIMIZATION ANALYSIS")
    print("="*60)
    
    df_results = pd.DataFrame(all_results)
    
    print(f"📊 SUMMARY STATISTICS:")
    print(f"   Total p values tested: {len(all_results)}")
    print(f"   Region count range: {df_results['num_regions'].min()} - {df_results['num_regions'].max()}")
    print(f"   Separation score range: {df_results['separation_score'].min():.3f} - {df_results['separation_score'].max():.3f}")
    
    # Find results by distance from target
    exact_matches = df_results[df_results['distance_from_target'] == 0]
    close_matches = df_results[df_results['distance_from_target'] == 1]
    
    print(f"\n🎯 TARGET ANALYSIS:")
    print(f"   Exact matches ({target_regions} regions): {len(exact_matches)}")
    print(f"   Close matches (±1 region): {len(close_matches)}")
    
    if len(exact_matches) > 0:
        best_exact = exact_matches.loc[exact_matches['separation_score'].idxmax()]
        print(f"   Best exact match: p={best_exact['p_value']}%, separation={best_exact['separation_score']:.3f}")
    
    # Show top 3 results by separation score
    top_3 = df_results.nlargest(3, 'separation_score')
    print(f"\n🏆 TOP 3 BY SEPARATION SCORE:")
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"   {i}. p={row['p_value']}%, regions={row['num_regions']}, separation={row['separation_score']:.3f}")
    
    return df_results

# Integration functions for the notebook
def replace_merge_close_points_with_optimization(df_c, detected_intervals, response, target_regions=4, tolerance=1, figure_dir='./output'):
    """
    Drop-in replacement for merge_close_points that uses optimization
    """
    print("🔄 Replacing manual p selection with automatic optimization...")
    
    # Run optimization
    best_result, target_results, all_results = find_optimal_p_for_target_regions(
        df_c, detected_intervals, response, target_regions, tolerance
    )
    
    # Plot results
    plot_constrained_optimization_results(all_results, target_results, target_regions, best_result, figure_dir)
    
    # Analyze results
    df_results = analyze_optimization_results(all_results, target_regions)
    
    # Return the optimally merged points
    optimal_p = best_result['p_value']
    merged_points = best_result['merged_points']
    
    print(f"\n✅ OPTIMIZATION COMPLETE")
    print(f"   Using optimal p = {optimal_p}%")
    print(f"   Resulting in {len(merged_points)} regions")
    print(f"   Separation score: {best_result['separation_score']:.3f}")
    
    return merged_points, optimal_p, best_result

def quick_p_optimization(df_c, detected_intervals, response, target_regions=4):
    """
    Quick optimization without plots for fast iteration
    """
    best_result, _, _ = find_optimal_p_for_target_regions(
        df_c, detected_intervals, response, target_regions, tolerance=1, verbose=False
    )
    
    return best_result['merged_points'], best_result['p_value']

if __name__ == "__main__":
    print("ConstrainedPOptimization module loaded successfully!")
    print("Available functions:")
    print("- find_optimal_p_for_target_regions()")
    print("- plot_constrained_optimization_results()")
    print("- replace_merge_close_points_with_optimization()")
    print("- quick_p_optimization()") 