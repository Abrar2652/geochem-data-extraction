#!/usr/bin/env python3
"""Compare v4 baseline with v5 multi-source extraction results."""
import json

# Load v5 results
with open('batch_results_v5/batch_metrics.json') as f:
    v5 = json.load(f)

print("=" * 70)
print("ACCURACY IMPROVEMENT: Multi-Source Extraction (v5)")
print("=" * 70)
print()
print("BASELINE COMPARISON:")
print("-" * 70)
print(f"v4 (baseline with elif):     59.21%")
print(f"v5 (multi-source with if):   {v5['aggregate']['mean_overall_%']:.2f}%")
print(f"Improvement:                 +{v5['aggregate']['mean_overall_%'] - 59.21:.2f} percentage points")
print()
print("DETAILED METRICS:")
print("-" * 70)
agg = v5['aggregate']
print(f"T1 Metadata accuracy:        {agg['mean_T1_metadata_%']:.2f}%")
print(f"T2 Numerical accuracy:       {agg['mean_T2_numerical_%']:.2f}%")
print(f"T3 Structural accuracy:      {agg['mean_T3_structural_%']:.2f}%")
print(f"T4 Null handling:            {agg['mean_T4_null_%']:.2f}%")
print()
print("EXTRACTION COVERAGE:")
print("-" * 70)
print(f"Papers processed:            {agg['n_papers']}")
print(f"Total samples extracted:     {agg['total_samples_predicted']:,}")
print(f"Ground truth samples:        {agg['total_samples_ground_truth']:,}")
print(f"Samples matched:             {agg['total_samples_matched']:,}")
coverage = 100 * agg['total_samples_matched'] / agg['total_samples_ground_truth']
print(f"Extraction coverage:         {coverage:.1f}%")
print()
print("=" * 70)

# Show top 5 papers by accuracy
print("\nTOP 5 PAPERS BY ACCURACY:")
print("-" * 70)
papers = [(k, v['scores']['overall_%']) for k, v in v5['per_paper'].items() if 'scores' in v]
papers.sort(key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(papers[:5], 1):
    print(f"{i}. {name:45s} {score:6.2f}%")
