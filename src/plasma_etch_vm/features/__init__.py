"""Feature builders.

cycles   per (wafer, BOSCH cycle, phase) aggregates from the partitioned Parquet store
         (tool-parameter means, OES line intensities, binned mean spectra) — memory-bounded streaming.
wafer    per-wafer feature table: phase statistics, drift over cycles, actinometry / product ratios,
         timing and run context.
"""
