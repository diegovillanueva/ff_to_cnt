#!/usr/bin/env python3
"""Digitize values from Wang et al. (2014) Figure 1.

Figure shows immersion freezing active fraction vs temperature for
CSU106 and ZINC106 dust samples. Six series are extracted:
  1. CSU106: Obs (red open circles) — discrete observation points
  2. CSU106: α-PDF (red solid line) — model curve
  3. CSU106: single α (red dashed line) — model curve
  4. ZINC106: Obs (blue open circles) — discrete observation points
  5. ZINC106: α-PDF (blue solid line) — model curve
  6. ZINC106: single α (blue dashed line) — model curve

Values were manually extracted by careful visual inspection of the
log-scale figure. Temperatures read relative to major gridlines at
every 5°C; active fractions read relative to log-decade gridlines
(10^-6 through 10^-1) and minor gridlines at 2×, 3×, 5× per decade.
"""
from __future__ import annotations

import csv
from pathlib import Path


def get_digitized_data() -> dict[str, dict[int | float, float]]:
    """Return all digitized series as {name: {temperature: active_fraction}}."""

    # ── Observation data points (discrete markers) ────────────────────────
    # CSU106: Obs — red open circles, 7 points identified
    # Points follow the CSU106 α-PDF curve closely (good model fit).
    csu106_obs: dict[int | float, float] = {
        -33: 2.5e-2,   # between -35 and -30, ~2/5 from -35; AF just above 2e-2 minor gridline
        -30: 1.5e-2,   # at -30 mark; AF between 1e-2 and 2e-2
        -28: 1.3e-2,   # ~2/5 from -30 to -25; AF just above 1e-2
        -25: 4.0e-3,   # at -25 mark; AF between 3e-3 and 5e-3
        -22: 1.5e-3,   # ~3/5 from -25 to -20; AF between 1e-3 and 2e-3
        -20: 6.0e-4,   # at -20 mark; AF ~60-70% up from 1e-4 to 1e-3 on log scale
        -17: 1.0e-4,   # ~3/5 from -20 to -15; AF right at 1e-4 gridline
    }

    # ZINC106: Obs — blue open circles, 5 clearly visible points
    # At warm temperatures (T > -25), ZINC obs drop to very low values
    # (near 1e-5 to 1e-6); these are barely visible in the figure.
    zinc106_obs: dict[int | float, float] = {
        -37: 6.0e-2,   # ~3/5 from -40 to -35; AF ~78% up from 1e-2 to 1e-1 (log)
        -35: 2.5e-2,   # at -35 mark; AF ~40% up from 1e-2 to 1e-1 (log)
        -33: 2.0e-2,   # ~2/5 from -35 to -30; AF at or just above 2e-2
        -30: 5.0e-3,   # at -30 mark; AF ~70% up from 1e-3 to 1e-2 (log)
        -28: 1.5e-3,   # ~2/5 from -30 to -25; AF between 1e-3 and 2e-3
    }

    # ── α-PDF model curves (solid lines) ──────────────────────────────────
    # CSU106: α-PDF — red solid line
    # Relatively flat from -40 to -33 (~2.5e-2), then gradual S-shaped
    # decrease on log scale, leveling off near 5e-6 around T=-12.
    csu106_alpha_pdf: dict[int | float, float] = {
        -40: 2.5e-2,
        -38: 2.5e-2,
        -36: 2.5e-2,
        -35: 2.5e-2,   # crossing point with ZINC106 α-PDF
        -34: 2.5e-2,
        -33: 2.5e-2,
        -32: 2.0e-2,
        -30: 1.3e-2,
        -28: 8.0e-3,
        -26: 5.0e-3,
        -25: 4.0e-3,
        -24: 3.0e-3,
        -22: 1.5e-3,
        -20: 6.0e-4,
        -18: 2.0e-4,
        -16: 5.0e-5,
        -15: 2.0e-5,
        -12: 5.0e-6,
        -10: 5.0e-6,   # leveling off
    }

    # CSU106: single α — red dashed line
    # Tracks α-PDF closely from -40 to ~-25, then drops much more steeply.
    # The sharper cutoff is characteristic of the single contact angle model.
    # Divergence from α-PDF becomes visible around T=-22.
    # Drops below 1e-6 around T=-16.
    csu106_single_alpha: dict[int | float, float] = {
        -40: 2.5e-2,
        -38: 2.5e-2,
        -36: 2.5e-2,
        -35: 2.5e-2,
        -34: 2.5e-2,
        -33: 2.5e-2,
        -32: 2.0e-2,
        -30: 1.2e-2,
        -28: 7.0e-3,
        -26: 3.5e-3,
        -25: 2.0e-3,
        -24: 1.0e-3,
        -22: 2.0e-4,   # diverging from α-PDF (1.5e-3)
        -20: 1.0e-5,   # far below α-PDF (6e-4)
        -18: 5.0e-7,   # near bottom of plot range
    }

    # ZINC106: α-PDF — blue solid line
    # Starts high (~8e-2 at -40), drops steeply through the transition
    # region (-35 to -25), levels off near 5e-6 around T=-22.
    zinc106_alpha_pdf: dict[int | float, float] = {
        -40: 8.0e-2,
        -38: 5.0e-2,
        -37: 4.0e-2,
        -36: 3.5e-2,
        -35: 2.8e-2,   # crossing point with CSU106 α-PDF
        -34: 2.0e-2,
        -33: 1.5e-2,
        -32: 1.0e-2,
        -30: 5.0e-3,
        -28: 1.5e-3,
        -26: 3.0e-4,
        -25: 1.0e-4,
        -24: 3.0e-5,
        -22: 5.0e-6,
        -20: 5.0e-6,   # leveling off
    }

    # ZINC106: single α — blue dashed line
    # Tracks α-PDF from -40 to ~-32, then drops much more steeply.
    # Divergence visible around T=-30. Drops below 1e-6 around T=-24.
    zinc106_single_alpha: dict[int | float, float] = {
        -40: 8.0e-2,
        -38: 5.0e-2,
        -37: 4.0e-2,
        -36: 3.5e-2,
        -35: 2.8e-2,
        -34: 2.0e-2,
        -33: 1.5e-2,
        -32: 8.0e-3,
        -30: 3.0e-3,   # diverging from α-PDF (5e-3)
        -28: 3.0e-4,   # far below α-PDF (1.5e-3)
        -26: 1.0e-5,   # near bottom
        -25: 5.0e-6,
    }

    return {
        "CSU106_obs": csu106_obs,
        "ZINC106_obs": zinc106_obs,
        "CSU106_alpha_PDF": csu106_alpha_pdf,
        "CSU106_single_alpha": csu106_single_alpha,
        "ZINC106_alpha_PDF": zinc106_alpha_pdf,
        "ZINC106_single_alpha": zinc106_single_alpha,
    }


def write_csv(data: dict[str, dict[int | float, float]], output_path: Path) -> None:
    """Write digitized data to CSV, merging all series on a common temperature grid."""
    # Collect all unique temperatures
    all_temps: set[int | float] = set()
    for series in data.values():
        all_temps.update(series.keys())
    all_temps_sorted = sorted(all_temps)

    columns = [
        "temperature_C",
        "CSU106_obs",
        "ZINC106_obs",
        "CSU106_alpha_PDF",
        "CSU106_single_alpha",
        "ZINC106_alpha_PDF",
        "ZINC106_single_alpha",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for t in all_temps_sorted:
            row = [t]
            for col in columns[1:]:
                val = data[col].get(t)
                row.append(f"{val:.1e}" if val is not None else "")
            writer.writerow(row)

    print(f"Wrote {len(all_temps_sorted)} rows to {output_path}")


def print_summary(data: dict[str, dict[int | float, float]]) -> None:
    """Print a summary of the digitized data."""
    print("\n=== Digitized data summary ===")
    for name, series in data.items():
        temps = sorted(series.keys())
        print(f"\n{name}: {len(series)} points")
        print(f"  T range: {temps[0]}°C to {temps[-1]}°C")
        print(f"  AF range: {min(series.values()):.1e} to {max(series.values()):.1e}")
        for t in temps:
            print(f"    T={t:6.0f}°C  AF={series[t]:.1e}")


def main() -> None:
    data = get_digitized_data()
    print_summary(data)

    output_path = Path(__file__).parent / "fig_1_wang_ff_vs_t.csv"
    write_csv(data, output_path)


if __name__ == "__main__":
    main()
