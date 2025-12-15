#!/usr/bin/env python3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from systematics import SystematicsApplier
# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_PARQUET = (
    "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/"
    "Run3_output/data_Run3_test_run_pretrain_0.parquet"
)

OUTPUT_JES_ONLY = "data_syst_JES_only.parquet"
OUTPUT_MET_ONLY = "data_syst_MET_only.parquet"
OUTPUT_JES_MET  = "data_syst_JES_MET.parquet"

N_DATASETS_PER_TYPE = 50

# slice for testing / memory control
BASE_SLICE = int(8_000_000 / 50)

# JES sampling
JES_SIGMA = 0.01
JES_MIN   = -0.10
JES_MAX   =  0.10

# MET sampling
MET_LOGNORM_MEAN  = 0.0
MET_LOGNORM_SIGMA = 1.0
MET_MIN = 0.0
MET_MAX = 5.0


# ------------------------------------------------------------
# Sampling utilities
# ------------------------------------------------------------

def sample_jes():
    """Sample JES scale factor."""
    jes = np.random.normal(0.0, JES_SIGMA)
    return float(np.clip(jes, JES_MIN, JES_MAX))


def sample_met():
    """Sample MET px, py shifts."""
    px = np.random.lognormal(MET_LOGNORM_MEAN, MET_LOGNORM_SIGMA)
    py = np.random.lognormal(MET_LOGNORM_MEAN, MET_LOGNORM_SIGMA)
    px = float(np.clip(px, MET_MIN, MET_MAX))
    py = float(np.clip(py, MET_MIN, MET_MAX))
    return px, py


# ------------------------------------------------------------
# Systematic shift builders
# ------------------------------------------------------------

def make_jes_object_shift(jes):
    """
    Build object_shifts dict for JES variation.
    Assumes column [..., 5] encodes object type (0 = jet).
    """
    return {
        "jet_pt": {
            "features": ["pt"],
            "select": lambda x: (x[..., 5] == 0),
            "apply": lambda v, s: v * (1.0 + s),
            "scale": jes,
        }
    }


# ------------------------------------------------------------
# Bookkeeping columns
# ------------------------------------------------------------

def add_syst_columns(table, jes, met_px, met_py, tag):
    """
    Add per-row systematic bookkeeping columns so that
    systematics can be traced after concatenation.
    """
    n = table.num_rows

    table = table.append_column(
        "syst_jes", pa.array([jes] * n, type=pa.float32())
    )
    table = table.append_column(
        "syst_met_px", pa.array([met_px] * n, type=pa.float32())
    )
    table = table.append_column(
        "syst_met_py", pa.array([met_py] * n, type=pa.float32())
    )
    table = table.append_column(
        "syst_tag", pa.array([tag] * n, type=pa.int64())
    )

    return table


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    # --------------------------------------------------------
    # Load base dataset
    # --------------------------------------------------------

    print("Reading base parquet...")
    base = pq.read_table(INPUT_PARQUET)
    base = base.slice(0, BASE_SLICE)
    print(f"Base rows per dataset: {base.num_rows:,}")

    # Instantiate your systematics applier
    syst = SystematicsApplier()

    tables_jes_only = []
    tables_met_only = []
    tables_jes_met  = []

    # --------------------------------------------------------
    # JES only
    # --------------------------------------------------------

    print("Generating JES-only datasets...")
    for i in range(N_DATASETS_PER_TYPE):
        jes = sample_jes()

        shifted = syst.apply(
            base,
            object_shifts=make_jes_object_shift(jes),
            met_shift=None,
            recompute_globals=True,
        )

        shifted = add_syst_columns(
            shifted,
            jes=jes,
            met_px=0.0,
            met_py=0.0,
            tag=i,
        )

        tables_jes_only.append(shifted)

        print(f"  JES-only [{i+1:02d}/50]  jes = {jes:+.4f}")

    table_jes_only = pa.concat_tables(tables_jes_only, promote_options="default")
    pq.write_table(table_jes_only, OUTPUT_JES_ONLY)
    print(f"  JES-only rows : {table_jes_only.num_rows:,}")

    # --------------------------------------------------------
    # MET only
    # --------------------------------------------------------

    print("Generating MET-only datasets...")
    for i in range(N_DATASETS_PER_TYPE):
        px, py = sample_met()

        shifted = syst.apply(
            base,
            object_shifts=None,
            met_shift={"px": px, "py": py},
            recompute_globals=True,
        )

        shifted = add_syst_columns(
            shifted,
            jes=0.0,
            met_px=px,
            met_py=py,
            tag=i,
        )

        tables_met_only.append(shifted)

        print(f"  MET-only [{i+1:02d}/50]  px = {px:.3f}, py = {py:.3f}")

    table_met_only = pa.concat_tables(tables_met_only, promote_options="default")
    pq.write_table(table_met_only, OUTPUT_MET_ONLY)
    print(f"  MET-only rows : {table_met_only.num_rows:,}")
    # --------------------------------------------------------
    # JES + MET
    # --------------------------------------------------------

    print("Generating JES+MET datasets...")
    for i in range(N_DATASETS_PER_TYPE):
        jes = sample_jes()
        px, py = sample_met()

        shifted = syst.apply(
            base,
            object_shifts=make_jes_object_shift(jes),
            met_shift={"px": px, "py": py},
            recompute_globals=True,
        )

        shifted = add_syst_columns(
            shifted,
            jes=jes,
            met_px=px,
            met_py=py,
            tag=i,
        )

        tables_jes_met.append(shifted)

        print(
            f"  JES+MET [{i+1:02d}/50]  "
            f"jes = {jes:+.4f}, px = {px:.3f}, py = {py:.3f}"
        )

    table_jes_met  = pa.concat_tables(tables_jes_met,  promote_options="default")
    pq.write_table(table_jes_met, OUTPUT_JES_MET)
    print(f"  JES+MET rows  : {table_jes_met.num_rows:,}")

    print("Done.")


if __name__ == "__main__":
    main()