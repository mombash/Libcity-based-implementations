#!/usr/bin/env python3
"""
Build combined results table CSV: normalized MAE, normalized RMSE, raw MAPE
in the format of the table in MCST-Mamba-Journal-Paper/main.tex.

Table columns: Model, Metric, PEMS04_Flow, PEMS04_Occupancy, PEMS04_Speed, PEMS04_Average,
               PEMS08_Flow, PEMS08_Occupancy, PEMS08_Speed, PEMS08_Average
Metrics: vMAE, vRMSE, MAPE (raw, as percentage)
Channel order: Flow (ch0), Occupancy (ch1), Speed (ch2)
Mamba4Traffic = MCSTMambaLST_Ablation
STAEformer from STAEformer/PEMS04_benchmarks and STAEformer/PEMS08_benchmarks (CSVs outside normalized folders for raw MAPE; normalized folders for vMAE/vRMSE)
"""
import csv
from pathlib import Path
from typing import Optional

BASELINES_ROOT = Path(__file__).resolve().parent / "_baselines"
STAE_PEMS04 = Path(__file__).resolve().parent.parent / "STAEformer" / "PEMS04_benchmarks"
STAE_PEMS08 = Path(__file__).resolve().parent.parent / "STAEformer" / "PEMS08_benchmarks"
HORIZON_ROW = 11  # 0-based index for horizon 12 (12th row)

# Paper table model order
MODELS_ORDER = [
    "GWNET",
    "MTGNN",
    "STGCN",
    "D2STGNN",
    "GMAN",
    "DCRNN",
    "GTS",
    "STAEformer",
    "Mamba4Traffic",
]

# LibCity folder name -> paper name
BASELINE_TO_PAPER = {
    "MCSTMambaLST_Ablation": "Mamba4Traffic",
}


def get_libcity_row(path: Path, row_index: int) -> Optional[dict]:
    """Read LibCity-format CSV (header MAE,MAPE,MSE,RMSE,...), return row as dict."""
    if not path.exists():
        return None
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if row_index >= len(rows):
        return None
    return rows[row_index]


def get_stae_row(path: Path, step_name: str = "step_12") -> Optional[dict]:
    """Read STAEformer CSV (step, mse, rmse, mae, mape or step, rmse, mae, mape), return row for step_12."""
    if not path.exists():
        return None
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("step") == step_name:
                return row
    return None


def collect_libcity_baseline(dataset: str, model: str) -> dict:
    """Return dict with vMAE_flow, vMAE_occ, vMAE_speed, vMAE_avg, vRMSE_*, MAPE_flow, MAPE_occ, MAPE_speed, MAPE_avg."""
    base = BASELINES_ROOT / dataset / model
    out = {}
    # Normalized per channel: channel_0=Flow, channel_1=Occupancy, channel_2=Speed
    for ch, name in [(0, "flow"), (1, "occupancy"), (2, "speed")]:
        path = base / "Normalized" / "Per Channel" / f"channel_{ch}_metrics.csv"
        row = get_libcity_row(path, HORIZON_ROW)
        if row:
            out[f"vMAE_{name}"] = float(row["MAE"])
            out[f"vRMSE_{name}"] = float(row["RMSE"])
        path_raw = base / "Raw" / f"channel_{ch}_metrics.csv"
        row_raw = get_libcity_row(path_raw, HORIZON_ROW)
        if row_raw:
            mape_val = row_raw.get("masked_MAPE") or row_raw.get("MAPE")
            if mape_val and str(mape_val).lower() != "inf":
                out[f"MAPE_{name}"] = float(mape_val) * 100.0  # fraction -> percentage
            else:
                out[f"MAPE_{name}"] = None
    # Average normalized
    avg_path = next((base / "Normalized" / "Average").glob("*.csv"), None)
    if avg_path:
        row = get_libcity_row(avg_path, HORIZON_ROW)
        if row:
            out["vMAE_avg"] = float(row["MAE"])
            out["vRMSE_avg"] = float(row["RMSE"])
    # Average raw MAPE
    if all(k in out for k in ["MAPE_flow", "MAPE_occupancy", "MAPE_speed"]):
        vals = [out["MAPE_flow"], out["MAPE_occupancy"], out["MAPE_speed"]]
        vals = [v for v in vals if v is not None]
        out["MAPE_avg"] = sum(vals) / len(vals) if vals else None
    else:
        out["MAPE_avg"] = None
    return out


def collect_staeformer() -> dict:
    """Return dict dataset -> { flow, occupancy, speed, avg } for vMAE, vRMSE, MAPE."""
    def read_stae_channel(dataset: str, channel: str, normalized: bool) -> Optional[dict]:
        root = STAE_PEMS04 if dataset == "PEMS04" else STAE_PEMS08
        folder = channel  # flow, speed, occupancy
        if normalized:
            sub = root / folder / "normalized"
            csvs = list(sub.glob("eval_*.csv"))
        else:
            sub = root / folder
            csvs = [f for f in sub.glob("eval_*.csv") if "normalized" not in str(f)]
        if not csvs:
            return None
        return get_stae_row(csvs[0])

    data = {}
    for dataset in ["PEMS04", "PEMS08"]:
        data[dataset] = {"flow": {}, "occupancy": {}, "speed": {}}
        for ch_name in ["flow", "speed", "occupancy"]:
            row_n = read_stae_channel(dataset, ch_name, normalized=True)
            row_r = read_stae_channel(dataset, ch_name, normalized=False)
            if row_n:
                data[dataset][ch_name]["vMAE"] = float(row_n.get("mae", 0))
                data[dataset][ch_name]["vRMSE"] = float(row_n.get("rmse", 0))
            else:
                data[dataset][ch_name]["vMAE"] = None
                data[dataset][ch_name]["vRMSE"] = None
            if row_r and row_r.get("mape"):
                data[dataset][ch_name]["MAPE"] = float(row_r["mape"])  # already percentage
            else:
                data[dataset][ch_name]["MAPE"] = None
        # averages
        for metric in ["vMAE", "vRMSE", "MAPE"]:
            vals = [data[dataset][c][metric] for c in ["flow", "occupancy", "speed"] if data[dataset][c][metric] is not None]
            data[dataset]["avg_" + metric] = sum(vals) / len(vals) if vals else None
    return data


def fmt(x) -> str:
    if x is None:
        return ""
    return f"{x:.3f}".rstrip("0").rstrip(".")


def main():
    # Collect LibCity baselines (excluding MCSTMambaLST for PEMS08 if missing Raw channel_1/2; we use MCSTMambaLST_Ablation as Mamba4Traffic)
    libcity_models = ["GWNET", "MTGNN", "STGCN", "D2STGNN", "GMAN", "DCRNN", "GTS", "MCSTMambaLST_Ablation"]
    results = {}  # paper_name -> { "PEMS04": {...}, "PEMS08": {...} }
    # Dataset folder names in _baselines are PEMSD4 and PEMSD8; we output as PEMS04, PEMS08
    for model in libcity_models:
        paper_name = BASELINE_TO_PAPER.get(model, model)
        results[paper_name] = {"PEMS04": {}, "PEMS08": {}}
        for folder_name, out_key in [("PEMSD4", "PEMS04"), ("PEMSD8", "PEMS08")]:
            folder = BASELINES_ROOT / folder_name / model
            if not folder.exists():
                continue
            results[paper_name][out_key] = collect_libcity_baseline(folder_name, model)

    # STAEformer
    stae_data = collect_staeformer()
    results["STAEformer"] = {
        "PEMS04": {
            "vMAE_flow": stae_data["PEMS04"]["flow"]["vMAE"],
            "vMAE_occupancy": stae_data["PEMS04"]["occupancy"]["vMAE"],
            "vMAE_speed": stae_data["PEMS04"]["speed"]["vMAE"],
            "vMAE_avg": stae_data["PEMS04"]["avg_vMAE"],
            "vRMSE_flow": stae_data["PEMS04"]["flow"]["vRMSE"],
            "vRMSE_occupancy": stae_data["PEMS04"]["occupancy"]["vRMSE"],
            "vRMSE_speed": stae_data["PEMS04"]["speed"]["vRMSE"],
            "vRMSE_avg": stae_data["PEMS04"]["avg_vRMSE"],
            "MAPE_flow": stae_data["PEMS04"]["flow"]["MAPE"],
            "MAPE_occupancy": stae_data["PEMS04"]["occupancy"]["MAPE"],
            "MAPE_speed": stae_data["PEMS04"]["speed"]["MAPE"],
            "MAPE_avg": stae_data["PEMS04"]["avg_MAPE"],
        },
        "PEMS08": {
            "vMAE_flow": stae_data["PEMS08"]["flow"]["vMAE"],
            "vMAE_occupancy": stae_data["PEMS08"]["occupancy"]["vMAE"],
            "vMAE_speed": stae_data["PEMS08"]["speed"]["vMAE"],
            "vMAE_avg": stae_data["PEMS08"]["avg_vMAE"],
            "vRMSE_flow": stae_data["PEMS08"]["flow"]["vRMSE"],
            "vRMSE_occupancy": stae_data["PEMS08"]["occupancy"]["vRMSE"],
            "vRMSE_speed": stae_data["PEMS08"]["speed"]["vRMSE"],
            "vRMSE_avg": stae_data["PEMS08"]["avg_vRMSE"],
            "MAPE_flow": stae_data["PEMS08"]["flow"]["MAPE"],
            "MAPE_occupancy": stae_data["PEMS08"]["occupancy"]["MAPE"],
            "MAPE_speed": stae_data["PEMS08"]["speed"]["MAPE"],
            "MAPE_avg": stae_data["PEMS08"]["avg_MAPE"],
        },
    }

    # Build CSV similar to paper table: Model, Metric, PEMS04_Flow, PEMS04_Occupancy, PEMS04_Speed, PEMS04_Average, PEMS08_Flow, PEMS08_Occupancy, PEMS08_Speed, PEMS08_Average
    cols = [
        "Model", "Metric",
        "PEMS04_Flow", "PEMS04_Occupancy", "PEMS04_Speed", "PEMS04_Average",
        "PEMS08_Flow", "PEMS08_Occupancy", "PEMS08_Speed", "PEMS08_Average",
    ]
    channel_keys = ["flow", "occupancy", "speed", "avg"]
    out_path = BASELINES_ROOT.parent / "vehicular_results_table.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for model in MODELS_ORDER:
            if model not in results:
                continue
            r4 = results[model]["PEMS04"]
            r8 = results[model]["PEMS08"]
            for metric_type, key_prefix in [("vMAE", "vMAE"), ("vRMSE", "vRMSE"), ("MAPE", "MAPE")]:
                row = [model, metric_type]
                for ds, r in [("PEMS04", r4), ("PEMS08", r8)]:
                    for k in channel_keys:
                        key = f"{key_prefix}_{k}"
                        val = r.get(key) if r else None
                        row.append(fmt(val) if val is not None else "")
                w.writerow(row)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
