import csv
import numpy as np


def load_te_inputs(csv_path):
    params = {}
    holdings = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            kind = row[0].strip()
            if not kind:
                continue
            if kind.lower() == "type":
                continue

            kind_upper = kind.upper()
            if kind_upper == "PARAM":
                if len(row) < 3:
                    raise ValueError("PARAM rows must have Type, Key, Value.")
                key = row[1].strip()
                value = row[2].strip()
                if not key:
                    raise ValueError("PARAM key cannot be blank.")
                params[key] = value
            elif kind_upper == "HOLDING":
                if len(row) < 5:
                    raise ValueError(
                        "HOLDING rows must have Type, Ticker, Sector Weight, Beta, in_benchmark."
                    )
                in_benchmark_raw = row[4].strip().lower()
                if in_benchmark_raw in ("true", "1"):
                    in_benchmark = True
                elif in_benchmark_raw in ("false", "0"):
                    in_benchmark = False
                else:
                    raise ValueError("in_benchmark must be TRUE/FALSE or 1/0.")

                holdings.append(
                    {
                        "Ticker": row[1].strip(),
                        "Sector Weight": float(row[2]),
                        "Beta": float(row[3]),
                        "in_benchmark": in_benchmark,
                    }
                )
            else:
                raise ValueError(f"Unknown row type: {kind}")

    numeric_keys = {
        "DIG_sector_weight",
        "SPXE_sector_weight",
        "sector_volatility",
        "DIG_sector_beta",
        "benchmark_sector_beta",
        "market_volatility",
        "sector_guideline",
        "target_total_te",
    }
    for key in numeric_keys:
        if key in params:
            params[key] = float(params[key])

    return params, holdings


def compute_te_components(
    weights,
    betas,
    in_benchmark,
    dig_sector_weight,
    spxe_sector_weight,
    sector_volatility,
    benchmark_sector_beta,
    market_volatility,
):
    weights = np.asarray(weights, dtype=float)
    betas = np.asarray(betas, dtype=float)
    in_benchmark = np.asarray(in_benchmark, dtype=bool)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0. Current sum={weights.sum():.6f}")

    portfolio_beta = float(np.dot(weights, betas))
    allocation_dev = abs(dig_sector_weight - spxe_sector_weight)
    te_allocation = allocation_dev * sector_volatility
    te_beta = dig_sector_weight * abs(portfolio_beta - benchmark_sector_beta) * market_volatility
    non_benchmark_pct = float(weights[~in_benchmark].sum())
    te_selection = dig_sector_weight * sector_volatility * non_benchmark_pct

    total_te = float(np.sqrt(te_allocation**2 + te_beta**2 + te_selection**2))

    return {
        "portfolio_beta": portfolio_beta,
        "non_benchmark_pct": non_benchmark_pct,
        "TE_allocation": te_allocation,
        "TE_beta": te_beta,
        "TE_selection": te_selection,
        "total_TE": total_te,
    }


def optimize_weights_for_target_te(
    weights,
    betas,
    in_benchmark,
    target_total_te,
    dig_sector_weight,
    spxe_sector_weight,
    sector_volatility,
    benchmark_sector_beta,
    market_volatility,
    steps=200,
):
    weights = np.asarray(weights, dtype=float)
    betas = np.asarray(betas, dtype=float)
    in_benchmark = np.asarray(in_benchmark, dtype=bool)

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0. Current sum={weights.sum():.6f}")

    nb_mask = ~in_benchmark
    nb_sum = float(weights[nb_mask].sum())
    b_sum = float(weights[in_benchmark].sum())
    if nb_sum <= 0 or b_sum <= 0:
        raise ValueError("Both benchmark and non-benchmark groups must have positive weight.")

    s_min = 0.0
    s_max = 1.0 / nb_sum
    s_grid = np.linspace(s_min, s_max, steps)

    best = None
    best_gap = float("inf")

    for s in s_grid:
        w_nb = weights[nb_mask] * s
        remaining = 1.0 - float(w_nb.sum())
        if remaining < 0:
            continue
        w_b = weights[in_benchmark] * (remaining / b_sum)
        w_new = weights.copy()
        w_new[nb_mask] = w_nb
        w_new[in_benchmark] = w_b

        metrics = compute_te_components(
            w_new,
            betas,
            in_benchmark,
            dig_sector_weight,
            spxe_sector_weight,
            sector_volatility,
            benchmark_sector_beta,
            market_volatility,
        )
        gap = abs(metrics["total_TE"] - target_total_te)
        if gap < best_gap:
            best_gap = gap
            best = (w_new, metrics)

    if best is None:
        raise ValueError("Unable to find feasible weights for the target TE.")

    return {
        "weights": best[0],
        "metrics": best[1],
        "target_TE": target_total_te,
        "gap": best_gap,
    }
