import pandas as pd
import numpy as np
from te_optimizer import load_te_inputs, compute_te_components, optimize_weights_for_target_te

inputs_path = r"C:\Users\gisel\DIG\te_inputs.xlsx"  # <- Change file path (copy from files)


def _parse_numeric(value):
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith("%"):
            return float(cleaned.rstrip("%")) / 100.0
        return float(cleaned)
    return float(value)


def load_te_inputs_excel(xlsx_path):
    try:
        sector_df = pd.read_excel(xlsx_path, sheet_name="Sector")
    except ImportError as exc:
        raise ImportError(
            "Reading .xlsx requires openpyxl. Install it or save the inputs as CSV."
        ) from exc
    if sector_df.empty:
        raise ValueError("Sector sheet is empty.")
    params = sector_df.iloc[0].to_dict()
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
        if key in params and params[key] is not None:
            params[key] = _parse_numeric(params[key])

    holdings_df = pd.read_excel(xlsx_path, sheet_name="Current_Holdings")
    if holdings_df.empty:
        raise ValueError("Current_Holdings sheet is empty.")
    holdings_df = holdings_df.rename(columns={c: str(c).strip() for c in holdings_df.columns})
    normalized_map = {}
    for col in holdings_df.columns:
        norm = str(col).strip().lower().replace(" ", "_")
        if norm in ("stocks", "ticker"):
            normalized_map[col] = "Ticker"
        elif norm in ("weight", "sector_weight", "sectorweight"):
            normalized_map[col] = "Sector Weight"
        elif norm == "beta":
            normalized_map[col] = "Beta"
        elif norm in ("in_benchmark", "inbenchmark"):
            normalized_map[col] = "in_benchmark"
    holdings_df = holdings_df.rename(columns=normalized_map)
    if "in_benchmark" not in holdings_df.columns:
        holdings_df["in_benchmark"] = False

    required_holdings_cols = {"Ticker", "Sector Weight", "Beta", "in_benchmark"}
    missing_holdings_cols = required_holdings_cols - set(holdings_df.columns)
    if missing_holdings_cols:
        raise ValueError(
            f"Missing required holdings columns: {sorted(missing_holdings_cols)}. "
            f"Found columns: {sorted(holdings_df.columns)}"
        )

    holdings = holdings_df[
        ["Ticker", "Sector Weight", "Beta", "in_benchmark"]
    ].to_dict(orient="records")

    return params, holdings


def load_proposed_holdings_excel(xlsx_path):
    proposed_df = pd.read_excel(xlsx_path, sheet_name="Proposed_Holdings")
    if proposed_df.empty:
        return None
    proposed_df = proposed_df.rename(
        columns={
            "new_holding": "Ticker",
            "proposed_weight": "Proposed Weight",
            "beta": "Beta",
            "inbenchmark": "in_benchmark",
        }
    )
    required_cols = {"Ticker", "Proposed Weight", "Beta"}
    missing_cols = required_cols - set(proposed_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required proposed holdings columns: {sorted(missing_cols)}")
    return proposed_df


## Inputs (from unified CSV or XLSX) ##

if inputs_path.lower().endswith((".xlsx", ".xls")):
    params, holdings = load_te_inputs_excel(inputs_path)
    proposed_holdings_df = load_proposed_holdings_excel(inputs_path)
else:
    params, holdings = load_te_inputs(inputs_path)
    proposed_holdings_df = None

required_params = {
    "sector_name",
    "DIG_sector_weight",
    "SPXE_sector_weight",
    "sector_volatility",
    "DIG_sector_beta",
    "benchmark_sector_beta",
    "market_volatility",
    "sector_guideline",
}
missing_params = required_params - set(params)
if missing_params:
    raise ValueError(f"Missing required PARAM keys: {sorted(missing_params)}")

sector_name = params["sector_name"]
DIG_sector_weight = params["DIG_sector_weight"]
SPXE_sector_weight = params["SPXE_sector_weight"]
sector_volatility = params["sector_volatility"]
DIG_sector_beta = params["DIG_sector_beta"]
benchmark_sector_beta = params["benchmark_sector_beta"]
market_volatility = params["market_volatility"]
sector_guideline = params["sector_guideline"]
target_total_te = params.get("target_total_te", sector_guideline)

holdings_df = pd.DataFrame(holdings)
if holdings_df.empty:
    raise ValueError("No HOLDING rows found in the inputs file.")

if "in_benchmark" in holdings_df.columns and holdings_df["in_benchmark"].dtype != bool:
    holdings_df["in_benchmark"] = (
        holdings_df["in_benchmark"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )
    if holdings_df["in_benchmark"].isna().any():
        raise ValueError("Invalid values in 'in_benchmark'. Use True/False (or yes/no, 1/0).")

weights_sum = holdings_df["Sector Weight"].sum()
if not np.isclose(weights_sum, 1.0):
    if np.isclose(weights_sum, 100.0):
        holdings_df["Sector Weight"] = holdings_df["Sector Weight"] / 100.0
    else:
        raise ValueError(
            "Holdings weights must sum to 1.0 (or 100 if provided as percentages). "
            f"The current sum is {weights_sum}"
        )

## Allocation TE ##

## allocation_gap ##

allocation_deviation = abs(DIG_sector_weight - SPXE_sector_weight)
TE_allocation = abs(allocation_deviation) * sector_volatility

## Beta TE ##

beta_deviation = abs(DIG_sector_beta - benchmark_sector_beta)
TE_beta = DIG_sector_weight * abs(beta_deviation) * market_volatility

## Selection TE ##

non_benchmark_pct = holdings_df.loc[holdings_df["in_benchmark"] == False, "Sector Weight"].sum()
TE_selection = DIG_sector_weight * sector_volatility * non_benchmark_pct    

## Total TE Diagnostic #

total_TE = np.sqrt(
    TE_allocation**2 + 
    TE_beta**2 + 
    TE_selection**2
)

## Comparison to Guideline ##

current_diagnostic = total_TE - sector_guideline
def sector_guidance():
    if current_diagnostic > 0:
        return "Over Guideline: Rebalancing Recommended"
    elif current_diagnostic < 0:
        return "Under Guideline: Room for Active Bets"
    else:
        return "Within Guideline: Good Position"


## Output ##

summary = pd.DataFrame({
    "Metric": [
        "Allocation TE", 
        "Beta TE", 
        "Selection TE", 
        "Total TE",
        "Sector Guidance",
        "Gap"
    ],
    "Value": [
        f"{round(TE_allocation * 100, 2)}%",
        f"{round(TE_beta * 100, 2)}%",
        f"{round(TE_selection * 100, 2)}%",
        f"{round(total_TE * 100, 2)}%",
        sector_guidance(),
        f"{round(current_diagnostic * 100, 2)}%"
    ]
})  

### Problem Identification ###

def order_te_components():
    components = {
        "Allocation TE": TE_allocation,
        "Beta TE": TE_beta,
        "Selection TE": TE_selection
    }
    ordered = sorted(components.items(), key=lambda x: x[1], reverse=True)
    
    return ordered




############################
#### STRATEGIC DECISIONS ###
############################

## Decision 1: Sector Allocation ##
new_DIG_allocation = DIG_sector_weight ##change
TE_new_allocation = abs(new_DIG_allocation - SPXE_sector_weight) * sector_volatility


## Decision 2: Beta Matching ##

new_DIG_sector_beta = benchmark_sector_beta ##change
TE_new_beta = new_DIG_allocation * abs(new_DIG_sector_beta - benchmark_sector_beta) * market_volatility


## Dection 3: Stock Selection ##

new_DIG_selection = non_benchmark_pct ## change
TE_new_selection = new_DIG_allocation * sector_volatility * new_DIG_selection


## New Total TE ##
new_total_TE = np.sqrt(
    TE_new_allocation**2 + 
    TE_new_beta**2 + 
    TE_new_selection**2
)   
print(new_total_TE)

####################################
########## Proposed Trades ##########
####################################


weights = holdings_df["Sector Weight"].to_numpy(dtype=float)
betas = holdings_df["Beta"].to_numpy(dtype=float)
in_benchmark = holdings_df["in_benchmark"].to_numpy(dtype=bool)

optimizer_result = optimize_weights_for_target_te(
    weights=weights,
    betas=betas,
    in_benchmark=in_benchmark,
    target_total_te=target_total_te,
    dig_sector_weight=DIG_sector_weight,
    spxe_sector_weight=SPXE_sector_weight,
    sector_volatility=sector_volatility,
    benchmark_sector_beta=benchmark_sector_beta,
    market_volatility=market_volatility,
)

optimized_weights = optimizer_result["weights"]
optimized_metrics = optimizer_result["metrics"]

proposed_trades = pd.DataFrame(
    {
        "Updated Tickers": holdings_df["Ticker"].values,
        "New Betas": holdings_df["Beta"].values,
        "in_benchmark": holdings_df["in_benchmark"].values,
        "Current Weight": holdings_df["Sector Weight"].values,
        "Proposed Weight": optimized_weights,
    }
)
proposed_trades["Delta"] = proposed_trades["Proposed Weight"] - proposed_trades["Current Weight"]







## Fun part: Optimizer ###
""" This is Beta (not like beta but like new and untested) """



print(f"\nSector: {sector_name}")
print(summary)
print("\nOrdered TE Components (Largest to Smallest):")
ordered_components = order_te_components()
for component, value in ordered_components:
    print(f"{component}: {value*100:.2f}%")  
print(f"New Total TE (manual inputs): {new_total_TE:.6f}")

if proposed_holdings_df is not None:
    proposed_holdings_df = proposed_holdings_df.rename(
        columns={c: str(c).strip() for c in proposed_holdings_df.columns}
    )
    if "in_benchmark" not in proposed_holdings_df.columns:
        proposed_holdings_df["in_benchmark"] = False

    proposed_holdings_df["Ticker"] = proposed_holdings_df["Ticker"].astype(str)
    if proposed_holdings_df["in_benchmark"].dtype != bool:
        proposed_holdings_df["in_benchmark"] = (
            proposed_holdings_df["in_benchmark"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        )
        if proposed_holdings_df["in_benchmark"].isna().any():
            raise ValueError(
                "Invalid values in Proposed_Holdings 'in_benchmark'. Use True/False (or yes/no, 1/0)."
            )

    proposed_weights = proposed_holdings_df["Proposed Weight"].astype(float).copy()
    proposed_betas = proposed_holdings_df["Beta"].astype(float).copy()
    total_proposed = float(proposed_weights.sum())
    if not np.isclose(total_proposed, 1.0):
        if np.isclose(total_proposed, 100.0):
            proposed_weights = proposed_weights / 100.0
        else:
            proposed_weights = proposed_weights / total_proposed

    manual_metrics = compute_te_components(
        weights=proposed_weights.values,
        betas=proposed_betas.values,
        in_benchmark=proposed_holdings_df["in_benchmark"].values,
        dig_sector_weight=DIG_sector_weight,
        spxe_sector_weight=SPXE_sector_weight,
        sector_volatility=sector_volatility,
        benchmark_sector_beta=benchmark_sector_beta,
        market_volatility=market_volatility,
    )
    print("\nManual Proposed_Holdings TE Metrics:")
    print(
        pd.DataFrame(
            {
                "Metric": ["Allocation TE", "Beta TE", "Selection TE", "Total TE", "Gap"],
                "Value": [
                    f"{manual_metrics['TE_allocation']*100:.2f}%",
                    f"{manual_metrics['TE_beta']*100:.2f}%",
                    f"{manual_metrics['TE_selection']*100:.2f}%",
                    f"{manual_metrics['total_TE']*100:.2f}%",
                    f"{(manual_metrics['total_TE'] - target_total_te)*100:.2f}%",
                ],
            }
        )
    )

    proposed_optimizer_result = optimize_weights_for_target_te(
        weights=proposed_weights.values,
        betas=proposed_betas.values,
        in_benchmark=proposed_holdings_df["in_benchmark"].values,
        target_total_te=target_total_te,
        dig_sector_weight=DIG_sector_weight,
        spxe_sector_weight=SPXE_sector_weight,
        sector_volatility=sector_volatility,
        benchmark_sector_beta=benchmark_sector_beta,
        market_volatility=market_volatility,
    )
    proposed_optimized_weights = proposed_optimizer_result["weights"]
    proposed_optimized_metrics = proposed_optimizer_result["metrics"]

    proposed_only_trades = pd.DataFrame(
        {
            "Proposed Tickers": proposed_holdings_df["Ticker"].values,
            "New Betas": proposed_holdings_df["Beta"].values,
            "in_benchmark": proposed_holdings_df["in_benchmark"].values,
            "Current Weight": proposed_weights.values,
            "Proposed Weight": proposed_optimized_weights,
        }
    )
    proposed_only_trades["Delta"] = (
        proposed_only_trades["Proposed Weight"] - proposed_only_trades["Current Weight"]
    )

print(f"\nOptimized Weights (Current Holdings, Target TE = {target_total_te:.4f}):")
print(proposed_trades)
print("\nOptimized TE Metrics (Current Holdings):")
print(
    pd.DataFrame(
        {
            "Metric": ["Allocation TE", "Beta TE", "Selection TE", "Total TE", "Gap"],
            "Value": [
                f"{optimized_metrics['TE_allocation']*100:.2f}%",
                f"{optimized_metrics['TE_beta']*100:.2f}%",
                f"{optimized_metrics['TE_selection']*100:.2f}%",
                f"{optimized_metrics['total_TE']*100:.2f}%",
                f"{(optimized_metrics['total_TE'] - target_total_te)*100:.2f}%",
            ],
        }
    )
)

if proposed_holdings_df is not None:
    print(f"\nOptimized Weights (Proposed Holdings, Target TE = {target_total_te:.4f}):")
    print(proposed_only_trades)
    print("\nOptimized TE Metrics (Proposed Holdings):")
    print(
        pd.DataFrame(
            {
                "Metric": ["Allocation TE", "Beta TE", "Selection TE", "Total TE", "Gap"],
                "Value": [
                    f"{proposed_optimized_metrics['TE_allocation']*100:.2f}%",
                    f"{proposed_optimized_metrics['TE_beta']*100:.2f}%",
                    f"{proposed_optimized_metrics['TE_selection']*100:.2f}%",
                    f"{proposed_optimized_metrics['total_TE']*100:.2f}%",
                    f"{(proposed_optimized_metrics['total_TE'] - target_total_te)*100:.2f}%",
                ],
            }
        )
    )
