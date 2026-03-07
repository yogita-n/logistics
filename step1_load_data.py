import pandas as pd

# ── Load orders ──────────────────────────────────────────────────
# Correct path based on YOUR folder structure
orders_df = pd.read_excel(
    "data/orders/orders.xlsx",
    sheet_name=0   # sheet index 0 = first sheet (Day 1)
)

print("=== ALL SHEET NAMES IN orders.xlsx ===")
# Check how many sheets (days) exist
import openpyxl
wb = openpyxl.load_workbook("data/orders/orders.xlsx")
print(wb.sheetnames)

print("\n=== COLUMNS IN orders.xlsx (Sheet 1) ===")
print(orders_df.columns.tolist())

print("\n=== FIRST 5 ROWS ===")
print(orders_df.head())

print(f"\n=== TOTAL ROWS (Day 1): {len(orders_df)} ===")


# ── Load time matrix (Day 1, normal traffic) ──────────────────────
# Correct path: data/time_and_distance_matrices/day_1/
time_df = pd.read_excel(
    "data/time_and_distance_matrices/day_1/time_matrix_mostlikely_1.xlsx",
    index_col=0
)

print("\n=== TIME MATRIX SHAPE ===")
print(f"Rows: {time_df.shape[0]}, Columns: {time_df.shape[1]}")

print("\n=== TIME MATRIX COLUMN NAMES (first 5) ===")
print(time_df.columns.tolist()[:5])

print("\n=== TIME MATRIX ROW INDEX NAMES (first 5) ===")
print(time_df.index.tolist()[:5])

print("\n=== TIME MATRIX (top-left 5x5 values) ===")
print(time_df.iloc[:5, :5])


# ── Load distance matrix (Day 1) ──────────────────────────────────
dist_df = pd.read_excel(
    "data/time_and_distance_matrices/day_1/distance_matrix_1.xlsx",
    index_col=0
)

print("\n=== DISTANCE MATRIX SHAPE ===")
print(f"Rows: {dist_df.shape[0]}, Columns: {dist_df.shape[1]}")

print("\n=== DISTANCE MATRIX (top-left 5x5 values) ===")
print(dist_df.iloc[:5, :5])
