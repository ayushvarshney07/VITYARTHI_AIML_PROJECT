

import os
import sys
import csv
import math
import subprocess
import random
from datetime import date



try:
    import numpy as np
except ImportError:
    print("Missing library. Run:  pip install numpy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Missing library. Run:  pip install pandas")
    sys.exit(1)

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
except ImportError:
    print("Missing library. Run:  pip install scikit-learn")
    sys.exit(1)

try:
    import matplotlib
    try:
        import tkinter
        matplotlib.use("TkAgg")
    except ImportError:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker
except ImportError:
    print("Missing library. Run:  pip install matplotlib")
    sys.exit(1)



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(SCRIPT_DIR, "data", "my_predictions.csv")
CHART_DIR    = os.path.join(SCRIPT_DIR, "charts")


RESERVED_TO_GENERAL_FACTOR = 1.8




def generate_realistic_dataset(num_records=10000) -> pd.DataFrame:
    """
    Generates a statistically realistic dataset in-memory.
    Uses normal distributions to simulate the JEE bell curve.
    """
    print("\n  [⚙️] Generating standalone realistic dataset in-memory...")
    np.random.seed(42)
    data = []
    for _ in range(num_records):
        year = random.randint(2013, 2025)

        category = np.random.choice(["General", "Reserved"], p=[0.55, 0.45])
        sex = np.random.choice(["M", "F"], p=[0.70, 0.30])
        
       
        total_raw = np.random.normal(loc=110, scale=55)
        tot = int(np.clip(total_raw, 10, 300))
        
       
        base_split = tot / 3.0
        m = int(np.clip(np.random.normal(base_split, 10), 0, 100))
        p = int(np.clip(np.random.normal(base_split, 10), 0, 100))
        
       
        c = tot - (m + p)

        if c < 0:
            diff = abs(c)
            c = 0
      
            if m > p: m -= diff
            else: p -= diff
        elif c > 100:
            diff = c - 100
            c = 100
  
            if m < p: m += diff
            else: p += diff
            

        tot = m + p + c 
        
        data.append([year, category, sex, "State", m, p, c, tot])
        
    df = pd.DataFrame(data, columns=["Year", "Category", "Sex", "State",
                                     "Maths_Marks", "Physics_Marks", "Chemistry_Marks",
                                     "Total_Marks"])
    
   
    df = df.sort_values(by=["Category", "Sex", "Total_Marks"], ascending=[True, True, False])
    
 
    df["Rank"] = df.groupby(["Category", "Sex"]).cumcount() + 1
    
    
    df["Percentile"] = 100.0 - ((df["Rank"] - 1) / df.groupby(["Category", "Sex"])["Rank"].transform('max') * 100.0)
    
    print(f"  [✓] Successfully generated {num_records:,} synthetic records.")
    return df


def load_data() -> pd.DataFrame:
 
    df = generate_realistic_dataset(num_records=12000)

    needed = ["Year", "Category", "Sex",
              "Maths_Marks", "Physics_Marks", "Chemistry_Marks",
              "Total_Marks", "Rank", "Percentile"]
    
    df = df[needed].copy()
    
    num_cols = ["Year", "Maths_Marks", "Physics_Marks",
                "Chemistry_Marks", "Total_Marks", "Rank", "Percentile"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df.reset_index(drop=True)

    print(f"\n  Dataset loaded successfully!")
    print(f"  Total records : {len(df):,}")
    print(f"  Years covered : {int(df['Year'].min())} to {int(df['Year'].max())}")
    print(f"  Marks range   : {df['Total_Marks'].min():.0f} to {df['Total_Marks'].max():.0f}")
    
    return df



def build_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, r in df.iterrows():
        m   = float(r["Maths_Marks"])
        p   = float(r["Physics_Marks"])
        c   = float(r["Chemistry_Marks"])
        tot = float(r["Total_Marks"])

        rows.append([
            m, p, c, tot,
            tot ** 2,            
            math.log1p(tot),     
            m - p,               
            m - c,               
            abs(m - p) + abs(p - c) + abs(m - c),  
        ])

    return np.array(rows, dtype=float)




def train_all_models(df: pd.DataFrame) -> dict:
    groups = [
        ("General",  "M", "General Male"),
        ("General",  "F", "General Female"),
        ("Reserved", "M", "Reserved Male"),
        ("Reserved", "F", "Reserved Female"),
    ]

    models = {}
    print("\n  Training ML models on standalone dataset...")
    print("  " + "─" * 50)

    for cat, sex, label in groups:
        subset = df[(df["Category"] == cat) & (df["Sex"] == sex)].copy()
        
        if len(subset) < 10:
            print(f"  {label:<20}  Not enough data for this group.")
            continue

        X = build_features(subset)
        y = subset["Rank"].values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            min_samples_leaf=3, subsample=0.85, random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)

        models[(cat, sex)] = model
        print(f"  {label:<20}  rows: {len(subset):>5}  MAE: {mae:.2f}  R²: {r2:.4f}")

    print("  " + "─" * 50)
    return models




def predict(models: dict, maths: float, physics: float, chemistry: float,
            category: str, sex: str) -> dict:
    total = maths + physics + chemistry


    if total >= 300:
        return {
            "maths":        maths,
            "physics":      physics,
            "chemistry":    chemistry,
            "total":        total,
            "category":     category,
            "sex":          sex,
            "cat_rank":     1,
            "cat_rank_lo":  1,
            "cat_rank_hi":  1,
            "gen_rank":     1,
            "gen_rank_lo":  1,
            "gen_rank_hi":  1,
            "percentile":   100.00,
            "date":         date.today().isoformat(),
        }

    row = pd.DataFrame([{
        "Maths_Marks":    maths,
        "Physics_Marks":  physics,
        "Chemistry_Marks": chemistry,
        "Total_Marks":    total,
    }])
    row["Year"]     = 2025
    row["Category"] = category
    row["Sex"]      = sex

    X = build_features(row)
    
    if (category, sex) not in models:
        print("  Error: Model for this demographic lacked training data.")
        return {}
        
    cat_rank_raw = float(models[(category, sex)].predict(X)[0])
    cat_rank     = max(1, int(round(cat_rank_raw)))

    
    rank_lo = max(1, int(cat_rank * 0.85))
    rank_hi = int(cat_rank * 1.15)

    if category == "General":
        gen_rank    = cat_rank
        gen_rank_lo = rank_lo
        gen_rank_hi = rank_hi
    else:
        gen_rank    = max(1, int(cat_rank * RESERVED_TO_GENERAL_FACTOR))
        gen_rank_lo = max(1, int(gen_rank * 0.85))
        gen_rank_hi = int(gen_rank * 1.15)

    scaled_rank = gen_rank * 100
    percentile_raw = 100.0 - (scaled_rank / 1_000_000 * 100.0)
    cat_percentile = min(99.99, max(0.01, percentile_raw))

    return {
        "maths":        maths,
        "physics":      physics,
        "chemistry":    chemistry,
        "total":        total,
        "category":     category,
        "sex":          sex,
        "cat_rank":     cat_rank,
        "cat_rank_lo":  rank_lo,
        "cat_rank_hi":  rank_hi,
        "gen_rank":     gen_rank,
        "gen_rank_lo":  gen_rank_lo,
        "gen_rank_hi":  gen_rank_hi,
        "percentile":   cat_percentile,
        "date":         date.today().isoformat(),
    }




def open_image(path: str):
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except Exception:
        pass

def save_and_show(fig, filename: str):
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if "agg" in matplotlib.get_backend().lower():
        print(f"  Chart saved → {path}")
        open_image(os.path.abspath(path))
    else:
        plt.show()
        print(f"  Chart saved → {path}")
    plt.close()

BG   = "#0D1117"
GRID = "#1E2530"

def style_ax(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2A2A2A")
    ax.tick_params(colors="#888888", labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)

def plot_prediction_chart(df: pd.DataFrame, result: dict):
    if not result: return
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        style_ax(ax)

    cat    = result["category"]
    sex    = result["sex"]
    total  = result["total"]
    c_rank = result["cat_rank"]
    g_rank = result["gen_rank"]

   
    df_plot = df[df["Total_Marks"] >= 150]

    
    gm = df_plot[(df_plot["Category"] == "General") & (df_plot["Sex"] == "M")]
    gf = df_plot[(df_plot["Category"] == "General") & (df_plot["Sex"] == "F")]

    ax1.scatter(gm["Total_Marks"], gm["Rank"], color="#4FC3F7", s=10, alpha=0.3, label="General Male")
    ax1.scatter(gf["Total_Marks"], gf["Rank"], color="#F48FB1", s=10, alpha=0.3, label="General Female")

    plot_rank = c_rank if cat == "General" else g_rank
    ax1.scatter([total], [plot_rank], color="#FF5252", s=200, zorder=6, marker="*", label="You")
    ax1.annotate(f" Rank ≈ {g_rank}", xy=(total, plot_rank), color="#FF5252", fontsize=9, fontweight="bold", va="center")

    ax1.set_xlabel("Total Marks", color="#AAAAAA")
    ax1.set_ylabel("Rank", color="#AAAAAA")
    ax1.set_title("General Category — Marks vs Rank", color="white", fontweight="bold")
    ax1.legend(facecolor="#161B22", labelcolor="white", fontsize=8)
    ax1.invert_yaxis()

    # Panel 2
    rm = df_plot[(df_plot["Category"] == "Reserved") & (df_plot["Sex"] == "M")]
    rf = df_plot[(df_plot["Category"] == "Reserved") & (df_plot["Sex"] == "F")]

    ax2.scatter(rm["Total_Marks"], rm["Rank"], color="#FFB74D", s=10, alpha=0.3, label="Reserved Male")
    ax2.scatter(rf["Total_Marks"], rf["Rank"], color="#A5D6A7", s=10, alpha=0.3, label="Reserved Female")

    res_rank = c_rank if cat == "Reserved" else max(1, int(c_rank / RESERVED_TO_GENERAL_FACTOR))
    ax2.scatter([total], [res_rank], color="#FF5252", s=200, zorder=6, marker="*", label="You")
    ax2.annotate(f" Rank ≈ {res_rank}", xy=(total, res_rank), color="#FF5252", fontsize=9, fontweight="bold", va="center")

    ax2.set_xlabel("Total Marks", color="#AAAAAA")
    ax2.set_ylabel("Rank", color="#AAAAAA")
    ax2.set_title("Reserved Category — Marks vs Rank", color="white", fontweight="bold")
    ax2.legend(facecolor="#161B22", labelcolor="white", fontsize=8)
    ax2.invert_yaxis()

  
    avg_m = df["Maths_Marks"].mean()
    avg_p = df["Physics_Marks"].mean()
    avg_c = df["Chemistry_Marks"].mean()

    subjects = ["Maths", "Physics", "Chemistry"]
    user_vals = [result["maths"], result["physics"], result["chemistry"]]
    avg_vals  = [avg_m, avg_p, avg_c]

    x     = np.arange(len(subjects))
    width = 0.35

    bars1 = ax3.bar(x - width / 2, user_vals, width, color="#4FC3F7", alpha=0.85, label="Your marks")
    bars2 = ax3.bar(x + width / 2, avg_vals,  width, color="#FFB74D", alpha=0.85, label="Dataset average")

    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{bar.get_height():.0f}", ha="center", va="bottom", color="white", fontsize=9)
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{bar.get_height():.1f}", ha="center", va="bottom", color="#AAAAAA", fontsize=9)

    ax3.set_xticks(x)
    ax3.set_xticklabels(subjects, color="#AAAAAA")
    ax3.set_ylabel("Marks", color="#AAAAAA")
    ax3.set_title("Your Marks vs Dataset Average", color="white", fontweight="bold")
    ax3.legend(facecolor="#161B22", labelcolor="white", fontsize=8)
    ax3.set_ylim(0, 115)

   
    yearly_gen = df[df["Category"] == "General"].groupby("Year")["Total_Marks"].mean()
    yearly_res = df[df["Category"] == "Reserved"].groupby("Year")["Total_Marks"].mean()

    ax4.plot(yearly_gen.index.astype(int), yearly_gen.values, "o-", color="#4FC3F7", linewidth=2, label="General")
    ax4.plot(yearly_res.index.astype(int), yearly_res.values, "s-", color="#FFB74D", linewidth=2, label="Reserved")

    ax4.set_xlabel("Year", color="#AAAAAA")
    ax4.set_ylabel("Avg Total Marks", color="#AAAAAA")
    ax4.set_title("Average Marks by Year (Simulated Data)", color="white", fontweight="bold")
    ax4.legend(facecolor="#161B22", labelcolor="white", fontsize=8)
    ax4.tick_params(axis="x", rotation=45)

    sex_label = "Male" if sex == "M" else "Female"
    fig.suptitle(f"JEE Rank Analysis | {cat} {sex_label} | Total: {total:.0f}/300 | Rank ≈ {c_rank}", color="white", fontsize=12, fontweight="bold", y=1.01)

    save_and_show(fig, "prediction_analysis.png")

def plot_dataset_overview(df: pd.DataFrame):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        style_ax(ax)

    gen = df[df["Category"] == "General"]["Total_Marks"]
    res = df[df["Category"] == "Reserved"]["Total_Marks"]
    mal = df[df["Sex"] == "M"]["Total_Marks"]
    fem = df[df["Sex"] == "F"]["Total_Marks"]

    bins = np.arange(0, 310, 10)
    ax1.hist(gen, bins=bins, color="#4FC3F7", alpha=0.7, label=f"General")
    ax1.hist(res, bins=bins, color="#FFB74D", alpha=0.7, label=f"Reserved")
    ax1.set_title("Marks Distribution: General vs Reserved (Bell Curve)", color="white", fontweight="bold")
    ax1.legend(facecolor="#161B22", labelcolor="white")

    ax2.hist(mal, bins=bins, color="#7986CB", alpha=0.7, label=f"Male")
    ax2.hist(fem, bins=bins, color="#F48FB1", alpha=0.7, label=f"Female")
    ax2.set_title("Marks Distribution: Male vs Female (Bell Curve)", color="white", fontweight="bold")
    ax2.legend(facecolor="#161B22", labelcolor="white")

    yearly = df.groupby("Year")[["Maths_Marks", "Physics_Marks", "Chemistry_Marks"]].mean()
    yrs = yearly.index.astype(int)

    ax3.plot(yrs, yearly["Maths_Marks"], "o-", color="#4FC3F7", linewidth=2, label="Maths")
    ax3.plot(yrs, yearly["Physics_Marks"], "s-", color="#FFB74D", linewidth=2, label="Physics")
    ax3.plot(yrs, yearly["Chemistry_Marks"], "^-", color="#A5D6A7", linewidth=2, label="Chemistry")
    ax3.set_title("Subject-wise Avg Marks", color="white", fontweight="bold")
    ax3.legend(facecolor="#161B22", labelcolor="white")
    ax3.tick_params(axis="x", rotation=45)

    rank_bins = np.arange(0, df["Rank"].max() + 500, 500)
    gen_ranks = df[df["Category"] == "General"]["Rank"]
    res_ranks = df[df["Category"] == "Reserved"]["Rank"]

    ax4.hist(gen_ranks, bins=rank_bins, color="#4FC3F7", alpha=0.7, label="General")
    ax4.hist(res_ranks, bins=rank_bins, color="#FFB74D", alpha=0.7, label="Reserved")
    ax4.set_title("Synthetic Rank Distribution", color="white", fontweight="bold")
    ax4.legend(facecolor="#161B22", labelcolor="white")

    fig.suptitle("JEE Main Dataset Overview (Synthetic Data)", color="white", fontsize=14, fontweight="bold", y=1.01)
    save_and_show(fig, "dataset_overview.png")




def save_to_history(result: dict):
    if not result: return
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    fields = ["date", "maths", "physics", "chemistry", "total",
              "category", "sex", "cat_rank", "gen_rank", "percentile"]
    exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        w.writerow({k: result[k] for k in fields})

def show_history():
    if not os.path.exists(HISTORY_FILE):
        print("\n  No predictions saved yet. Use option 1 first!\n")
        return
    df = pd.read_csv(HISTORY_FILE)
    if df.empty: return

    print("\n" + "═" * 80)
    print("  📋  Your Prediction History")
    print("═" * 80)
    for _, r in df.iterrows():
        sex_label = "Male" if str(r["sex"]) == "M" else "Female"
        print(f"  {str(r['date']):<12} {int(r['maths']):>4} {int(r['physics']):>4} {int(r['chemistry']):>4} "
              f"{int(r['total']):>6}  {str(r['category']):<10} {sex_label:>7}  "
              f"{int(r['cat_rank']):>9}  {int(r['gen_rank']):>9}")
    print()




def show_dataset_info(df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  📊  Synthetic Dataset Information")
    print("═" * 60)
    print(f"  Type          : 100% In-Memory Auto-Generated")
    print(f"  Total records : {len(df):,}")
    print(f"  Total marks   : {df['Total_Marks'].min():.0f} to {df['Total_Marks'].max():.0f}")
    print()




def pick_option(prompt: str, options: list) -> str:
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1): print(f"  {i}. {o}")
    while True:
        try:
            n = int(input("  Your choice: ").strip())
            if 1 <= n <= len(options): return options[n - 1]
        except ValueError:
            pass
        print("  Invalid choice.")

def enter_marks(subject: str) -> float:
    while True:
        try:
            val = float(input(f"  {subject} marks (0 to 100): ").strip())
            if 0 <= val <= 100: return round(val, 1)
        except ValueError:
            pass
        print("  Invalid input.")




def show_result(result: dict):
    if not result: return
    sex_label = "Male" if result["sex"] == "M" else "Female"
    
    print("\n" + "═" * 60)
    print("  🎯  Prediction Result")
    print("═" * 60)
    print(f"  Total marks      : {result['total']:.0f} / 300")
    print(f"  Category         : {result['category']}")
    print(f"  Gender           : {sex_label}")
    print("\n" + "─" * 60)
    print(f"  📌  Your Category Rank  : {result['cat_rank']}")
    print(f"  🏆  Your General Rank   : {result['gen_rank']}")
    print(f"  📈  Approx Percentile   : {result['percentile']:.2f}%")
    print("\n" + "─" * 60)




def menu():
    os.makedirs("data", exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

    print("\n" + "═" * 60)
    print("  🎯  JEE Main Rank Predictor (Standalone Version)")
    print("═" * 60)

    df     = load_data()
    models = train_all_models(df)
    last_result = None

    while True:
        print("\n" + "─" * 40)
        print("  1. Predict my rank")
        print("  2. View my prediction history")
        print("  3. Show dataset overview chart (Bell Curves)")
        print("  4. Show prediction analysis chart (last prediction)")
        print("  5. Show dataset information")
        print("  0. Exit")
        print()

        choice = input("  Choose an option: ").strip()

        if choice == "1":
            print("\n" + "═" * 60)
            maths     = enter_marks("Maths")
            physics   = enter_marks("Physics")
            chemistry = enter_marks("Chemistry")
            category  = pick_option("Select your category:", ["General", "Reserved"])
            gender    = pick_option("Select your gender:", ["Male", "Female"])
            sex       = "M" if gender == "Male" else "F"

            result      = predict(models, maths, physics, chemistry, category, sex)
            last_result = result
            show_result(result)
            save_to_history(result)

            if result:
                show = input("  Show analysis chart for this prediction? (y/n): ").strip().lower()
                if show == "y": plot_prediction_chart(df, result)

        elif choice == "2": show_history()
        elif choice == "3": plot_dataset_overview(df)
        elif choice == "4": 
            if last_result: plot_prediction_chart(df, last_result)
            else: print("\n  Please make a prediction first (option 1).")
        elif choice == "5": show_dataset_info(df)
        elif choice == "0": break
        else: print("  Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
