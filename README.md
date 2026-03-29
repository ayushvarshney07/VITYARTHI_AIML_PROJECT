
# JEE Main Rank Predictor

A command-line Python tool that predicts your JEE Main rank from your subject-wise marks using a machine learning model. Just enter your Maths, Physics, and Chemistry scores along with your category and gender — and it gives you an estimated rank, percentile, and visual charts to understand where you stand.

---

## What It Does

- Predicts your **JEE Main rank** based on Maths, Physics, and Chemistry marks
- Gives separate **category rank** (General or Reserved) and **general rank**
- Shows your **approximate percentile**
- Generates **visual charts** — bell curves, scatter plots, and subject comparisons
- Saves your **prediction history** locally so you can track multiple mock test attempts
- Works completely **offline** — no internet required, no external data files needed

---

## How It Works (Quick Overview)

The tool generates a realistic synthetic dataset of 12,000 student records in-memory using normal distribution statistics that match the actual JEE score pattern. It then trains a **Gradient Boosting Regressor** model — four separate models actually, one each for General Male, General Female, Reserved Male, and Reserved Female. When you enter your marks, the right model predicts your rank.

---

## Requirements

Make sure you have Python 3.7 or above installed. Then install the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib
```

That's it. No dataset downloads. No API keys. No setup files.

---

## How to Run

1. Clone or download this repository
2. Open a terminal and navigate to the project folder
3. Run the script:

```bash
python jee_rank_calculator.py
```

The program will:
- Automatically generate the training dataset in memory
- Train the ML models (takes about 15–30 seconds)
- Show you a menu to start predicting

---

## Menu Options

Once the program starts, you will see this menu:

```
1. Predict my rank
2. View my prediction history
3. Show dataset overview chart (Bell Curves)
4. Show prediction analysis chart (last prediction)
5. Show dataset information
0. Exit
```

**Option 1** is the main one. It will ask you:
- Maths marks (0 to 100)
- Physics marks (0 to 100)
- Chemistry marks (0 to 100)
- Your category (General or Reserved)
- Your gender (Male or Female)

After that, it shows your predicted rank and optionally opens a chart.

---

## Output Example

```
════════════════════════════════════════════════
  🎯  Prediction Result
════════════════════════════════════════════════
  Total marks      : 210 / 300
  Category         : General
  Gender           : Male

  📌  Your Category Rank  : 8420
  🏆  Your General Rank   : 8420
  📈  Approx Percentile   : 99.16%
```

---

## Charts Generated

After a prediction, you can view two charts:

**Dataset Overview** — shows:
- Marks distribution bell curve (General vs Reserved, Male vs Female)
- Subject-wise average marks over the years
- Rank distribution across the synthetic dataset

**Prediction Analysis** — shows:
- Scatter plot of all students with your position marked as a red star
- Bar chart comparing your marks to the dataset average
- Yearly average marks trend by category

All charts are saved to the `charts/` folder inside the project directory as PNG files.

---

## Prediction History

Every prediction you make is saved to `data/my_predictions.csv`. You can view this history any time using Option 2 in the menu. This is useful if you're taking multiple mock tests and want to see how your predicted rank changes over time.

---

## Project Structure

```
jee_rank_calculator.py   ← main script (everything is here)
data/
  my_predictions.csv     ← auto-created, stores your prediction history
charts/
  prediction_analysis.png   ← auto-created when you view charts
  dataset_overview.png      ← auto-created when you view charts
```

---

## Known Limitations

- The training data is synthetic (statistically realistic, but not real NTA data)
- The reserved-to-general rank conversion uses an approximate factor of 1.8
- State-wise rank is not predicted
- Runs in the terminal only — no web interface

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| numpy | any recent | Dataset generation, array math |
| pandas | any recent | Data manipulation |
| scikit-learn | any recent | Gradient Boosting model |
| matplotlib | any recent | Charts and visualisation |

---

## Author

**ANMOL KUMAR | 25BCE11219**

The problem was chosen because rank prediction is something literally every JEE student needs after their exam, and existing tools are either overly simplistic or behind paywalls.
