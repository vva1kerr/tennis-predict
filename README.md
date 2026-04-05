# ATP Tennis Match Predictor

A Random Forest model that predicts the winner of any ATP men's singles match given two player names and a court surface.

## How it works

The model is trained on ~30 years of ATP match data (1991–2025). For every match in the dataset, it engineers a feature vector representing the **difference** between the two players across several categories, then randomly flips half the rows so the classifier sees both player orderings equally. At prediction time the same features are computed from each player's current rolling state.

### Features

| Category | Features |
|---|---|
| Rankings | ATP ranking points diff, ranking diff |
| Biometrics | Age diff, height diff |
| Match context | Best-of (3/5), draw size |
| Head-to-head | Overall H2H wins diff, surface-specific H2H wins diff |
| Experience | Career matches played diff |
| Win rate | Rolling win-rate diff over last 3, 5, 10, 25, 50, 100 matches |
| Serve stats | Rolling diff (last 3–2000 matches) for: ace %, double-fault %, 1st-serve-in %, 1st-serve-won %, 2nd-serve-won %, break-point-saved % |
| ELO | Overall ELO diff, surface-specific ELO diff |
| ELO trend | ELO slope (linear fit) diff over last 5, 10, 20, 35, 50, 100, 250 matches |

### Model

`RandomForestClassifier` with:
- 200 trees, max depth 9, `max_features="log2"`
- 85/15 train/test split
- Typical test accuracy: ~65–68 %

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Command line

```bash
python predict.py "Novak Djokovic" "Carlos Alcaraz"
python predict.py "Novak Djokovic" "Carlos Alcaraz" --surface clay
python predict.py "Jannik Sinner" "Daniil Medvedev" --surface hard
```

Player names are matched with fuzzy search, so minor spelling differences are tolerated.

### Jupyter notebook

```bash
jupyter notebook predict.ipynb
```

The notebook provides the full pipeline with charts:
- Surface distribution and matches-per-year plots
- Feature correlation heatmap
- Feature importance bar chart
- Win-probability visualization for any match-up
- Surface-by-surface comparison
- ELO trajectory plot for any player

## Data

Data is sourced from the [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) repository and should be placed under `data/`:

```
data/
  all/
    atp_matches_1991.csv
    atp_matches_1992.csv
    ...
    atp_matches_2024.csv
    aus_open_2025.csv
  rankings/
    atp_rankings_current.csv
  atp_players.csv
```

## Resources

- **Data**: [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) — the canonical open ATP dataset
- **ELO rating system**: [Wikipedia — Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system)
- **Random Forests**: Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- **scikit-learn RandomForestClassifier**: [scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- **Tennis prediction research**: Kovalchik, S. A. (2016). *Searching for the GOAT of tennis win prediction*. Journal of Quantitative Analysis in Sports, 12(3), 127–138.
