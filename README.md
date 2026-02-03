# ValueSauce
PropEdge - Player Prop Betting Model

## Overview

ValueSauce is a Python-based statistical model for analyzing player prop bets in sports betting. It helps identify value betting opportunities by calculating expected values and probabilities based on historical player performance.

## Features

- **Statistical Analysis**: Analyze player props using historical performance data
- **Expected Value Calculation**: Calculate EV for both over and under bets
- **Multiple Sports Support**: Built-in support for NBA, NFL, MLB, and more
- **Value Bet Detection**: Automatically identify positive EV betting opportunities
- **Odds Conversion**: Convert between American, decimal, and implied probability formats

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zachringnight/ValueSauce.git
cd ValueSauce
```

2. Install dependencies (optional, as the core model uses only Python standard library):
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from player_prop_model import PlayerPropModel, PlayerProp, PlayerStats, PropType

# Create the model
model = PlayerPropModel(min_ev_threshold=0.05)

# Define a player prop
prop = PlayerProp(
    player_name="LeBron James",
    prop_type=PropType.POINTS,
    line=25.5,
    over_odds=-110,
    under_odds=-110
)

# Add historical stats
stats = PlayerStats(
    player_name="LeBron James",
    prop_type=PropType.POINTS,
    recent_games=[28, 31, 24, 27, 29, 26, 30, 25, 32, 27],
    season_average=27.5
)

# Analyze the prop
analysis = model.analyze_prop(prop, stats)
print(analysis)
```

## Usage Examples

Run the example script to see the model in action:

```bash
python example_usage.py
```

This will demonstrate:
- Analyzing individual player props
- Finding value bets from multiple props
- Understanding odds conversions
- Interpreting expected value calculations

## Model Components

### PropType
Supported prop types include:
- `POINTS`, `REBOUNDS`, `ASSISTS`, `THREE_POINTERS` (Basketball)
- `PASSING_YARDS`, `RUSHING_YARDS`, `RECEIVING_YARDS`, `TOUCHDOWNS` (Football)
- `HITS`, `HOME_RUNS`, `STRIKEOUTS` (Baseball)

### PlayerProp
Represents a betting line with:
- Player name
- Prop type
- Line (e.g., 25.5 points)
- Over/Under odds (American format)

### PlayerStats
Historical performance data:
- Recent game statistics
- Season averages
- Used to estimate probabilities

### PropAnalysis
Complete analysis output:
- Estimated player value
- Probability of over/under
- Expected value calculations
- Betting recommendations

## How It Works

1. **Data Input**: Provide player props and historical statistics
2. **Probability Estimation**: Calculate likelihood of player going over/under the line based on recent performance
3. **EV Calculation**: Compute expected value using true probabilities vs. implied odds
4. **Value Detection**: Identify bets where true probability suggests positive expected value
5. **Recommendation**: Suggest bets that exceed the minimum EV threshold

## Expected Value (EV)

The model calculates EV using the formula:
```
EV = (True Probability × Profit) - ((1 - True Probability) × Stake)
```

Positive EV indicates a potentially profitable bet over the long term.

## Customization

Adjust the minimum EV threshold when creating the model:

```python
# More aggressive (3% minimum EV)
model = PlayerPropModel(min_ev_threshold=0.03)

# More conservative (10% minimum EV)
model = PlayerPropModel(min_ev_threshold=0.10)
```

## Project Structure

```
ValueSauce/
├── __init__.py              # Package initialization
├── player_prop_model.py     # Core model implementation
├── example_usage.py         # Usage examples
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Contributing

This is an open-source project. Contributions are welcome!

## License

Open source - feel free to use and modify.

## Disclaimer

This tool is for educational and research purposes only. Sports betting involves risk. Always bet responsibly and within your means. Past performance does not guarantee future results.
