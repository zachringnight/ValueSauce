"""
Example usage of the Player Prop Betting Model

This script demonstrates how to use the PlayerPropModel to analyze
player prop bets and find value betting opportunities.
"""

from player_prop_model import (
    PlayerPropModel, PlayerProp, PlayerStats,
    PropType, BetDirection
)


def main():
    """Demonstrate the Player Prop Betting Model"""
    
    print("=" * 70)
    print("Player Prop Betting Model - Example Usage")
    print("=" * 70)
    print()
    
    # Create the model with a 5% minimum EV threshold
    model = PlayerPropModel(min_ev_threshold=0.05)
    
    # Example: NBA Player Props
    print("Example 1: NBA Player Points Prop")
    print("-" * 70)
    
    # Create a player prop for LeBron James - Points
    lebron_prop = PlayerProp(
        player_name="LeBron James",
        prop_type=PropType.POINTS,
        line=25.5,
        over_odds=-110,  # Risk $110 to win $100
        under_odds=-110
    )
    
    # Historical stats for LeBron (last 10 games)
    lebron_stats = PlayerStats(
        player_name="LeBron James",
        prop_type=PropType.POINTS,
        recent_games=[28, 31, 24, 27, 29, 26, 30, 25, 32, 27],  # Last 10 games
        season_average=27.5
    )
    
    # Analyze the prop
    analysis = model.analyze_prop(lebron_prop, lebron_stats)
    print(analysis)
    print()
    
    # Example 2: NFL Player Passing Yards
    print("Example 2: NFL Player Passing Yards Prop")
    print("-" * 70)
    
    qb_prop = PlayerProp(
        player_name="Patrick Mahomes",
        prop_type=PropType.PASSING_YARDS,
        line=285.5,
        over_odds=-115,
        under_odds=-105
    )
    
    qb_stats = PlayerStats(
        player_name="Patrick Mahomes",
        prop_type=PropType.PASSING_YARDS,
        recent_games=[312, 278, 320, 289, 305, 267, 298, 315],
        season_average=295.5
    )
    
    analysis = model.analyze_prop(qb_prop, qb_stats)
    print(analysis)
    print()
    
    # Example 3: Finding Value Bets
    print("Example 3: Finding Value Bets from Multiple Props")
    print("-" * 70)
    
    # Create multiple props
    props = [
        PlayerProp("Stephen Curry", PropType.THREE_POINTERS, 4.5, -110, -110),
        PlayerProp("Giannis Antetokounmpo", PropType.REBOUNDS, 11.5, -105, -115),
        PlayerProp("Luka Doncic", PropType.ASSISTS, 8.5, -120, +100),
    ]
    
    # Create stats dictionary
    stats_dict = {
        "Stephen Curry": PlayerStats(
            "Stephen Curry",
            PropType.THREE_POINTERS,
            [5, 6, 4, 7, 5, 5, 6, 4, 5, 7],  # Often hits 5+ threes
            season_average=5.2
        ),
        "Giannis Antetokounmpo": PlayerStats(
            "Giannis Antetokounmpo",
            PropType.REBOUNDS,
            [12, 13, 10, 14, 11, 12, 13, 11, 10, 12],
            season_average=11.8
        ),
        "Luka Doncic": PlayerStats(
            "Luka Doncic",
            PropType.ASSISTS,
            [9, 10, 8, 11, 9, 10, 8, 9, 10, 11],
            season_average=9.5
        ),
    }
    
    # Find value bets
    value_bets = model.find_value_bets(props, stats_dict)
    
    if value_bets:
        print(f"Found {len(value_bets)} value bet(s):\n")
        for i, bet in enumerate(value_bets, 1):
            print(f"{i}. {bet}")
            print()
    else:
        print("No value bets found with current threshold.")
    
    # Example 4: Understanding Odds Conversion
    print("Example 4: Understanding Odds Conversion")
    print("-" * 70)
    
    test_odds = [-110, -150, +120, +200]
    print("American Odds → Decimal Odds → Implied Probability\n")
    for odds in test_odds:
        decimal = model.american_odds_to_decimal(odds)
        implied_prob = model.american_odds_to_implied_probability(odds)
        print(f"{odds:>6} → {decimal:>5.3f} → {implied_prob:>6.1%}")
    
    print()
    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
