"""
Unit tests for the Player Prop Betting Model

Run with: python -m pytest test_player_prop_model.py
or simply: python test_player_prop_model.py
"""

import unittest
from player_prop_model import (
    PlayerPropModel, PlayerProp, PlayerStats,
    PropType, BetDirection
)


class TestPlayerPropModel(unittest.TestCase):
    """Test cases for PlayerPropModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = PlayerPropModel(min_ev_threshold=0.05)
    
    def test_american_odds_to_decimal(self):
        """Test conversion from American to decimal odds"""
        # Negative odds
        self.assertAlmostEqual(
            self.model.american_odds_to_decimal(-110), 
            1.909, 
            places=3
        )
        self.assertAlmostEqual(
            self.model.american_odds_to_decimal(-150), 
            1.667, 
            places=3
        )
        
        # Positive odds
        self.assertAlmostEqual(
            self.model.american_odds_to_decimal(120), 
            2.2, 
            places=3
        )
        self.assertAlmostEqual(
            self.model.american_odds_to_decimal(200), 
            3.0, 
            places=3
        )
    
    def test_american_odds_to_implied_probability(self):
        """Test conversion from American odds to implied probability"""
        # Negative odds
        self.assertAlmostEqual(
            self.model.american_odds_to_implied_probability(-110),
            0.5238,
            places=3
        )
        
        # Positive odds
        self.assertAlmostEqual(
            self.model.american_odds_to_implied_probability(120),
            0.4545,
            places=3
        )
    
    def test_estimate_probability_over(self):
        """Test probability estimation"""
        stats = PlayerStats(
            player_name="Test Player",
            prop_type=PropType.POINTS,
            recent_games=[20, 25, 30, 22, 28, 26, 24, 29, 27, 25]
        )
        
        # Line at 23 - most games are over
        prob_over = self.model.estimate_probability_over(stats, 23.0)
        self.assertGreater(prob_over, 0.5)
        
        # Line at 30 - most games are under
        prob_over = self.model.estimate_probability_over(stats, 30.0)
        self.assertLess(prob_over, 0.5)
    
    def test_calculate_expected_value(self):
        """Test expected value calculation"""
        # 60% probability with -110 odds should have positive EV
        ev = self.model.calculate_expected_value(0.60, -110)
        self.assertGreater(ev, 0)
        
        # 40% probability with -110 odds should have negative EV
        ev = self.model.calculate_expected_value(0.40, -110)
        self.assertLess(ev, 0)
    
    def test_analyze_prop(self):
        """Test full prop analysis"""
        prop = PlayerProp(
            player_name="Test Player",
            prop_type=PropType.POINTS,
            line=25.5,
            over_odds=-110,
            under_odds=-110
        )
        
        stats = PlayerStats(
            player_name="Test Player",
            prop_type=PropType.POINTS,
            recent_games=[28, 30, 26, 29, 27, 31, 25, 28, 30, 29],
            season_average=28.3
        )
        
        analysis = self.model.analyze_prop(prop, stats)
        
        # Check that analysis is complete
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.prop, prop)
        self.assertAlmostEqual(analysis.estimated_value, 28.3, places=1)
        self.assertGreater(analysis.probability_over, 0)
        self.assertLess(analysis.probability_over, 1)
        self.assertEqual(
            analysis.probability_over + analysis.probability_under, 
            1.0
        )
    
    def test_find_value_bets(self):
        """Test finding value bets from multiple props"""
        props = [
            PlayerProp("Player A", PropType.POINTS, 20.5, -110, -110),
            PlayerProp("Player B", PropType.POINTS, 25.5, -110, -110),
        ]
        
        stats_dict = {
            "Player A": PlayerStats(
                "Player A",
                PropType.POINTS,
                [25, 26, 24, 27, 25, 26, 24, 25, 26, 27],  # Consistently over 20.5
                season_average=25.5
            ),
            "Player B": PlayerStats(
                "Player B",
                PropType.POINTS,
                [24, 25, 26, 25, 24, 26, 25, 24, 25, 26],  # Around 25
                season_average=25.0
            ),
        }
        
        value_bets = self.model.find_value_bets(props, stats_dict)
        
        # Player A should be a strong value bet (all games over 20.5)
        self.assertGreater(len(value_bets), 0)
        self.assertIn("Player A", [bet.prop.player_name for bet in value_bets])
    
    def test_prop_type_enum(self):
        """Test PropType enum values"""
        self.assertEqual(PropType.POINTS.value, "points")
        self.assertEqual(PropType.ASSISTS.value, "assists")
        self.assertEqual(PropType.REBOUNDS.value, "rebounds")
    
    def test_bet_direction_enum(self):
        """Test BetDirection enum values"""
        self.assertEqual(BetDirection.OVER.value, "over")
        self.assertEqual(BetDirection.UNDER.value, "under")


class TestPlayerProp(unittest.TestCase):
    """Test cases for PlayerProp dataclass"""
    
    def test_player_prop_creation(self):
        """Test creating a PlayerProp"""
        prop = PlayerProp(
            player_name="Test Player",
            prop_type=PropType.POINTS,
            line=25.5,
            over_odds=-110,
            under_odds=-110
        )
        
        self.assertEqual(prop.player_name, "Test Player")
        self.assertEqual(prop.prop_type, PropType.POINTS)
        self.assertEqual(prop.line, 25.5)
        self.assertEqual(prop.over_odds, -110)
        self.assertEqual(prop.under_odds, -110)
    
    def test_player_prop_str(self):
        """Test string representation of PlayerProp"""
        prop = PlayerProp(
            player_name="LeBron James",
            prop_type=PropType.POINTS,
            line=25.5,
            over_odds=-110,
            under_odds=-110
        )
        
        str_repr = str(prop)
        self.assertIn("LeBron James", str_repr)
        self.assertIn("points", str_repr)
        self.assertIn("25.5", str_repr)


class TestPlayerStats(unittest.TestCase):
    """Test cases for PlayerStats dataclass"""
    
    def test_player_stats_with_average(self):
        """Test PlayerStats with provided average"""
        stats = PlayerStats(
            player_name="Test Player",
            prop_type=PropType.POINTS,
            recent_games=[20, 25, 30],
            season_average=27.0
        )
        
        self.assertEqual(stats.season_average, 27.0)
    
    def test_player_stats_auto_average(self):
        """Test PlayerStats with automatic average calculation"""
        stats = PlayerStats(
            player_name="Test Player",
            prop_type=PropType.POINTS,
            recent_games=[20, 25, 30]
        )
        
        self.assertAlmostEqual(stats.season_average, 25.0, places=1)
    
    def test_player_stats_empty_games(self):
        """Test PlayerStats with empty recent games"""
        stats = PlayerStats(
            player_name="Test Player",
            prop_type=PropType.POINTS,
            recent_games=[]
        )
        
        self.assertIsNone(stats.season_average)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
