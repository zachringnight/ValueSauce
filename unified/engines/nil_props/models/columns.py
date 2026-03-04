"""Feature column definitions for each model layer."""

MINUTES_FEATURES = [
    "min_last_3",
    "min_last_5",
    "min_last_10",
    "min_std_5",
    "started_last",
    "starter_rate_10",
    "foul_rate_5",
    "rest_days",
    "is_back_to_back",
    "games_played_last_10",
    "is_home_int",
    "game_month",
    "position_code",
]

MINUTES_TARGET = "minutes"

OPPORTUNITY_FEATURES = [
    "min_last_5",
    "ast_per_min_last_5",
    "ast_per_min_last_10",
    "past_per_min_last_5",
    "past_per_min_last_10",
    "touches_per_min_last_5",
    "passes_per_min_last_5",
    "usg_last_5",
    "ast_last_3",
    "ast_last_5",
    "ast_last_10",
    "ast_shrunk_5",
    "past_shrunk_5",
    "pos_group_avg_ast",
    "prior_game_count",
    "opp_pace_5",
    "opp_drtg_5",
    "opp_ast_allowed_5",
    "team_pace_5",
    "is_home_int",
    "rest_days",
    "game_month",
    "position_code",
    "teammates_out",
    "guard_teammates_out",
]

OPPORTUNITY_TARGET = "potential_assists"

CONVERSION_FEATURES = [
    "ast_conversion_last_5",
    "ast_conversion_last_10",
    "past_per_min_last_5",
    "opp_drtg_5",
    "is_home_int",
    "rest_days",
    "game_month",
    "position_code",
]

CONVERSION_TARGET = "ast_conversion"

# Direct assist model — bypasses three-layer decomposition to anchor projections
DIRECT_ASSIST_FEATURES = [
    "ast_last_3",
    "ast_last_5",
    "ast_last_10",
    "ast_shrunk_5",
    "pos_group_avg_ast",
    "prior_game_count",
    "min_last_5",
    "opp_ast_allowed_5",
    "opp_drtg_5",
    "team_pace_5",
    "is_home_int",
    "rest_days",
    "position_code",
    "teammates_out",
    "guard_teammates_out",
]

DIRECT_ASSIST_TARGET = "assists"

# Baseline: just rolling 10-game average assists
BASELINE_FEATURES = ["ast_last_10"]
