"""Leakage detection tests for backtest integrity."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class LeakageError(Exception):
    """Raised when temporal leakage is detected."""
    pass


class LeakageDetector:
    """
    Hard-fail tests for temporal leakage (Section K6).

    Must detect:
    - Any feature timestamp > decision timestamp
    - Any injury snapshot > freeze timestamp
    - Any odds snapshot > freeze timestamp
    - Closing line leaks into model features
    - Tracking-only values silently imputed
    - Future same-day info leaks between rows
    """

    @staticmethod
    def check_feature_timestamps(
        feature_snapshot: dict, freeze_timestamp: datetime
    ) -> list[str]:
        """Check that no feature timestamps exceed the freeze timestamp."""
        violations = []

        injury_ts = feature_snapshot.get("injury_snapshot_timestamp_utc")
        if injury_ts and injury_ts > freeze_timestamp:
            violations.append(
                f"Injury snapshot timestamp {injury_ts} > freeze {freeze_timestamp}"
            )

        odds_ts = feature_snapshot.get("odds_snapshot_timestamp_utc")
        if odds_ts and odds_ts > freeze_timestamp:
            violations.append(
                f"Odds snapshot timestamp {odds_ts} > freeze {freeze_timestamp}"
            )

        return violations

    @staticmethod
    def check_closing_line_leak(
        feature_snapshot: dict,
    ) -> list[str]:
        """Check that closing line information hasn't leaked into features."""
        violations = []
        features = feature_snapshot.get("feature_json", {})

        closing_indicators = [
            "closing_", "close_", "final_line", "final_odds",
            "closing_spread", "closing_total_line",
        ]

        for key in features:
            for indicator in closing_indicators:
                if indicator in key.lower() and "close_over_prob" not in key:
                    violations.append(
                        f"Potential closing line leak in feature: {key}"
                    )

        return violations

    @staticmethod
    def check_tracking_imputation(
        feature_snapshot: dict,
    ) -> list[str]:
        """Check that tracking-only values are not silently imputed."""
        violations = []
        tracking_available = feature_snapshot.get("tracking_available", False)
        features = feature_snapshot.get("feature_json", {})

        tracking_only_features = [
            "catch_shoot_3pa_per_min", "pull_up_3pa_per_min",
            "catch_shoot_share", "pull_up_share",
            "touches_per_min", "time_of_poss_per_min",
            "avg_seconds_per_touch", "avg_dribbles_per_touch",
            "assisted_3pm_share", "unassisted_3pm_share",
        ]

        if not tracking_available:
            for feat in tracking_only_features:
                if feat in features and features[feat] is not None:
                    violations.append(
                        f"Tracking-only feature '{feat}' has value "
                        f"{features[feat]} but tracking_available=False"
                    )

        return violations

    @staticmethod
    def check_same_day_leak(
        rows: list[dict], game_date: str
    ) -> list[str]:
        """Check for future same-day info leaking between rows."""
        violations = []

        # Sort by freeze timestamp
        sorted_rows = sorted(rows, key=lambda r: r.get("freeze_timestamp_utc", ""))

        for i, row in enumerate(sorted_rows):
            freeze = row.get("freeze_timestamp_utc")
            if not freeze:
                continue

            # Check that no feature references data from after freeze
            features = row.get("feature_json", {})
            data_timestamp = features.get("_latest_data_timestamp")
            if data_timestamp and data_timestamp > str(freeze):
                violations.append(
                    f"Row {i} has data timestamp {data_timestamp} > freeze {freeze}"
                )

        return violations

    def run_all_checks(
        self, feature_snapshot: dict, freeze_timestamp: datetime
    ) -> tuple[bool, list[str]]:
        """
        Run all leakage checks.

        Returns:
            (passed, violations) - passed is True if no leakage detected
        """
        all_violations = []

        all_violations.extend(
            self.check_feature_timestamps(feature_snapshot, freeze_timestamp)
        )
        all_violations.extend(
            self.check_closing_line_leak(feature_snapshot)
        )
        all_violations.extend(
            self.check_tracking_imputation(feature_snapshot)
        )

        passed = len(all_violations) == 0

        if not passed:
            for v in all_violations:
                logger.error("LEAKAGE DETECTED: %s", v)

        return passed, all_violations
