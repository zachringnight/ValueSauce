"""Feature snapshot builder – orchestrator for all feature sub-builders.

The ``FeatureSnapshotBuilder`` ties together the four independent
feature builders (minutes, opportunity, make-rate) and the archetype
classifier to produce a single, frozen feature snapshot suitable for
model inference or for persisting as a historical record.

Each snapshot is tagged with metadata:
- ``freeze_timestamp`` – the point-in-time at which the snapshot was
  created (all input data must predate this).
- ``injury_snapshot_timestamp`` – when the injury data was captured.
- ``odds_snapshot_timestamp`` – when the odds/line data was captured.
- ``tracking_available`` – whether tracking data was present.
- ``feature_json_hash`` – SHA-256 hash of the feature payload for
  deduplication / integrity checking.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from .archetype import ArchetypeClassifier
from .make_rate_features import MakeRateFeatureBuilder
from .minutes_features import MinutesFeatureBuilder
from .opportunity_features import OpportunityFeatureBuilder

logger = logging.getLogger(__name__)


class FeatureSnapshotBuilder:
    """Orchestrate all feature builders into a single snapshot."""

    def __init__(
        self,
        minutes_builder: MinutesFeatureBuilder | None = None,
        opportunity_builder: OpportunityFeatureBuilder | None = None,
        make_rate_builder: MakeRateFeatureBuilder | None = None,
        archetype_classifier: ArchetypeClassifier | None = None,
    ) -> None:
        self.minutes_builder = minutes_builder or MinutesFeatureBuilder()
        self.opportunity_builder = opportunity_builder or OpportunityFeatureBuilder()
        self.make_rate_builder = make_rate_builder or MakeRateFeatureBuilder()
        self.archetype_classifier = archetype_classifier or ArchetypeClassifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_snapshot(
        self,
        player_id: str,
        game_id: str,
        freeze_timestamp: str | datetime,
        player_games: list[dict[str, Any]],
        tracking_games: list[dict[str, Any]] | None = None,
        opponent_shooting: list[dict[str, Any]] | None = None,
        game_context: dict[str, Any] | None = None,
        injury_snapshot: dict[str, Any] | None = None,
        teammate_statuses: list[dict[str, Any]] | None = None,
        odds_snapshot_timestamp: str | datetime | None = None,
    ) -> dict[str, Any]:
        """Build the complete feature snapshot for one player-game.

        Parameters
        ----------
        player_id:
            Unique player identifier.
        game_id:
            Unique game identifier.
        freeze_timestamp:
            Point-in-time cutoff – no data after this timestamp should
            be reflected in the snapshot.  ISO-8601 string or datetime.
        player_games:
            Boxscore game dicts, most-recent-first, all with
            ``game_date <= freeze_timestamp``.
        tracking_games:
            Tracking game dicts (may be ``None``).
        opponent_shooting:
            Opponent defensive shooting records.
        game_context:
            Pregame context dict (spread, totals, pace, etc.).
        injury_snapshot:
            Player's own injury report info.
        teammate_statuses:
            List of teammate status dicts with ``role`` and ``status``.
        odds_snapshot_timestamp:
            When the odds data was captured (for metadata only).

        Returns
        -------
        dict
            The complete feature snapshot including metadata.
        """
        game_context = game_context or {}
        freeze_ts = self._normalise_timestamp(freeze_timestamp)

        # 1. Classify archetype
        logger.info(
            "Building feature snapshot for player=%s game=%s",
            player_id,
            game_id,
        )
        archetype = self.archetype_classifier.classify(player_games, tracking_games)

        # 2. Minutes features (H1)
        minutes_features = self.minutes_builder.build(
            player_games=player_games,
            game_context=game_context,
            injury_snapshot=injury_snapshot,
            teammate_statuses=teammate_statuses,
        )

        # 3. Opportunity features (H2 / H3)
        opportunity_features = self.opportunity_builder.build(
            player_games=player_games,
            tracking_games=tracking_games,
            opponent_shooting=opponent_shooting,
            game_context=game_context,
            archetype=archetype,
        )

        # 4. Make-rate features (H4)
        make_rate_features = self.make_rate_builder.build(
            player_games=player_games,
            tracking_games=tracking_games,
            opponent_shooting=opponent_shooting,
            game_context=game_context,
            archetype=archetype,
        )

        # 5. Combine into a single dict, namespacing by sub-model
        snapshot: dict[str, Any] = {
            "player_id": player_id,
            "game_id": game_id,
            "archetype": archetype,
        }

        # Prefix keys to avoid collisions
        for key, val in minutes_features.items():
            snapshot[f"min_{key}"] = val

        for key, val in opportunity_features.items():
            snapshot[f"opp_{key}"] = val

        for key, val in make_rate_features.items():
            snapshot[f"mr_{key}"] = val

        # 6. Metadata
        tracking_available = bool(tracking_games)
        injury_ts = None
        if injury_snapshot:
            injury_ts = injury_snapshot.get("timestamp") or injury_snapshot.get(
                "snapshot_timestamp"
            )
            if injury_ts is not None:
                injury_ts = self._normalise_timestamp(injury_ts)

        odds_ts = (
            self._normalise_timestamp(odds_snapshot_timestamp)
            if odds_snapshot_timestamp
            else None
        )

        snapshot["meta_freeze_timestamp"] = freeze_ts
        snapshot["meta_injury_snapshot_timestamp"] = injury_ts
        snapshot["meta_odds_snapshot_timestamp"] = odds_ts
        snapshot["meta_tracking_available"] = tracking_available
        snapshot["meta_feature_json_hash"] = self._compute_hash(snapshot)

        logger.info(
            "Feature snapshot built: archetype=%s tracking=%s hash=%s",
            archetype,
            tracking_available,
            snapshot["meta_feature_json_hash"][:12],
        )

        return snapshot

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_no_leakage(
        self,
        snapshot_dict: dict[str, Any],
        freeze_timestamp: str | datetime,
    ) -> bool:
        """Check that all embedded timestamps are <= freeze_timestamp.

        This is a sanity check to catch accidental look-ahead bias.

        Returns
        -------
        bool
            ``True`` if no leakage is detected, ``False`` otherwise.
        """
        freeze_ts = self._normalise_timestamp(freeze_timestamp)

        timestamp_keys = [
            "meta_freeze_timestamp",
            "meta_injury_snapshot_timestamp",
            "meta_odds_snapshot_timestamp",
        ]

        for key in timestamp_keys:
            ts = snapshot_dict.get(key)
            if ts is None:
                continue

            ts_str = self._normalise_timestamp(ts)
            if ts_str > freeze_ts:
                logger.error(
                    "Leakage detected: %s=%s > freeze=%s",
                    key,
                    ts_str,
                    freeze_ts,
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Hash computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(feature_dict: dict[str, Any]) -> str:
        """Compute a deterministic SHA-256 hex digest of the feature
        dict.

        Metadata keys (prefixed with ``meta_``) are excluded from the
        hash so that the hash represents only the *feature* content.
        """
        # Filter out metadata keys to hash only feature content
        payload = {
            k: v for k, v in sorted(feature_dict.items()) if not k.startswith("meta_")
        }

        # Convert to a canonical JSON string.  We handle non-serialisable
        # types by coercing to str as a fallback.
        json_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Timestamp normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_timestamp(ts: str | datetime | Any) -> str:
        """Coerce a timestamp to an ISO-8601 string for comparison."""
        if isinstance(ts, datetime):
            return ts.isoformat()
        if isinstance(ts, str):
            return ts
        return str(ts)
