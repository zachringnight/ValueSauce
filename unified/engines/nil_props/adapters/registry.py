"""Adapter registry — selects adapters based on source mode."""

from __future__ import annotations

import logging

from engines.nil_props.config.settings import Settings, SourceMode
from engines.nil_props.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


def get_nba_adapter(settings: Settings) -> BaseAdapter:
    """Get NBA stats adapter. Only real data sources."""
    if settings.source_mode == SourceMode.MOCK:
        # Only tests should use mock mode
        from engines.nil_props.adapters.mock_adapter import MockNBAAdapter

        return MockNBAAdapter(settings)

    # Try nba_api (stats.nba.com) first — no API key needed
    try:
        from engines.nil_props.adapters.nba_stats_adapter import NBAStatsAdapter

        adapter = NBAStatsAdapter(settings)
        if adapter.authenticate():
            return adapter
        logger.warning("NBAStats adapter failed, trying premium adapter")
    except ImportError:
        logger.warning("nba_api not installed, trying premium adapter")

    # Try premium vendor adapter
    if settings.premium_nba_api_key:
        from engines.nil_props.adapters.premium_adapter import PremiumNBAAdapter

        adapter = PremiumNBAAdapter(settings)
        if adapter.authenticate():
            return adapter

    raise ConnectionError(
        "No NBA data adapter available. Install nba_api or configure PREMIUM_NBA_API_KEY."
    )


def get_injury_adapter(settings: Settings) -> BaseAdapter:
    """Get injury adapter. Mock for now — no real injury API integrated yet."""
    from engines.nil_props.adapters.mock_adapter import MockInjuryAdapter

    return MockInjuryAdapter(settings)


def get_odds_adapter(settings: Settings) -> BaseAdapter:
    """Get odds adapter. Only real data sources."""
    if settings.source_mode == SourceMode.MOCK:
        from engines.nil_props.adapters.mock_adapter import MockOddsAdapter

        return MockOddsAdapter(settings)

    # Try SportsGameOdds first
    if settings.sportsgameodds_api_key:
        try:
            from engines.nil_props.adapters.sportsgameodds_adapter import SportsgameoddsAdapter

            adapter = SportsgameoddsAdapter(settings)
            if adapter.authenticate():
                return adapter
            logger.warning("SportsGameOdds auth failed, trying next adapter")
        except ImportError:
            pass

    # Try The Odds API
    if settings.the_odds_api_key:
        try:
            from engines.nil_props.adapters.theoddsapi_adapter import TheOddsAPIAdapter

            adapter = TheOddsAPIAdapter(settings)
            if adapter.authenticate():
                return adapter
            logger.warning("The Odds API auth failed, trying next adapter")
        except ImportError:
            pass

    # Try generic premium odds adapter
    if settings.premium_odds_api_key:
        from engines.nil_props.adapters.premium_adapter import PremiumOddsAdapter

        adapter = PremiumOddsAdapter(settings)
        if adapter.authenticate():
            return adapter

    raise ConnectionError(
        "No odds adapter available. Configure SPORTSGAMEODDS_API_KEY or THE_ODDS_API_KEY."
    )


def get_all_adapters(settings: Settings) -> dict[str, BaseAdapter]:
    return {
        "nba": get_nba_adapter(settings),
        "injury": get_injury_adapter(settings),
        "odds": get_odds_adapter(settings),
    }
