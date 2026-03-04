from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

from .models import GameOverrides, GameState, MinutesOverride, PropRequest, VALID_MARKETS
from .utils import name_key, norm_name, parse_asof, split_semicolon_list

class CardError(Exception):
    pass

def _parse_float_or_none(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        raise CardError(f"Expected float, got '{v}'")

def _parse_minutes_overrides(value: str) -> Dict[str, MinutesOverride]:
    out: Dict[str, MinutesOverride] = {}
    if not value or not str(value).strip():
        return out
    for entry in split_semicolon_list(value):
        parts = [p.strip() for p in entry.split(":")]
        if len(parts) != 3:
            raise CardError(f"Bad minutes override '{entry}'. Expected Name:cap:NN or Name:target:NN")
        name, kind, num = parts
        kind = kind.lower()
        if kind not in {"cap", "target"}:
            raise CardError(f"Bad minutes override kind '{kind}' in '{entry}'")
        try:
            value = float(num)
        except Exception:
            raise CardError(f"Bad minutes override value '{num}' in '{entry}'")
        out[norm_name(name)] = MinutesOverride(kind=kind, value=value)
    return out

def load_game_card(path: str, asof: str) -> Dict[str, GameState]:
    p = Path(path)
    if not p.exists():
        raise CardError(f"Game card not found: {path}")
    games: Dict[str, GameState] = {}
    asof_dt = parse_asof(asof)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"game_key", "player_team", "opponent", "home"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise CardError(f"Game card missing required columns: {sorted(required)}")
        for row in reader:
            game_key = (row.get("game_key") or "").strip()
            if not game_key:
                continue
            home_raw = str(row.get("home") or "").strip()
            if home_raw not in {"0", "1"}:
                raise CardError(f"home must be 0 or 1 for game_key={game_key}")

            ov = GameOverrides(
                team_out=set(split_semicolon_list(row.get("override_team_out", ""))),
                team_in=set(split_semicolon_list(row.get("override_team_in", ""))),
                opp_out=set(split_semicolon_list(row.get("override_opp_out", ""))),
                opp_in=set(split_semicolon_list(row.get("override_opp_in", ""))),
                notes=(row.get("notes") or "").strip(),
            )
            caps = _parse_minutes_overrides(row.get("minutes_caps", ""))
            targets = _parse_minutes_overrides(row.get("minutes_targets", ""))
            ov.minutes_overrides = {**caps, **targets}

            gs = GameState(
                game_key=game_key,
                asof_utc=asof_dt,
                game_date_local=(row.get("game_date_local") or "").strip() or None,
                player_team_abbr=(row.get("player_team") or "").strip().upper(),
                opponent_abbr=(row.get("opponent") or "").strip().upper(),
                is_home=home_raw == "1",
                overrides=ov,
                spread=_parse_float_or_none(row.get("spread")),
                vegas_team_total=_parse_float_or_none(row.get("vegas_team_total")),
                vegas_game_total=_parse_float_or_none(row.get("vegas_game_total")),
            )
            if game_key in games:
                raise CardError(f"Duplicate game_key in game card: {game_key}")
            games[game_key] = gs

    _validate_game_overrides(games)
    return games

def load_props_card(path: str) -> List[PropRequest]:
    p = Path(path)
    if not p.exists():
        raise CardError(f"Props card not found: {path}")

    props: List[PropRequest] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"game_key", "player", "market", "line"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise CardError(f"Props card missing required columns: {sorted(required)}")
        for row in reader:
            game_key = (row.get("game_key") or "").strip()
            if not game_key:
                continue
            player = norm_name(row.get("player", ""))
            market = (row.get("market") or "").strip().upper()
            if market not in VALID_MARKETS:
                raise CardError(f"Invalid market '{market}' for {player} {game_key}")
            try:
                line = float(row.get("line"))
            except Exception:
                raise CardError(f"Invalid line for {player} {market} {game_key}: {row.get('line')}")

            def _parse_american(raw: object, *, field_name: str) -> Optional[int]:
                if not str(raw or "").strip():
                    return None
                s = str(raw).strip().upper()
                if s.startswith("+"):
                    s = s[1:]
                try:
                    out = int(s)
                    return out
                except Exception:
                    try:
                        f = float(s)
                    except Exception:
                        raise CardError(f"Invalid American odds ({field_name}) for {player} {market} {game_key}: {raw}")
                    if not math.isfinite(f):
                        raise CardError(f"Invalid American odds ({field_name}) for {player} {market} {game_key}: {raw}")
                    rounded = int(round(f))
                    if abs(f - float(rounded)) > 1e-6:
                        raise CardError(f"Invalid American odds ({field_name}) for {player} {market} {game_key}: {raw}")
                    return rounded

            odds_legacy = _parse_american(row.get("odds"), field_name="odds")
            odds_over = _parse_american(row.get("odds_over"), field_name="odds_over")
            odds_under = _parse_american(row.get("odds_under"), field_name="odds_under")
            if odds_over is None:
                odds_over = odds_legacy

            bookmaker = str(row.get("bookmaker") or "").strip() or None
            event_id = str(row.get("event_id") or "").strip() or None
            event_start_utc = str(row.get("event_start_utc") or "").strip() or None
            props.append(PropRequest(
                game_key=game_key,
                player_name=player,
                market=market,
                line=line,
                odds_american=odds_over,
                odds_over_american=odds_over,
                odds_under_american=odds_under,
                bookmaker=bookmaker,
                event_id=event_id,
                event_start_utc=event_start_utc,
            ))
    return props

def validate_cards(games: Dict[str, GameState], props: List[PropRequest]) -> None:
    game_keys = set(games.keys())
    for pr in props:
        if pr.game_key not in game_keys:
            raise CardError(f"Prop references unknown game_key '{pr.game_key}'")
    _validate_game_overrides(games)

def _validate_game_overrides(games: Dict[str, GameState]) -> None:
    for gk, gs in games.items():
        ov = gs.overrides
        team_out_keys = {name_key(x) for x in ov.team_out if name_key(x)}
        team_in_keys = {name_key(x) for x in ov.team_in if name_key(x)}
        opp_out_keys = {name_key(x) for x in ov.opp_out if name_key(x)}
        opp_in_keys = {name_key(x) for x in ov.opp_in if name_key(x)}

        bad_team = team_out_keys & team_in_keys
        bad_opp = opp_out_keys & opp_in_keys
        if bad_team:
            raise CardError(f"{gk}: players in both override_team_out and override_team_in: {sorted(bad_team)}")
        if bad_opp:
            raise CardError(f"{gk}: players in both override_opp_out and override_opp_in: {sorted(bad_opp)}")

        all_out = team_out_keys | opp_out_keys
        for name in ov.minutes_overrides:
            if name_key(name) in all_out:
                raise CardError(f"{gk}: minutes override provided for OUT player '{name}'")
