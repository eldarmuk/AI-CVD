"""Explicit recorded-outcome codes; no source notes or private classification lexicon."""

import hashlib
import json
from datetime import datetime, timedelta, timezone


def as_time(value):
    at = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if at.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return at.astimezone(timezone.utc)


def iso(value):
    return as_time(value).isoformat().replace("+00:00", "Z")


def build_episodes(alerts, burst_minutes=10):
    if burst_minutes <= 0:
        raise ValueError("Positive burst duration required")
    normalized, seen = [], set()
    for alert in alerts:
        sid, identifier = str(alert["subject_id"]), str(alert["event_id"])
        if not sid or not identifier or (sid, identifier) in seen:
            raise ValueError("Empty or duplicate event identity")
        seen.add((sid, identifier))
        severity = int(alert["outcome_code"])
        if str(severity) != str(alert["outcome_code"]) or severity not in (0, 1, 2, 3):
            raise ValueError("Explicit integer recorded-outcome code from 0 to 3 required")
        normalized.append((sid, as_time(alert["timestamp"]), identifier, severity))
    groups = []
    for row in sorted(normalized):
        if (
            not groups
            or row[0] != groups[-1][-1][0]
            or row[1] - groups[-1][-1][1] > timedelta(minutes=burst_minutes)
        ):
            groups.append([])
        groups[-1].append(row)
    result = []
    for group in groups:
        sid, at, identifier, _ = group[0]
        identity = json.dumps([sid, iso(at), identifier])
        result.append(
            {
                "senior_id": sid,
                "patient": sid,
                "anchor": iso(at),
                "episode_id": "episode_" + hashlib.sha256(identity.encode()).hexdigest(),
                "label": max(r[3] for r in group),
                "constituent_count": len(group),
                "constituents": [
                    {"timestamp": iso(r[1]), "severity": r[3], "event_id": r[2]} for r in group
                ],
            }
        )
    return result


def event_time(episode, severities):
    """Retained rule: first alarm retrospectively assigned a qualifying code."""
    return next(
        (as_time(a["timestamp"]) for a in episode["constituents"] if a["severity"] in severities),
        None,
    )
