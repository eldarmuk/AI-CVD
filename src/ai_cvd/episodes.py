"""Retain every alert; distinguish recorded information from episode outcomes."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math


def as_time(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Timestamps must carry an explicit timezone")
    return value.astimezone(timezone.utc)


def iso(value):
    return as_time(value).isoformat().replace("+00:00", "Z")


def classify_severity(note):
    """Versioned keyword proxy, not clinical adjudication; preserve original notes privately."""
    if note is None or (isinstance(note, float) and math.isnan(note)):
        return -1
    text = str(note).lower()
    no_dispatch = "brak wskazań do interwencji zrm" in text
    # Explicit escalation takes priority over an incidental 'accidental' phrase.
    if not no_dispatch and any(term in text for term in (
        "zdecydowano się na interwencję zrm", "zdecydowano wezwać zrm", "wezwanie zrm"
    )):
        return 3
    if any(term in text for term in ("alarm przypadkowy", "alarm testowy", "alert techniczny")):
        return 0
    if "nie nawiązano kontaktu" in text:
        return 2
    if "nawiązano kontakt" in text or "opiekun powiadomiony" in text:
        return 1
    return -1


def build_episodes(alerts, task):
    """Adjacent gaps <= burst_minutes belong to one episode (including chained bursts)."""
    ordered = sorted(alerts, key=lambda a: (str(a["senior_id"]), as_time(a["alert_date"]), str(a["alert_id"])))
    seen, groups = set(), []
    for source in ordered:
        a = dict(source)
        key = (str(a["senior_id"]), str(a["alert_id"]))
        if key in seen:
            raise ValueError("Duplicate source alert ID")
        seen.add(key)
        a["senior_id"] = str(a["senior_id"])
        a["alert_date"] = iso(a["alert_date"])
        a["severity"] = int(a.get("severity", classify_severity(a.get("sos_note"))))
        if a["severity"] not in {-1, 0, 1, 2, 3}:
            raise ValueError("Invalid alert severity")
        if (not groups or groups[-1][-1]["senior_id"] != a["senior_id"] or
            as_time(a["alert_date"]) - as_time(groups[-1][-1]["alert_date"]) > timedelta(minutes=task.burst_minutes)):
            groups.append([])
        groups[-1].append(a)
    result = []
    for group in groups:
        first = group[0]
        maximum = max(a["severity"] for a in group)
        onset = next((a["alert_date"] for a in group if a["severity"] > first["severity"]), None)
        identity = json.dumps([first["senior_id"], str(first["alert_id"]), first["alert_date"]])
        result.append({
            "senior_id": first["senior_id"],
            "episode_id": "episode_" + hashlib.sha256(identity.encode()).hexdigest(),
            "episode_start": first["alert_date"], "episode_end": group[-1]["alert_date"],
            "first_severity": first["severity"], "maximum_severity": maximum,
            "final_recorded_severity": group[-1]["severity"],
            "escalation_recorded_at": onset,
            "maximum_severity_recorded_at": next(a["alert_date"] for a in group if a["severity"] == maximum),
            "constituent_count": len(group),
            "constituents": [{"alert_id": str(a["alert_id"]), "timestamp": a["alert_date"], "severity": a["severity"]} for a in group],
            "outcome_basis": "maximum_recorded_keyword_proxy_not_clinician_adjudicated",
        })
    return result


def event_time(episode, severities):
    """First qualifying recorded alert, never first lower-severity alert backdated."""
    return next((as_time(a["timestamp"]) for a in episode["constituents"] if a["severity"] in severities), None)


def write_episode_tables(connection, episodes):
    """Write new versioned derived tables; refuse to overwrite an earlier extraction."""
    connection.execute("CREATE TABLE alert_episodes_v2 (episode_id TEXT PRIMARY KEY, senior_id TEXT, episode_start TEXT, episode_end TEXT, first_severity INTEGER, maximum_severity INTEGER, final_recorded_severity INTEGER, escalation_recorded_at TEXT, maximum_severity_recorded_at TEXT, constituent_count INTEGER, outcome_basis TEXT)")
    connection.execute("CREATE TABLE alert_episode_members_v2 (episode_id TEXT, alert_id TEXT, timestamp TEXT, severity INTEGER, PRIMARY KEY(episode_id,alert_id))")
    for e in episodes:
        keys = ["episode_id", "senior_id", "episode_start", "episode_end", "first_severity", "maximum_severity", "final_recorded_severity", "escalation_recorded_at", "maximum_severity_recorded_at", "constituent_count", "outcome_basis"]
        connection.execute("INSERT INTO alert_episodes_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?)", [e[k] for k in keys])
        connection.executemany("INSERT INTO alert_episode_members_v2 VALUES (?,?,?,?)", [(e["episode_id"], a["alert_id"], a["timestamp"], a["severity"]) for a in e["constituents"]])
