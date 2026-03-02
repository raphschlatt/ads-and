from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


UID_REGISTRY_SCHEMA_VERSION = "v1"


@dataclass
class UidRegistryAssignmentMeta:
    clusters_total: int
    clusters_reused: int
    clusters_new: int
    clusters_merged_conflicts: int
    mentions_total: int
    mentions_previously_mapped: int
    mentions_newly_mapped: int
    registry_size_after: int
    next_id_after: int
    local_to_global_max_nunique: int
    global_to_local_max_nunique: int
    local_to_global_violations: int
    global_to_local_violations: int
    local_to_global_valid: bool


def _resolve_alias(uid: str, aliases: dict[str, str]) -> str:
    current = str(uid)
    seen: set[str] = set()
    while current in aliases and current not in seen:
        seen.add(current)
        current = str(aliases[current])
    return current


def _default_registry(namespace: str) -> dict[str, Any]:
    return {
        "schema_version": UID_REGISTRY_SCHEMA_VERSION,
        "uid_namespace": str(namespace),
        "next_id": 1,
        "mention_to_uid": {},
        "aliases": {},
    }


def load_uid_registry(path: str | Path, *, namespace: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return _default_registry(namespace=namespace)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != UID_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported UID registry schema_version={schema_version!r}; "
            f"expected {UID_REGISTRY_SCHEMA_VERSION!r}."
        )

    payload_namespace = str(payload.get("uid_namespace", "")).strip()
    if payload_namespace != str(namespace):
        raise ValueError(
            "UID registry namespace mismatch: "
            f"registry={payload_namespace!r} requested={namespace!r}"
        )

    payload["mention_to_uid"] = dict(payload.get("mention_to_uid", {}) or {})
    payload["aliases"] = dict(payload.get("aliases", {}) or {})
    payload["next_id"] = int(payload.get("next_id", 1))
    if payload["next_id"] < 1:
        payload["next_id"] = 1
    return payload


def save_uid_registry(path: str | Path, registry: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    return p


def assign_registry_uids(
    *,
    clusters: pd.DataFrame,
    registry: dict[str, Any],
    uid_namespace: str,
) -> tuple[pd.DataFrame, dict[str, Any], UidRegistryAssignmentMeta]:
    required = {"mention_id", "author_uid"}
    missing = [col for col in required if col not in clusters.columns]
    if missing:
        raise ValueError(f"clusters missing required columns: {missing}")

    out = clusters.copy()
    out["mention_id"] = out["mention_id"].astype(str)
    if out["mention_id"].duplicated().any():
        raise ValueError("clusters contains duplicate mention_id; expected one row per mention_id.")
    if "author_uid_local" not in out.columns:
        out["author_uid_local"] = out["author_uid"].astype(str)
    else:
        out["author_uid_local"] = out["author_uid_local"].astype(str)

    mention_to_uid = dict(registry.get("mention_to_uid", {}) or {})
    aliases = dict(registry.get("aliases", {}) or {})
    next_id = int(registry.get("next_id", 1))
    if next_id < 1:
        next_id = 1

    clusters_reused = 0
    clusters_new = 0
    clusters_merged_conflicts = 0
    mentions_previously_mapped = 0
    mentions_newly_mapped = 0
    mention_to_assigned_uid: dict[str, str] = {}

    for _, grp in out.groupby("author_uid_local", sort=False):
        mention_ids = grp["mention_id"].astype(str).tolist()
        known_uids = {
            _resolve_alias(str(mention_to_uid[mid]), aliases)
            for mid in mention_ids
            if mid in mention_to_uid
        }

        if len(known_uids) == 0:
            assigned_uid = f"{uid_namespace}::au{next_id:09d}"
            next_id += 1
            clusters_new += 1
        elif len(known_uids) == 1:
            assigned_uid = next(iter(known_uids))
            clusters_reused += 1
        else:
            assigned_uid = sorted(known_uids)[0]
            clusters_merged_conflicts += 1
            for uid in known_uids:
                if uid != assigned_uid:
                    aliases[str(uid)] = str(assigned_uid)

        for mid in mention_ids:
            if mid in mention_to_uid:
                mentions_previously_mapped += 1
            else:
                mentions_newly_mapped += 1
            mention_to_uid[mid] = str(assigned_uid)
            mention_to_assigned_uid[mid] = str(assigned_uid)

    out["author_uid"] = out["mention_id"].map(mention_to_assigned_uid)
    if out["author_uid"].isna().any():
        missing_mentions = out.loc[out["author_uid"].isna(), "mention_id"].astype(str).head(5).tolist()
        raise RuntimeError(
            "Registry UID assignment produced null author_uid values for mention_ids: "
            f"{missing_mentions}"
        )
    out["author_uid"] = out["author_uid"].astype(str)

    local_to_global = out.groupby("author_uid_local")["author_uid"].nunique()
    global_to_local = out.groupby("author_uid")["author_uid_local"].nunique()
    local_to_global_max_nunique = int(local_to_global.max()) if len(local_to_global) else 0
    global_to_local_max_nunique = int(global_to_local.max()) if len(global_to_local) else 0
    local_to_global_violations = int((local_to_global > 1).sum()) if len(local_to_global) else 0
    global_to_local_violations = int((global_to_local > 1).sum()) if len(global_to_local) else 0
    local_to_global_valid = bool(local_to_global_max_nunique <= 1)

    registry_out = {
        "schema_version": UID_REGISTRY_SCHEMA_VERSION,
        "uid_namespace": str(uid_namespace),
        "next_id": int(next_id),
        "mention_to_uid": mention_to_uid,
        "aliases": aliases,
    }

    meta = UidRegistryAssignmentMeta(
        clusters_total=int(out["author_uid_local"].nunique()),
        clusters_reused=int(clusters_reused),
        clusters_new=int(clusters_new),
        clusters_merged_conflicts=int(clusters_merged_conflicts),
        mentions_total=int(len(out)),
        mentions_previously_mapped=int(mentions_previously_mapped),
        mentions_newly_mapped=int(mentions_newly_mapped),
        registry_size_after=int(len(mention_to_uid)),
        next_id_after=int(next_id),
        local_to_global_max_nunique=int(local_to_global_max_nunique),
        global_to_local_max_nunique=int(global_to_local_max_nunique),
        local_to_global_violations=int(local_to_global_violations),
        global_to_local_violations=int(global_to_local_violations),
        local_to_global_valid=bool(local_to_global_valid),
    )
    return out, registry_out, meta
