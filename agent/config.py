"""
config.py — loads a tenant's configuration + reference data by tenant_id.

Multi-tenant seam: ALL client-specific data lives under tenants/<tenant_id>/.
The agent code is 100% client-agnostic. Onboarding a customer = adding a folder.

  tenants/TENANT_BIO_99/
    ├── config.yaml          (taxonomy + thresholds)
    ├── catalog.json         (Gate 5)
    ├── hcp_registry.json    (Gate 6)
    └── srd_registry.json    (SRD suggestions)
"""

import json
from functools import lru_cache
from pathlib import Path

import yaml

TENANTS_DIR = Path(__file__).parent / "tenants"


def _tenant_path(tenant_id: str, filename: str) -> Path:
    path = TENANTS_DIR / tenant_id / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {filename} for tenant '{tenant_id}' at {path}")
    return path


@lru_cache(maxsize=None)
def load_config(tenant_id: str) -> dict:
    with _tenant_path(tenant_id, "config.yaml").open() as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def _load_json(tenant_id: str, filename: str) -> dict:
    with _tenant_path(tenant_id, filename).open() as f:
        return json.load(f)


# --- config accessors ---

def get_taxonomy(tenant_id: str) -> dict:
    """{class_name: description} for the tenant's taxonomy."""
    return {t["name"]: t["description"] for t in load_config(tenant_id)["taxonomy"]}


def get_thresholds(tenant_id: str) -> dict:
    return load_config(tenant_id)["thresholds"]


def get_reporting_deadlines(tenant_id: str) -> dict:
    """Calendar-day deadlines (from Day 0 = MedInfo intake) by regime + seriousness."""
    return load_config(tenant_id)["reporting_deadlines_days"]


# --- reference-data accessors (used by the gates) ---

def get_catalog(tenant_id: str) -> list:
    return _load_json(tenant_id, "catalog.json")["products"]


def get_hcps(tenant_id: str) -> list:
    return _load_json(tenant_id, "hcp_registry.json")["hcps"]


def get_srds(tenant_id: str) -> list:
    return _load_json(tenant_id, "srd_registry.json")["standard_response_documents"]
