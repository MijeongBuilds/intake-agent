"""
schemas.py — the data contract for the Medical Information intake agent.

Every model here maps directly to the locked design spec. Pydantic enforces
structure at runtime, which IS Gate 7 (Schema Validation): if the LLM returns
data that breaks a constraint, Pydantic raises an error and the agent routes
the case to human review instead of auto-approving.

Layers (matching the spec):
  1. Enums            — controlled vocabularies (allowed values)
  2. CommonMetadata   — extracted for every document
  3. ClassifierOutput — the classify node's prediction (+ 4 AE pillars)
  4. 8 transactional record models — one per class that triggers a record
  5. ProcessingDecision — system-populated audit + routing fields
  6. ProcessedCase    — top-level container bundling all of the above
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


# ============================================================
# 1. ENUMS — controlled vocabularies
# ============================================================

class DocClass(str, Enum):
    """The 10-class taxonomy (demo subset of the customer's 100)."""
    ON_LABEL = "On-Label Medical Inquiry"
    OFF_LABEL = "Off-Label Medical Inquiry"
    STANDARD_RESPONSE = "Standard Response Request"
    FORMULATION = "Formulation & Excipient Information"
    EXPANDED_ACCESS = "Expanded Access / Compassionate Use"
    IIT = "Investigator-Initiated Trial Proposal"
    ADVERSE_EVENT = "Adverse Event Report"
    PQC = "Product Quality Complaint"
    LEGAL_REG = "Legal / Regulatory Authority Correspondence"
    PUBLIC_FAQ = "Public Health / FAQ Deflection"
    OUT_OF_SCOPE = "Out of scope"   # sentinel: not a medical inquiry at all (spam/marketing/wrong-company); set from the out_of_scope flag, not scored by the classifier


class SourceChannel(str, Enum):
    EMAIL = "Email"
    WEB_PORTAL = "Web Portal"
    NETWORK_API = "Network API"


class ReporterType(str, Enum):
    PHYSICIAN = "HCP - Physician"
    PHARMACIST = "HCP - Pharmacist"
    NURSE = "Nurse"
    CONSUMER = "Consumer"
    OTHER = "Other"


class Seriousness(str, Enum):
    SERIOUS = "Serious"
    NON_SERIOUS = "Non-Serious"


class RecordStatus(str, Enum):
    DRAFT = "Draft"
    PENDING_REVIEW = "Pending Review"
    APPROVED = "Approved"


class RoutingTarget(str, Enum):
    AUTO_APPROVE = "Auto-Approve"
    MIS_QUEUE = "MIS Queue"
    PV_QUEUE = "PV Queue"
    UNREADABLE = "Unreadable Queue"   # pre-classify halt: OCR below the unreadable floor


# --- per-class field enums (open-ended ones include OTHER as a safe fallback) ---

class InsertSection(str, Enum):
    DOSAGE = "Dosage and Administration"
    CONTRAINDICATIONS = "Contraindications"
    ADVERSE_REACTIONS = "Adverse Reactions"
    DRUG_INTERACTIONS = "Drug Interactions"
    OTHER = "Other"


class DemographicFlag(str, Enum):
    PEDIATRIC = "Pediatric under 12"
    GERIATRIC = "Geriatric"
    PREGNANCY = "Pregnancy/Lactation"
    NONE = "None"


class DeliveryMethod(str, Enum):
    SECURE_EMAIL = "Secure Email"
    VAULT_PORTAL = "Vault Portal Download"
    DIRECT_MAIL = "Direct Mail"


class UrgencyTier(str, Enum):
    EMERGENCY = "Emergency Single Patient Access"
    INTERMEDIATE = "Intermediate Size Cohort Ingress"


class SupportType(str, Enum):
    FINANCIAL = "Pure Financial Grant"
    DRUG_SUPPLY = "Drug Supply Provision Only"
    HYBRID = "Matched Hybrid Support"


class DefectType(str, Enum):
    CRACKED_VIAL = "Cracked Vial Body"
    DEVICE_JAM = "Device Needle Mechanical Jam"
    LABEL_MISPRINT = "Label Text Misprint"
    CLOUDINESS = "Solution Cloudiness/Precipitate"
    OTHER = "Other"


class SampleReturnStatus(str, Enum):
    RETURN_ARRANGED = "Return Arranged"
    DESTROYED = "Destroyed by Clinic"
    AWAITING = "Awaiting Shipping Label"


# ============================================================
# 2. COMMON METADATA — extracted for every document
# ============================================================

class CommonMetadata(BaseModel):
    tenant_id: str                          # multi-tenant identifier (system)
    case_id: str                            # links child attachments (system)
    document_id: str                        # assigned at ingestion (system)
    source_channel: SourceChannel
    received_date: date
    customer_org: Optional[str] = None      # hospital / clinic / pharmacy (None for private consumers)
    reporter_name: str
    reporter_type: ReporterType
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None     # E.164
    country_code: str                       # ISO 2-letter
    language: str
    product_mentioned: Optional[str] = None # validated against catalog
    active_study_flag: bool = False         # patient in an active trial / product used as comparator -> clinical-trial reporting clock
    study_id: Optional[str] = None          # trial or protocol identifier, if the reporter names one


class ExtractedMetadata(BaseModel):
    """The subset of CommonMetadata the LLM reads from the DOCUMENT BODY.

    Envelope fields (tenant_id, case_id, document_id, source_channel,
    received_date) are supplied by the ingestion pipeline, NOT the model — the
    agent must never guess what the system already knows. extract() merges this
    with the envelope to build a full CommonMetadata.
    """
    customer_org: Optional[str] = None      # the org the reporter represents; None for a private consumer
    reporter_name: str
    reporter_type: ReporterType
    contact_email: Optional[str] = None      # the reporter's OWN email only (never a company/system address)
    contact_phone: Optional[str] = None      # E.164
    country_code: str                        # ISO 2-letter
    language: str
    product_mentioned: Optional[str] = None  # brand name AS WRITTEN (catalog match happens at Gate 5)
    active_study_flag: bool = False           # true only if the doc states the patient is in an active trial OR the product is a comparator/concomitant in one
    study_id: Optional[str] = None            # trial/protocol id (e.g. NCT number) if named, else null


# ============================================================
# 3. CLASSIFIER OUTPUT — the classify node's prediction
# ============================================================

class ClassScore(BaseModel):
    """One class + how well the document fits it (0-1)."""
    doc_class: DocClass
    fit_score: float = Field(ge=0.0, le=1.0)


class ClassifierScores(BaseModel):
    """RAW classifier output (what the LLM returns): a fit score for EVERY class + the
    binary judgments. predicted_class / class_confidence / classification_conflict_margin
    are DERIVED in code from class_scores — so the routing-critical numbers are *computed*
    (argmax + top1-top2 gap), not self-reported by the model (which was noisy/uncalibrated)."""
    class_scores: list[ClassScore]
    classification_rationale: str = ""
    adverse_event_flag: bool
    out_of_scope: bool
    has_patient_pillar: bool = False
    has_reporter_pillar: bool = False
    has_product_pillar: bool = False
    has_event_pillar: bool = False


class ClassifierOutput(BaseModel):
    predicted_class: DocClass                              # DERIVED: argmax(class_scores)
    class_confidence: float = Field(ge=0.0, le=1.0)        # DERIVED: top fit_score
    classification_conflict_margin: float = Field(ge=0.0, le=1.0)  # DERIVED: top1 - top2
    classification_rationale: str = ""   # one-line: why this class + why classes compete
    class_scores: list[ClassScore] = Field(default_factory=list)   # full per-class distribution (audit/UI)
    adverse_event_flag: bool
    out_of_scope: bool
    # Four Legal Pillars (AE-only; default False for non-AE docs)
    has_patient_pillar: bool = False
    has_reporter_pillar: bool = False
    has_product_pillar: bool = False
    has_event_pillar: bool = False


# ============================================================
# 4. PER-CLASS TRANSACTIONAL RECORDS (the 8 that trigger one)
# ============================================================

class OnLabelRecord(BaseModel):
    record_type: str = "On-Label Inquiry"
    target_package_insert_section: InsertSection
    inquiry_summary_text: str               # free-text -> semantic eval
    suggested_srd_match: Optional[str] = None


class OffLabelRecord(BaseModel):
    record_type: str = "Off-Label Inquiry"
    unapproved_demographic_flag: Optional[DemographicFlag] = None
    off_label_indication: str
    unsolicited_verification_flag: bool


class StandardResponseRecord(BaseModel):
    record_type: str = "Standard Response Request"
    requested_srd_id: str
    delivery_method_preference: DeliveryMethod


class FormulationRecord(BaseModel):
    record_type: str = "Formulation & Excipient Info"
    queried_substance_allergen: str
    clinical_justification: Optional[str] = None  # free-text -> semantic eval


class ExpandedAccessRecord(BaseModel):
    record_type: str = "Expanded Access / Compassionate Use"
    patient_urgency_tier: UrgencyTier
    treating_physician_dea_number: str
    investigational_compound_id: str


class IITRecord(BaseModel):
    record_type: str = "Investigator-Initiated Trial Proposal"
    proposed_study_title: str
    requested_support_type: SupportType
    therapeutic_area_alignment: str


class AEReportRecord(BaseModel):
    """Draft ICSR. seriousness computed at extraction from the 5 ICH criteria."""
    record_type: str = "Adverse Event Report"
    patient_initials: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    onset_date: Optional[date] = None
    adverse_symptoms_list: list[str]
    concomitant_medications_list: list[str] = Field(default_factory=list)
    ae_seriousness: Seriousness
    seriousness_rationale: Optional[str] = None  # which ICH criteria fired + basis (audit trail)
    is_valid_icsr: bool                     # derived: all 4 pillars present


class PQCRecord(BaseModel):
    record_type: str = "Product Quality Complaint"
    defect_category_type: DefectType
    lot_number: str
    sample_return_status: SampleReturnStatus


# Union of all record types (transactional_record holds whichever applies).
# Classes 9 (Legal/Reg) and 10 (Public FAQ) produce no record -> None.
TransactionalRecord = Union[
    OnLabelRecord,
    OffLabelRecord,
    StandardResponseRecord,
    FormulationRecord,
    ExpandedAccessRecord,
    IITRecord,
    AEReportRecord,
    PQCRecord,
]


# ============================================================
# 5. PROCESSING & DECISION METADATA — system-populated
# ============================================================

class ProcessingDecision(BaseModel):
    model_version: str                      # e.g. "claude-sonnet-4.6"
    prompt_version: str                     # e.g. "classify_v3"
    processed_timestamp: datetime
    ocr_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    catalog_match_result: Optional[str] = None   # resolved product code, else None
    hcp_system_id: Optional[str] = None          # resolved HCP id, else None
    record_status: RecordStatus
    routing_target: RoutingTarget
    failed_gates: list[str] = Field(default_factory=list)
    day_zero: Optional[date] = None              # MedInfo intake date = start of the regulatory clock
    regulatory_due_date: Optional[date] = None   # computed reporting deadline (AE cases only)
    reporting_regime: Optional[str] = None       # "Post-Marketing (Spontaneous)" | "Clinical Trial (SUSAR)"


# ============================================================
# 6. TOP-LEVEL CONTAINER — the full processed case
# ============================================================

class ProcessedCase(BaseModel):
    common: CommonMetadata
    classification: ClassifierOutput
    transactional_record: Optional[TransactionalRecord] = None
    processing: ProcessingDecision
