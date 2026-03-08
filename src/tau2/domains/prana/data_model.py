"""
PRANA domain data model.

Four isolated datastores simulate real hospital information system silos.
No single datastore contains a complete patient record — the agent must
discover which fields are missing only when it attempts to file the KARS report.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from tau2.environment.db import DB


class LabResult(DB):
    """A single timestamped lab measurement."""

    value: Optional[float] = Field(description="Measured value")
    recorded_at: str = Field(description="Date recorded (YYYY-MM-DD)")


class Patient(DB):
    """PatientDB record — demographics and lab history (KARS Form Section 2)."""

    patient_id: str = Field(description="Unique patient identifier")
    name: str = Field(description="Patient full name")
    age: int = Field(description="Patient age in years")
    blood_type: str = Field(description="ABO blood type (e.g. A+, O-) — stable, no recency requirement")
    hba1c: List[LabResult] = Field(default_factory=list, description="HbA1c measurement history, chronological")
    gfr: List[LabResult] = Field(default_factory=list, description="GFR measurement history, chronological")
    creatinine: List[LabResult] = Field(default_factory=list, description="Creatinine measurement history, chronological")
    pra: Optional[int] = Field(default=None, description="Panel reactive antibody percentage — stable")


class ClinicalNote(DB):
    """ClinicalNotesDB record — physician notes with implicitly coded diagnoses (KARS Form Section 4)."""

    note_id: str = Field(description="Unique note identifier")
    patient_id: str = Field(description="Associated patient identifier")
    author: str = Field(description="Physician name")
    date: str = Field(description="Note date (YYYY-MM-DD)")
    text: str = Field(description="Full note text")
    coded_diagnoses: List[str] = Field(
        default_factory=list,
        description="Explicitly coded diagnoses extracted from notes",
    )


class Medication(DB):
    """PharmacyDB record — active medications with indirect comorbidity evidence (KARS Form Section 4 supporting)."""

    med_id: str = Field(description="Medication record identifier")
    patient_id: str = Field(description="Associated patient identifier")
    drug_name: str = Field(description="Drug name")
    dosage: str = Field(description="Dosage and frequency")
    indication: Optional[str] = Field(default=None, description="Prescribing indication if documented")


class WaitlistEntry(DB):
    """WaitlistDB record — OPTN waitlist registration data (OPTN Waiting List Policy 18.1)."""

    patient_id: str = Field(description="Associated patient identifier")
    registration_date: str = Field(description="Waitlist registration date (YYYY-MM-DD)")
    status: str = Field(description="Current waitlist status (active | inactive | removed)")
    dialysis_start_date: Optional[str] = Field(default=None, description="Dialysis start date (YYYY-MM-DD)")
    cpra: Optional[int] = Field(default=None, description="Calculated panel reactive antibody percentage")
    exceptions: List[str] = Field(default_factory=list, description="Active OPTN policy exceptions")


class PranaDB(DB):
    """
    Combined PRANA database — four isolated clinical datastores.

    Intentionally fragmented: no single datastore contains a complete
    KARS-required patient record. The agent discovers missing fields
    only at SRTR report filing (task t5).
    """

    patient_db: Dict[str, Patient] = Field(
        description="PatientDB: demographics and lab values, keyed by patient_id"
    )
    clinical_notes_db: Dict[str, List[ClinicalNote]] = Field(
        description="ClinicalNotesDB: physician notes keyed by patient_id"
    )
    pharmacy_db: Dict[str, List[Medication]] = Field(
        description="PharmacyDB: active medications keyed by patient_id"
    )
    waitlist_db: Dict[str, WaitlistEntry] = Field(
        description="WaitlistDB: OPTN waitlist entries keyed by patient_id"
    )

    def get_statistics(self) -> dict[str, Any]:
        return {
            "num_patients": len(self.patient_db),
            "num_clinical_notes": sum(len(v) for v in self.clinical_notes_db.values()),
            "num_medications": sum(len(v) for v in self.pharmacy_db.values()),
            "num_waitlist_entries": len(self.waitlist_db),
        }
