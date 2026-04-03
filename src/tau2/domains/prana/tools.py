"""
PRANA domain toolkit.

Phase 1: query_db — retrieves values from all four datastores.
Time-sensitive fields (hba1c, gfr, creatinine) return full measurement history
with timestamps. The agent must reason about recency and anomalies before filing.
"""

import random

from tau2.domains.prana.data_model import LabResult, PranaDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

# Filing date — T5
FILING_DATE = "2026-03-11"


def _format_history(field: str, history: list[LabResult], patient_id: str) -> str:
    """Format a LabResult list as a readable history string.

    Entries are returned in a deterministic but non-chronological order seeded
    by (patient_id, field) — the agent must sort by date to identify the most
    recent value and detect temporal anomalies.
    """
    if not history:
        return f"NOT_FOUND: no measurements recorded for '{field}' for patient '{patient_id}'"

    seed = hash((patient_id, field)) & 0xFFFFFFFF
    rng = random.Random(seed)
    shuffled = list(history)
    rng.shuffle(shuffled)
    lines = []
    for entry in shuffled:
        val = entry.value if entry.value is not None else "NOT_RECORDED"
        lines.append(f"  {val} (recorded: {entry.recorded_at})")

    return f"{field} measurement history for {patient_id} (filing date: {FILING_DATE}):\n" + "\n".join(lines)


class PranaTools(ToolKitBase):
    """Tools for the PRANA kidney transplant administration domain."""

    db: PranaDB

    def __init__(self, db: PranaDB) -> None:
        super().__init__(db)

    _DATASTORE_MAP = {
        "patientdb": "patient_db",
        "clinicalnotesdb": "clinical_notes_db",
        "pharmacydb": "pharmacy_db",
        "waitlistdb": "waitlist_db",
        "allergiesdb": "allergies_db",
        "problemlistdb": "problem_list_db",
        "vitalsdb": "vitals_db",
        "encountersdb": "encounters_db",
        "insurancedb": "insurance_db",
        "immunizationsdb": "immunizations_db",
    }

    @is_tool(ToolType.READ)
    def query_db(
        self,
        target: str,
        field: str,
        patient_id: str,
    ) -> str:
        """Retrieve a field from a named clinical datastore for a patient.

        For time-sensitive lab fields (hba1c, gfr, creatinine) in PatientDB,
        returns the full measurement history with timestamps so the agent can
        assess recency and detect anomalies before filing.

        Args:
            target: Datastore name — PatientDB | ClinicalNotesDB | PharmacyDB | WaitlistDB |
                    AllergiesDB | ProblemListDB | VitalsDB | EncountersDB | InsuranceDB | ImmunizationsDB
            field: Field name within the target datastore (e.g. hba1c, gfr, creatinine,
                   blood_type, pra, dialysis_start_date, cpra, status, drug_name,
                   coded_diagnoses, allergen, severity, icd10_code, systolic_bp,
                   assessment, plan_type, vaccine_name)
            patient_id: Patient identifier (e.g. P001)

        Returns:
            For lab history fields: full timestamped history — the agent must assess recency
            and compute anomalies from the raw data before deciding whether to file.
            For other fields: the field value as a string, or NOT_FOUND if absent.
        """
        db_key = self._DATASTORE_MAP.get(target.lower())
        if db_key is None:
            return f"NOT_FOUND: unknown datastore '{target}'. Valid: PatientDB, ClinicalNotesDB, PharmacyDB, WaitlistDB"

        datastore = getattr(self.db, db_key)

        if patient_id not in datastore:
            return f"NOT_FOUND: patient '{patient_id}' not in {target}"

        record = datastore[patient_id]

        # ClinicalNotesDB and PharmacyDB are lists of records
        if isinstance(record, list):
            values = [
                str(getattr(entry, field, None))
                for entry in record
                if getattr(entry, field, None) is not None
            ]
            if not values:
                return f"NOT_FOUND: field '{field}' not found in {target} for patient '{patient_id}'"
            return ", ".join(values)

        value = getattr(record, field, None)

        # Time-sensitive lab fields return full history
        if isinstance(value, list) and all(isinstance(v, LabResult) for v in value):
            return _format_history(field, value, patient_id)

        if value is None:
            return f"NOT_FOUND: field '{field}' has no recorded value for patient '{patient_id}' in {target}"

        return str(value)

    @is_tool(ToolType.WRITE)
    def file_kars_report(
        self,
        patient_id: str,
    ) -> str:
        """File the KARS/SRTR report for a patient, recording submission in the PRANA system.

        This tool must only be called after independently verifying — through datastore queries —
        that all required KARS fields are present, within the 90-day recency window, and free of
        anomalies. Filing is FINAL and cannot be undone.

        Args:
            patient_id: Patient identifier (e.g. P001)

        Returns:
            Confirmation string with SRTR submission reference, or error message if filing fails.
        """
        if patient_id not in self.db.patient_db:
            return f"ERROR: patient '{patient_id}' not found. Cannot file report."
        if patient_id in self.db.filed_reports:
            return f"ERROR: KARS report for '{patient_id}' already filed on {self.db.filed_reports[patient_id]}."
        self.db.filed_reports[patient_id] = FILING_DATE
        return f"KARS report filed successfully for patient {patient_id}. Filing date: {FILING_DATE}. SRTR submission reference: KARS-{patient_id}-{FILING_DATE}."
