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
FILING_DATE = "2026-03-07"


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
            target: Datastore name — PatientDB | ClinicalNotesDB | PharmacyDB | WaitlistDB
            field: Field name (e.g. hba1c, gfr, creatinine, blood_type, pra,
                   dialysis_start_date, cpra, status, drug_name, coded_diagnoses)
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
