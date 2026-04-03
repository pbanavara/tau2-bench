"""
PRANA domain data model.

Nine clinical datastores modeled after Epic EHR simulate a real hospital
information system with intentional fragmentation across silos.
No single datastore contains a complete patient record — the agent must
discover which fields are missing only when it attempts to file the KARS report.

Datastores:
    PatientDB          — demographics + longitudinal lab values
    ClinicalNotesDB    — physician progress notes, consults, H&Ps
    PharmacyDB         — active prescriptions with prescriber/route details
    WaitlistDB         — OPTN waitlist registration (UNOS/SRTR)
    AllergiesDB        — drug/food/environmental allergy records
    ProblemListDB      — ICD-10 coded active and chronic problems
    VitalsDB           — timestamped flowsheet vital signs
    EncountersDB       — clinical visits and hospitalizations
    InsuranceDB        — active payer/coverage information
    ImmunizationsDB    — administered vaccine records
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from tau2.environment.db import DB


# ─── Core lab type ───────────────────────────────────────────────────────────

class LabResult(DB):
    """A single timestamped lab measurement."""

    value: Optional[float] = Field(description="Measured value")
    recorded_at: str = Field(description="Date recorded (YYYY-MM-DD)")


# ─── PatientDB ───────────────────────────────────────────────────────────────

class Patient(DB):
    """PatientDB record — full demographics and longitudinal lab history (KARS Form Section 2)."""

    patient_id: str = Field(description="Unique patient identifier (e.g. P001)")
    mrn: str = Field(description="Epic Medical Record Number (e.g. MRN-010000)")
    name: str = Field(description="Patient full name")
    dob: str = Field(description="Date of birth (YYYY-MM-DD)")
    age: int = Field(description="Patient age in years")
    gender: str = Field(description="Gender (Male / Female / Non-binary)")
    race: str = Field(description="Race (e.g. White, Black or African American, Asian)")
    ethnicity: str = Field(description="Ethnicity (Hispanic or Latino / Not Hispanic or Latino)")
    preferred_language: str = Field(default="English", description="Preferred spoken language")
    address: str = Field(description="Full mailing address")
    phone: str = Field(description="Primary contact phone number")
    emergency_contact_name: str = Field(description="Emergency contact full name")
    emergency_contact_phone: str = Field(description="Emergency contact phone number")
    pcp: str = Field(description="Primary care physician name")
    blood_type: Optional[str] = Field(
        default=None,
        description="ABO blood type (e.g. A+, O-) — stable, no recency requirement",
    )
    hba1c: List[LabResult] = Field(
        default_factory=list,
        description="HbA1c measurement history — agent must sort by date to find most recent",
    )
    gfr: List[LabResult] = Field(
        default_factory=list,
        description="eGFR measurement history — agent must sort by date to find most recent",
    )
    creatinine: List[LabResult] = Field(
        default_factory=list,
        description="Serum creatinine history — agent must sort by date to find most recent",
    )
    pra: Optional[float] = Field(
        default=None,
        description="Panel reactive antibody percentage — stable immunologic value",
    )


# ─── AllergiesDB ─────────────────────────────────────────────────────────────

class Allergy(DB):
    """AllergiesDB record — documented drug, food, or environmental allergy/intolerance."""

    allergy_id: str = Field(description="Unique allergy record identifier (e.g. ALG-P001-01)")
    patient_id: str = Field(description="Associated patient identifier")
    allergen: str = Field(description="Allergen or substance name")
    reaction: str = Field(description="Documented reaction description")
    severity: str = Field(description="Reaction severity: mild | moderate | severe | life-threatening")
    documented_date: str = Field(description="Date first documented (YYYY-MM-DD)")
    status: str = Field(default="active", description="Record status: active | inactive | entered-in-error")


# ─── ProblemListDB ───────────────────────────────────────────────────────────

class Problem(DB):
    """ProblemListDB record — ICD-10 coded active or chronic diagnosis."""

    problem_id: str = Field(description="Unique problem identifier (e.g. PRB-P001-01)")
    patient_id: str = Field(description="Associated patient identifier")
    icd10_code: str = Field(description="ICD-10-CM diagnosis code (e.g. N18.5)")
    description: str = Field(description="Human-readable diagnosis description")
    status: str = Field(description="Problem status: active | chronic | resolved | inactive")
    onset_date: str = Field(description="Approximate onset date (YYYY-MM-DD or YYYY)")
    documented_by: str = Field(description="Documenting clinician")
    documented_date: str = Field(description="Date added to problem list (YYYY-MM-DD)")


# ─── VitalsDB ────────────────────────────────────────────────────────────────

class VitalSign(DB):
    """VitalsDB record — a single flowsheet vital signs entry."""

    recorded_at: str = Field(description="Date and time of measurement (YYYY-MM-DD HH:MM)")
    recorded_by: str = Field(description="Staff member who recorded the vitals")
    systolic_bp: Optional[int] = Field(default=None, description="Systolic blood pressure (mmHg)")
    diastolic_bp: Optional[int] = Field(default=None, description="Diastolic blood pressure (mmHg)")
    heart_rate: Optional[int] = Field(default=None, description="Heart rate (beats per minute)")
    temperature_f: Optional[float] = Field(default=None, description="Body temperature (°F)")
    weight_kg: Optional[float] = Field(default=None, description="Body weight (kg)")
    height_cm: Optional[float] = Field(default=None, description="Height (cm)")
    bmi: Optional[float] = Field(default=None, description="Body mass index (kg/m²)")
    o2_saturation: Optional[float] = Field(default=None, description="Peripheral oxygen saturation (%)")


# ─── EncountersDB ────────────────────────────────────────────────────────────

class Encounter(DB):
    """EncountersDB record — clinical visit or hospitalization."""

    encounter_id: str = Field(description="Unique encounter identifier (e.g. ENC-P001-001)")
    patient_id: str = Field(description="Associated patient identifier")
    visit_type: str = Field(description="Visit type: office | inpatient | telehealth | procedure | ED")
    date: str = Field(description="Encounter date (YYYY-MM-DD)")
    provider: str = Field(description="Attending or treating provider")
    department: str = Field(description="Clinical department (e.g. Nephrology, Transplant Surgery)")
    facility: str = Field(description="Facility or clinic name")
    chief_complaint: str = Field(description="Chief complaint or reason for visit")
    assessment: str = Field(description="Clinical assessment / impression")
    plan: str = Field(description="Treatment plan documentation")
    diagnoses: List[str] = Field(default_factory=list, description="ICD-10 codes assigned at this encounter")
    discharge_disposition: Optional[str] = Field(
        default=None,
        description="Discharge disposition for inpatient encounters (e.g. Home, Skilled nursing facility)",
    )


# ─── InsuranceDB ─────────────────────────────────────────────────────────────

class Insurance(DB):
    """InsuranceDB record — active insurance coverage for a patient."""

    patient_id: str = Field(description="Associated patient identifier")
    payer_name: str = Field(description="Insurance payer organization name")
    plan_name: str = Field(description="Specific plan name")
    member_id: str = Field(description="Member or subscriber ID")
    group_number: Optional[str] = Field(default=None, description="Group number (commercial plans)")
    plan_type: str = Field(
        description="Plan type: commercial | medicare | medicaid | medicare_advantage | self-pay | VA"
    )
    effective_date: str = Field(description="Coverage effective date (YYYY-MM-DD)")
    termination_date: Optional[str] = Field(default=None, description="Coverage end date, null if currently active")


# ─── ImmunizationsDB ─────────────────────────────────────────────────────────

class Immunization(DB):
    """ImmunizationsDB record — administered vaccine."""

    immunization_id: str = Field(description="Unique immunization record identifier (e.g. IMM-P001-01)")
    patient_id: str = Field(description="Associated patient identifier")
    vaccine_name: str = Field(description="Vaccine name (CVX description)")
    date_administered: str = Field(description="Administration date (YYYY-MM-DD)")
    lot_number: Optional[str] = Field(default=None, description="Vaccine lot number")
    administered_by: str = Field(description="Administering clinician or facility")
    series: Optional[str] = Field(default=None, description="Series dose (e.g. 1 of 2, booster)")


# ─── ClinicalNotesDB ─────────────────────────────────────────────────────────

class ClinicalNote(DB):
    """ClinicalNotesDB record — physician note with implicit and explicit diagnoses (KARS Form Section 4)."""

    note_id: str = Field(description="Unique note identifier")
    patient_id: str = Field(description="Associated patient identifier")
    note_type: str = Field(
        default="Progress Note",
        description="Note type: Progress Note | H&P | Nephrology Consult | Transplant Evaluation | Discharge Summary",
    )
    author: str = Field(description="Authoring physician name")
    department: str = Field(default="Nephrology", description="Authoring clinical department")
    date: str = Field(description="Note date (YYYY-MM-DD)")
    text: str = Field(description="Full note text")
    coded_diagnoses: List[str] = Field(
        default_factory=list,
        description="Explicitly coded diagnoses extracted from the note",
    )


# ─── PharmacyDB ──────────────────────────────────────────────────────────────

class Medication(DB):
    """PharmacyDB record — active prescription with full dispensing details (KARS Form Section 4 supporting)."""

    med_id: str = Field(description="Medication record identifier")
    patient_id: str = Field(description="Associated patient identifier")
    drug_name: str = Field(description="Drug generic name")
    dosage: str = Field(description="Dosage and frequency (e.g. 10mg daily)")
    route: Optional[str] = Field(default=None, description="Route: oral | intravenous | subcutaneous | topical")
    indication: Optional[str] = Field(default=None, description="Prescribing indication")
    prescriber: Optional[str] = Field(default=None, description="Prescribing physician")
    start_date: Optional[str] = Field(default=None, description="Prescription start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date, null if still active")
    refills_remaining: Optional[int] = Field(default=None, description="Number of authorized refills remaining")
    pharmacy: Optional[str] = Field(default=None, description="Dispensing pharmacy name")


# ─── WaitlistDB ──────────────────────────────────────────────────────────────

class WaitlistEntry(DB):
    """WaitlistDB record — OPTN/UNOS waitlist registration (OPTN Waiting List Policy 18.1)."""

    patient_id: str = Field(description="Associated patient identifier")
    registration_date: str = Field(description="Waitlist registration date (YYYY-MM-DD)")
    status: str = Field(description="Current waitlist status: active | inactive | removed")
    dialysis_start_date: Optional[str] = Field(default=None, description="Dialysis initiation date (YYYY-MM-DD)")
    cpra: Optional[int] = Field(default=None, description="Calculated panel reactive antibody percentage")
    exceptions: List[str] = Field(default_factory=list, description="Active OPTN policy exceptions")
    listing_center: Optional[str] = Field(default=None, description="Transplant center name")
    listing_physician: Optional[str] = Field(default=None, description="Transplant physician who listed the patient")
    opo_region: Optional[str] = Field(default=None, description="UNOS OPO region (e.g. Region 7)")
    previous_transplants: Optional[int] = Field(default=0, description="Number of prior kidney transplants")
    evaluation_status: Optional[str] = Field(
        default=None, description="Transplant evaluation status: complete | pending | deferred"
    )
    primary_diagnosis: Optional[str] = Field(
        default=None, description="Primary ESRD-causing diagnosis"
    )


# ─── Combined PranaDB ────────────────────────────────────────────────────────

class PranaDB(DB):
    """
    Combined PRANA database — ten Epic EHR-modeled clinical datastores.

    Intentionally fragmented: no single datastore contains a complete
    KARS-required patient record. The agent discovers missing fields
    only at SRTR report filing (task t5).
    """

    patient_db: Dict[str, Patient] = Field(
        description="PatientDB: demographics and longitudinal lab values, keyed by patient_id"
    )
    clinical_notes_db: Dict[str, List[ClinicalNote]] = Field(
        description="ClinicalNotesDB: physician notes keyed by patient_id"
    )
    pharmacy_db: Dict[str, List[Medication]] = Field(
        description="PharmacyDB: active prescriptions keyed by patient_id"
    )
    waitlist_db: Dict[str, WaitlistEntry] = Field(
        description="WaitlistDB: OPTN waitlist entries keyed by patient_id"
    )
    allergies_db: Dict[str, List[Allergy]] = Field(
        default_factory=dict,
        description="AllergiesDB: allergy and intolerance records keyed by patient_id",
    )
    problem_list_db: Dict[str, List[Problem]] = Field(
        default_factory=dict,
        description="ProblemListDB: ICD-10 coded active problems keyed by patient_id",
    )
    vitals_db: Dict[str, List[VitalSign]] = Field(
        default_factory=dict,
        description="VitalsDB: flowsheet vital signs entries keyed by patient_id",
    )
    encounters_db: Dict[str, List[Encounter]] = Field(
        default_factory=dict,
        description="EncountersDB: clinical visits and hospitalizations keyed by patient_id",
    )
    insurance_db: Dict[str, Insurance] = Field(
        default_factory=dict,
        description="InsuranceDB: active insurance coverage keyed by patient_id",
    )
    immunizations_db: Dict[str, List[Immunization]] = Field(
        default_factory=dict,
        description="ImmunizationsDB: vaccine administration records keyed by patient_id",
    )
    filed_reports: Dict[str, str] = Field(
        default_factory=dict,
        description="KARS report filing registry — keyed by patient_id, value is filing date. Populated by file_kars_report.",
    )

    def get_statistics(self) -> dict[str, Any]:
        return {
            "num_patients": len(self.patient_db),
            "num_clinical_notes": sum(len(v) for v in self.clinical_notes_db.values()),
            "num_medications": sum(len(v) for v in self.pharmacy_db.values()),
            "num_waitlist_entries": len(self.waitlist_db),
            "num_allergy_records": sum(len(v) for v in self.allergies_db.values()),
            "num_problems": sum(len(v) for v in self.problem_list_db.values()),
            "num_vital_entries": sum(len(v) for v in self.vitals_db.values()),
            "num_encounters": sum(len(v) for v in self.encounters_db.values()),
            "num_insured_patients": len(self.insurance_db),
            "num_immunization_records": sum(len(v) for v in self.immunizations_db.values()),
            "num_filed_reports": len(self.filed_reports),
        }
