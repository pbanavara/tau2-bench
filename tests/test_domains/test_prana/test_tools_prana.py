from tau2.domains.prana.environment import get_environment, get_tasks, get_tasks_split


def test_prana_environment_loads_tasks_and_splits():
    env = get_environment()
    tasks = get_tasks()
    splits = get_tasks_split()

    assert env.domain_name == "prana"
    assert len(tasks) == 33
    assert "base" in splits
    assert "easy" in splits
    assert "adversarial" in splits

    task_ids = {task.id for task in tasks}
    assert task_ids == set(str(i) for i in range(len(tasks)))


def test_prana_query_db_reads_fragmented_clinical_data():
    env = get_environment()

    blood_type = env.tools.query_db("PatientDB", "blood_type", "P001")
    assert blood_type == "A+"

    hba1c = env.tools.query_db("PatientDB", "hba1c", "P001")
    assert "hba1c measurement history for P001" in hba1c
    assert "2025-11-07" in hba1c
    assert "2026-03-07" in hba1c

    insurance = env.tools.query_db("InsuranceDB", "plan_name", "P001")
    assert insurance == "Medicare Part A & B"

    waitlist_status = env.tools.query_db("WaitlistDB", "status", "P001")
    assert waitlist_status == "active"


def test_prana_file_kars_report_is_final_and_idempotent():
    env = get_environment()

    first = env.tools.file_kars_report("P001")
    assert "KARS report filed successfully" in first
    assert "KARS-P001-2026-03-11" in first

    second = env.tools.file_kars_report("P001")
    assert "already filed" in second
    assert "P001" in second
