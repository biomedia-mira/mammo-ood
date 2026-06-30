from pathlib import Path

from mammo_benchmark.inspection.sample_selection import validate_sample_csv


def test_fixed_inspection_samples_have_required_ids():
    repo = Path(__file__).resolve().parents[1]
    vindr = validate_sample_csv(repo / "data/inspection_samples/vindr_inspection_sample.csv")
    embed = validate_sample_csv(repo / "data/inspection_samples/embed_inspection_sample.csv")
    assert len(vindr) == 4334
    assert len(embed) == 10202

