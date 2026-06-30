from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm


OPTIMAM_ROOT = Path(os.environ.get("OPTIMAM_ROOT", "/path/to/optimam"))
OPTIMAM_CSV_ROOT = OPTIMAM_ROOT
OPTIMAM_DATA_ROOT = OPTIMAM_ROOT / "FFDM" / "DATA"
OPTIMAM_IMAGE_ROOT = OPTIMAM_ROOT / "FFDM" / "IMAGES"
FOR_PRESENTATION = "FOR PRESENTATION"


IMAGE_COLUMNS = [
    "client_id",
    "site",
    "client_status",
    "client_status_text",
    "episode_id",
    "episode_status",
    "episode_status_text",
    "episode_type",
    "episode_type_text",
    "episode_action",
    "episode_action_text",
    "episode_opened_date",
    "episode_closed_date",
    "episode_diagnosis_date",
    "episode_is_closed",
    "episode_year",
    "has_malignant_opinions",
    "has_benign_opinions",
    "is_interval_cancer",
    "screening_left_opinion",
    "screening_left_opinion_text",
    "screening_right_opinion",
    "screening_right_opinion_text",
    "screening_left_radiology_opinion",
    "screening_left_radiology_opinion_text",
    "screening_right_radiology_opinion",
    "screening_right_radiology_opinion_text",
    "screening_date",
    "screening_equipment_make_model",
    "assessment_left_opinion",
    "assessment_left_opinion_text",
    "assessment_right_opinion",
    "assessment_right_opinion_text",
    "assessment_date",
    "clinical_left_opinion",
    "clinical_left_opinion_text",
    "clinical_right_opinion",
    "clinical_right_opinion_text",
    "clinical_date",
    "biopsy_wide_left_opinion",
    "biopsy_wide_left_opinion_text",
    "biopsy_wide_right_opinion",
    "biopsy_wide_right_opinion_text",
    "biopsy_wide_date",
    "biopsy_fine_left_opinion",
    "biopsy_fine_left_opinion_text",
    "biopsy_fine_right_opinion",
    "biopsy_fine_right_opinion_text",
    "biopsy_fine_date",
    "surgery_left_opinion",
    "surgery_left_opinion_text",
    "surgery_right_opinion",
    "surgery_right_opinion_text",
    "surgery_date",
    "study_uid",
    "study_date",
    "study_event_type",
    "study_event_type_text",
    "series_uid",
    "sop_uid",
    "dcm_path",
    "dicom_json_path",
    "image_path",
    "image_id",
    "ImageLateralityFinal",
    "ViewPosition",
    "PresentationIntentType",
    "Manufacturer",
    "ManufacturerModelName",
    "Modality",
    "Rows",
    "Columns",
    "AgeAtScreening",
    "PatientAgeDICOM",
    "num_marks",
]

MARK_COLUMNS = [
    "client_id",
    "client_status",
    "client_status_text",
    "episode_id",
    "episode_status",
    "episode_status_text",
    "study_uid",
    "series_uid",
    "sop_uid",
    "image_path",
    "ImageLateralityFinal",
    "ViewPosition",
    "AgeAtScreening",
    "PatientAgeDICOM",
    "MarkID",
    "LinkedNBSSLesionNumber",
    "X1",
    "Y1",
    "X2",
    "Y2",
    "Width",
    "Height",
    "Conspicuity",
    "Mass",
    "MassClassification",
    "BenignClassification",
    "MilkOfCalcium",
    "OtherBenignCluster",
    "PlasmaCellMastitis",
    "BenignSkinFeature",
    "SuspiciousCalcifications",
    "Calcifications",
    "SutureCalcification",
    "VascularFeature",
    "FocalAsymmetry",
    "ArchitecturalDistortion",
    "DystrophicCalcification",
    "FatNecrosis",
    "LinkedLesionStatus",
    "LinkedLesionDescription",
    "LinkedLesionGrade",
    "LinkedLesionClinicalOpinion",
    "LinkedLesionClinicalOpinionText",
    "LinkedLesionBiopsyWideOpinion",
    "LinkedLesionBiopsyWideOpinionText",
    "LinkedLesionBiopsyFineOpinion",
    "LinkedLesionBiopsyFineOpinionText",
    "LinkedLesionSurgeryOpinion",
    "LinkedLesionSurgeryOpinionText",
]


def _import_omidb():
    try:
        import omidb
    except ImportError as exc:
        raise ImportError(
            "OPTIMAM preprocessing uses the official PyPI `omidb` package. "
            "Activate the `omidb` conda environment before running this code."
        ) from exc
    return omidb


def _enum_name(value: Any) -> Any:
    return value.name if value is not None else pd.NA


def _enum_value(value: Any) -> Any:
    return value.value if value is not None else pd.NA


def _date_text(value: Any) -> Any:
    return value.isoformat() if value is not None else pd.NA


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _join(values: Iterable[Any]) -> Any:
    clean = [str(value) for value in values if not _is_missing(value)]
    return "|".join(clean) if clean else pd.NA


def _unique_join(values: Iterable[Any]) -> Any:
    clean = []
    for value in values:
        if _is_missing(value) or value in clean:
            continue
        clean.append(value)
    return _join(clean)


def _dicom_value(image: Any, tag: str) -> Any:
    item = image.attributes.get(tag) if image.attributes else None
    values = item.get("Value") if isinstance(item, dict) else None
    if not values:
        return pd.NA

    value = values[0]
    if isinstance(value, dict):
        return value.get("Alphabetic", str(value))
    return value


def _screening_summary(episode: Any) -> dict[str, Any]:
    screenings = episode.events.screening if episode.events and episode.events.screening else []
    return {
        "screening_left_opinion": _join(_enum_name(screen.left_opinion) for screen in screenings),
        "screening_left_opinion_text": _join(_enum_value(screen.left_opinion) for screen in screenings),
        "screening_right_opinion": _join(_enum_name(screen.right_opinion) for screen in screenings),
        "screening_right_opinion_text": _join(_enum_value(screen.right_opinion) for screen in screenings),
        "screening_left_radiology_opinion": _join(
            _enum_name(screen.left.opinion)
            for screen in screenings
            if screen.left is not None
        ),
        "screening_left_radiology_opinion_text": _join(
            _enum_value(screen.left.opinion)
            for screen in screenings
            if screen.left is not None
        ),
        "screening_right_radiology_opinion": _join(
            _enum_name(screen.right.opinion)
            for screen in screenings
            if screen.right is not None
        ),
        "screening_right_radiology_opinion_text": _join(
            _enum_value(screen.right.opinion)
            for screen in screenings
            if screen.right is not None
        ),
        "screening_date": _join(
            _date_text(date)
            for screen in screenings
            for date in (screen.dates or [])
        ),
        "screening_equipment_make_model": _unique_join(
            equipment
            for screen in screenings
            for side in [screen.left, screen.right]
            if side is not None
            for equipment in [side.equipment_make_model]
        ),
    }


def _event_summary(episode: Any, event_name: str) -> dict[str, Any]:
    event = getattr(episode.events, event_name, None) if episode.events else None
    return {
        f"{event_name}_left_opinion": _enum_name(event.left_opinion) if event else pd.NA,
        f"{event_name}_left_opinion_text": _enum_value(event.left_opinion) if event else pd.NA,
        f"{event_name}_right_opinion": _enum_name(event.right_opinion) if event else pd.NA,
        f"{event_name}_right_opinion_text": _enum_value(event.right_opinion) if event else pd.NA,
        f"{event_name}_date": _join(_date_text(date) for date in (event.dates or [])) if event else pd.NA,
    }


def _episode_event_summary(episode: Any) -> dict[str, Any]:
    row = _screening_summary(episode)
    for event_name in ["assessment", "clinical", "biopsy_wide", "biopsy_fine", "surgery"]:
        row.update(_event_summary(episode, event_name))
    return row


@lru_cache(maxsize=None)
def _load_nbss_json(nbss_path: str) -> dict[str, Any]:
    with open(nbss_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _age_at_screening(data_root: Path, client_id: str, episode_id: str) -> Any:
    nbss_path = data_root / client_id / f"NBSS_{client_id}.json"
    if not nbss_path.exists():
        return pd.NA

    episode = _load_nbss_json(str(nbss_path)).get(str(episode_id))
    if not isinstance(episode, dict):
        return pd.NA

    screening = episode.get("SCREENING")
    ages: list[int] = []
    if isinstance(screening, dict):
        for side in ("L", "R"):
            side_data = screening.get(side)
            if not isinstance(side_data, dict):
                continue
            age = side_data.get("AgeAtScreening")
            if age is None:
                continue
            try:
                age = int(age)
            except (TypeError, ValueError):
                continue
            if age not in ages:
                ages.append(age)

    if len(ages) == 1:
        return ages[0]

    try:
        return int(episode.get("PatientAge"))
    except (TypeError, ValueError):
        return pd.NA


def _patient_ids(
    data_root: Path = OPTIMAM_DATA_ROOT,
    patient_ids: Iterable[str] | None = None,
    limit_patients: int | None = None,
) -> list[str] | None:
    if patient_ids is not None:
        ids = [str(patient_id) for patient_id in patient_ids]
    elif limit_patients is not None:
        ids = []
        with os.scandir(data_root) as entries:
            for entry in entries:
                if entry.is_dir():
                    ids.append(entry.name)
                    if len(ids) >= limit_patients:
                        break
    else:
        return None

    if limit_patients is not None:
        ids = ids[:limit_patients]

    return [patient_id for patient_id in ids if (data_root / patient_id).exists()]


def iter_optimam_clients(
    data_root: Path = OPTIMAM_DATA_ROOT,
    image_root: Path = OPTIMAM_IMAGE_ROOT,
    patient_ids: Iterable[str] | None = None,
    limit_patients: int | None = None,
    show_progress: bool = True,
):
    omidb = _import_omidb()
    selected_patient_ids = _patient_ids(data_root, patient_ids, limit_patients)
    clients = omidb.DB(
        data_root,
        image_dir=image_root,
        clients=selected_patient_ids,
        ignore_missing_images=True,
    )
    if show_progress:
        total = len(selected_patient_ids) if selected_patient_ids is not None else None
        clients = tqdm(clients, total=total, desc="Parsing OPTIMAM")
    yield from clients


def _is_cancer_status_name(episode_status: Any) -> Any:
    if _is_missing(episode_status):
        return pd.NA
    return str(episode_status) in {"M", "CI"}


def build_optimam_dataframe(
    data_root: Path = OPTIMAM_DATA_ROOT,
    image_root: Path = OPTIMAM_IMAGE_ROOT,
    patient_ids: Iterable[str] | None = None,
    limit_patients: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    _import_omidb()
    rows: list[dict[str, Any]] = []

    for client in iter_optimam_clients(data_root, image_root, patient_ids, limit_patients, show_progress):
        client_status = client.status
        for episode in client.episodes:
            episode_status = episode.status
            event_summary = _episode_event_summary(episode)
            age_at_screening = _age_at_screening(data_root, client.id, episode.id)
            for study in episode.studies:
                study_event_type = _join(_enum_name(event) for event in (study.event_type or []))
                study_event_type_text = _join(_enum_value(event) for event in (study.event_type or []))
                for series in study.series:
                    for image in series.images:
                        dcm_path = Path(client.id) / study.id / f"{image.id}.dcm"
                        image_path = dcm_path.with_suffix(".png")
                        row = {
                            "client_id": client.id,
                            "site": client.site,
                            "client_status": _enum_name(client_status),
                            "client_status_text": _enum_value(client_status),
                            "episode_id": episode.id,
                            "episode_status": _enum_name(episode_status),
                            "episode_status_text": _enum_value(episode_status),
                            "episode_type": _enum_name(episode.type),
                            "episode_type_text": _enum_value(episode.type),
                            "episode_action": _enum_name(episode.action),
                            "episode_action_text": _enum_value(episode.action),
                            "episode_opened_date": _date_text(episode.opened_date),
                            "episode_closed_date": _date_text(episode.closed_date),
                            "episode_diagnosis_date": _date_text(episode.diagnosis_date),
                            "episode_is_closed": episode.is_closed,
                            "episode_year": episode.actual_opened_year,
                            "has_malignant_opinions": episode.has_malignant_opinions,
                            "has_benign_opinions": episode.has_benign_opinions,
                            "is_interval_cancer": episode.is_interval_cancer,
                            "study_uid": study.id,
                            "study_date": _date_text(study.date),
                            "study_event_type": study_event_type,
                            "study_event_type_text": study_event_type_text,
                            "series_uid": series.id,
                            "sop_uid": image.id,
                            "dcm_path": str(dcm_path),
                            "dicom_json_path": str(Path(client.id) / study.id / f"{image.id}.dcm.json"),
                            "image_path": str(image_path),
                            "image_id": image_path.name,
                            "ImageLateralityFinal": _dicom_value(image, "00200062"),
                            "ViewPosition": _dicom_value(image, "00185101"),
                            "PresentationIntentType": _dicom_value(image, "00080068"),
                            "Manufacturer": _dicom_value(image, "00080070"),
                            "ManufacturerModelName": _dicom_value(image, "00081090"),
                            "Modality": _dicom_value(image, "00080060"),
                            "Rows": _dicom_value(image, "00280010"),
                            "Columns": _dicom_value(image, "00280011"),
                            "AgeAtScreening": age_at_screening,
                            "PatientAgeDICOM": _dicom_value(image, "00101010"),
                            "num_marks": len(image.marks),
                        }
                        rows.append(row | event_summary)

    return pd.DataFrame(rows, columns=IMAGE_COLUMNS)


def filter_optimam_for_presentation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    presentation_intent = df["PresentationIntentType"].fillna("").astype(str).str.upper().str.strip()
    return df[presentation_intent == FOR_PRESENTATION].reset_index(drop=True)


def build_optimam_classification_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_optimam_for_presentation(df)
    df = df.copy()
    df["is_cancer"] = df["episode_status"].apply(_is_cancer_status_name)
    df = df[df["is_cancer"].notna()].copy()
    df["is_cancer"] = df["is_cancer"].astype(int)
    return df.reset_index(drop=True)


def _mark_lesion_ids(mark: Any) -> list[Any]:
    lesion_ids = getattr(mark, "lesion_ids", None)
    if lesion_ids is None:
        return []
    return sorted(lesion_ids)


def _linked_lesions(episode: Any, mark: Any) -> list[Any]:
    lesions = getattr(episode, "lesions", None) or {}
    return [
        lesions[lesion_id]
        for lesion_id in _mark_lesion_ids(mark)
        if lesion_id in lesions
    ]


def build_optimam_marks_dataframe(
    df_images: pd.DataFrame,
    data_root: Path = OPTIMAM_DATA_ROOT,
    image_root: Path = OPTIMAM_IMAGE_ROOT,
    show_progress: bool = True,
) -> pd.DataFrame:
    if df_images.empty:
        return pd.DataFrame(columns=MARK_COLUMNS)

    df_images = filter_optimam_for_presentation(df_images)
    df_images = df_images.drop_duplicates(subset=["client_id", "study_uid", "sop_uid"], keep="last")
    image_lookup = df_images.set_index(["client_id", "study_uid", "sop_uid"], drop=False)
    patient_ids = df_images["client_id"].dropna().astype(str).drop_duplicates().tolist()

    rows: list[dict[str, Any]] = []
    for client in iter_optimam_clients(data_root, image_root, patient_ids, show_progress=show_progress):
        client_status = client.status
        for episode in client.episodes:
            episode_status = episode.status
            for study in episode.studies:
                for series in study.series:
                    for image in series.images:
                        image_key = (client.id, study.id, image.id)
                        if not image.marks or image_key not in image_lookup.index:
                            continue

                        image_row = image_lookup.loc[image_key]
                        for mark in image.marks:
                            bbox = mark.boundingBox
                            lesions = _linked_lesions(episode, mark)
                            rows.append(
                                {
                                    "client_id": client.id,
                                    "client_status": _enum_name(client_status),
                                    "client_status_text": _enum_value(client_status),
                                    "episode_id": episode.id,
                                    "episode_status": _enum_name(episode_status),
                                    "episode_status_text": _enum_value(episode_status),
                                    "study_uid": study.id,
                                    "series_uid": series.id,
                                    "sop_uid": image.id,
                                    "image_path": image_row["image_path"],
                                    "ImageLateralityFinal": image_row["ImageLateralityFinal"],
                                    "ViewPosition": image_row["ViewPosition"],
                                    "AgeAtScreening": image_row.get("AgeAtScreening", pd.NA),
                                    "PatientAgeDICOM": image_row.get("PatientAgeDICOM", pd.NA),
                                    "MarkID": mark.id,
                                    "LinkedNBSSLesionNumber": _join(_mark_lesion_ids(mark)),
                                    "X1": bbox.x1,
                                    "Y1": bbox.y1,
                                    "X2": bbox.x2,
                                    "Y2": bbox.y2,
                                    "Width": bbox.x2 - bbox.x1,
                                    "Height": bbox.y2 - bbox.y1,
                                    "Conspicuity": _enum_value(mark.conspicuity),
                                    "Mass": mark.mass,
                                    "MassClassification": _enum_value(mark.mass_classification),
                                    "BenignClassification": _enum_value(mark.benign_classification),
                                    "MilkOfCalcium": mark.milk_of_calcium,
                                    "OtherBenignCluster": mark.other_benign_cluster,
                                    "PlasmaCellMastitis": mark.plasma_cell_mastitis,
                                    "BenignSkinFeature": mark.benign_skin_feature,
                                    "SuspiciousCalcifications": mark.suspicious_calcifications,
                                    "Calcifications": mark.calcifications,
                                    "SutureCalcification": mark.suture_calcification,
                                    "VascularFeature": mark.vascular_feature,
                                    "FocalAsymmetry": mark.focal_asymmetry,
                                    "ArchitecturalDistortion": mark.architectural_distortion,
                                    "DystrophicCalcification": mark.dystrophic_calcification,
                                    "FatNecrosis": mark.fat_necrosis,
                                    "LinkedLesionStatus": _join(_enum_name(lesion.status) for lesion in lesions),
                                    "LinkedLesionDescription": _join(_enum_value(lesion.description) for lesion in lesions),
                                    "LinkedLesionGrade": _join(_enum_value(lesion.grade) for lesion in lesions),
                                    "LinkedLesionClinicalOpinion": _join(
                                        _enum_name(lesion.clinical.opinion)
                                        for lesion in lesions
                                        if lesion.clinical is not None
                                    ),
                                    "LinkedLesionClinicalOpinionText": _join(
                                        _enum_value(lesion.clinical.opinion)
                                        for lesion in lesions
                                        if lesion.clinical is not None
                                    ),
                                    "LinkedLesionBiopsyWideOpinion": _join(
                                        _enum_name(lesion.biopsy_wide.opinion)
                                        for lesion in lesions
                                        if lesion.biopsy_wide is not None
                                    ),
                                    "LinkedLesionBiopsyWideOpinionText": _join(
                                        _enum_value(lesion.biopsy_wide.opinion)
                                        for lesion in lesions
                                        if lesion.biopsy_wide is not None
                                    ),
                                    "LinkedLesionBiopsyFineOpinion": _join(
                                        _enum_name(lesion.biopsy_fine.opinion)
                                        for lesion in lesions
                                        if lesion.biopsy_fine is not None
                                    ),
                                    "LinkedLesionBiopsyFineOpinionText": _join(
                                        _enum_value(lesion.biopsy_fine.opinion)
                                        for lesion in lesions
                                        if lesion.biopsy_fine is not None
                                    ),
                                    "LinkedLesionSurgeryOpinion": _join(
                                        _enum_name(lesion.surgery.opinion)
                                        for lesion in lesions
                                        if lesion.surgery is not None
                                    ),
                                    "LinkedLesionSurgeryOpinionText": _join(
                                        _enum_value(lesion.surgery.opinion)
                                        for lesion in lesions
                                        if lesion.surgery is not None
                                    ),
                                }
                            )

    return pd.DataFrame(rows, columns=MARK_COLUMNS)
