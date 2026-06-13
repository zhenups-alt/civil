from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class UrbanDefect:
    incident_id: str
    defect_type: str
    severity_level: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    normalized_area: float
    surface_impact: float
    frame_anchor: tuple[float, float]


@dataclass(frozen=True)
class DetectionReport:
    source_kind: str
    source_size: tuple[int, int]
    generated_at: datetime
    annotated_frame: Image.Image
    detected_infrastructure_defects: list[UrbanDefect]
    model_catalog: dict[int, str]


class InfrastructureDetector:
    def __init__(self, model_artifact: Path) -> None:
        self.model_artifact = Path(model_artifact)
        self._vision_model: YOLO | None = None

    def analyze_urban_environment(
        self,
        source_frame: Image.Image | np.ndarray,
        confidence_floor: float,
        source_kind: str = "image",
    ) -> DetectionReport:
        prepared_frame = self._normalize_frame(source_frame)
        frame_height, frame_width = prepared_frame.shape[:2]
        detection_payload = self._model.predict(
            prepared_frame,
            conf=confidence_floor,
            verbose=False,
        )

        if not detection_payload:
            annotated_frame = Image.fromarray(prepared_frame)
            return DetectionReport(
                source_kind=source_kind,
                source_size=(frame_width, frame_height),
                generated_at=datetime.utcnow(),
                annotated_frame=annotated_frame,
                detected_infrastructure_defects=[],
                model_catalog=self._read_model_catalog(),
            )

        primary_payload = detection_payload[0]
        detected_defects = self._extract_defect_events(
            primary_payload,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        annotated_frame = self._render_detection_overlay(primary_payload)

        return DetectionReport(
            source_kind=source_kind,
            source_size=(frame_width, frame_height),
            generated_at=datetime.utcnow(),
            annotated_frame=annotated_frame,
            detected_infrastructure_defects=detected_defects,
            model_catalog=self._read_model_catalog(),
        )

    @property
    def _model(self) -> YOLO:
        if self._vision_model is None:
            self._vision_model = self._load_model()
        return self._vision_model

    def _load_model(self) -> YOLO:
        if not self.model_artifact.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_artifact}")

        try:
            import torch
        except Exception:
            return YOLO(str(self.model_artifact))

        original_loader = torch.load

        def compatible_loader(*loader_args: Any, **loader_kwargs: Any) -> Any:
            loader_kwargs.setdefault("weights_only", False)
            return original_loader(*loader_args, **loader_kwargs)

        torch.load = compatible_loader
        try:
            return YOLO(str(self.model_artifact))
        finally:
            torch.load = original_loader

    def _extract_defect_events(
        self,
        model_response: Any,
        frame_width: int,
        frame_height: int,
    ) -> list[UrbanDefect]:
        detection_boxes = getattr(model_response, "boxes", None)
        if detection_boxes is None or len(detection_boxes) == 0:
            return []

        model_catalog = self._read_model_catalog()
        frame_area = max(float(frame_width * frame_height), 1.0)
        urban_defects: list[UrbanDefect] = []

        for defect_index, raw_box in enumerate(detection_boxes):
            x_minimum, y_minimum, x_maximum, y_maximum = (
                float(value) for value in raw_box.xyxy[0].tolist()
            )
            detected_class_index = int(raw_box.cls[0].item())
            detected_label = model_catalog.get(
                detected_class_index,
                f"infrastructure_defect_{detected_class_index}",
            )
            confidence = float(raw_box.conf[0].item())
            defect_area = max(0.0, x_maximum - x_minimum) * max(0.0, y_maximum - y_minimum)
            normalized_area = defect_area / frame_area
            severity_level = self._classify_defect_severity(detected_label, detected_class_index)
            surface_impact = self._estimate_surface_impact(
                normalized_area=normalized_area,
                severity_level=severity_level,
                confidence=confidence,
            )

            urban_defects.append(
                UrbanDefect(
                    incident_id=f"CIV-{datetime.utcnow().strftime('%H%M%S')}-{defect_index + 1:03d}",
                    defect_type=self._normalize_defect_name(detected_label, detected_class_index),
                    severity_level=severity_level,
                    confidence=confidence,
                    bounding_box=(x_minimum, y_minimum, x_maximum, y_maximum),
                    normalized_area=normalized_area,
                    surface_impact=surface_impact,
                    frame_anchor=(
                        (x_minimum + x_maximum) / 2.0 / max(frame_width, 1),
                        (y_minimum + y_maximum) / 2.0 / max(frame_height, 1),
                    ),
                )
            )

        return urban_defects

    def _read_model_catalog(self) -> dict[int, str]:
        model_names = getattr(self._model, "names", {}) if self._vision_model else {}
        if isinstance(model_names, dict):
            return {int(class_id): str(label) for class_id, label in model_names.items()}
        return {index: str(label) for index, label in enumerate(model_names)}

    @staticmethod
    def _normalize_frame(source_frame: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(source_frame, Image.Image):
            return np.array(source_frame.convert("RGB"))

        if source_frame.ndim == 2:
            return cv2.cvtColor(source_frame, cv2.COLOR_GRAY2RGB)

        if source_frame.shape[-1] == 4:
            return cv2.cvtColor(source_frame, cv2.COLOR_BGRA2RGB)

        return cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _render_detection_overlay(model_response: Any) -> Image.Image:
        overlay_frame = model_response.plot()
        if overlay_frame.ndim == 3 and overlay_frame.shape[-1] == 3:
            overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(overlay_frame)

    @staticmethod
    def _classify_defect_severity(defect_label: str, class_index: int) -> str:
        normalized_label = defect_label.lower()
        critical_terms = ("pothole", "hole", "pit", "rut", "depression", "яма", "выбоина")
        medium_terms = ("crack", "fissure", "fracture", "alligator", "трещ")

        if any(term in normalized_label for term in critical_terms) or class_index == 0:
            return "Критический"
        if any(term in normalized_label for term in medium_terms):
            return "Средний"
        return "Наблюдение"

    @staticmethod
    def _normalize_defect_name(defect_label: str, class_index: int) -> str:
        normalized_label = defect_label.lower()
        if class_index == 0 or any(term in normalized_label for term in ("pothole", "hole", "pit")):
            return "Яма дорожного полотна"
        if any(term in normalized_label for term in ("crack", "fissure", "fracture")):
            return "Трещина дорожного полотна"
        if "debris" in normalized_label:
            return "Посторонний объект на дороге"
        if "vegetation" in normalized_label:
            return "Зарастание дорожной зоны"
        return "Дефект городской инфраструктуры"

    @staticmethod
    def _estimate_surface_impact(
        normalized_area: float,
        severity_level: str,
        confidence: float,
    ) -> float:
        severity_weight = {
            "Критический": 1.0,
            "Средний": 0.62,
            "Наблюдение": 0.34,
        }.get(severity_level, 0.34)
        return normalized_area * severity_weight * confidence
