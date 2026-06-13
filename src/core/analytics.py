from dataclasses import dataclass

import pandas as pd

from src.core.detector import UrbanDefect


@dataclass(frozen=True)
class RoadQualitySnapshot:
    road_quality_score: float
    critical_incidents: int
    medium_incidents: int
    observation_incidents: int
    affected_surface_share: float
    operational_status: str
    defect_table: pd.DataFrame


class RoadQualityEngine:
    def build_city_surface_snapshot(
        self,
        detected_infrastructure_defects: list[UrbanDefect],
    ) -> RoadQualitySnapshot:
        critical_incidents = self._count_by_severity(detected_infrastructure_defects, "Критический")
        medium_incidents = self._count_by_severity(detected_infrastructure_defects, "Средний")
        observation_incidents = self._count_by_severity(detected_infrastructure_defects, "Наблюдение")
        affected_surface_share = min(
            1.0,
            sum(defect.normalized_area for defect in detected_infrastructure_defects),
        )

        impact_index = sum(defect.surface_impact for defect in detected_infrastructure_defects)
        incident_penalty = critical_incidents * 8.5 + medium_incidents * 4.0 + observation_incidents * 1.25
        surface_penalty = min(55.0, impact_index * 420.0)
        road_quality_score = max(0.0, min(100.0, 100.0 - incident_penalty - surface_penalty))

        return RoadQualitySnapshot(
            road_quality_score=round(road_quality_score, 2),
            critical_incidents=critical_incidents,
            medium_incidents=medium_incidents,
            observation_incidents=observation_incidents,
            affected_surface_share=round(affected_surface_share, 4),
            operational_status=self._resolve_status(road_quality_score),
            defect_table=self._compose_defect_registry(detected_infrastructure_defects),
        )

    @staticmethod
    def _count_by_severity(
        detected_infrastructure_defects: list[UrbanDefect],
        severity_level: str,
    ) -> int:
        return sum(1 for defect in detected_infrastructure_defects if defect.severity_level == severity_level)

    @staticmethod
    def _resolve_status(road_quality_score: float) -> str:
        if road_quality_score >= 85:
            return "Норма"
        if road_quality_score >= 65:
            return "Требуется мониторинг"
        if road_quality_score >= 40:
            return "Плановый ремонт"
        return "Аварийное реагирование"

    @staticmethod
    def _compose_defect_registry(
        detected_infrastructure_defects: list[UrbanDefect],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Инцидент": defect.incident_id,
                    "Тип дефекта": defect.defect_type,
                    "Критичность": defect.severity_level,
                    "Уверенность ИИ": f"{defect.confidence:.1%}",
                    "Доля кадра": f"{defect.normalized_area:.2%}",
                }
                for defect in detected_infrastructure_defects
            ]
        )
