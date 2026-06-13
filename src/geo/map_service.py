from dataclasses import dataclass
from random import Random

import folium

from src.core.detector import UrbanDefect


@dataclass(frozen=True)
class CityGeoProfile:
    name: str
    center_latitude: float
    center_longitude: float
    marker_spread: float = 0.012


class UrbanMapManager:
    def __init__(self, city_profile: CityGeoProfile) -> None:
        self.city_profile = city_profile

    def render_incident_map(
        self,
        detected_infrastructure_defects: list[UrbanDefect],
    ) -> folium.Map:
        city_map = folium.Map(
            location=[self.city_profile.center_latitude, self.city_profile.center_longitude],
            zoom_start=13,
            tiles="CartoDB positron",
            control_scale=True,
        )

        folium.Circle(
            location=[self.city_profile.center_latitude, self.city_profile.center_longitude],
            radius=950,
            color="#2563eb",
            fill=True,
            fill_opacity=0.05,
            weight=2,
            tooltip=f"Зона мониторинга: {self.city_profile.name}",
        ).add_to(city_map)

        for defect_index, infrastructure_defect in enumerate(detected_infrastructure_defects):
            incident_latitude, incident_longitude = self._derive_incident_coordinate(
                infrastructure_defect,
                defect_index,
            )
            folium.Marker(
                location=[incident_latitude, incident_longitude],
                tooltip=f"{infrastructure_defect.defect_type}: {infrastructure_defect.severity_level}",
                popup=self._build_popup(infrastructure_defect),
                icon=folium.Icon(
                    color=self._marker_color(infrastructure_defect.severity_level),
                    icon="warning-sign",
                ),
            ).add_to(city_map)

        return city_map

    def _derive_incident_coordinate(
        self,
        infrastructure_defect: UrbanDefect,
        defect_index: int,
    ) -> tuple[float, float]:
        deterministic_noise = Random(f"{infrastructure_defect.incident_id}-{defect_index}")
        anchor_x, anchor_y = infrastructure_defect.frame_anchor
        latitude_offset = (anchor_y - 0.5) * self.city_profile.marker_spread
        longitude_offset = (anchor_x - 0.5) * self.city_profile.marker_spread
        latitude_jitter = deterministic_noise.uniform(-0.0015, 0.0015)
        longitude_jitter = deterministic_noise.uniform(-0.0015, 0.0015)

        return (
            self.city_profile.center_latitude + latitude_offset + latitude_jitter,
            self.city_profile.center_longitude + longitude_offset + longitude_jitter,
        )

    @staticmethod
    def _marker_color(severity_level: str) -> str:
        if severity_level == "Критический":
            return "red"
        if severity_level == "Средний":
            return "orange"
        return "blue"

    @staticmethod
    def _build_popup(infrastructure_defect: UrbanDefect) -> str:
        return (
            f"<b>{infrastructure_defect.incident_id}</b><br>"
            f"{infrastructure_defect.defect_type}<br>"
            f"Критичность: {infrastructure_defect.severity_level}<br>"
            f"Уверенность: {infrastructure_defect.confidence:.1%}"
        )
