from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

# Патч для поддержки старых версий streamlit (< 1.36.0), где on_change в кастомных компонентах вызывает ошибку сериализации
try:
    streamlit_version = [int(x) for x in st.__version__.split(".")]
    if len(streamlit_version) >= 2 and (streamlit_version[0] < 1 or (streamlit_version[0] == 1 and streamlit_version[1] < 36)):
        import streamlit_folium
        original_component_func = streamlit_folium._component_func
        def patched_component_func(*args, **kwargs):
            kwargs.pop("on_change", None)
            return original_component_func(*args, **kwargs)
        streamlit_folium._component_func = patched_component_func
except Exception:
    pass

from streamlit_folium import st_folium

from src.core.analytics import RoadQualityEngine, RoadQualitySnapshot
from src.core.detector import DetectionReport, InfrastructureDetector, UrbanDefect
from src.geo.map_service import CityGeoProfile, UrbanMapManager


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ARTIFACT = PROJECT_ROOT / "weights" / "DATASET_W_w.pt"
DEFAULT_CITY = CityGeoProfile(
    name="Алматы",
    center_latitude=43.238949,
    center_longitude=76.889709,
)


st.set_page_config(
    page_title="CivicEye AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_infrastructure_detector(model_artifact: Path) -> InfrastructureDetector:
    return InfrastructureDetector(model_artifact=model_artifact)


def render_visual_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 1.6rem; }
        div[data-testid="stMetric"] {
            border: 1px solid #d7dde8;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
        }
        .civiceye-score {
            border: 1px solid #d7dde8;
            border-radius: 8px;
            padding: 18px;
            background: #ffffff;
        }
        .civiceye-score-row {
            display: flex;
            align-items: center;
            gap: 18px;
        }
        .civiceye-score-ring {
            width: 132px;
            height: 132px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: conic-gradient(#16a34a var(--score-angle), #e5e7eb 0deg);
        }
        .civiceye-score-core {
            width: 94px;
            height: 94px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: #ffffff;
            color: #111827;
            font-size: 28px;
            font-weight: 700;
        }
        .civiceye-caption {
            color: #5b6472;
            font-size: 13px;
            line-height: 1.45;
        }
        .civiceye-panel-title {
            font-size: 18px;
            font-weight: 700;
            margin: 0 0 10px 0;
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_score_widget(quality_snapshot: RoadQualitySnapshot) -> None:
    score_value = quality_snapshot.road_quality_score
    score_angle = max(0.0, min(100.0, score_value)) * 3.6
    st.markdown(
        f"""
        <div class="civiceye-score" style="--score-angle: {score_angle:.1f}deg;">
            <div class="civiceye-score-row">
                <div class="civiceye-score-ring">
                    <div class="civiceye-score-core">{score_value:.0f}</div>
                </div>
                <div>
                    <div class="civiceye-panel-title">Road Quality Score</div>
                    <div style="font-size: 22px; font-weight: 700;">{quality_snapshot.operational_status}</div>
                    <div class="civiceye-caption">
                        Индекс снижается за критические дефекты, средние повреждения и площадь затронутого покрытия.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_snapshot() -> RoadQualitySnapshot:
    return RoadQualitySnapshot(
        road_quality_score=100.0,
        critical_incidents=0,
        medium_incidents=0,
        observation_incidents=0,
        affected_surface_share=0.0,
        operational_status="Ожидание данных",
        defect_table=pd.DataFrame(
            columns=["Инцидент", "Тип дефекта", "Критичность", "Уверенность ИИ", "Доля кадра"]
        ),
    )


def analyze_uploaded_image(
    uploaded_media,
    detector: InfrastructureDetector,
    confidence_floor: float,
) -> DetectionReport:
    source_frame = Image.open(uploaded_media).convert("RGB")
    return detector.analyze_urban_environment(
        source_frame=source_frame,
        confidence_floor=confidence_floor,
        source_kind="image",
    )


def analyze_uploaded_video(
    uploaded_media,
    detector: InfrastructureDetector,
    confidence_floor: float,
    frame_stride: int,
    frame_limit: int,
) -> tuple[Image.Image | None, list[UrbanDefect]]:
    captured_defects: list[UrbanDefect] = []
    annotated_preview: Image.Image | None = None

    with NamedTemporaryFile(suffix=".mp4") as media_buffer:
        media_buffer.write(uploaded_media.getvalue())
        media_buffer.flush()
        video_stream = cv2.VideoCapture(media_buffer.name)

        frame_position = 0
        analyzed_frames = 0
        while video_stream.isOpened() and analyzed_frames < frame_limit:
            frame_available, raw_frame = video_stream.read()
            if not frame_available:
                break

            if frame_position % frame_stride == 0:
                frame_report = detector.analyze_urban_environment(
                    source_frame=raw_frame,
                    confidence_floor=confidence_floor,
                    source_kind="video",
                )
                annotated_preview = frame_report.annotated_frame
                captured_defects.extend(frame_report.detected_infrastructure_defects)
                analyzed_frames += 1

            frame_position += 1

        video_stream.release()

    return annotated_preview, captured_defects


def render_detection_registry(quality_snapshot: RoadQualitySnapshot) -> None:
    if quality_snapshot.defect_table.empty:
        st.info("Инциденты не обнаружены. Загрузите медиафайл и запустите ИИ-анализ.")
        return
    st.dataframe(quality_snapshot.defect_table, use_container_width=True, hide_index=True)


render_visual_style()

st.title("CivicEye AI")
st.caption("Ситуационный центр анализа дефектов дорожной инфраструктуры")

if not MODEL_ARTIFACT.exists():
    st.error(f"Весовой файл модели не найден: {MODEL_ARTIFACT}")
    st.stop()

road_quality_engine = RoadQualityEngine()
map_manager = UrbanMapManager(DEFAULT_CITY)
infrastructure_detector = load_infrastructure_detector(MODEL_ARTIFACT)

left_column, right_column = st.columns([1.05, 1.2], gap="large")

with left_column:
    st.subheader("Панель ИИ-анализа медиапотока")
    media_mode = st.radio(
        "Тип медиапотока",
        options=["Фото", "Видео"],
        index=0,
        horizontal=True,
    )
    confidence_floor = st.slider(
        "Порог уверенности модели",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
    )

    frame_stride = 30
    frame_limit = 8
    if media_mode == "Видео":
        frame_stride = st.slider("Шаг анализа кадров", 5, 90, 30, 5)
        frame_limit = st.slider("Максимум кадров для анализа", 1, 24, 8, 1)

    uploaded_media = st.file_uploader(
        "Загрузите фото или видео дорожного участка",
        type=["jpg", "jpeg", "png", "webp"] if media_mode == "Фото" else ["mp4", "avi", "mov", "mkv"],
    )

    run_analysis = st.button(
        "Запустить анализ",
        type="primary",
        use_container_width=True,
        disabled=uploaded_media is None,
    )

    active_defects: list[UrbanDefect] = []
    preview_frame: Image.Image | None = None

    if uploaded_media is not None and not run_analysis:
        if media_mode == "Фото":
            st.image(uploaded_media, caption="Исходный медиапоток", use_column_width=True)
        else:
            st.video(uploaded_media)

    if uploaded_media is not None and run_analysis:
        with st.spinner("CivicEye AI анализирует дорожную сцену..."):
            if media_mode == "Фото":
                detection_report = analyze_uploaded_image(
                    uploaded_media=uploaded_media,
                    detector=infrastructure_detector,
                    confidence_floor=confidence_floor,
                )
                active_defects = detection_report.detected_infrastructure_defects
                preview_frame = detection_report.annotated_frame
            else:
                preview_frame, active_defects = analyze_uploaded_video(
                    uploaded_media=uploaded_media,
                    detector=infrastructure_detector,
                    confidence_floor=confidence_floor,
                    frame_stride=frame_stride,
                    frame_limit=frame_limit,
                )

        if preview_frame is not None:
            st.image(preview_frame, caption="Разметка обнаруженных дефектов", use_column_width=True)

    quality_snapshot = (
        road_quality_engine.build_city_surface_snapshot(active_defects)
        if active_defects
        else render_empty_snapshot()
    )

with right_column:
    st.subheader("Интерактивная ГИС-карта инцидентов ЖКХ")
    city_incident_map = map_manager.render_incident_map(active_defects)
    st_folium(city_incident_map, height=370, use_container_width=True)

    st.subheader("Метрики состояния городской среды")
    render_score_widget(quality_snapshot)

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Критические", quality_snapshot.critical_incidents)
    metric_b.metric("Средние", quality_snapshot.medium_incidents)
    metric_c.metric("Площадь дефектов", f"{quality_snapshot.affected_surface_share:.1%}")

    st.subheader("Реестр инцидентов")
    render_detection_registry(quality_snapshot)
