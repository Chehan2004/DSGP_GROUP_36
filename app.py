from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from QualityGrader import QualityGrader
from TeaDetector import TeaDetector
from TeaDiseaseClassifier import TeaDiseaseClassifier
from TeaFertilizerRecommender import TeaFertilizerRecommender
from TeaLeafPredictor import TeaLeafPredictor

st.set_page_config(
    page_title="TeaVision · Leaf Intelligence",
    layout="centered",
    page_icon=None,
    initial_sidebar_state="expanded"
)


# ── Load CSS ──────────────────────────────────────────────────────────────────
def load_css(path: str) -> None:
    """Read an external CSS file and inject it into the Streamlit page."""
    css = Path(path).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("style.css")


# ── Session State ─────────────────────────────────────────────────────────────
if 'processed_crops' not in st.session_state:
    st.session_state.processed_crops = None
if 'annotated_img' not in st.session_state:
    st.session_state.annotated_img = None
if 'selected_leaf' not in st.session_state:
    st.session_state.selected_leaf = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## TeaVision")
    st.divider()
    st.markdown("**Image Input**")
    uploaded_file = st.file_uploader(
        "Upload a tea leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Detection Parameters**")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.30, step=0.05,
        help="Determines the minimum score required to identify a region as a tea leaf. Lowering this can help in cluttered backgrounds."
    )

    padding = st.slider(
        "Crop Padding (px)",
        min_value=0, max_value=50, value=15, step=1,
        help="Adds a border of pixels around the leaf before analysis. This helps the Quality Assessment model see the entire leaf margin."
    )
    st.divider()
    st.caption("TeaVision · Leaf Intelligence System\nv1.0 · DSGP Project")


# ── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models(pad):
    try:
        yolo_detector          = TeaDetector(padding=pad)
        leaf_predictor         = TeaLeafPredictor()
        disease_classifier     = TeaDiseaseClassifier()
        quality_grader         = QualityGrader()
        fertilizer_recommender = TeaFertilizerRecommender()
        return yolo_detector, leaf_predictor, disease_classifier, quality_grader, fertilizer_recommender
    except Exception as e:
        st.error(f"Model initialisation failed: {e}")
        return None, None, None, None, None

detector, predictor, disease_classifier, quality_grader, fertilizer_recommender = load_models(padding)


# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("# TeaVision")
st.markdown(
    "<p class='tv-subtitle'>"
    "This application automates tea leaf quality grading and disease detection for the Sri Lankan tea industry. It uses machine learning to verify if an image contains a tea leaf. The system identifies specific pests and diseases. If a disease is found, it recommends the right fertilizer and treatment. For healthy leaves, it grades quality based on color and texture. This tool uses deep learning to make tea assessment faster and more accurate than traditional manual methods."
    "</p>",
    unsafe_allow_html=True
)
st.divider()


# ── Image Upload & Detection ──────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown("")

    if all([detector, predictor, disease_classifier, quality_grader, fertilizer_recommender]):
        if st.button("Detect and Verify Leaves", type="primary"):
            st.session_state.selected_leaf = None

            with st.spinner("Detecting and verifying leaves..."):
                annotated_img, crops = detector.detect_and_crop(image, conf_threshold=conf_threshold)
                st.session_state.annotated_img = annotated_img

                processed_crops = []
                if crops:
                    for unique_id, crop in enumerate(crops):
                        try:
                            label, pred_conf, metadata = predictor.predict(crop['image'])
                            crop['label']     = label
                            crop['pred_conf'] = pred_conf
                            crop['id']        = unique_id
                        except Exception as e:
                            crop['label']     = "Error"
                            crop['pred_conf'] = 0.0
                            crop['id']        = unique_id
                            print(f"Prediction error: {e}")
                        processed_crops.append(crop)

                st.session_state.processed_crops = processed_crops

    # ── Detection Results ─────────────────────────────────────────────────────
    if st.session_state.processed_crops is not None:
        st.markdown("### Detection Results")
        st.image(
            st.session_state.annotated_img,
            caption="Annotated image with bounding boxes",
            use_container_width=True
        )
        st.markdown("")

        crops = st.session_state.processed_crops

        if crops:
            show_only_tea = st.toggle("Show only verified tea leaves", value=False)
            display_crops = (
                [c for c in crops if c.get('label') == 'Tea Leaf']
                if show_only_tea else crops
            )

            if display_crops:
                tea_count   = sum(1 for c in display_crops if c.get('label') == 'Tea Leaf')
                other_count = len(display_crops) - tea_count

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Detected",      len(display_crops))
                m2.metric("Verified Tea Leaves", tea_count)
                m3.metric("Other / Uncertain",   other_count)
                st.markdown("")

                cols = st.columns(4)
                for idx, crop in enumerate(display_crops):
                    with cols[idx % 4]:
                        lbl   = crop.get('label', 'N/A')
                        conf  = crop.get('pred_conf', 0)
                        badge = "badge-verified" if lbl == "Tea Leaf" else "badge-other"

                        st.image(crop['image'], use_container_width=True)
                        st.markdown(
                            f"<div class='tv-leaf-label'>"
                            f"<span class='leaf-badge {badge}'>{lbl}</span></div>"
                            f"<div class='tv-leaf-conf'>{conf:.0%} confidence</div>",
                            unsafe_allow_html=True
                        )

                        _, center_col, _ = st.columns([1, 2, 1])
                        with center_col:
                            st.button(
                                "Select",
                                key=f"select_btn_{crop['id']}",
                                on_click=lambda c=crop: st.session_state.update(selected_leaf=c),
                                use_container_width=True
                            )
            else:
                st.info(
                    "No verified tea leaves found. Toggle off the filter to see all detections."
                )
        else:
            st.info(
                "No objects detected. Try lowering the Confidence Threshold in the sidebar."
            )

else:
    # ── Empty State ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='tv-empty'>"
        "<p class='tv-empty-title'>Upload a tea leaf image to begin</p>"
        "<p class='tv-empty-sub'>"
        "Supports JPG, JPEG and PNG &nbsp;&middot;&nbsp; "
        "Configure detection parameters in the sidebar"
        "</p></div>",
        unsafe_allow_html=True
    )


# ── Selected Leaf Analysis ────────────────────────────────────────────────────
if st.session_state.get('selected_leaf') is not None:
    selected = st.session_state.selected_leaf

    st.divider()
    st.markdown("### Leaf Analysis")

    c_img, c_meta = st.columns([1, 2])
    with c_img:
        st.image(selected['image'], use_container_width=True)
    with c_meta:
        lbl   = selected.get('label', 'N/A')
        badge = "badge-verified" if lbl == "Tea Leaf" else "badge-other"
        st.markdown(
            f"<p class='section-label'>Verification Result</p>"
            f"<span class='leaf-badge {badge}'>{lbl}</span>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if lbl == "Tea Leaf":
            st.success("Confirmed as a genuine tea leaf.")
        else:
            st.warning("This crop did not pass tea leaf verification.")

    st.markdown("")

    # ── Quality Assessment ────────────────────────────────────────────────────
    st.markdown("<p class='section-label'>Quality Assessment</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='tv-subtitle'>"
        "This component evaluates the premium status of healthy tea leaves by analyzing visual characteristics "
        "such as color and texture. Using the EfficientNet-B0 architecture, it classifies leaves into "
        "five distinct grades (T0-T4) to ensure standardized quality control before industrial processing."
        "</p>",
        unsafe_allow_html=True
    )

    if 'quality_label' in selected:
        qa1, qa2 = st.columns(2)
        qa1.success(f"**Grade:** {selected['quality_label']}")
        qa2.info(f"**Confidence:** {selected['quality_conf']:.1%}")
    else:
        st.caption("Grade the physical quality of this leaf on a T0 to T4 scale.")
        if st.button("Assess Quality", key=f"qual_btn_{selected['id']}"):
            with st.spinner("Running quality grading model..."):
                try:
                    img_array = np.array(selected['image'])
                    q_label, q_conf = quality_grader.predict(img_array)
                    selected['quality_label'] = q_label
                    selected['quality_conf']  = q_conf
                    st.rerun()
                except Exception as e:
                    st.error(f"Quality assessment error: {e}")

    st.markdown("")

    # ── Disease Diagnosis ─────────────────────────────────────────────────────
    st.markdown("<p class='section-label'>Disease Diagnosis</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='tv-subtitle'>"
        "This module identifies 16 different tea leaf diseases and pests, including Anthracnose, "
        "Gray Blight, and Red Spider mites. It utilizes an ensemble deep learning approach "
        "combining MobileNetV2, ResNet50, and EfficientNet architectures to achieve high "
        "accuracy even in varied field conditions."
        "</p>",
        unsafe_allow_html=True
    )
    if 'disease_label' in selected:
        dd1, dd2 = st.columns(2)
        dd1.success(f"**Condition:** {selected['disease_label']}")
        dd2.info(f"**Confidence:** {selected['disease_conf']:.2f}%")

        if selected['disease_label'] == 'Healthy':
            st.success("This leaf is healthy. No fertilizer treatment is required.")
        else:
            st.markdown("")
            st.markdown(
                "<p class='section-label'>Fertilizer Recommendation</p>",
                unsafe_allow_html=True
            )
            # ── Fertilizer Recommendation ─────────────────────────────────────────────
            st.markdown(
                "<p class='tv-subtitle'>"
                "This system provides data-driven advice on the best fertilizers and precise dosages "
                "to mitigate crop loss. By processing environmental factors like soil pH, "
                "rainfall, and temperature alongside the detected disease, it utilizes a "
                "Decision Tree model to offer transparent and actionable recovery steps."
                "</p>",
                unsafe_allow_html=True
            )
            st.divider()
            st.caption("Provide field environment details below to generate a treatment plan.")

            with st.form(key=f"fert_form_{selected['id']}"):
                f1, f2 = st.columns(2)

                soil_type     = f1.selectbox("Soil Type",          ["Loamy", "Sandy", "Clay"])
                soil_pH       = f2.number_input("Soil pH",          min_value=0.0, max_value=14.0, value=6.0,   step=0.1)
                rainfall      = f1.number_input("Rainfall (mm/wk)", min_value=0.0, value=45.0)
                humidity      = f2.number_input("Humidity (%)",     min_value=0.0, max_value=100.0, value=80.0)
                temp          = f1.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
                disease_input = f2.text_input("Disease Name",       value=selected['disease_label'])
                severity      = f1.selectbox("Severity",            ["Low", "Medium", "High"])
                model_choice  = f2.selectbox("ML Model",            ["random_forest", "decision_tree", "gradient_boosting"])

                submit_button = st.form_submit_button("Generate Recommendation")

            if submit_button:
                try:
                    fert, dose = fertilizer_recommender.predict(
                        soil_type, soil_pH, rainfall, humidity, temp,
                        disease_input, severity, model_choice
                    )
                    r1, r2 = st.columns(2)
                    r1.success(f"**Recommended Fertilizer:**\n\n{fert}")
                    r2.info(f"**Recommended Dosage:**\n\n{dose}")
                except Exception as e:
                    st.error(f"Recommendation error: {e}")
    else:
        st.caption("Run the CNN classifier to check this leaf for infections or disease.")
        if st.button("Diagnose Disease", type="primary", key=f"diag_btn_{selected['id']}"):
            with st.spinner("Analysing leaf with CNN models..."):
                try:
                    disease_class, disease_conf = disease_classifier.predict(selected['image'])
                    selected['disease_label'] = disease_class
                    selected['disease_conf']  = disease_conf
                    st.rerun()
                except Exception as e:
                    st.error(f"Disease diagnosis error: {e}")
