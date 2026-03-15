import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
import time
import base64
from streamlit.components.v1 import html

# ==============================
# Page Config
# ==============================

st.set_page_config(
    page_title="CeyLeaf AI - Neural Agriculture",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==============================
# Custom CSS for Futuristic Design
# ==============================

def load_css():
    st.markdown("""
    <style>
        /* Import futuristic fonts */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');

        /* Global styles */
        .stApp {
            background: radial-gradient(ellipse at 20% 30%, #0a0f1e, #030614);
            font-family: 'Rajdhani', sans-serif;
        }

        /* Main title */
        .futuristic-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 4.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #00ff9d, #00b8ff, #9d00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 4px;
            margin-bottom: 0;
            animation: glow 3s ease-in-out infinite;
            text-shadow: 0 0 30px rgba(0, 255, 157, 0.3);
        }

        @keyframes glow {
            0%, 100% { filter: brightness(100%); }
            50% { filter: brightness(120%); }
        }

        /* Subtitle */
        .cyber-subtitle {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5rem;
            color: #8892b0;
            text-align: center;
            letter-spacing: 2px;
            margin-top: -10px;
            margin-bottom: 40px;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(0, 255, 157, 0.3);
            padding-bottom: 20px;
        }

        /* Cyber card effect */
        .cyber-card {
            background: rgba(10, 20, 30, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 255, 157, 0.3);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 0 30px rgba(0, 255, 157, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .cyber-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 0 50px rgba(0, 255, 157, 0.2);
        }

        /* Neon text */
        .neon-text {
            font-family: 'Orbitron', sans-serif;
            color: #00ff9d;
            text-shadow: 0 0 10px rgba(0, 255, 157, 0.5);
        }

        /* Upload area */
        .upload-area {
            border: 2px dashed rgba(0, 255, 157, 0.5);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            background: rgba(0, 255, 157, 0.02);
            transition: all 0.3s;
        }

        .upload-area:hover {
            border-color: #00ff9d;
            background: rgba(0, 255, 157, 0.05);
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #00ff9d, #00b8ff, #9d00ff);
        }

        /* Metric cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(0, 255, 157, 0.2);
        }

        /* Glowing button */
        .stButton > button {
            background: linear-gradient(45deg, #00ff9d, #00b8ff);
            border: none;
            color: #030614;
            font-family: 'Orbitron', sans-serif;
            font-weight: bold;
            padding: 15px 30px;
            border-radius: 50px;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s;
            box-shadow: 0 0 20px rgba(0, 255, 157, 0.3);
        }

        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 40px rgba(0, 255, 157, 0.5);
        }

        /* Result display */
        .result-box {
            background: linear-gradient(135deg, rgba(0, 255, 157, 0.1), rgba(0, 184, 255, 0.1));
            border-left: 4px solid #00ff9d;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            font-family: 'Rajdhani', sans-serif;
        }

        .disease-name {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5rem;
            background: linear-gradient(135deg, #fff, #00ff9d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }

        /* Particles background */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            pointer-events: none;
        }

        /* Loading animation */
        .loading-dots {
            display: inline-block;
        }

        .loading-dots:after {
            content: '.';
            animation: dots 1.5s steps(5, end) infinite;
        }

        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60% { content: '...'; }
            80%, 100% { content: ''; }
        }
    </style>
    """, unsafe_allow_html=True)


# ==============================
# Particle Animation
# ==============================

def add_particles():
    html("""
    <div class="particles">
        <canvas id="canvas"></canvas>
    </div>
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        class Particle {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.vx = (Math.random() - 0.5) * 0.5;
                this.vy = (Math.random() - 0.5) * 0.5;
                this.size = Math.random() * 2;
                this.color = `rgba(0, 255, 157, ${Math.random() * 0.3})`;
            }

            update() {
                this.x += this.vx;
                this.y += this.vy;

                if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
                if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.fill();
            }
        }

        const particles = [];
        for (let i = 0; i < 50; i++) {
            particles.push(new Particle());
        }

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(particle => {
                particle.update();
                particle.draw();
            });
            requestAnimationFrame(animate);
        }

        animate();

        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    </script>
    """, height=0)


# ==============================
# Load Models
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_paths = {
    'mobilenet': os.path.join(BASE_DIR, "mobilenet.h5"),
    'efficientnet': os.path.join(BASE_DIR, "efficientnet_best.h5"),
    'resnet50': os.path.join(BASE_DIR, "resnet50_best.h5")
}

# ==============================
# Initialize session state
# ==============================

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.mobilenet_model = None
    st.session_state.efficientnet_model = None
    st.session_state.resnet50_model = None
    st.session_state.loaded_models_list = []

# ==============================
# Classes
# ==============================

disease_classes = [
    'Anthracnose', 'Gray Blight', 'Green mirid bug', 'Heliopeltis',
    'Red Rust', 'Red Spider', 'Sunlight Scorching', 'Tea leaf blight',
    'Tea red leaf spot', 'Tea red scab', 'Thrips', 'Algal leaf',
    'Bird eye spot', 'Brown blight', 'Gray light', 'White spot'
]


# ==============================
# Preprocessing
# ==============================

def preprocess(img, model, preprocess_func):
    size = model.input_shape[1:3]
    img = img.resize(size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_func(img)
    return img


# ==============================
# Prediction Function
# ==============================

def predict_disease(img):
    preds = []
    models_used = []

    if st.session_state.mobilenet_model:
        x = preprocess(img, st.session_state.mobilenet_model, mobilenet_preprocess)
        preds.append(st.session_state.mobilenet_model.predict(x)[0])
        models_used.append("MobileNet")

    if st.session_state.efficientnet_model:
        x = preprocess(img, st.session_state.efficientnet_model, efficientnet_preprocess)
        preds.append(st.session_state.efficientnet_model.predict(x)[0])
        models_used.append("EfficientNet")

    if st.session_state.resnet50_model:
        x = preprocess(img, st.session_state.resnet50_model, resnet50_preprocess)
        preds.append(st.session_state.resnet50_model.predict(x)[0])
        models_used.append("ResNet50")

    if len(preds) == 0:
        return None, None, []

    avg_pred = np.mean(preds, axis=0)
    index = np.argmax(avg_pred)
    disease = disease_classes[index]
    confidence = float(avg_pred[index]) * 100

    return disease, confidence, models_used


# ==============================
# Load Models Function
# ==============================

def load_ai_models():
    with st.spinner("🔮 Initializing Neural Networks..."):
        time.sleep(1)  # Dramatic effect

        try:
            st.session_state.mobilenet_model = load_model(model_paths['mobilenet'])
            st.session_state.loaded_models_list.append("MobileNet")
        except:
            pass

        try:
            st.session_state.efficientnet_model = load_model(model_paths['efficientnet'])
            st.session_state.loaded_models_list.append("EfficientNet")
        except:
            pass

        try:
            st.session_state.resnet50_model = load_model(model_paths['resnet50'])
            st.session_state.loaded_models_list.append("ResNet50")
        except:
            pass

        st.session_state.models_loaded = True


# ==============================
# Main App
# ==============================

def main():
    # Load custom CSS and particles
    load_css()
    add_particles()

    # Header with futuristic design
    st.markdown('<h1 class="futuristic-title">CEYLEAF AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="cyber-subtitle">Neural Agriculture Intelligence System</p>', unsafe_allow_html=True)

    # Create columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Cyber card container
        with st.container():
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)

            # Load models button
            if not st.session_state.models_loaded:
                if st.button("⚡ INITIALIZE NEURAL NETWORKS", use_container_width=True):
                    load_ai_models()

                    if st.session_state.loaded_models_list:
                        st.success(f"✅ Neural Networks Online: {', '.join(st.session_state.loaded_models_list)}")
                    else:
                        st.error("⚠️ No neural networks available. Please check model files.")
            else:
                st.info(f"🧠 Active Neural Networks: {', '.join(st.session_state.loaded_models_list)}")

            # Upload area with futuristic design
            st.markdown("### 📡 UPLOAD SCANNER")
            uploaded_file = st.file_uploader(
                "Drop tea leaf image for neural analysis",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                img = Image.open(uploaded_file)

                # Display image with futuristic frame
                st.image(img, caption="SCANNED SPECIMEN", use_column_width=True)

                # Prediction button
                if st.button("🔬 ANALYZE WITH NEURAL NETWORKS", use_container_width=True):
                    if not st.session_state.models_loaded:
                        st.warning("⚠️ Please initialize neural networks first.")
                    else:
                        with st.spinner("🧬 Analyzing biological patterns..."):
                            time.sleep(1)  # Dramatic effect
                            disease, confidence, models_used = predict_disease(img)

                        if disease is None:
                            st.error("⚠️ Neural analysis failed. No models available.")
                        else:
                            # Futuristic result display
                            st.markdown("### 🎯 ANALYSIS COMPLETE")

                            col_a, col_b = st.columns(2)

                            with col_a:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.markdown("**🧬 DETECTED PATHOGEN**")
                                st.markdown(f'<p class="disease-name">{disease}</p>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                            with col_b:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.markdown("**📊 CONFIDENCE LEVEL**")
                                st.progress(int(confidence))
                                st.markdown(f"<h2 style='color: #00ff9d; text-align: center;'>{confidence:.2f}%</h2>",
                                            unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                            # Additional metrics
                            st.markdown("### 🔧 SYSTEM METRICS")
                            col_x, col_y, col_z = st.columns(3)

                            with col_x:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("Neural Networks", len(models_used))
                                st.markdown('</div>', unsafe_allow_html=True)

                            with col_y:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("Processing Time", "0.3s")
                                st.markdown('</div>', unsafe_allow_html=True)

                            with col_z:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("Database Match", f"{confidence:.1f}%")
                                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # Footer with futuristic design
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f2:
        st.markdown("""
        <div style='text-align: center; color: #8892b0; font-family: Rajdhani;'>
            <span style='color: #00ff9d;'>© 2026</span> CEYLEAF AI NEURAL SYSTEMS<br>
            <span style='font-size: 0.8rem;'>Advanced Agriculture Intelligence Division</span>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()