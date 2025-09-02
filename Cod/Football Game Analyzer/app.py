import os
import streamlit as st
from Processing import load_model
from ReadVideo import ReadVideo

# Streamlit setup
st.set_page_config(page_title="Football Analytics", layout="wide")
st.title("Analiză video a unui meci de fotbal")

# Persistență în session_state
if "model" not in st.session_state:
    st.session_state.model = None

# Căi fixe
WEIGHTS_PATH = r"C:\Users\dariu\OneDrive\Desktop\Licenta\Cod\Football Game Analyzer\Models\best.pt"
VIDEO_PATH = r"C:\Users\dariu\OneDrive\Desktop\Licenta\Videoclipuri\fcsb_modificat.mp4"
OUTPUT_PATH = r"C:\Users\dariu\OneDrive\Desktop\Licenta\Rezultate Videoclipuri\nu_conteaza.mp4"

# Încarcă modelul YOLO (CPU)
if st.button("Încarcă modelul pre-antrenat"):
    with st.spinner("Se încarcă modelul..."):
        # al doilea argument False forțează rularea pe CPU
        st.session_state.model = load_model(WEIGHTS_PATH, False)
    st.success("Model încărcat cu succes! ✅")

# Afișăm informația despre model, dacă e prezent
if st.session_state.model is not None:
    st.info(f"Model activ: `{WEIGHTS_PATH}`")

# Afișăm videoclipul sursă
st.subheader("Videoclip sursă")
st.markdown(f"`{VIDEO_PATH}`")
st.video(VIDEO_PATH)

# Procesează videoclipul
if st.button("Procesează videoclipul"):
    if st.session_state.model is None:
        st.error("❗ Trebuie mai întâi să încarci modelul.")
    else:
        # Creăm folderul de output, dacă nu există
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        with st.spinner("Se procesează videoclipul... (pe CPU, poate dura câteva minute)"):
            # Instanțiem ReadVideo și injectăm modelul și căile
            rv = ReadVideo()
            rv.model = st.session_state.model
            rv.video_path = VIDEO_PATH
            rv.output_path = OUTPUT_PATH

            # Apelăm procesarea fără argumente, atributele sunt deja setate
            rv.process_video_with_tracker()

        st.success(f"✅ Procesare finalizată! Video salvat la:\n`{OUTPUT_PATH}`")
        st.subheader("Rezultat")
        st.video(OUTPUT_PATH)
