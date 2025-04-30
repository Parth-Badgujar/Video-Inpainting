import tempfile
import os
import streamlit as st
# Set the page layout to wide\ nst.set_page_config(layout="wide")

# Add padding between columns
st.set_page_config(layout = "wide")
st.title("Towards Improving Video Inpainting")

# --- Session State Initialization ---
for key in ['original_video', 'processed_video', 'final_video']:
    if key not in st.session_state:
        st.session_state[key] = None



# --- Section 1: Initial Processing ---
st.subheader("Initial editing using an image model")
col1, xx, col2, yy = st.columns([0.5, 0.1, 0.3, 0.1])
with col1:
    # Upload & Initial Hyperparameters
    uploaded = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
    if uploaded:
        tmp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp_orig.write(uploaded.read())
        tmp_orig.flush()
        st.session_state['original_video'] = tmp_orig.name

    choice = st.selectbox("Select processing type", ['Option A', 'Option B', 'Option C'])
    prompt1 = st.text_input("Prompt 1")
    prompt2 = st.text_input("Prompt 2")
    slider1 = st.slider("Parameter 1", 0.0, 1.0, 0.5)
    slider2 = st.slider("Parameter 2", 0, 100, 50)
    if st.button("Process Video") and st.session_state['original_video']:
        # Placeholder: your processing code here
        tmp_proc = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        with open(st.session_state['original_video'], 'rb') as src, open(tmp_proc.name, 'wb') as dst:
            dst.write(src.read())
        st.session_state['processed_video'] = tmp_proc.name

st.divider()

with col2:
    # Display Original & Processed Videos
    if st.session_state['original_video']:
        st.subheader("Original Video")
        st.video(st.session_state['original_video'])
    if st.session_state['processed_video']:
        st.subheader("Processed Video")
        st.video(st.session_state['processed_video'])

# --- Section 2: Additional Editing ---
st.subheader("Mixed Inversion + Video ")
col3, aa, col4, bb = st.columns([0.5, 0.1, 0.3, 0.1])
with col3:
    # Additional Hyperparameters
    slider3 = st.slider("Parameter 3", 0.0, 1.0, 0.5)
    slider4 = st.slider("Parameter 4", 0, 100, 25)
    if st.button("Apply Additional Edits") and st.session_state['processed_video']:
        # Placeholder: your editing code here
        tmp_final = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        with open(st.session_state['processed_video'], 'rb') as src, open(tmp_final.name, 'wb') as dst:
            dst.write(src.read())
        st.session_state['final_video'] = tmp_final.name

with col4:
    if st.session_state['final_video']:
        st.subheader("Final Edited Video")
        st.video(st.session_state['final_video'])

