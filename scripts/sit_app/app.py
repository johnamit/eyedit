import streamlit as st
import json
import os
import sys
import io

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    from feature_extractor import extract_features
    from inference import SiTGenerator
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# --- CONFIGURATION ---
MAPPING_PATH = os.path.join(project_root, "data", "class_mapping.json")
CKPT_PATH = os.path.join(project_root, "SiT", "weights", "results-gene0error-fix", "000-SiT-XL-2-Linear-velocity-None", "checkpoints", "0110000.pt")

# --- RESOURCE LOADING ---
@st.cache_data
def load_valid_genes():
    if os.path.exists(MAPPING_PATH):
        try:
            with open(MAPPING_PATH, 'r') as f:
                return list(json.load(f).keys())
        except Exception:
            return []
    return ["ABCA4", "USH2A", "RPGR"] # Fallback

@st.cache_resource(show_spinner="Initializing SiT Clinical Engine...")
def load_model():
    if not os.path.exists(CKPT_PATH):
        st.error(f"Checkpoint not found at: {CKPT_PATH}")
        return None
    try:
        return SiTGenerator(CKPT_PATH, MAPPING_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# --- UI SETUP ---
st.set_page_config(page_title="SiT: FAF Generation", layout="centered")
st.title("SiT: FAF Generation")

# Load Resources
valid_genes = load_valid_genes()
engine = load_model()

if engine is None:
    st.stop()

# --- CHAT HISTORY ---
# Initialize chat history so messages persist
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an initial greeting from the assistant
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Describe a patient case, and I will generate a corresponding FAF image for you. For example: *'Show me a 60 year old male with Stargardt disease.'*"
    })

# 1. Use 'enumerate' to get an index 'i' for the unique key
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if isinstance(message["content"], io.BytesIO):
            st.image(message["content"], width=300)
            
            # 2. Add 'key', 'type="tertiary"', and a simple emoji label
            st.download_button(
                label="📥",  # Clean icon-only look
                data=message["content"],
                file_name=f"SiT_generated_image_{i}.png", # Unique filename
                mime="image/png",
                key=f"download_btn_{i}", # UNIQUE KEY (Fixes your error)
                help="Download Image",
                type="tertiary" # Removes background & border
            )
        else:
            st.markdown(message["content"])

# --- CHAT INPUT (Bottom of Page) ---
if prompt := st.chat_input("Describe a patient case..."):
    # 1. Display User Message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        # Step 2a: Interpretation (LLM)
        with st.spinner("Analyzing clinical request..."):
            params = extract_features(prompt, valid_genes)

        # Validation
        error_msg = None
        if "error" in params:
             error_msg = f"**LLM Error:** {params['error']}"
        elif params.get('gene') is None:
             error_msg = f"**Unsupported Gene.** I can only generate images for: `{', '.join(valid_genes[:5])}...`"

        if error_msg:
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.stop() # Stop processing for this turn

        # Step 2b: Generation (SiT Engine)
        with st.spinner(f"Generating FAF image for **{params['gene']}**..."):
            try:
                # Generate Image
                image = engine.generate(
                    gene=params['gene'],
                    laterality=params['laterality'],
                    age=params['age']
                )
                
                # Prepare for display & history
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                img_buffer.seek(0) # Rewind buffer pointer
                
                # Display Image
                st.image(img_buffer, width=300, caption=f"{params['gene']} | Age {params['age']} | {params['laterality']} Eye")
                
                # Add Image to history so it stays on screen
                st.session_state.messages.append({"role": "assistant", "content": img_buffer})

            except Exception as e:
                err = f"**Generation Failed:** {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})