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
    from feature_extractor import extract_features, extract_edit_features, classify_intent
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
st.set_page_config(page_title="SiT: FAF Generation & Editing", layout="centered")
st.title("SiT: FAF Generation & Editing")

# Load Resources
valid_genes = load_valid_genes()
engine = load_model()

if engine is None:
    st.stop()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Describe a patient case, and I will **generate** a corresponding FAF image.\n\n"
            "Once an image is generated, you can **edit** it by describing the changes "
            "(e.g. *'Age it down to 25'* or *'Switch to left eye with gene USH2A'*).\n\n"
            "To generate a brand-new image, use words like *'generate'*, *'create'*, or *'show me'*."
        )
    })

# Current image state: stores the inverted noise + params of the active image
if "current_image" not in st.session_state:
    st.session_state.current_image = None  # Will hold: { 'z_noise', 'gene', 'laterality', 'age', 'pil_image' }

# --- RENDER CHAT HISTORY ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if isinstance(message["content"], io.BytesIO):
            st.image(message["content"], width=300)
            st.download_button(
                label="📥",
                data=message["content"],
                file_name=f"SiT_image_{i}.png",
                mime="image/png",
                key=f"download_btn_{i}",
                help="Download Image",
                type="tertiary"
            )
        else:
            st.markdown(message["content"])


# --- HELPER: build a change summary for edits ---
def _build_edit_summary(old_params, new_params):
    """Build a human-readable summary of what changed."""
    changes = []
    if old_params["gene"] != new_params["gene"]:
        changes.append(f"Gene: **{old_params['gene']}** → **{new_params['gene']}**")
    if old_params["laterality"] != new_params["laterality"]:
        lat_names = {"L": "Left", "R": "Right"}
        changes.append(f"Laterality: **{lat_names.get(old_params['laterality'], old_params['laterality'])}** → **{lat_names.get(new_params['laterality'], new_params['laterality'])}**")
    if old_params["age"] != new_params["age"]:
        changes.append(f"Age: **{old_params['age']}** → **{new_params['age']}**")
    
    if not changes:
        return "No changes detected — image unchanged."
    
    summary = "**Edited image.** Changes:\n" + "\n".join(f"- {c}" for c in changes)
    
    # List unchanged params
    unchanged = []
    if old_params["gene"] == new_params["gene"]:
        unchanged.append(f"Gene={new_params['gene']}")
    if old_params["laterality"] == new_params["laterality"]:
        unchanged.append(f"Laterality={new_params['laterality']}")
    if old_params["age"] == new_params["age"]:
        unchanged.append(f"Age={new_params['age']}")
    if unchanged:
        summary += f"\n- Unchanged: {', '.join(unchanged)}"
    
    return summary


# --- CHAT INPUT ---
if prompt := st.chat_input("Describe a patient case or edit the current image..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- Determine intent ---
    has_current_image = st.session_state.current_image is not None
    intent = classify_intent(prompt)
    
    # If no image exists yet, force generate regardless of intent
    if not has_current_image and intent == "edit":
        intent = "generate"

    with st.chat_message("assistant"):

        # ===================== GENERATE =====================
        if intent == "generate":
            # --- Extract generation parameters via LLM ---
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
                st.stop()

            gene = params['gene']
            laterality = params['laterality']
            age = params['age']

            # --- Generate image ---
            with st.spinner(f"Generating FAF image for **{gene}**..."):
                try:
                    image = engine.generate(gene=gene, laterality=laterality, age=age)
                except Exception as e:
                    err = f"**Generation Failed:** {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    st.stop()

            # --- Invert image to get noise representation for future edits ---
            with st.spinner("Preparing image for future edits..."):
                try:
                    z_noise = engine.invert(image, gene, laterality, age)
                except Exception as e:
                    # Inversion failure is non-fatal — user just can't edit
                    z_noise = None
                    st.warning(f"Inversion failed (edits disabled): {e}")

            # --- Store current image state ---
            st.session_state.current_image = {
                "z_noise": z_noise,
                "gene": gene,
                "laterality": laterality,
                "age": age,
                "pil_image": image,
            }

            # --- Display result ---
            summary = f"**Generated image.** Gene=**{gene}** | Age=**{age}** | Laterality=**{laterality}**"
            st.markdown(summary)
            st.session_state.messages.append({"role": "assistant", "content": summary})

            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            st.image(img_buffer, width=300)
            st.session_state.messages.append({"role": "assistant", "content": img_buffer})

        # ===================== EDIT =====================
        elif intent == "edit":
            current = st.session_state.current_image
            old_params = {"gene": current["gene"], "laterality": current["laterality"], "age": current["age"]}

            # Check inversion is available
            if current["z_noise"] is None:
                err = "**Cannot edit:** The current image was not successfully inverted. Please generate a new image."
                st.markdown(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.stop()

            # --- Extract edit parameters via LLM ---
            with st.spinner("Understanding your edit request..."):
                edit_params = extract_edit_features(prompt, valid_genes, old_params)

            if "error" in edit_params:
                err = f"**LLM Error:** {edit_params['error']}"
                st.markdown(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.stop()

            # Merge: keep current values for anything not changed
            new_gene = edit_params.get("gene") or current["gene"]
            new_laterality = edit_params.get("laterality") or current["laterality"]
            new_age = edit_params.get("age") if edit_params.get("age") is not None else current["age"]

            # Validate gene
            if new_gene not in valid_genes:
                err = f"**Unsupported Gene '{new_gene}'.** Valid genes: `{', '.join(valid_genes[:5])}...`"
                st.markdown(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.stop()

            new_params = {"gene": new_gene, "laterality": new_laterality, "age": new_age}

            # --- Run edit: forward ODE with new conditions ---
            with st.spinner(f"Editing image..."):
                try:
                    edited_image = engine.edit(
                        z_noise=current["z_noise"],
                        target_gene=new_gene,
                        target_laterality=new_laterality,
                        target_age=new_age,
                    )
                except Exception as e:
                    err = f"**Edit Failed:** {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
                    st.stop()

            # --- Re-invert the edited image for further edits ---
            with st.spinner("Preparing edited image for future edits..."):
                try:
                    z_noise_new = engine.invert(edited_image, new_gene, new_laterality, new_age)
                except Exception as e:
                    z_noise_new = current["z_noise"]  # fall back to old noise
                    st.warning(f"Re-inversion failed, future edits may be less accurate: {e}")

            # --- Update current image state ---
            st.session_state.current_image = {
                "z_noise": z_noise_new,
                "gene": new_gene,
                "laterality": new_laterality,
                "age": new_age,
                "pil_image": edited_image,
            }

            # --- Display result ---
            summary = _build_edit_summary(old_params, new_params)
            st.markdown(summary)
            st.session_state.messages.append({"role": "assistant", "content": summary})

            img_buffer = io.BytesIO()
            edited_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            st.image(img_buffer, width=300)
            st.session_state.messages.append({"role": "assistant", "content": img_buffer})