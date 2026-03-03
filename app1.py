# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import requests
import pandas as pd
from groq import Groq
from langfuse import Langfuse
import os
import time
import json
import re
import tempfile
import speech_recognition as sr
from datetime import datetime
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder
import io
from pydub import AudioSegment


# ============================================================================
# SYSTEM PROMPT - Expert AI Pharmacist
# ============================================================================
SYSTEM_PROMPT = """
### ROLE: SENIOR MEDICAL EXPERT & COMPASSIONATE MASTER DOCTOR

### MISSION:
You are an expert AI Pharmacist. Your primary goal is patient well-being. You combine medical expertise with genuine empathy. Language: Professional Hinglish (Hindi-English mix). Tone: Authoritative yet Caring.

---

### LOGIC FLOW (Always follow this order):

#### 1. NORMAL GREETING LOGIC:
If user says ONLY: 'Hi', 'Hello', 'Hello ji', 'Namaste', 'How are you', 'Kaise ho', or any simple greeting (NO symptoms mentioned):
- Response: "Hello! I am your Senior AI Pharmacist. How can I help you today?"
- Keep it SHORT and PROFESSIONAL. No extra empathy needed.

#### 2. CARING & MEDICAL LOGIC:
If user mentions ANY pain, symptom, or disease (like: kamar dard, fever, headache, dard, bukhar, cough, cold, infection, etc.):

**STEP 1 - DEEP EMPATHY (Very Important):**
- Show genuine care and warmth

**STEP 2 - EXPERT MEDICAL ADVICE:**
- Recommend the world's best generic medicine for their condition
- Example: Paracetamol for fever, Ibuprofen for pain, Amoxicillin for bacterial infection
- Give 2-3 practical home remedies (e.g., "Keep hydrated", "Take rest", "Apply warm compress")

**STEP 3 - INVENTORY CHECK (CRITICAL):**
- Check the {inventory_data} provided
- If medicine IN STOCK: "Great news! We have this medicine in stock. Would you like me to help you order it?"
- If medicine OUT OF STOCK: "Unfortunately, this medicine is not available in our stock. But don't worry - please get it from any trusted pharmacy nearby as it is the right treatment for your condition."

**STEP 4 - SAFETY DISCLAIMER (Always add):**
- "Please consult a physical doctor for formal diagnosis. This is advisory only."
- If symptoms are serious (chest pain, severe bleeding, breathing difficulty): "URGENT: Please call emergency services immediately!"

---

### IMPORTANT RULES:
1. NEVER diagnose definitively - always defer to medical professionals
2. Be 100% honest about stock availability - only use medicines from {inventory_data}
3. If medicine is out of stock, suggest alternatives ONLY if you're medically certain
4. Never hallucinate about medicine availability
5. Always prioritize patient safety over making a sale
6. Use simple language that anyone can understand
"""


# ============================================================================
# CONFIGURATION & OBSERVABILITY
# ============================================================================

# ========================================
# GOOGLE SHEET CSV EXPORT URL - LIVE INVENTORY DATA
# ========================================
GOOGLE_SHEET_CSV_URL = ""

# ========================================
# PRODUCTION WEBHOOK URL - CHANGE THIS TO YOUR ACTUAL n8n WEBHOOK
# ========================================
N8N_WEBHOOK_URL = ""


# ============================================================================
# HELPER FUNCTIONS - LOCAL DATA LOADING (NO BACKEND DEPENDENCY)
# ============================================================================

@st.cache_data(ttl=10)
def load_inventory_data():
    """Load inventory data from LIVE Google Sheet.
    
    Returns:
        DataFrame with inventory data
    """
    try:
        # Load from Google Sheet CSV export - no backend dependency
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        # Fallback: Return empty dataframe if sheet not accessible
        st.error(f"Failed to load inventory from Google Sheet: {e}")
        return pd.DataFrame()


def get_available_medicines():
    """Get list of available medicine names from local CSV."""
    try:
        df = load_inventory_data()
        if not df.empty and 'medicine_name' in df.columns:
            return df['medicine_name'].dropna().astype(str).tolist()
        return []
    except Exception as e:
        return []


# FFmpeg path configuration for pydub
import os
import shutil

def find_ffmpeg_system():
    """Try to find ffmpeg in system PATH or common locations."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    
    if ffmpeg_path and ffprobe_path:
        return ffmpeg_path, ffprobe_path
    
    common_paths = [
        r'C:\ffmpeg\bin',
        r'C:\ffmpeg\ffmpeg\bin',
        r'C:\Program Files\ffmpeg\bin',
        r'C:\Program Files (x86)\ffmpeg\bin',
    ]
    
    for base_path in common_paths:
        ffmpeg_check = os.path.join(base_path, 'ffmpeg.exe')
        ffprobe_check = os.path.join(base_path, 'ffprobe.exe')
        
        if os.path.exists(ffmpeg_check) and os.path.exists(ffprobe_check):
            return ffmpeg_check, ffprobe_check
    
    return None, None

system_ffmpeg, system_ffprobe = find_ffmpeg_system()

if system_ffmpeg:
    FFMPEG_PATH = system_ffmpeg
    FFPROBE_PATH = system_ffprobe
else:
    FFMPEG_PATH = r'C:\ffmpeg\ffmpeg\bin\ffmpeg.exe'
    FFPROBE_PATH = r'C:\ffmpeg\ffmpeg\bin\ffprobe.exe'

def check_ffmpeg_tools():
    """Check if ffmpeg and ffprobe executables exist."""
    missing_tools = []
    
    if not os.path.exists(FFMPEG_PATH):
        missing_tools.append(f"FFmpeg not found at: {FFMPEG_PATH}")
    
    if not os.path.exists(FFPROBE_PATH):
        missing_tools.append(f"FFprobe not found at: {FFPROBE_PATH}")
    
    return missing_tools

def init_ffmpeg():
    """Initialize FFmpeg configuration with proper paths and validation."""
    missing = check_ffmpeg_tools()
    
    if missing:
        return False, missing
    
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffprobe = FFPROBE_PATH
    return True, []

ffmpeg_initialized, ffmpeg_errors = init_ffmpeg()

if "ffmpeg_initialized" not in st.session_state:
    st.session_state.ffmpeg_initialized = ffmpeg_initialized
    st.session_state.ffmpeg_errors = ffmpeg_errors

# Langfuse setup for observability
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_HOST"] = ""

langfuse = Langfuse()

# Groq API Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)


# ============================================================================
# ORDER MANAGEMENT FUNCTIONS
# ============================================================================

def approve_order(idx):
    """Approve a pending order and send notification to n8n.
    
    CRITICAL: This function:
    1. Takes order from session_state.pending_orders
    2. Sends webhook to n8n Production Webhook URL
    3. On success, pops the order from the list
    4. Calls st.rerun() to update UI
    
    Args:
        idx: The index of the order in pending_orders list
    """
    if 0 <= idx < len(st.session_state.pending_orders):
        order = st.session_state.pending_orders[idx]
        
        # Send webhook to n8n
        try:
            webhook_url = N8N_WEBHOOK_URL  # Use production webhook URL
            data = {
                "medicine_name": order.get("medicine_name", "Unknown"),
                "customer_name": order.get("customer_name", "User"),
                "status": "Confirmed",
                "action": "order_approved"
            }
            
            response = requests.post(webhook_url, json=data, timeout=10)
            
            if response.status_code == 200:
                # Success: Remove order from pending list
                st.session_state.pending_orders.pop(idx)
                st.success(f"✅ Order for {order.get('medicine_name')} approved and customer notified!")
                st.rerun()
            else:
                st.warning(f"Webhook returned status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to send approval notification: {e}")
        
        return True
    return False


def create_pending_order(medicine_name, quantity=1, customer_name="User"):
    """Create a pending order and add to session state.
    
    CRITICAL: This function creates the order dict with correct keys:
    - medicine_name
    - quantity_bought
    - customer_name
    - status
    
    Args:
        medicine_name: Name of the medicine
        quantity: Quantity to order (default: 1)
        customer_name: Name of the customer (default: "User")
    """
    order = {
        "medicine_name": medicine_name,
        "quantity_bought": quantity,
        "customer_name": customer_name,
        "status": "Pending",
        "order_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Check if order already exists (avoid duplicates)
    existing_order = any(
        o.get('medicine_name', '').lower() == medicine_name.lower() 
        and o.get('status', '').lower() == 'pending'
        for o in st.session_state.pending_orders
    )
    
    if not existing_order:
        st.session_state.pending_orders.append(order)
        # CRITICAL: Immediately rerun so Admin Panel updates instantly
        st.rerun()
    
    return order


# ============================================================================
# HELPER FUNCTIONS (UNCHANGED)
# ============================================================================

def extract_medicine_name(prompt, available_medicines):
    """Extract medicine name from user prompt by matching against available medicines."""
    prompt_lower = prompt.lower()
    
    for medicine in available_medicines:
        if medicine.lower() in prompt_lower:
            return medicine
    
    for medicine in available_medicines:
        med_words = medicine.lower().split()
        for word in med_words:
            if len(word) > 3 and word in prompt_lower:
                return medicine
    
    return None


def requires_prescription(medicine_name, df_meds):
    """Check if a medicine requires prescription."""
    med_filter = df_meds['medicine_name'].str.lower() == medicine_name.lower()
    if not df_meds[med_filter].empty:
        prescription_req = str(df_meds.loc[med_filter, 'prescription_required'].values[0]).lower()
        return prescription_req in ['yes', 'true', '1']
    return False


def extract_medicine_from_image(image_bytes):
    """Extract medicine name from prescription image using Google Gemini 2.5 Flash."""
    try:
        GEMINI_API_KEY = ""
        genai.configure(api_key=GEMINI_API_KEY)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
            f.write(image_bytes)
            temp_path = f.name
        
        uploaded_file = genai.upload_file(path=temp_path)
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        
        response = model.generate_content([
            "Extract ONLY the generic/chemical medicine name from this prescription image. Examples: Paracetamol, Amoxicillin, Ibuprofen, Azithromycin, Cough Syrup. Do NOT include any extra text, periods, or explanations. Also identify if there's a quantity mentioned. Return the response as a JSON with keys: 'medicine_name' and 'quantity'. If no quantity is specified, default to 1. If you cannot determine the medicine name, return medicine_name as 'UNKNOWN'.",
            uploaded_file
        ])
        
        os.remove(temp_path)
        response_text = response.text.strip()
        
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        
        if json_match:
            try:
                result = json.loads(json_match.group())
                medicine_name = result.get('medicine_name', '').strip().rstrip('.').strip()
                quantity = int(result.get('quantity', 1))
                
                if medicine_name and medicine_name.upper() != 'UNKNOWN':
                    return {"valid": True, "medicine_name": medicine_name, "quantity": quantity}
                else:
                    return {"valid": False, "reason": "Could not identify medicine name from the prescription image. Please try again with a clearer image."}
            except json.JSONDecodeError:
                pass
        
        medicine_name = response_text.strip().rstrip('.').strip()
        if medicine_name and medicine_name.upper() != 'UNKNOWN':
            return {"valid": True, "medicine_name": medicine_name, "quantity": 1}
        
        return {"valid": False, "reason": "Could not extract medicine name from the prescription image. Please try with a clearer image."}
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {"valid": False, "reason": f"Error calling Gemini API: {str(e)}"}


def extract_prescription_data(image_bytes):
    """Extract medicine name, quantity, and check for formal document markers using Gemini."""
    try:
        result = extract_medicine_from_image(image_bytes)
        
        if result.get('valid', False):
            return result
        else:
            return {"valid": False, "reason": result.get('reason', 'Failed to extract medicine from image')}

    except Exception as e:
        return {"valid": False, "reason": str(e)}


def verify_prescription(extracted_data, df_meds):
    """Verify prescription authenticity and check medicine availability using fuzzy matching."""
    if not extracted_data:
        return False, "Failed to extract data from image"
    
    if not extracted_data.get('valid', False):
        return False, extracted_data.get('reason', 'Invalid Prescription')
    
    extracted_medicine = extracted_data.get('medicine_name', '').strip().lower()
    extracted_quantity = extracted_data.get('quantity', 1)
    
    matched_medicine = None
    
    for idx, row in df_meds.iterrows():
        inventory_name = str(row['medicine_name']).strip().lower()
        if inventory_name == extracted_medicine:
            matched_medicine = row['medicine_name']
            break
    
    if not matched_medicine:
        for idx, row in df_meds.iterrows():
            inventory_name = str(row['medicine_name']).strip().lower()
            if extracted_medicine in inventory_name or inventory_name in extracted_medicine:
                matched_medicine = row['medicine_name']
                break
    
    if not matched_medicine:
        return False, "Medicine not found in our inventory."
    
    med_filter = df_meds['medicine_name'] == matched_medicine
    current_stock = df_meds.loc[med_filter, 'stock_level'].values[0]
    
    if current_stock <= 0:
        return False, f"Medicine is out of stock."
    
    if current_stock < extracted_quantity:
        return False, f"Insufficient stock. Available: {current_stock}"
    
    return True, {"medicine": matched_medicine, "quantity": extracted_quantity}


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_med" not in st.session_state:
    st.session_state.pending_med = None

if "requested_medicine" not in st.session_state:
    st.session_state.requested_medicine = None

if "prescription_verified" not in st.session_state:
    st.session_state.prescription_verified = False

if "verification_data" not in st.session_state:
    st.session_state.verification_data = None

if "voice_text" not in st.session_state:
    st.session_state.voice_text = None

if "voice_prompt_added" not in st.session_state:
    st.session_state.voice_prompt_added = False

if "recorder_counter" not in st.session_state:
    st.session_state.recorder_counter = 0

if "voice_processed" not in st.session_state:
    st.session_state.voice_processed = False

# Pending Orders Workflow - stores orders waiting for admin approval
if "pending_orders" not in st.session_state:
    st.session_state.pending_orders = []


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Expert AI Pharmacist", page_icon="💊", layout="wide")

st.title("🏥 Pharm-Agent: Autonomous Pharmacy Ecosystem")
st.info("System is behaving as an Expert Pharmacist with Safety Guardrails.")

# Check FFmpeg initialization
if not st.session_state.get('ffmpeg_initialized', True):
    st.error("🔴 **FFmpeg Configuration Error**")
    for error in st.session_state.get('ffmpeg_errors', []):
        st.error(f"• {error}")
    st.warning("Please ensure FFmpeg is installed at the correct path and restart the application.")


# ============================================================================
# SIDEBAR - Voice Assistant, Prescription Upload & Admin Controls
# ============================================================================
with st.sidebar:
    st.header("🎙️ Voice Assistant")
    st.caption("🎯 Tips: Speak clearly. Mention the medicine name clearly.")

    if 'recorder_counter' not in st.session_state:
        st.session_state.recorder_counter = 0
    
    recorder_key = f"voice_recorder_{st.session_state.recorder_counter}"
    
    audio_info = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop",
        key=recorder_key
    )

    if audio_info:
        try:
            with st.spinner("🎧 Processing your voice..."):
                audio_bytes = audio_info['bytes']
                
                if not audio_bytes or len(audio_bytes) == 0:
                    st.error("❌ No audio captured.")
                else:
                    recognizer = sr.Recognizer()
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
                    audio_segment = audio_segment.normalize()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                        temp_wav_path = temp_wav.name
                        audio_segment.export(temp_wav_path, format="wav")
                    
                    with sr.AudioFile(temp_wav_path) as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio_data = recognizer.record(source)
                    
                    if os.path.exists(temp_wav_path):
                        os.remove(temp_wav_path)

                    try:
                        text = recognizer.recognize_google(audio_data, language='hi-IN')
                    except:
                        try:
                            text = recognizer.recognize_google(audio_data, language='en-IN')
                        except:
                            text = None

                    if text:
                        st.success(f"✅ Captured: \"{text}\"")
                        
                        try:
                            df_meds = load_inventory_data()
                            col_name = 'medicine_name' 
                            if col_name in df_meds.columns:
                                med_list = df_meds[col_name].dropna().astype(str).tolist()
                                detected_medicine = extract_medicine_name(text, med_list)
                            else:
                                st.error(f"❌ Column '{col_name}' not found in inventory!")
                                detected_medicine = None
                                
                        except Exception as e:
                            print(f"Inventory Error: {e}")
                            detected_medicine = None

                        # Order detection in voice - use correct keys
                        if detected_medicine:
                            with st.spinner("Processing order..."):
                                # CRITICAL: Create order with correct keys and rerun immediately
                                create_pending_order(detected_medicine, quantity=1, customer_name="User")
                                st.info(f"📨 Order captured for {detected_medicine}!")
                        else:
                            st.info("💬 No medicine detected. Processing as chat message...")
                            st.session_state.voice_prompt = text
                            st.session_state.voice_prompt_added = False
                            st.rerun()
                    else:
                        st.error("❌ Samajh nahi aaya, dobara boliye.")

        except Exception as audio_error:
            st.error(f"❌ Audio processing error: {audio_error}")
        
        st.session_state.recorder_counter += 1
        st.rerun()

    if not audio_info:
        st.info("👆 Click 'Start Recording' to speak")

    st.markdown("---")
    
    # --- Prescription Upload & Verification ---
    st.header("📷 Prescription Upload & Verification")
    
    try:
        df_meds = load_inventory_data()
        
        uploaded_file = st.file_uploader(
            "Upload prescription image for verification",
            type=["jpg", "jpeg", "png"],
            key="sidebar_prescription_upload"
        )
        
        if uploaded_file is not None:
            with st.spinner("Verifying prescription..."):
                extracted_data = extract_prescription_data(uploaded_file.getvalue())
                
                if extracted_data:
                    verification_result, verification_data = verify_prescription(extracted_data, df_meds)
                    
                    if verification_result:
                        st.success("✅ Prescription Verified Successfully!")
                        st.session_state.prescription_verified = True
                        st.session_state.verification_data = verification_data
                        
                        st.markdown("### Verification Details:")
                        st.markdown(f"- **Medicine:** {verification_data['medicine']}")
                        st.markdown(f"- **Quantity:** {verification_data['quantity']}")
                        
                        if st.button("Place Order", key="sidebar_place_order"):
                            medicine_name = verification_data.get('medicine')
                            if not medicine_name:
                                st.error("❌ Medicine name is empty. Cannot place order.")
                            else:
                                # CRITICAL: Use create_pending_order with correct keys
                                create_pending_order(medicine_name, quantity=verification_data['quantity'], customer_name="User")
                                st.success("Order submitted for admin approval! ✅")
                    else:
                        st.error(f"❌ {verification_data}")
                        st.session_state.prescription_verified = False
                        st.session_state.verification_data = None
                else:
                    st.error("Failed to extract data from prescription image")
                    st.session_state.prescription_verified = False
        else:
            st.info("Upload a prescription image to verify")
    
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # --- User Medicine Search ---
    st.header("🔍 Search Medicines")
    
    try:
        if 'df_meds' not in locals():
            df_meds = load_inventory_data()
        
        user_search_term = st.text_input("🔎 Search for medicines:", placeholder="Type medicine name to search...")
        
        if user_search_term:
            user_df_filtered = df_meds[df_meds['medicine_name'].str.lower().str.contains(user_search_term.lower(), na=False)]
        else:
            user_df_filtered = df_meds
        
        def highlight_low_stock(row):
            if row['stock_level'] < 10:
                return ['background-color: #ffcccc; color: #990000'] * len(row)
            return [''] * len(row)
        
        user_styled_df = user_df_filtered.style.apply(highlight_low_stock, axis=1)
        st.dataframe(user_styled_df, hide_index=True, use_container_width=True)
        
        low_stock_items = df_meds[df_meds['stock_level'] < 10]
        if not low_stock_items.empty:
            st.warning(f"⚠️ Low Stock Alert: {len(low_stock_items)} medicine(s) have stock level below 10!")
    
    except Exception as e:
        st.error(f"Error loading inventory: {e}")
    
    st.markdown("---")
    
    # --- Admin Control Panel ---
    st.header("🏪 Admin Control Panel")
    
    try:
        if 'df_meds' not in locals():
            df_meds = load_inventory_data()
        
        st.success("✅ Inventory Loaded (Live Google Sheet Mode)")
        
        # --- Admin Inventory Table ---
        st.subheader("📦 Stock Inventory")
        
        def highlight_low_stock_admin(row):
            if row['stock_level'] < 10:
                return ['background-color: #ffcccc; color: #990000'] * len(row)
            return [''] * len(row)
        
        admin_styled_df = df_meds.style.apply(highlight_low_stock_admin, axis=1)
        st.dataframe(admin_styled_df, hide_index=True, use_container_width=True)
        
        # --- Pending Orders for Admin Approval - CRITICAL UI FIX ---
        st.subheader("📋 Pending Orders")
        
        # Read from session_state.pending_orders with correct keys
        if st.session_state.get('pending_orders'):
            pending_orders_list = [o for o in st.session_state.pending_orders if o.get('status', '').lower() == 'pending']
            
            if pending_orders_list:
                st.markdown("**Pending Orders (Session State)**")
                
                for idx, order in enumerate(pending_orders_list):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # CRITICAL: Use correct keys from order dict
                        st.write(f"**{order.get('medicine_name', 'N/A')}** - Qty: {order.get('quantity_bought', 1)} - Customer: {order.get('customer_name', 'User')}")
                    with col2:
                        # CRITICAL: Use approve_order function with correct idx
                        if st.button(f"✅ Approve", key=f"approve_{idx}"):
                            # Find the actual index in pending_orders
                            actual_idx = st.session_state.pending_orders.index(order)
                            approve_order(actual_idx)
            else:
                st.info("No pending orders.")
        else:
            st.info("No pending orders.")
    
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # --- Live Traceability ---
    st.header("🔍 Live Traceability")
    langfuse_project_id = ""
    langfuse_dashboard_url = f"https://cloud.langfuse.com/project/{langfuse_project_id}"
    st.markdown(f"📊 **[Langfuse Public Dashboard]({langfuse_dashboard_url})**")


# ============================================================================
# CHAT INTERFACE
# ============================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Priority Logic: Check voice_prompt first, then use chat_input
voice_prompt = st.session_state.get('voice_prompt', None)
prompt = st.chat_input("Dawai mangiye...", key="pharmacy_input")

if voice_prompt and not st.session_state.voice_prompt_added:
    prompt = voice_prompt
    st.session_state.voice_prompt_added = True
    st.session_state.voice_prompt = None


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get available medicines from local CSV
    available_medicines = get_available_medicines()
    detected_medicine = extract_medicine_name(prompt, available_medicines)
    
    # Check if this is an order request
    order_keywords = ["order", "buy", "chahiye", "lena", "purchase", "get me", "i want"]
    is_order_request = any(keyword in prompt.lower() for keyword in order_keywords)
    
    # CRITICAL: Order detection in chat - create order with correct keys
    if detected_medicine and is_order_request:
        # Create order dictionary with CORRECT keys
        order_details = {
            "medicine_name": detected_medicine,
            "quantity_bought": 1,
            "customer_name": "User",
            "status": "Pending"
        }
        
        # Append to session_state.pending_orders
        st.session_state.pending_orders.append(order_details)
        
        # CRITICAL: Immediately rerun so Admin Panel updates instantly
        st.rerun()
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with langfuse.start_as_current_span(name="Pharmacist-Consultation-Session") as trace:
        trace.update_trace(
            user_id="customer_001",
            metadata={"model": "meta-llama/llama-4-scout-17b-16e-instruct"}
        )
        
        with st.chat_message("assistant"):
            try:
                df_meds = load_inventory_data()
                
                if not st.session_state.requested_medicine:
                    med_name_from_prompt = extract_medicine_name(prompt, df_meds['medicine_name'].tolist())
                    if med_name_from_prompt:
                        st.session_state.requested_medicine = med_name_from_prompt
                
                inventory_details = []
                for _, row in df_meds.iterrows():
                    stock_status = "OUT OF STOCK" if row['stock_level'] <= 0 else f"Stock: {int(row['stock_level'])}"
                    prescription = "REQUIRES PRESCRIPTION" if str(row['prescription_required']).lower() in ['yes', 'true'] else "No prescription needed"
                    inventory_details.append(f"{row['medicine_name']} ({stock_status}, {prescription})")
                
                inventory_list = ", ".join(inventory_details)
                
                chat_history = ""
                for msg in st.session_state.messages[-5:]:
                    chat_history += f"{msg['role']}: {msg['content']}\n"
                
                current_medicine = st.session_state.get('requested_medicine', 'Not specified')
                
                system_instruction = f"""{SYSTEM_PROMPT}

CURRENT CONVERSATION CONTEXT:
- Requested Medicine: {current_medicine}

AVAILABLE MEDICINES WITH STOCK STATUS:
{inventory_list}

CRITICAL RULES:
Aap ek "Senior Expert Doctor" hain. Agar user ko dard hai, toh:
1. Pehle hamdardi (empathy) dikhao. 
2. Us dard ke liye duniya ki sabse best dawai aur gharelu upay batao, chahe wo hamare paas ho ya na ho.
3. pahle tum probelm solve kro nhi to aakhir mai Doctor se milne ki salah do.
4. Sabse aakhir mein check karo hamare stock mein kya hai. Agar match nahi hota, toh saaf bolo ki "Ye mere stock mein nahi hai par aap bahar se le lo kyunki aapka thik hona zaroori hai."
1. STOCK CHECK: If OUT OF STOCK, REJECT and say "This medicine is currently out of stock."
2. PRESCRIPTION CHECK: If "REQUIRES PRESCRIPTION", ask for a prescription image in the sidebar.
3. NO HALLUCINATION: Only use the inventory list provided above.
3. STRICT INVENTORY CHECK (CRITICAL):
   - You MUST look at the provided Inventory Data: {inventory_list}
   - If the medicine is NOT in the list or has 0 stock: DO NOT say you have it. 
   - Instead, say: "While I don't have this medicine in my current stock, it is the best treatment for you. Please purchase it from a local pharmacy immediately."

4. ORDERING LOGIC:
   - Only offer to process an order if the medicine is actually present in the {inventory_list}.
   - If the user says "Confirm/Order", check availability one last time.
   - When user wants to order, respond with ORDER_DETECTED: medicine_name
RESPONSE STYLE:
Professional, empathetic, and 100% honest about stock availability.
"""
                
                model_to_use = "meta-llama/llama-4-scout-17b-16e-instruct"
                
                ans = ""
                for attempt in range(5):
                    try:
                        response = client.chat.completions.create(
                            model=model_to_use,
                            messages=[
                                {"role": "system", "content": system_instruction},
                                {"role": "user", "content": f"{chat_history}\n\nCURRENT USER MESSAGE: {prompt}"}
                            ],
                            temperature=0.7,
                            max_tokens=1024,
                            top_p=1,
                            stream=False
                        )
                        
                        ans = response.choices[0].message.content.strip()
                        break
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "429" in error_msg or "rate_limit" in error_msg:
                            if attempt < 4:
                                wait_time = (2 ** attempt) * 5
                                st.warning(f"⚠️ Groq API rate limit. Retrying in {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                continue
                        else:
                            st.error(f"System Error: {e}")
                            break
                
                if not ans:
                    st.error("🚨 API Service unavailable. Please check your quota.")
                    st.stop()
                
                # Check for ORDER_DETECTED in response
                if "ORDER_DETECTED:" in ans:
                    parts = ans.split("ORDER_DETECTED:")
                    if len(parts) > 1:
                        order_med_name = parts[1].strip().split()[0] if parts[1].strip() else None
                        ans = parts[0].strip()
                        
                        if order_med_name:
                            # CRITICAL: Use create_pending_order with correct keys
                            create_pending_order(order_med_name, quantity=1, customer_name="User")
                
                # --- AGENTIC ACTION: REAL-WORLD TOOL USE ---
                if "#CONFIRM#" in ans:
                    med = st.session_state.pending_med if st.session_state.pending_med else prompt
                    med_name = extract_medicine_name(med, df_meds['medicine_name'].tolist())
                    
                    if med_name:
                        med_filter = df_meds['medicine_name'].str.lower() == med_name.lower()
                        current_stock = df_meds.loc[med_filter, 'stock_level'].values[0]
                        
                        if current_stock <= 0:
                            ans = ans.replace("#CONFIRM#", "") + f"\n\n❌ **Stock Check Failed:** {med_name} is currently out of stock. Please choose another medicine."
                            st.session_state.pending_med = None
                        else:
                            if requires_prescription(med_name, df_meds):
                                if st.session_state.prescription_verified and st.session_state.verification_data:
                                    verification_data = st.session_state.verification_data
                                    
                                    # CRITICAL: Use create_pending_order with correct keys
                                    create_pending_order(
                                        verification_data['medicine'], 
                                        quantity=verification_data['quantity'], 
                                        customer_name="User"
                                    )
                                    
                                    ans = ans.replace("#CONFIRM#", "") + f"\n\n✅ **Order Submitted for Approval!** Your prescription for {verification_data['medicine']} has been verified and is pending admin approval. 🚀"
                                    st.balloons()
                                    st.session_state.pending_med = None
                                    st.session_state.prescription_verified = False
                                    st.session_state.verification_data = None
                                else:
                                    ans = ans.replace("#CONFIRM#", "") + f"\n\n⚠️ **Prescription Required:** {med_name} requires a prescription. Please upload your prescription image in the sidebar for verification before proceeding with the order."
                            else:
                                # CRITICAL: Use create_pending_order with correct keys
                                create_pending_order(med_name, quantity=1, customer_name="User")
                                
                                ans = ans.replace("#CONFIRM#", "") + f"\n\n✅ **Order Submitted for Approval!** {med_name} is pending admin approval. 🚀"
                                st.balloons()
                                st.session_state.pending_med = None
                    else:
                        ans = ans.replace("#CONFIRM#", "")
                        st.session_state.pending_med = None
                
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                
                trace.update(input=prompt, output=ans)
                
                generation = trace.start_observation(
                    as_type="generation",
                    name="Pharmacist-Reasoning-Step",
                    model=model_to_use,
                    input=prompt,
                    output=ans
                )
                generation.end()
            
            except Exception as e:
                st.error(f"System Error: {e}")