import os
import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from groq import Groq
from langfuse import Langfuse
import google.generativeai as genai
import tempfile
import re
import json

# --- GEMINI VISION FUNCTION ---
def extract_medicine_from_image(image_bytes):
    """
    Extract medicine name from prescription image using Google Gemini 2.5 Flash.
    
    Args:
        image_bytes: The image data as bytes
        
    Returns:
        dict: Contains medicine_name and quantity if successful, or error info
    """
    try:
        # Configure Gemini client with the provided API key
        GEMINI_API_KEY = ""
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Save image to temporary file for upload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
            f.write(image_bytes)
            temp_path = f.name
        
        uploaded_file = genai.upload_file(path=temp_path)
        
        # Use Gemini 2.5 Flash model for vision
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        response = model.generate_content([
            "Extract the medicine name from this prescription image. Also identify if there's a quantity mentioned. Return the response as a JSON with keys: 'medicine_name' and 'quantity'. If no quantity is specified, default to 1. Only return the medicine name if it is clearly legible on the prescription. If you cannot determine the medicine name, return medicine_name as 'UNKNOWN'.",
            uploaded_file
        ])
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Parse the response to extract medicine name and quantity
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        
        if json_match:
            try:
                result = json.loads(json_match.group())
                medicine_name = result.get('medicine_name', '').strip()
                quantity = int(result.get('quantity', 1))
                
                if medicine_name and medicine_name.upper() != 'UNKNOWN':
                    return {
                        "valid": True,
                        "medicine_name": medicine_name,
                        "quantity": quantity
                    }
                else:
                    return {
                        "valid": False,
                        "reason": "Could not identify medicine name from the prescription image."
                    }
            except json.JSONDecodeError:
                pass
        
        # Fallback: Try to extract medicine name directly from response text
        medicine_name = response_text.strip()
        if medicine_name and medicine_name.upper() != 'UNKNOWN':
            return {
                "valid": True,
                "medicine_name": medicine_name,
                "quantity": 1
            }
        
        return {
            "valid": False,
            "reason": "Could not extract medicine name from the prescription image."
        }
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {
            "valid": False,
            "reason": f"Error calling Gemini API: {str(e)}"
        }

# --- 1. SETUP & OBSERVABILITY ---
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
langfuse = Langfuse()

# Configure Groq client with API key
# Use environment variable if available, otherwise use the provided key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

client = Groq(api_key=GROQ_API_KEY)

# --- 2. PREDICTIVE INTELLIGENCE ---
def check_refills():
    try:
        res = requests.get("http://127.0.0.1:8000/refill-alerts")
        if res.status_code == 200:
            data = res.json()
            return data.get("alerts", [])
        return []
    except Exception as e:
        print(f"Refill Check Error: {e}")
        return []

# --- 3. REAL-WORLD ACTION ---
def handle_order(medicine_name):
    try:
        res = requests.post(f"http://127.0.0.1:8000/place-order/{medicine_name.strip()}")
        if res.status_code == 200:
            data = res.json()
            return f"✅ Order Successful! Remaining stock: {data.get('remaining_stock', 'Updated')}"
        return "❌ Order failed: Stock issue."
    except Exception:
        return "❌ Connection Error: Kya main.py chal raha hai?"

# --- 4. EXPERT AGENT LOGIC ---
def ask_pharmacist(user_input, last_medicine=None):
    trace = langfuse.start_span(name="expert-pharmacy-consultation")
    trace.update_trace(user_id="customer_1")

    try:
        # Naya Expert System Prompt jo Brain ki tarah kaam karega
        system_instruction = (
            "You are the World's Best AI Pharmacist & Medical Expert. Your goal is to provide a holistic consultation.\n"
            "1. Empathize & Analyze: Pehle user ki problem samjhein aur doctor ki tarah empathy dikhayein sabse pahle tum doctor banake uski problem solve kro.\n"
            "2. Expert Advice:dawai and lifestyle badlav batayein (diet, exercise, precautions).\n"
            "3. Medical Suggestions: Apni internal knowledge se best medicines suggest karein.\n"
            "4. Inventory Check: Check karein ki kya wo medicines stock mein hain.\n"
            "5. Safety First: Agar dawai ke liye prescription zaroori hai, toh user se Doctor ka naam aur ID maangein.\n"
            "6. Order Detection: If user wants to order a medicine, respond with ORDER_DETECTED: medicine_name"
        )

        # Groq Call for Expert Reasoning
        prompt = f"{system_instruction}\n\nUser Input: {user_input}\nLast Medicine Context: {last_medicine}"

        # Use Groq Chat Completions API
        generation = trace.start_generation(name="expert-reasoning", model="meta-llama/llama-4-scout-17b-16e-instruct", input=prompt)

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"User Input: {user_input}\nLast Medicine Context: {last_medicine}"}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )

        response_text = response.choices[0].message.content.strip()
        generation.end(output=response_text)

        # Check for order detection
        order_details = None
        if "ORDER_DETECTED:" in response_text:
            parts = response_text.split("ORDER_DETECTED:")
            if len(parts) > 1:
                medicine_name = parts[1].strip()
                order_details = {"medicine_name": medicine_name, "quantity": 1}
                response_text = parts[0].strip()

        # Stock check logic if a medicine is mentioned
        # Hum response text mein medicine name dhoondne ki koshish karenge
        res_data = {}
        if last_medicine or any(word in user_input.lower() for word in ["order", "buy", "chahiye"]):
            med_to_check = last_medicine if last_medicine else user_input
            stock_res = requests.get(f"http://127.0.0.1:8000/check-stock/{med_to_check.strip()}")
            if stock_res.status_code == 200:
                res_data = stock_res.json()

                # Prescription Check Logic
                if str(res_data.get('prescription_required')).lower() == "yes":
                    trace.create_event(name="policy-block", input=med_to_check, output="Verification Needed")
                    # Agar user ne abhi tak Dr ka naam nahi diya toh maangein
                    if "dr." not in user_input.lower():
                        trace.end()
                        return f"Senior Pharmacist: Main aapki help zaroori karunga, par safety ke liye mujhe Dr. ka naam aur Registration ID chahiye. {response_text}", med_to_check, order_details

        trace.end()
        return response_text, None, order_details

    except Exception as e:
        trace.update(status_message=f"Error: {str(e)}")
        trace.end()
        return f"❌ AI Error: {str(e)}", None, None

# --- 5. MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    print("\n🏥 EXPERT AI PHARMACIST SYSTEM LIVE")
    print("Using Groq with Llama3-8b-8192 model")
    
    current_med = None
    while True:
        user_in = input("\nUser: ").strip()
        if user_in.lower() in ['exit', 'quit']: break
        
        # Expert AI se baat karein
        ai_msg, med_context, order_details = ask_pharmacist(user_in, current_med)
        if med_context: current_med = med_context
        
        print(f"\n💊 AI Pharmacist: {ai_msg}")
        
        # Handle order details if detected
        if order_details:
            print(f"📦 Order Detected: {order_details}")

        # Order confirmation logic
        if "confirm" in user_in.lower() or "verify" in user_in.lower() or "id:" in user_in.lower():
            if current_med:
                print(f"⚙️ Action: Validating & Updating Database...")
                result = handle_order(current_med)
                print(f"⚙️ Result: {result}")
                current_med = None # Reset context after order
