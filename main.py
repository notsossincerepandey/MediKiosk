from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
import logging
import traceback
import hashlib
import secrets
import datetime
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medikiosk")

# --- PASSWORD HELPERS -------------------------------------------------------
# Stdlib-only salted PBKDF2 hashing (no extra dependency like bcrypt/passlib
# required). Every user's default password is their date of birth in
# MMDDYY format; they can reset it afterwards to anything they like, and
# only the hash below is ever stored.

PBKDF2_ITERATIONS = 100_000

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return f"{salt}${digest.hex()}"

def verify_password(password: str, stored_hash: Optional[str]) -> bool:
    if not stored_hash or "$" not in stored_hash:
        return False
    salt, digest_hex = stored_hash.split("$", 1)
    try:
        check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    except ValueError:
        return False
    return secrets.compare_digest(check.hex(), digest_hex)

def default_password_from_dob(dob: datetime.date) -> str:
    """MMDDYY, e.g. 14 May 1998 -> '051498'."""
    return dob.strftime("%m%d%y")

# Centralized so a future Gemini deprecation only requires one change.
# The 404 you hit ("gemini-2.5-flash is no longer available to new users")
# is Google retiring the model early for newer projects. If this model
# also gets deprecated later, Google's error message will tell you the
# exact replacement name to put here.
GEMINI_MODEL = "gemini-3.6-flash"

app = FastAPI()

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google GenAI client for Gemini 2.5 Flash
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
@app.get("/")
def serve_frontend():
    if not os.path.exists("test.html"):
        return {"error": "test.html not found in the current directory."}
    return FileResponse("test.html")

def get_db():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
    return psycopg2.connect(
        dbname="medikiosk_test",
        user="postgres",
        password="password",
        host="localhost",
        port="5432"
    )

class IntakeRequest(BaseModel):
    full_name: str
    phone_number: str
    gender: str
    age: Optional[int] = 0
    symptoms: str
    medical_history: Optional[str] = ""
    full_transcript: Optional[str] = ""  # raw patient/AI turn history, for audit purposes
    department: Optional[str] = None     # AI's own final department decision, if available
    urgency: Optional[str] = None        # AI's own final urgency decision, if available
    priority_level: Optional[int] = None # AI's own final priority decision, if available
    allergies: Optional[List[str]] = []
    chronic: Optional[List[str]] = []
    # --- Returning-patient / consultation-type additions ---
    patient_id: Optional[int] = None        # if the patient already exists (looked up at login), reuse this row
                                             # instead of inserting a duplicate patient record
    consultation_type: Optional[str] = "FRESH"  # "FRESH" or "FOLLOWUP"
    follow_up_of: Optional[int] = None      # triage_session_id of the earlier visit this follows up on (optional)

class AbhaLinkRequest(BaseModel):
    patient_id: int
    abha_number: str
    abha_address: Optional[str] = None

class ChatTurn(BaseModel):
    role: str  # "user" or "model"
    text: str

class AIChatQuery(BaseModel):
    message: str
    language: Optional[str] = "Hindi"
    history: Optional[List[ChatTurn]] = []  # prior turns, so the model has memory of the conversation so far

TRIAGE_SYSTEM_INSTRUCTION = """
You are an advanced AI clinical intake and triage assistant for a multi-specialty hospital kiosk.

Each request includes the full prior conversation (as alternating user/model turns) plus the
patient's newest message. Read the ENTIRE conversation before deciding what to do — you must
never ask the patient for information they already gave earlier in the conversation. If you
find yourself about to ask something already covered above, stop and move to the next missing
piece of information (or finalize, if nothing is missing) instead.

You conduct a short, multi-turn intake conversation before recommending a doctor. Do NOT match
a department/doctor on the very first message — gather information first.

Your responsibilities:
1. Systematically prompt and guide the patient to provide, across multiple turns:
   a) Their current symptoms / reason for today's visit
   b) Past medical history or chronic conditions (e.g. diabetes, asthma, prior surgeries)
   c) Known drug or food allergies
   Ask ONE clear follow-up question at a time in the "reply" field, and ONLY about whichever of
   (a)/(b)/(c) is still genuinely missing from the conversation so far (a brief "no allergies" /
   "no history" counts as covered). Never repeat a question the patient has already answered,
   even if they answered it several turns ago or phrased it differently than expected.
2. Accurately transcribe audio or text input (supporting English, Hindi, or Hinglish).
3. Classify EACH patient message as either:
   - "CURRENT_SYMPTOM": what the patient is here for today (e.g. "I have a headache", "chest pain since morning")
   - "MEDICAL_HISTORY": pre-existing/background conditions, allergies, or past diagnoses the patient mentions
     (e.g. "I have asthma", "I'm allergic to dust", "I had surgery 2 years ago")
   If a single message contains both, classify it by whichever is the primary content of that message.
4. Maintain two running, cumulative clinical summaries built from the ENTIRE conversation so far
   (not just the latest message), always written in clear, professional, concise clinical English
   regardless of what language the patient is speaking — these go directly into the doctor's chart:
   - "symptom_summary": current visit's symptoms as a short clinical phrase, e.g.
     "Headache, nasal congestion, and abdominal pain, onset this morning."
   - "history_summary": relevant past history and allergies as a short clinical phrase, e.g.
     "History of hypertension, type 2 diabetes, and asthma. Reports dust allergy."
   Update both of these on every turn to reflect everything gathered so far, deduplicated and
   condensed — never a raw copy-paste of the patient's own wording, and never just the latest
   message alone.
5. Set "ready_for_triage":
   - false, while you are still actively gathering (a), (b), or (c) above — in this case you may
     leave "department"/"doctor"/"room"/"urgency"/"priority_level" as your best current guess, but
     the frontend will NOT act on them yet, so focus "reply" on asking the next genuinely-missing
     intake question.
   - true, once you have gathered enough about symptoms, history, and allergies to safely
     recommend a department. At that point, determine the correct hospital department and doctor:
     - Cardiology -> Dr. Amit Sharma (Cardiology - Room 102) [Priority Level 1 if chest pain/emergency]
     - Orthopedics -> Dr. Rajesh Nair (Orthopedics - Room 305) [Priority Level 2]
     - Gastroenterology -> Dr. Neha Gupta (Gastroenterology - Room 401) [Priority Level 2]
     - General Medicine -> Dr. Priya Varma (General Medicine - Room 204) [Priority Level 3]
   - true immediately, without delay, if EITHER: (i) the patient describes a clear emergency
     (e.g. severe chest pain, difficulty breathing), or (ii) the patient signals frustration or
     repetition — e.g. "I already told you", "I said this already", "just refer me" — in which
     case finalize using whatever information has been gathered so far rather than asking again.
6. Provide a compassionate response back to the patient in their preferred language.

You MUST respond strictly in valid JSON format using these exact keys:
{
  "transcript": "The cleaned text or speech transcription",
  "symptom_type": "CURRENT_SYMPTOM" or "MEDICAL_HISTORY",
  "reply": "Your conversational response — a follow-up intake question, or the final triage summary once ready",
  "ready_for_triage": true or false,
  "symptom_summary": "Cumulative professional clinical summary of current symptoms, in English",
  "history_summary": "Cumulative professional clinical summary of past history/allergies, in English",
  "department": "Cardiology or Orthopedics or Gastroenterology or General Medicine",
  "doctor": "Dr. Amit Sharma (Cardiology - Room 102)",
  "room": "Room 102",
  "urgency": "EMERGENCY" or "URGENT" or "ROUTINE",
  "priority_level": 1 or 2 or 3
}
"""

def extract_gemini_text(response) -> str:
    """
    Safely pull text out of a Gemini response.
    response.text raises instead of returning a string when there's no valid
    text part (blocked by safety filters, truncated, empty audio, etc).
    We check candidates/finish_reason ourselves so we get a clear error
    instead of a generic 500 with no explanation.
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise ValueError("Gemini returned no candidates (empty or fully blocked response).")

    candidate = candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)

    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content else None

    if not parts:
        # This is the case that silently breaks response.text
        raise ValueError(
            f"Gemini returned no usable text (finish_reason={finish_reason}). "
            f"This usually means the audio was blocked, empty, or unintelligible."
        )

    text = "".join(getattr(p, "text", "") or "" for p in parts)
    if not text.strip():
        raise ValueError(f"Gemini returned an empty response (finish_reason={finish_reason}).")

    return text


def parse_gemini_json(response):
    """Safely parse JSON from a Gemini response even if markdown code blocks are present, and map keys to frontend expectations."""
    raw_text = extract_gemini_text(response)

    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    try:
        data = json.loads(cleaned.strip())
    except json.JSONDecodeError as e:
        logger.error("Gemini did not return valid JSON. Raw text was:\n%s", raw_text)
        raise ValueError(f"Gemini response was not valid JSON: {e}")

    # Ensures keys align precisely with what test.html is trying to render
    return {
        "transcript": data.get("transcript", "Audio processed successfully."),
        "symptom_type": data.get("symptom_type", "CURRENT_SYMPTOM"),
        "reply": data.get("reply", "Symptoms noted."),
        "ready_for_triage": bool(data.get("ready_for_triage", False)),
        "symptom_summary": data.get("symptom_summary", ""),
        "history_summary": data.get("history_summary", ""),
        "matched_department": data.get("department", "General Medicine"),
        "doctor_name": data.get("doctor", "Dr. Priya Varma (General Medicine - Room 204)"),
        "room": data.get("room", "Room 204"),
        "urgency": data.get("urgency", "ROUTINE"),
        "priority_level": data.get("priority_level", 3)
    }

def build_history_contents(history):
    """Convert prior {role, text} turns into Gemini multi-turn Content objects,
    so the model can see what's already been said instead of only the latest message."""
    contents = []
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", None)
        text = turn.get("text") if isinstance(turn, dict) else getattr(turn, "text", None)
        if not text:
            continue
        role = "model" if role == "model" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
    return contents

# 1. TEXT CHAT ENDPOINT USING GEMINI 2.5 FLASH
@app.post("/api/ai/chat")
def multilingual_ai_triage(data: AIChatQuery):
    history_contents = build_history_contents(data.history)
    latest_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"Patient Message: '{data.message}'\nPreferred Language: {data.language}")]
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=history_contents + [latest_content],
            config=types.GenerateContentConfig(
                system_instruction=TRIAGE_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return parse_gemini_json(response)
    except Exception as e:
        logger.error("Text chat failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

# 2. MULTIMODAL VOICE CHAT ENDPOINT USING GEMINI 2.5 FLASH
@app.post("/api/ai/voice-chat")
async def process_voice_chat(file: UploadFile = File(...), language: str = Form(...), history: str = Form(default="[]")):
    try:
        audio_bytes = await file.read()
        prompt = f"Listen to this patient audio recording. Accurately transcribe speech, extract current medical issues, past history, and allergies. Preferred language for reply: {language}"

        if not audio_bytes:
            raise ValueError("Received an empty audio file from the browser (0 bytes).")

        mime_type = file.content_type or "audio/wav"
        logger.info("Voice upload: %d bytes, content_type=%s", len(audio_bytes), mime_type)

        try:
            history_list = json.loads(history) if history else []
        except json.JSONDecodeError:
            history_list = []

        history_contents = build_history_contents(history_list)
        latest_content = types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                types.Part.from_text(text=prompt),
            ]
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=history_contents + [latest_content],
            config=types.GenerateContentConfig(
                system_instruction=TRIAGE_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        return parse_gemini_json(response)

    except Exception as e:
        logger.error("Voice chat failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Gemini Voice Processing Error: {str(e)}")

# 2B. PATIENT LOOKUP + ABHA LINK/DELINK ENDPOINTS
#
# ABHA linking is a ONE-TIME action per patient profile. A profile is identified by its
# phone_number; the ABHA number/address are additional fields stored on that same patient
# row. The mobile number used to log in to MediKiosk does NOT need to match whatever mobile
# is registered against the ABHA account itself — the two identifiers are independent, and
# we never cross-check them against each other.
#
# Flow: on login, the frontend calls POST /api/patient/login with the mobile number. If a
# patient already exists and abha_number is already set, the mandatory ABHA-link gate is
# skipped entirely. It's only shown again if the patient explicitly delinks (which clears
# abha_number/abha_address).

class PatientLoginRequest(BaseModel):
    mobile: str
    full_name: Optional[str] = None
    dob: Optional[str] = None  # "YYYY-MM-DD" — required only the first time this mobile number is seen

class PatientAuthRequest(BaseModel):
    mobile: str
    password: str
    full_name: Optional[str] = None
    dob: Optional[str] = None  # "YYYY-MM-DD" — only used to REGISTER a brand-new mobile number

class PatientPasswordResetRequest(BaseModel):
    current_password: str
    new_password: str

@app.post("/api/patient/login")
def patient_login(data: PatientLoginRequest):
    """Get-or-create a patient profile by mobile number at login time. This gives every
    patient a persistent profile from their very first login (not only once they submit
    an intake), so ABHA-link status and past consultations can be tied to it — and so a
    returning patient who already linked ABHA never has to link it again.

    A brand-new profile also needs a date of birth: the default password for every user
    is their DOB in MMDDYY format, set once here and then only ever readable as a hash."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            "SELECT id, full_name, abha_number, abha_address FROM patients WHERE phone_number = %s;",
            (data.mobile,)
        )
        row = cur.fetchone()
        if row:
            return {
                "patient_id": row["id"],
                "full_name": row["full_name"],
                "abha_linked": bool(row["abha_number"]),
                "abha_number": row["abha_number"],
                "abha_address": row["abha_address"],
                "is_new": False,
            }

        if not data.dob:
            raise HTTPException(
                status_code=400,
                detail="This mobile number isn't registered yet — please provide your date of birth to create your profile."
            )
        try:
            dob_date = datetime.date.fromisoformat(data.dob)
        except ValueError:
            raise HTTPException(status_code=400, detail="Date of birth must be in YYYY-MM-DD format.")

        default_password_hash = hash_password(default_password_from_dob(dob_date))

        cur.execute(
            """
            INSERT INTO patients
                (full_name, phone_number, gender, address, known_allergies, chronic_conditions, dob, password_hash, password_updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            RETURNING id;
            """,
            (data.full_name or "New Patient", data.mobile, "Other", "", [], [], dob_date, default_password_hash)
        )
        patient_id = cur.fetchone()["id"]
        conn.commit()
        return {
            "patient_id": patient_id,
            "full_name": data.full_name or "New Patient",
            "abha_linked": False,
            "abha_number": None,
            "abha_address": None,
            "is_new": True,
        }
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.post("/api/patient/authenticate")
def authenticate_patient(data: PatientAuthRequest):
    """Password-based login. The default password is DOB in MMDDYY format until the
    patient resets it via /api/patient/{id}/reset-password.

    If the mobile number has never been seen before, this also REGISTERS the patient
    (mirroring what /api/patient/login does for the OTP flow) — as long as a date of
    birth was supplied. That lets the Password tab work for first-time sign-ups too,
    not just returning patients. The entered password must match the DOB-derived
    default (MMDDYY) for the very first sign-in; after that they can change it via
    /api/patient/{id}/reset-password."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            "SELECT id, full_name, password_hash, abha_number, abha_address FROM patients WHERE phone_number = %s;",
            (data.mobile,)
        )
        row = cur.fetchone()

        if not row:
            # Brand-new mobile number — register it now, same as the OTP path does.
            if not data.dob:
                raise HTTPException(
                    status_code=400,
                    detail="This mobile number isn't registered yet — please provide your date of birth to create your profile."
                )
            try:
                dob_date = datetime.date.fromisoformat(data.dob)
            except ValueError:
                raise HTTPException(status_code=400, detail="Date of birth must be in YYYY-MM-DD format.")

            default_password_hash = hash_password(default_password_from_dob(dob_date))

            # The password they typed must equal the DOB-derived default the first time —
            # verify it against the hash we're about to store, not the stored one, since
            # nothing's been stored yet.
            if not verify_password(data.password, default_password_hash):
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect password for a new account — your default password is your date of birth in MMDDYY format."
                )

            cur.execute(
                """
                INSERT INTO patients
                    (full_name, phone_number, gender, address, known_allergies, chronic_conditions, dob, password_hash, password_updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id, full_name, abha_number, abha_address;
                """,
                (data.full_name or "New Patient", data.mobile, "Other", "", [], [], dob_date, default_password_hash)
            )
            row = cur.fetchone()
            conn.commit()
            return {
                "patient_id": row["id"],
                "full_name": row["full_name"],
                "abha_linked": False,
                "abha_number": None,
                "abha_address": None,
                "is_new": True,
            }

        if not verify_password(data.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid mobile number or password.")
        return {
            "patient_id": row["id"],
            "full_name": row["full_name"],
            "abha_linked": bool(row["abha_number"]),
            "abha_number": row["abha_number"],
            "abha_address": row["abha_address"],
            "is_new": False,
        }
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.post("/api/patient/{patient_id}/reset-password")
def reset_patient_password(patient_id: int, data: PatientPasswordResetRequest):
    """Resets a patient's password. Requires the current password (default or previously
    chosen) so a stolen session alone can't take over the account."""
    if len(data.new_password) < 4:
        raise HTTPException(status_code=400, detail="New password must be at least 4 characters.")

    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT password_hash FROM patients WHERE id = %s;", (patient_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Patient not found.")
        if not verify_password(data.current_password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect.")

        cur.execute(
            "UPDATE patients SET password_hash = %s, password_updated_at = NOW() WHERE id = %s;",
            (hash_password(data.new_password), patient_id)
        )
        conn.commit()
        return {"status": "success"}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.post("/api/patient/link-abha")
def link_abha(data: AbhaLinkRequest):
    """One-time ABHA link for an existing patient profile. Safe to call again with the
    same values, but the frontend should only surface this step when abha_linked is False."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            UPDATE patients
            SET abha_number = %s, abha_address = %s, abha_linked_at = NOW()
            WHERE id = %s
            RETURNING id, abha_number, abha_address;
            """,
            (data.abha_number, data.abha_address, data.patient_id)
        )
        updated = cur.fetchone()
        if not updated:
            raise HTTPException(status_code=404, detail="Patient not found.")
        conn.commit()
        return {"status": "success", "patient": updated}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.post("/api/patient/{patient_id}/delink-abha")
def delink_abha(patient_id: int):
    """Explicitly delink ABHA from a profile. After this, the mandatory-link gate
    will be shown again on next login until the patient re-links."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            UPDATE patients
            SET abha_number = NULL, abha_address = NULL, abha_linked_at = NULL
            WHERE id = %s
            RETURNING id;
            """,
            (patient_id,)
        )
        updated = cur.fetchone()
        if not updated:
            raise HTTPException(status_code=404, detail="Patient not found.")
        conn.commit()
        return {"status": "success"}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.get("/api/patient/{patient_id}/consultations")
def get_past_consultations(patient_id: int):
    """Past visits for this patient, so the frontend can let them pick which one
    a new follow-up consultation relates to."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            SELECT t.id AS triage_session_id, t.created_at, t.ai_structured_summary,
                   t.triage_priority_level, t.consultation_type
            FROM triage_sessions t
            WHERE t.patient_id = %s
            ORDER BY t.created_at DESC
            LIMIT 20;
            """,
            (patient_id,)
        )
        return {"consultations": cur.fetchall()}
    finally:
        cur.close()
        conn.close()

@app.get("/api/patient/{patient_id}/medical-history")
def get_patient_medical_history(patient_id: int):
    """Everything on file for this patient, pre-sorted into the same clinical
    buckets a doctor thinks in — chronic illnesses, allergies, past surgeries,
    past consultations, and anything pulled from uploaded documents via OCR —
    instead of one long undifferentiated wall of text. Powers the doctor
    dashboard's 'View Past Medical History' page."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            SELECT id, full_name, dob, gender, phone_number,
                   known_allergies, chronic_conditions, past_surgeries
            FROM patients WHERE id = %s;
            """,
            (patient_id,)
        )
        patient = cur.fetchone()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")

        cur.execute(
            """
            SELECT id AS triage_session_id, created_at, consultation_type,
                   triage_priority_level, ai_structured_summary, speech_to_text_transcript
            FROM triage_sessions
            WHERE patient_id = %s
            ORDER BY created_at DESC;
            """,
            (patient_id,)
        )
        consultations = cur.fetchall()

        cur.execute(
            """
            SELECT id, document_image_url, extracted_data, is_doctor_verified,
                   doctor_notes, created_at
            FROM prescriptions_ocr
            WHERE patient_id = %s
            ORDER BY created_at DESC;
            """,
            (patient_id,)
        )
        documents = cur.fetchall()

        return {
            "patient": patient,
            "consultations": consultations,
            "documents": documents,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

# 3. INTAKE ENDPOINT 
@app.post("/api/intake")
def register_and_triage(data: IntakeRequest):
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        if data.patient_id:
            # Returning patient (already looked up at login) — reuse the existing profile
            # instead of creating a duplicate patient row.
            cur.execute("SELECT id FROM patients WHERE id = %s;", (data.patient_id,))
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail=f"patient_id {data.patient_id} not found.")
            patient_id = existing["id"]
        else:
            cur.execute(
                """
                INSERT INTO patients (full_name, phone_number, gender, address, known_allergies, chronic_conditions)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (data.full_name, data.phone_number, data.gender, f"Age: {data.age}", data.allergies, data.chronic)
            )
            patient_id = cur.fetchone()["id"]

        # Prefer the AI's own final triage decision from the conversation itself — it had the
        # full context (symptoms, history, allergies) to work with. Only fall back to a crude
        # keyword guess if the frontend didn't supply one (e.g. an older client).
        if data.priority_level in (1, 2, 3):
            priority = data.priority_level
        else:
            priority = 1 if any(k in data.symptoms.lower() for k in ["chest", "pain", "seene", "saans"]) else 3

        department = data.department or ("Cardiology" if priority == 1 else "General Medicine")
        urgency = data.urgency or ("EMERGENCY" if priority == 1 else "ROUTINE")

        summary = {
            "primary_complaint": data.symptoms,
            "history_notes": data.medical_history or "",
            "department": department,
            "urgency": urgency
        }

        # speech_to_text_transcript stores the raw conversation (for audit) when available,
        # falling back to the clean symptom summary for older clients that don't send one.
        transcript_for_record = data.full_transcript or data.symptoms

        # "FRESH" or "FOLLOWUP" — normalize/validate rather than trusting the client blindly.
        consultation_type = data.consultation_type if data.consultation_type in ("FRESH", "FOLLOWUP") else "FRESH"

        cur.execute(
            """
            INSERT INTO triage_sessions
                (patient_id, speech_to_text_transcript, ai_structured_summary, triage_priority_level,
                 consultation_type, follow_up_of_triage_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (patient_id, transcript_for_record, json.dumps(summary), priority,
             consultation_type, data.follow_up_of)
        )
        triage_id = cur.fetchone()["id"]

        # Tokens must be unique per queue entry, not hardcoded — start at 101
        # and increment from whatever's already been issued for this doctor.
        cur.execute(
            """
            SELECT COALESCE(MAX(token_number), 100) + 1 AS next_token
            FROM opd_queues WHERE doctor_id = 1;
            """
        )
        next_token = cur.fetchone()["next_token"]

        cur.execute(
            """
            INSERT INTO opd_queues (doctor_id, patient_id, triage_session_id, token_number, status)
            VALUES (1, %s, %s, %s, 'WAITING')
            RETURNING token_number;
            """,
            (patient_id, triage_id, next_token)
        )
        token = cur.fetchone()["token_number"]
        conn.commit()

        return {"status": "success", "patient_id": patient_id, "token": f"OPD-{token}"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

class PatientReviewRequest(BaseModel):
    doctor_id: int
    patient_id: int
    rating: int          
    comment: Optional[str] = None

class DoctorReviewRequest(BaseModel):
    doctor_id: int
    patient_id: Optional[int] = None
    rating: int          # 1-5
    comment: Optional[str] = None

@app.post("/api/reviews")
def submit_doctor_review(data: DoctorReviewRequest):
    """Anonymous-to-other-patients (but stored with patient_id for the doctor's own
    records) star rating + optional comment for a doctor. Requires an actual 1-5
    rating — there is no way to submit this with nothing filled in."""
    if data.rating < 1 or data.rating > 5:
        raise HTTPException(status_code=400, detail="Please select a star rating from 1 to 5.")

    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            INSERT INTO doctor_reviews (doctor_id, patient_id, rating, comment)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (data.doctor_id, data.patient_id, data.rating, data.comment)
        )
        review_id = cur.fetchone()["id"]
        conn.commit()
        return {"status": "success", "review_id": review_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.get("/api/doctor/{doctor_id}/reviews")
def get_doctor_reviews(doctor_id: int):
    """All reviews for a doctor plus the average rating, for a future 'my reviews'
    view on the doctor dashboard."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            "SELECT rating, comment, created_at FROM doctor_reviews WHERE doctor_id = %s ORDER BY created_at DESC;",
            (doctor_id,)
        )
        reviews = cur.fetchall()
        avg_rating = round(sum(r["rating"] for r in reviews) / len(reviews), 2) if reviews else None
        return {"reviews": reviews, "average_rating": avg_rating, "count": len(reviews)}
    finally:
        cur.close()
        conn.close()

@app.get("/api/patient/check")
def check_patient_exists(mobile: str):
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT id FROM patients WHERE phone_number = %s;", (mobile,))
        row = cur.fetchone()
        return {"exists": bool(row)}
    finally:
        cur.close()
        conn.close()

@app.post("/api/patient-reviews")
def submit_patient_review(data: PatientReviewRequest):
    if data.rating < 1 or data.rating > 5:
        raise HTTPException(status_code=400, detail="Please select a star rating from 1 to 5.")

    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            INSERT INTO patient_reviews (doctor_id, patient_id, rating, comment)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (data.doctor_id, data.patient_id, data.rating, data.comment)
        )
        review_id = cur.fetchone()["id"]
        conn.commit()
        return {"status": "success", "review_id": review_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()


        

# 3B. DOCTOR AUTHENTICATION + PASSWORD RESET
#
# Doctors also default to their DOB in MMDDYY format. Existing rows (seeded via migration)
# may only have a DOB on file and no password_hash yet — the first successful login backfills
# the hash from that DOB so every account gets one lazily rather than needing a one-off script.

class DoctorAuthRequest(BaseModel):
    staff_id: str
    password: str

class DoctorPasswordResetRequest(BaseModel):
    current_password: str
    new_password: str

@app.post("/api/doctor/authenticate")
def authenticate_doctor(data: DoctorAuthRequest):
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            "SELECT id, full_name, department, room, dob, password_hash FROM doctors WHERE staff_id = %s;",
            (data.staff_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Invalid Staff ID or password.")

        password_hash = row["password_hash"]
        if not password_hash:
            if not row["dob"]:
                raise HTTPException(status_code=401, detail="Invalid Staff ID or password.")
            password_hash = hash_password(default_password_from_dob(row["dob"]))
            cur.execute("UPDATE doctors SET password_hash = %s WHERE id = %s;", (password_hash, row["id"]))
            conn.commit()

        if not verify_password(data.password, password_hash):
            raise HTTPException(status_code=401, detail="Invalid Staff ID or password.")

        return {
            "doctor_id": row["id"],
            "full_name": row["full_name"],
            "department": row["department"],
            "room": row["room"],
        }
    except HTTPException:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

@app.post("/api/doctor/{doctor_id}/reset-password")
def reset_doctor_password(doctor_id: int, data: DoctorPasswordResetRequest):
    if len(data.new_password) < 4:
        raise HTTPException(status_code=400, detail="New password must be at least 4 characters.")

    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        # 1. Fetch dob as well so we can auto-initialize the hash if it is NULL
        cur.execute("SELECT password_hash, dob FROM doctors WHERE id = %s;", (doctor_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Doctor not found.")
        
        password_hash = row["password_hash"]
        
        # 2. Auto-initialize the hash if the DB was reset to NULL
        if not password_hash:
            if not row["dob"]:
                raise HTTPException(status_code=401, detail="Current password is incorrect.")
            password_hash = hash_password(default_password_from_dob(row["dob"]))
            cur.execute("UPDATE doctors SET password_hash = %s WHERE id = %s;", (password_hash, doctor_id))
            conn.commit()

        # 3. Verify the current password
        if not verify_password(data.current_password, password_hash):
            raise HTTPException(status_code=401, detail="Current password is incorrect.")

        # 4. Save the new password
        cur.execute(
            "UPDATE doctors SET password_hash = %s WHERE id = %s;",
            (hash_password(data.new_password), doctor_id)
        )
        conn.commit()
        return {"status": "success"}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

# 4. QUEUE RETRIEVAL
@app.get("/api/doctor/queue")
def fetch_queue():
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT q.id AS queue_id, q.token_number, q.status, p.id AS patient_id, p.full_name, p.gender, p.address as age_meta, p.known_allergies, 
               p.chronic_conditions, t.speech_to_text_transcript, t.ai_structured_summary,
               t.consultation_type, t.follow_up_of_triage_id
        FROM opd_queues q
        JOIN patients p ON q.patient_id = p.id
        LEFT JOIN triage_sessions t ON q.triage_session_id = t.id
        WHERE q.doctor_id = 1 AND q.status = 'WAITING'
        ORDER BY q.created_at DESC;
        """ 
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"queue": rows}

# 5. REMOVE FROM QUEUE (used by both the standalone "remove" button and
#    "Complete Consultation" — soft delete so patient/triage history stays intact)
@app.delete("/api/doctor/queue/{queue_id}")
def remove_from_queue(queue_id: int):
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            "UPDATE opd_queues SET status = 'REMOVED' WHERE id = %s RETURNING id;",
            (queue_id,)
        )
        updated = cur.fetchone()
        if not updated:
            raise HTTPException(status_code=404, detail=f"Queue entry {queue_id} not found.")
        conn.commit()
        return {"status": "success", "removed_queue_id": queue_id}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()