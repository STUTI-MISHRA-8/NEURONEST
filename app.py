import streamlit as st
import sounddevice as sd
import wavio
import numpy as np
import os
import datetime
import time
import threading
from queue import Queue, Empty
from faster_whisper import WhisperModel
from transformers import pipeline
import spacy
import sqlite3 
import json 
import requests

# --- Configuration ---
SAMPLE_RATE = 44100; CHANNELS = 1; AUDIO_DIR = "recordings"
MAX_RECORDING_MINUTES = 5; BLOCK_DURATION_SECONDS = 0.5
WHISPER_MODEL_NAME = "base"; SENTIMENT_MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"; SPACY_MODEL_NAME = "en_core_web_sm"
DB_NAME = "neuronest.db"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "mistral" 

# --- INITIALIZE SESSION STATE ---
if 'is_recording' not in st.session_state: st.session_state.is_recording = False
if 'current_filename' not in st.session_state: st.session_state.current_filename = None
if 'start_time' not in st.session_state: st.session_state.start_time = None
if 'recording_error' not in st.session_state: st.session_state.recording_error = None
if '_save_process_initiated' not in st.session_state: st.session_state._save_process_initiated = False
if 'model_load_error_message' not in st.session_state: st.session_state.model_load_error_message = None
if 'last_sentiment' not in st.session_state: st.session_state.last_sentiment = None
if 'last_summary' not in st.session_state: st.session_state.last_summary = None
if 'last_tags' not in st.session_state: st.session_state.last_tags = []
if 'whisper_model' not in st.session_state: st.session_state.whisper_model = None
if 'sentiment_analyzer' not in st.session_state: st.session_state.sentiment_analyzer = None
if 'summarizer' not in st.session_state: st.session_state.summarizer = None
if 'spacy_nlp' not in st.session_state: st.session_state.spacy_nlp = None
if 'refresh_entries_toggle' not in st.session_state: st.session_state.refresh_entries_toggle = False
if 'disclaimer_shown' not in st.session_state: st.session_state.disclaimer_shown = False
if 'last_llm_insights' not in st.session_state: st.session_state.last_llm_insights = None
if 'ollama_connection_error' not in st.session_state: st.session_state.ollama_connection_error = False
if "sentiment_filter_val" not in st.session_state: st.session_state.sentiment_filter_val = []
if "search_term_val" not in st.session_state: st.session_state.search_term_val = ""
if 'selected_journal_week_start_str' not in st.session_state: st.session_state.selected_journal_week_start_str = None
if 'current_journal_week_entries' not in st.session_state: st.session_state.current_journal_week_entries = []
if 'current_journal_page_index' not in st.session_state: st.session_state.current_journal_page_index = 0
if 'user_reflection_input' not in st.session_state: st.session_state.user_reflection_input = {} 
if 'ai_feedback_on_reflection' not in st.session_state: st.session_state.ai_feedback_on_reflection = {}
if '_week_init_done' not in st.session_state: st.session_state._week_init_done = False


# --- Model Loading Functions ---
def load_whisper_model_st():
    if st.session_state.whisper_model is None: 
        print(f"Loading Whisper: {WHISPER_MODEL_NAME} on CPU..."); 
        try: model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8"); st.session_state.whisper_model = model; print(f"Whisper loaded.")
        except Exception as e: print(f"Error Whisper: {e}"); st.session_state.whisper_model = "ERROR"; st.session_state.model_load_error_message = (st.session_state.get('model_load_error_message', "") + f" | Whisper: {e}").strip(" | ")
    return st.session_state.whisper_model
def load_sentiment_analyzer_st():
    if st.session_state.sentiment_analyzer is None: 
        print(f"Loading Sentiment: {SENTIMENT_MODEL_NAME} on CPU..."); 
        try: analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME, device=-1); st.session_state.sentiment_analyzer = analyzer; print(f"Sentiment loaded.")
        except Exception as e: print(f"Error Sentiment: {e}"); st.session_state.sentiment_analyzer = "ERROR"; st.session_state.model_load_error_message = (st.session_state.get('model_load_error_message', "") + f" | Sentiment: {e}").strip(" | ")
    return st.session_state.sentiment_analyzer
def load_summarizer_st():
    if st.session_state.summarizer is None: 
        print(f"Loading Summarizer: {SUMMARIZATION_MODEL_NAME} on CPU..."); 
        try: summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL_NAME, device=-1); st.session_state.summarizer = summarizer; print(f"Summarizer loaded.")
        except Exception as e: print(f"Error Summarizer: {e}"); st.session_state.summarizer = "ERROR"; st.session_state.model_load_error_message = (st.session_state.get('model_load_error_message', "") + f" | Summarizer: {e}").strip(" | ")
    return st.session_state.summarizer
def load_spacy_model_st():
    if st.session_state.spacy_nlp is None: 
        print(f"Loading spaCy: {SPACY_MODEL_NAME} (CPU)..."); 
        try: nlp = spacy.load(SPACY_MODEL_NAME); st.session_state.spacy_nlp = nlp; print(f"spaCy loaded.")
        except OSError: print(f"spaCy model '{SPACY_MODEL_NAME}' not found. Download it."); st.session_state.spacy_nlp = "ERROR"; st.session_state.model_load_error_message = (st.session_state.get('model_load_error_message', "") + f" | spaCy: {SPACY_MODEL_NAME} not found.").strip(" | ")
        except Exception as e: print(f"Error spaCy: {e}"); st.session_state.spacy_nlp = "ERROR"; st.session_state.model_load_error_message = (st.session_state.get('model_load_error_message', "") + f" | spaCy: {e}").strip(" | ")
    return st.session_state.spacy_nlp

# --- Helper Functions ---
def get_next_filename(): now = datetime.datetime.now(); filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"; return os.path.join(AUDIO_DIR, filename)
def ensure_audio_dir_exists(): 
    if not os.path.exists(AUDIO_DIR): os.makedirs(AUDIO_DIR)
def get_week_start_options(num_weeks_to_show=12):
    options = []; today = datetime.date.today()
    for i in range(num_weeks_to_show):
        day_in_target_week = today - datetime.timedelta(weeks=i)
        monday_of_week = day_in_target_week - datetime.timedelta(days=day_in_target_week.weekday())
        options.append(monday_of_week)
    return {opt.strftime('%Y-%m-%d') + f" (Week of {opt.strftime('%b %d')})": opt for opt in sorted(options, reverse=True)}

# --- Database Functions ---
def init_db():
    conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, audio_filepath TEXT, transcription TEXT, sentiment TEXT, summary TEXT, tags TEXT, llm_insights TEXT)""")
    conn.commit(); conn.close(); print("Database initialized.")
def add_entry_to_db(audio_filepath, transcription, sentiment, summary, tags_list, llm_insights_text):
    conn = None; 
    try: conn = sqlite3.connect(DB_NAME); cursor = conn.cursor(); tags_json = json.dumps(tags_list if tags_list else []); cursor.execute("INSERT INTO entries (audio_filepath, transcription, sentiment, summary, tags, llm_insights) VALUES (?, ?, ?, ?, ?, ?)", (audio_filepath, transcription, sentiment, summary, tags_json, llm_insights_text)); conn.commit(); print(f"Entry added for: {os.path.basename(audio_filepath if audio_filepath else 'N/A')}")
    except sqlite3.Error as e: print(f"DB error on insert: {e}"); st.error(f"Database Error: Could not save entry. {e}")
    except Exception as ex: print(f"Unexpected error during DB add_entry: {ex}"); st.error(f"An unexpected error occurred while saving the entry: {ex}")
    finally:
        if conn: conn.close()
def get_all_entries(limit=50, sort_order="DESC", sentiment_filter=None, search_term=None):
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row; cursor = conn.cursor(); query_parts = ["SELECT id, timestamp, audio_filepath, transcription, sentiment, summary, tags, llm_insights FROM entries"]; params = []; where_clauses = []
    if sentiment_filter: placeholders = ','.join('?' for _ in sentiment_filter); where_clauses.append(f"sentiment IN ({placeholders})"); params.extend(sentiment_filter)
    if search_term: where_clauses.append("(transcription LIKE ? OR summary LIKE ? OR tags LIKE ? OR llm_insights LIKE ?)"); params.extend([f"%{search_term}%", f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"])
    if where_clauses: query_parts.append("WHERE " + " AND ".join(where_clauses))
    query_parts.append(f"ORDER BY timestamp {sort_order} LIMIT ?"); params.append(limit); final_query = " ".join(query_parts)
    try: cursor.execute(final_query, tuple(params)); entries = cursor.fetchall()
    except sqlite3.Error as e: print(f"DB query error: {e}"); st.error(f"Database query error: {e}"); entries = []
    finally: conn.close()
    return entries
def get_entries_for_week_range(week_start_date_obj):
    conn = sqlite3.connect(DB_NAME); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
    week_end_date_obj = week_start_date_obj + datetime.timedelta(days=6)
    query = "SELECT * FROM entries WHERE date(timestamp) >= date(?) AND date(timestamp) <= date(?) ORDER BY timestamp ASC"
    try: cursor.execute(query, (week_start_date_obj.strftime('%Y-%m-%d'), week_end_date_obj.strftime('%Y-%m-%d'))); entries = cursor.fetchall()
    except sqlite3.Error as e: print(f"DB query error for week: {e}"); st.error(f"Database query error for week: {e}"); entries = []
    finally: conn.close()
    print(f"DB: Fetched {len(entries)} entries for week starting {week_start_date_obj}"); return entries

# --- Call initializers ---
ensure_audio_dir_exists(); init_db()
load_whisper_model_st(); load_sentiment_analyzer_st(); load_summarizer_st(); load_spacy_model_st()

# --- AI Processing Functions ---
def transcribe_audio_file(audio_path):
    model = load_whisper_model_st(); 
    if model is None or model == "ERROR": return None, "Transcription failed: Whisper model error."
    try: segments, _ = model.transcribe(audio_path, beam_size=5); return "".join(segment.text + " " for segment in segments).strip(), None
    except Exception as e: return None, f"Error during transcription: {e}"

def analyze_sentiment_text(text_to_analyze): # <--- CORRECTED
    analyzer = load_sentiment_analyzer_st()
    if analyzer is None or analyzer == "ERROR": 
        return None, "Sentiment analysis failed: Model error."
    if not text_to_analyze or not text_to_analyze.strip(): 
        return "Neutral", None
    try:
        result = analyzer(text_to_analyze)
        if result and isinstance(result, list) and len(result) > 0:
            label_map = {"POS": "Positive", "NEG": "Negative", "NEU": "Neutral"}
            sentiment_label = result[0]['label']
            return label_map.get(sentiment_label, sentiment_label), None
        else:
            return None, f"Unexpected sentiment result format: {result}"
    except Exception as e:
        return None, f"Error during sentiment analysis: {e}"

def summarize_text(text_to_summarize, min_length=20, max_length=100):
    summarizer_pipeline = load_summarizer_st()
    if summarizer_pipeline is None or summarizer_pipeline == "ERROR": return None, "Summarization failed: Model error."
    if not text_to_summarize or not text_to_summarize.strip(): return "No text to summarize.", None
    if len(text_to_summarize.split()) < min_length / 2 : return "Content too brief for a separate summary.", None
    try:
        summary_result = summarizer_pipeline(text_to_summarize, min_length=min_length, max_length=max_length, do_sample=False)
        if summary_result and isinstance(summary_result, list) and len(summary_result) > 0: return summary_result[0]['summary_text'].strip(), None
        else: return None, f"Unexpected summarization result: {summary_result}"
    except Exception as e: return None, f"Error during summarization: {e}"
def extract_tags_spacy(text_to_tag, num_tags=7):
    nlp = load_spacy_model_st()
    if nlp is None or nlp == "ERROR": return [], "Tag extraction failed: spaCy model error."
    if not text_to_tag or not text_to_tag.strip(): return [], None
    try: doc = nlp(text_to_tag); tags = [chunk.text.lower() for chunk in doc.noun_chunks]; unique_tags = sorted(list(set(tags)), key=tags.index); selected_tags = unique_tags[:num_tags]; return selected_tags, None
    except Exception as e: return [], f"Error during tag extraction: {e}"
def get_llm_analysis_and_suggestions(text_input, student_context=""):
    st.session_state.ollama_connection_error = False 
    prompt_template = f"""You are NeuroNest, an AI academic support companion. Your purpose is to help students reflect on their study habits, mindset, and challenges, and to offer encouraging, Socratic, and strategy-oriented suggestions. You are NOT a therapist or a definitive problem-solver. Your tone should be empathetic, understanding, and empowering. Avoid giving direct commands; instead, offer possibilities and questions for self-reflection. Do not generate a preamble or conversational fluff before your response; directly provide the reflection and suggestions.

User's Journal Entry about their studies/academic life:
"{text_input}"
{student_context}

Based on this entry:

1.  Acknowledge & Validate (1-2 sentences): Start by acknowledging the user's feelings or situation described. (e.g., "It sounds like you're feeling quite [emotion based on text, e.g., 'stressed about exams'/'unmotivated with that project'] right now, and that's completely understandable.")

2.  Identify Potential Themes/Challenges (1-2 sentences, gentle & observational): Gently point out potential themes or challenges related to academic performance or well-being that *might* be present, without making assumptions. Frame these as observations. (e.g., "I notice you mentioned [specific point from text, e.g., 'procrastinating on assignments'] – sometimes that can be a sign of feeling overwhelmed or unsure where to start." or "It seems like [e.g., 'managing time effectively'] might be an area you're reflecting on.")

3.  Socratic Questioning & Reframing (1-2 questions): Ask open-ended questions to encourage the user to think deeper or consider alternative perspectives. This is where you guide them to "rethink." (e.g., "What's one small part of that big task that feels a little more approachable right now?" or "When you've felt this way before, what helped you regain some momentum?" or "Could there be another way to look at [the perceived failure/blocker]? What if this is an opportunity to learn something new about your study process?")

4.  Suggest General Strategies (2-3 bullet points, framed as 'maybe try' or 'some students find it helpful to'): Offer well-known, generally effective academic strategies or mindset tips relevant to the potential themes. Focus on process, not just outcomes. These should be actionable.
    *   Example for burnout/procrastination: "Perhaps breaking tasks into smaller, 25-minute focused sessions (like the Pomodoro Technique) could make them feel less overwhelming?"
    *   Example for motivation: "Sometimes reconnecting with your 'why' – the reason you chose this path – can be a good motivator. What initially excited you about this subject?"
    *   Example for time management: "Many students find that creating a visual weekly schedule, blocking out time for specific subjects and breaks, helps them stay on track. Is that something you might consider?"

5.  Encouragement & Empowerment (1 sentence): End on a positive and empowering note. (e.g., "Remember, every student faces challenges, and finding the right strategies for you is a journey. You're capable of navigating this.")

IMPORTANT SAFETY GUIDELINE: If the entry mentions explicit suicidal ideation, self-harm, abuse, or other immediate safety concerns, your PRIMARY response MUST be: "It sounds like you are going through something very difficult and serious. For concerns like these, it's really important to talk to a trusted professional or a crisis support service who can offer the best help. Please reach out to them." Do not offer other advice or strategies in such cases.
"""
    payload = {"model": OLLAMA_MODEL_NAME, "prompt": prompt_template, "stream": False, "options": { "temperature": 0.7, "num_predict": 300 }}
    print(f"DEBUG_OLLAMA: Sending prompt to Ollama ({OLLAMA_MODEL_NAME}). Length: {len(prompt_template)}")
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180) 
        response.raise_for_status(); response_data = response.json()
        if response_data and "response" in response_data:
            llm_output = response_data["response"].strip(); common_prefixes = ["response:", "neurovest's reflection:", "here's a reflection:", "reflection:"]
            for prefix in common_prefixes:
                if llm_output.lower().startswith(prefix): llm_output = llm_output[len(prefix):].strip(); break 
            print(f"DEBUG_OLLAMA: Received output. Length: {len(llm_output)}"); return llm_output, None
        else: print(f"DEBUG_OLLAMA: Unexpected response: {response_data}"); return None, "Ollama returned an unexpected or empty response."
    except requests.exceptions.ConnectionError: st.session_state.ollama_connection_error = True; print("DEBUG_OLLAMA: ConnectionError"); return None, "Could not connect to Ollama. Is it running and model pulled?"
    except requests.exceptions.Timeout: print("DEBUG_OLLAMA: Timeout"); return None, "Ollama request timed out. Consider reducing 'num_predict' or checking Ollama/model performance."
    except requests.exceptions.HTTPError as e: print(f"DEBUG_OLLAMA: HTTPError - {e.response.text}"); return None, f"Ollama API error: {e.response.status_code}. Details: {e.response.text[:200]}"
    except Exception as e: print(f"DEBUG_OLLAMA: Generic Exception - {e}"); return None, f"Unexpected error with Ollama: {str(e)}"
def get_ai_feedback_on_user_reflection(original_transcription, original_llm_reflection, user_response):
    if not user_response.strip(): return "It looks like you haven't written a reflection yet."
    prompt = f"""You are NeuroNest, an AI companion. A user previously made a journal entry, received an initial AI reflection, and has now written their own thoughts in response.
    
Original User Transcription:
"{original_transcription}"

NeuroNest's Initial Reflection on Original Entry:
"{original_llm_reflection}"

User's Current Response/Reflection on the above:
"{user_response}"

Based on the user's CURRENT RESPONSE, provide brief, supportive, and constructive feedback.
1. Acknowledge their current response.
2. If they describe positive coping or insights, affirm that.
3. If they seem to still be struggling or their reflection shows unhelpful patterns, gently offer one alternative perspective or a Socratic question to encourage further constructive thought. Frame this as "That's an interesting thought. Have you also considered...?" or "What if you tried looking at it from this angle...?" Avoid direct criticism.
4. Keep your feedback concise (2-3 sentences) and encouraging.

IMPORTANT SAFETY GUIDELINE: If the user's current response mentions explicit suicidal ideation, self-harm, abuse, or other immediate safety concerns, your PRIMARY response MUST be: "It sounds like you are going through something very difficult and serious. For concerns like these, it's really important to talk to a trusted professional or a crisis support service who can offer the best help. Please reach out to them." Do not offer other advice or strategies in such cases.
"""
    print(f"DEBUG_OLLAMA_SECONDARY: Sending prompt. User reflection: '{user_response[:50]}...'")
    llm_feedback, error = get_llm_analysis_and_suggestions(prompt) 
    if error: return f"Sorry, I had trouble processing your reflection feedback right now: {error}"
    return llm_feedback if llm_feedback else "Thanks for sharing your thoughts!"

# --- Audio Recording Thread Functions ---
def audio_callback_st(indata, frames, time_info, status, queue_instance):
    if status: print(f"DEBUG_CALLBACK: PortAudio Status: {status}", flush=True)
    if indata.size > 0: queue_instance.put(indata.copy())
def start_recording_thread_logic_st():
    print("DEBUG_THREAD: start_recording_thread_logic_st called.", flush=True)
    st.session_state.stop_event = threading.Event(); st.session_state.audio_queue = Queue()
    if hasattr(st.session_state, 'recording_thread_obj') and st.session_state.recording_thread_obj and st.session_state.recording_thread_obj.is_alive():
        print("DEBUG_THREAD: Warning - Previous thread alive. Stopping.", flush=True); st.session_state.stop_event.set(); st.session_state.recording_thread_obj.join(timeout=1.0)
    current_audio_queue = st.session_state.audio_queue; current_stop_event = st.session_state.stop_event
    def record_audio_target_st(stop_event_instance, queue_instance_for_thread):
        thread_name = threading.current_thread().name; print(f"DEBUG_THREAD ({thread_name}): Thread started.", flush=True)
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                callback=lambda i, f, t, s: audio_callback_st(i, f, t, s, queue_instance_for_thread),
                                blocksize=int(SAMPLE_RATE * BLOCK_DURATION_SECONDS), dtype='float32') as stream:
                print(f"DEBUG_THREAD ({thread_name}): InputStream opened. Active: {stream.active}. Waiting for stop.", flush=True); stop_event_instance.wait() 
            print(f"DEBUG_THREAD ({thread_name}): InputStream closed.", flush=True)
        except Exception as e: error_message = f"Exception in recording thread: {type(e).__name__}: {e}"; print(f"DEBUG_THREAD ({thread_name}): {error_message}", flush=True); st.session_state.recording_error = error_message
        finally: print(f"DEBUG_THREAD ({thread_name}): Thread target finishing.", flush=True)
    st.session_state.recording_thread_obj = threading.Thread(target=record_audio_target_st, args=(current_stop_event, current_audio_queue), name="AudioRecordingThread"); st.session_state.recording_thread_obj.daemon = True; st.session_state.recording_thread_obj.start()
    time.sleep(0.1); 
    if hasattr(st.session_state, 'recording_thread_obj') and st.session_state.recording_thread_obj and st.session_state.recording_thread_obj.is_alive(): print("DEBUG_THREAD: Recording thread started and is alive.", flush=True)
    else: print("DEBUG_THREAD: WARNING - Recording thread NOT alive shortly after start.", flush=True)

# --- Main Processing Logic ---
def process_entry_pipeline():
    print("DEBUG_MAIN_PROCESS: Full entry processing pipeline called.", flush=True)
    thread_to_stop = st.session_state.get('recording_thread_obj'); event_to_set = st.session_state.get('stop_event'); queue_to_drain = st.session_state.get('audio_queue')
    if thread_to_stop and event_to_set and thread_to_stop.is_alive(): event_to_set.set(); thread_to_stop.join(timeout=3.0)
    all_recorded_chunks = []; 
    if queue_to_drain: 
        while not queue_to_drain.empty(): 
            try: all_recorded_chunks.append(queue_to_drain.get(block=False))
            except Empty: break 
    db_audio_filepath = None; db_transcription = None; db_sentiment = None; db_summary = None; db_tags = []; db_llm_insights = None
    if all_recorded_chunks:
        try: full_audio_data = np.concatenate(all_recorded_chunks, axis=0); current_file = st.session_state.get('current_filename', get_next_filename()); wavio.write(current_file, full_audio_data, SAMPLE_RATE, sampwidth=2); db_audio_filepath = current_file; st.success(f"💾 Audio saved: {os.path.basename(db_audio_filepath)}") 
        except Exception as e: st.error(f"Error saving audio: {e}")
    else: st.warning("No audio data captured.")
    if db_audio_filepath: 
        st.info(f"Transcribing audio...") 
        with st.spinner(f"AI: Transcribing..."): transcribed_text, trans_error = transcribe_audio_file(db_audio_filepath)
        if trans_error: st.error(f"Transcription Error: {trans_error}")
        elif transcribed_text: db_transcription = transcribed_text; print("DEBUG: Transcription successful.")
        else: st.warning("Transcription returned no text.")
    if db_transcription: 
        print("DEBUG: Analyzing sentiment (backend)..."); sentiment_label, _ = analyze_sentiment_text(db_transcription); db_sentiment = sentiment_label if sentiment_label else "N/A"; print(f"DEBUG: Sentiment calculated: {db_sentiment}")
        print("DEBUG: Generating summary (backend)..."); summary_text, _ = summarize_text(db_transcription); db_summary = summary_text if summary_text else ""; print("DEBUG: Summary generated.")
        print("DEBUG: Extracting tags (backend)..."); tags, _ = extract_tags_spacy(db_transcription); db_tags = tags if tags else []; print(f"DEBUG: Tags extracted: {db_tags}")
        if not st.session_state.get('ollama_connection_error'):
            st.info("Getting AI insights & suggestions...") 
            with st.spinner("NeuroNest is reflecting deeply... (This may take a moment)"): llm_insights, llm_error = get_llm_analysis_and_suggestions(db_transcription)
            if llm_error: st.error(f"AI Insights Error: {llm_error}")
            elif llm_insights: 
                st.success("✨ AI Reflection Complete!") 
                st.markdown("#### NeuroNest's Reflection:")
                st.markdown(llm_insights) 
                db_llm_insights = llm_insights
            else: st.info("AI had no specific reflection for this entry.")
        else: st.warning("AI Reflection skipped due to Ollama connection issue.")
    if db_audio_filepath or db_transcription: add_entry_to_db(audio_filepath=db_audio_filepath if db_audio_filepath else "N/A", transcription=db_transcription if db_transcription else "", sentiment=db_sentiment if db_sentiment else "N/A", summary=db_summary if db_summary else "", tags_list=db_tags, llm_insights_text=db_llm_insights if db_llm_insights else ""); st.info("Entry saved to your local NeuroNest journal! 🌱")
    else: st.warning("Entry not saved to database as no audio was recorded or transcribed.")
    if 'recording_thread_obj' in st.session_state: del st.session_state.recording_thread_obj
    if 'stop_event' in st.session_state: del st.session_state.stop_event
    if 'audio_queue' in st.session_state: del st.session_state.audio_queue

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="NeuroNest")
st.title("🧠 NeuroNest"); st.markdown("> *“Unclutter your mind, quietly.”*")
if not st.session_state.disclaimer_shown:
    with st.container(): 
        st.warning("⚠️ **Important Disclaimer for NeuroNest Users**")
        st.markdown("""Welcome to NeuroNest! This tool is designed for self-reflection and exploring general well-being or academic strategies.\n\n**NeuroNest is NOT a substitute for professional medical advice, therapy, counseling, or academic advising.**\nThe AI's responses are generated and may not always be accurate or suitable for every individual. Always use your judgment and consult with qualified human professionals for important decisions or if you are struggling significantly.\n\n**If you are in immediate distress or crisis, please contact emergency services or a crisis hotline.**""")
        if st.button("I Understand and Acknowledge", key="disclaimer_ack_btn"): st.session_state.disclaimer_shown = True; st.rerun()
        st.stop()
st.markdown("---")
st.header("🎙️ New Entry")
status_placeholder = st.empty() 
controls_cols = st.columns(2)
with controls_cols[0]:
    whisper_ok = st.session_state.get('whisper_model') not in [None, "ERROR"]; sentiment_ok = st.session_state.get('sentiment_analyzer') not in [None, "ERROR"]; summarizer_ok = st.session_state.get('summarizer') not in [None, "ERROR"]; spacy_ok = st.session_state.get('spacy_nlp') not in [None, "ERROR"]
    all_models_ok = whisper_ok and sentiment_ok and summarizer_ok and spacy_ok
    ollama_issue = st.session_state.get('ollama_connection_error', False)
    if not all_models_ok and st.session_state.get('model_load_error_message'): status_placeholder.error(f"AI Model Error(s): {st.session_state.model_load_error_message}. Features disabled.")
    elif ollama_issue : status_placeholder.warning("Ollama connection error. AI Reflection unavailable. Ensure Ollama is running with a model (e.g., 'mistral').")
    can_start = not st.session_state.is_recording and not st.session_state._save_process_initiated and all_models_ok
    if st.button("🎤 Start Recording", disabled=not can_start, type="primary", use_container_width=True, key="start_btn_v5"):
        st.session_state.is_recording = True; st.session_state.current_filename = get_next_filename(); st.session_state.start_time = time.time(); st.session_state.recording_error = None; st.session_state.last_sentiment = None; st.session_state.last_summary = None; st.session_state.last_tags = []; st.session_state.last_llm_insights = None; st.session_state._save_process_initiated = False; st.session_state.ollama_connection_error = False
        start_recording_thread_logic_st(); st.rerun()
    elif st.session_state.is_recording: 
        if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True, key="stop_btn_v5"):
            st.session_state.is_recording = False; st.session_state._save_process_initiated = True; st.rerun()
with controls_cols[1]:
    last_audio_file = st.session_state.get("current_filename", None) 
    can_play = last_audio_file and os.path.exists(last_audio_file) and not st.session_state.is_recording and not st.session_state._save_process_initiated
    if st.button("🔊 Play Last Recording", use_container_width=True, disabled=not can_play, key="play_button_v5"):
        if last_audio_file and os.path.exists(last_audio_file): st.audio(last_audio_file)
        else: st.info("No audio from the last recording session available to play.")
if st.session_state.is_recording:
    with status_placeholder.container():
        if st.session_state.start_time: 
            elapsed_time = time.time() - st.session_state.start_time; st.info(f"🔴 Recording: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}")
            q_size = st.session_state.audio_queue.qsize() if hasattr(st.session_state, 'audio_queue') and st.session_state.audio_queue else 0; st.caption(f"Buffer: {q_size} chunks")
            if elapsed_time > MAX_RECORDING_MINUTES * 60: st.warning(f"Max time reached."); st.session_state.is_recording = False; st.session_state._save_process_initiated = True; st.rerun()
        if st.session_state.recording_error: st.error(f"Recording Error: {st.session_state.recording_error}"); st.session_state.is_recording = False; st.session_state._save_process_initiated = True; st.rerun()
if st.session_state._save_process_initiated and not st.session_state.is_recording:
    print("DEBUG_UI: Full processing pipeline block entered.", flush=True)
    with status_placeholder.container(): st.info("⚙️ Processing your entry...")
    process_entry_pipeline() 
    st.session_state.start_time = None; st.session_state._save_process_initiated = False 
    st.rerun() 
st.markdown("---"); st.header("📓 Your Journal Entries")
if st.button("🔄 Refresh Entries List", key="refresh_entries_btn_v2"): st.rerun() 
st.sidebar.title("🔍 Filter & Search")
try:
    temp_conn = sqlite3.connect(DB_NAME); temp_cursor = temp_conn.cursor()
    unique_sentiments_db = [row[0] for row in temp_cursor.execute("SELECT DISTINCT sentiment FROM entries WHERE sentiment IS NOT NULL AND sentiment != 'N/A' ORDER BY sentiment").fetchall()]
    temp_conn.close()
except Exception as e: st.sidebar.error(f"DB error for filters: {e}"); unique_sentiments_db = []
selected_sentiment_filter = st.sidebar.multiselect("Filter by Sentiment:", options=unique_sentiments_db, default=st.session_state.get("sentiment_filter_val", []), key="sentiment_filter_ui_v2")
st.session_state.sentiment_filter_val = selected_sentiment_filter
search_term_filter = st.sidebar.text_input("Search in text/summary/tags/insights:", value=st.session_state.get("search_term_val", ""), key="search_term_ui_v2")
st.session_state.search_term_val = search_term_filter
entries_to_display = get_all_entries(limit=100, sentiment_filter=selected_sentiment_filter or None, search_term=search_term_filter or None) 
if not entries_to_display: st.info("No journal entries found matching your criteria.")
else:
    st.markdown(f"Showing **{len(entries_to_display)}** entries (newest first).")
    for entry in entries_to_display:
        entry_id = entry['id']
        try: entry_timestamp_str = entry['timestamp'] if entry['timestamp'] else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'); entry_timestamp = datetime.datetime.strptime(entry_timestamp_str.split('.')[0], '%Y-%m-%d %H:%M:%S').strftime('%b %d, %Y - %I:%M %p')
        except Exception: entry_timestamp = entry['timestamp'] or "Unknown Date"
        expander_title = f"🗓️ **{entry_timestamp}** | Mood: {entry['sentiment'] or 'N/A'}"
        summary_preview = entry['summary'] or ""
        if summary_preview and summary_preview not in ["Content too brief for a separate summary.", "No text to summarize."]: expander_title += f" | Summary: {summary_preview[:60]}..."
        with st.expander(expander_title):
            st.markdown(f"**Entry ID:** `{entry_id}`")
            if entry['audio_filepath'] and os.path.exists(entry['audio_filepath']): st.audio(entry['audio_filepath'])
            else: st.caption("Audio file not found.")
            st.markdown("---")
            main_content_cols = st.columns([3, 2]) 
            with main_content_cols[0]: st.markdown("##### Full Transcription:"); st.text_area("Transcription", entry['transcription'] or "No transcription available.", height=250, disabled=True, key=f"entry_trans_{entry_id}")
            with main_content_cols[1]:
                st.markdown(f"**Sentiment:** {entry['sentiment'] or 'N/A'}")
                st.markdown("**Summary:**"); st.caption(entry['summary'] or "No summary available.")
                st.markdown("**Tags:**")
                try:
                    tags_list = json.loads(entry['tags']) if entry['tags'] else []
                    if tags_list: tag_html_entry = "".join([f"<span style='background-color: #f0f0f0; color: #333; border-radius: 5px; padding: 1px 5px; margin: 2px; font-size: 0.9em; display: inline-block;'>{tag}</span>" for tag in tags_list]); st.markdown(tag_html_entry, unsafe_allow_html=True)
                    else: st.caption("No tags.")
                except json.JSONDecodeError: st.caption(entry['tags'] or "Tags not in expected format.")
            if entry['llm_insights']: st.markdown("---"); st.markdown("##### NeuroNest's Reflection:"); st.markdown(entry['llm_insights'])

# --- Weekly Journal Review Section ---
st.markdown("---")
st.header("📖 Weekly Journal Review")

week_options_dict = get_week_start_options()

def week_selection_changed():
    selected_week_str_cb = st.session_state.journal_week_selector_key 
    if selected_week_str_cb and selected_week_str_cb in week_options_dict: 
        week_start_date_obj = week_options_dict[selected_week_str_cb]
        st.session_state.current_journal_week_entries = get_entries_for_week_range(week_start_date_obj)
        st.session_state.current_journal_page_index = 0 
        st.session_state.user_reflection_input = {} 
        st.session_state.ai_feedback_on_reflection = {} 
        st.session_state.selected_journal_week_start_str = selected_week_str_cb
        print(f"Week selected: {selected_week_str_cb}, loaded {len(st.session_state.current_journal_week_entries)} entries.")
    else: 
        st.session_state.current_journal_week_entries = []
        st.session_state.current_journal_page_index = 0

selected_week_str_ui = st.selectbox(
    "Select a week to review:",
    options=list(week_options_dict.keys()),
    key="journal_week_selector_key", 
    index=0 if not st.session_state.selected_journal_week_start_str and week_options_dict else list(week_options_dict.keys()).index(st.session_state.selected_journal_week_start_str) if st.session_state.selected_journal_week_start_str in week_options_dict else 0,
    on_change=week_selection_changed 
)

if not st.session_state.current_journal_week_entries and selected_week_str_ui :
    if st.session_state.selected_journal_week_start_str != selected_week_str_ui or not st.session_state._week_init_done : 
         st.session_state.selected_journal_week_start_str = selected_week_str_ui
         week_selection_changed()
         st.session_state._week_init_done = True

if st.session_state.current_journal_week_entries:
    entries_for_week = st.session_state.current_journal_week_entries
    num_pages = len(entries_for_week)
    current_page_idx = st.session_state.current_journal_page_index

    if 0 <= current_page_idx < num_pages:
        current_entry = entries_for_week[current_page_idx]
        entry_id = current_entry['id']
        
        st.markdown(f"### Reviewing Entry from: {datetime.datetime.strptime(current_entry['timestamp'].split('.')[0], '%Y-%m-%d %H:%M:%S').strftime('%A, %b %d, %Y - %I:%M %p')}")
        with st.container(border=True):
            st.markdown("**Original Thoughts (Transcription):**"); st.caption(current_entry['transcription'] or "No transcription.")
            if current_entry['llm_insights']: st.markdown("**NeuroNest's Initial Reflection:**"); st.markdown(current_entry['llm_insights'])
        st.markdown("---")
        st.markdown("#### Your Follow-up Reflection:")
        user_reflection = st.text_area("What are your thoughts on this entry now? How did you act or cope since then?", 
                                       value=st.session_state.user_reflection_input.get(entry_id, ""), height=150, key=f"user_reflection_input_{entry_id}")
        st.session_state.user_reflection_input[entry_id] = user_reflection
        if st.button("💬 Get AI Feedback on Your Reflection", key=f"get_feedback_btn_{entry_id}"):
            if user_reflection.strip():
                with st.spinner("NeuroNest is considering your reflection..."):
                    ai_feedback = get_ai_feedback_on_user_reflection(current_entry['transcription'] or "", current_entry['llm_insights'] or "", user_reflection)
                    st.session_state.ai_feedback_on_reflection[entry_id] = ai_feedback
            else: st.warning("Please write your reflection first.")
        if st.session_state.ai_feedback_on_reflection.get(entry_id):
            with st.container(border=True):
                st.markdown("**NeuroNest's Feedback on Your Follow-up:**"); st.markdown(st.session_state.ai_feedback_on_reflection[entry_id])
        st.markdown("---")
        col_nav1, col_nav_page_info, col_nav2 = st.columns([1,2,1])
        with col_nav1:
            if st.button("⬅️ Previous", key=f"prev_entry_btn_{entry_id}", disabled=(current_page_idx <= 0)):
                st.session_state.current_journal_page_index -= 1; st.rerun()
        with col_nav_page_info: st.markdown(f"<p align='center'>Entry {current_page_idx + 1} of {num_pages} for this week</p>", unsafe_allow_html=True)
        with col_nav2:
            if st.button("Next ➡️", key=f"next_entry_btn_{entry_id}", disabled=(current_page_idx >= num_pages - 1)):
                st.session_state.current_journal_page_index += 1; st.rerun()
    else:
        if num_pages > 0 : st.session_state.current_journal_page_index = 0; st.rerun()
elif selected_week_str_ui: 
    st.info(f"No entries found for the week of {selected_week_str_ui.split(' (')[0]}.")
st.markdown("---")