# 🧠 NeuroNest – Your Offline AI Voice Journal

> *"Unclutter your mind, quietly."*

NeuroNest is a privacy-first, offline, AI-powered voice journaling application built with Python. It allows users to verbally "brain dump" their thoughts, which are then instantly transcribed, analyzed for sentiment, summarized, tagged by theme, and reflected upon by a local AI companion.

**All processing and data storage happen 100% on your local machine.** No data ever leaves your computer, ensuring complete privacy. There are no APIs, no subscriptions, and no hidden costs.

**catch the demo here**
https://drive.google.com/file/d/1U3BCKEz06mOel_k_WvY-YmSp6rxxE7s0/view?usp=sharing
>>>please watch at 720p


---

## ✨ Features

-   🎤 **Voice-to-Text Journaling**: Speak your thoughts, and NeuroNest transcribes them using `faster-whisper`.
-   😊 **Sentiment Analysis**: Each entry is automatically analyzed for its emotional tone using a BERT-based model.
-   📜 **AI Summarization**: Long thoughts are condensed into a concise summary using a BART model.
-   🏷️ **Automatic Tagging**: Key topics and themes are extracted as tags using `spaCy`.
-   🤔 **AI Reflection (Socratic Companion)**: A locally-run Large Language Model (via Ollama) provides gentle, supportive reflections and Socratic questions on your entries.
-   💾 **100% Offline & Private**: All data is stored in a local SQLite database (`neuronest.db`).
-   📓 **Journal Dashboard**: A clean UI built with Streamlit to browse, review, and filter past entries.
-   📖 **Weekly Review**: An interactive "book" view to reflect on entries from a specific week.

---

## 🧱 Tech Stack

| Component                | Technology / Library                                      |
| ------------------------ | --------------------------------------------------------- |
| **UI Framework**         | `Streamlit`                                               |
| **Audio Processing**     | `sounddevice`, `wavio`                                    |
| **Speech-to-Text**       | `faster-whisper`                                          |
| **Sentiment Analysis**   | `transformers` (`finiteautomata/bertweet-base...`)        |
| **Summarization**        | `transformers` (`facebook/bart-large-cnn`)                |
| **NLP / Tagging**        | `spaCy`                                                   |
| **AI Reflection**        | `Ollama` + `mistral` (or other local LLM)                 |
| **Database**             | `sqlite3`                                                 |

---

## ⚙️ Setup & Installation

### Prerequisites

1.  **Python**: Python 3.9+
2.  **Ollama**: [Ollama](https://ollama.ai/) installed and running.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/NeuroNest.git
    cd NeuroNest
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy language model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Download a Large Language Model via Ollama:**
    ```bash
    ollama pull mistral
    ```
    *(Ensure the Ollama application/service is running in the background.)*

### Running the Application

1.  Make sure your Python virtual environment is activated and the Ollama service is running.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---
