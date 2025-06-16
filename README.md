# Scripture AI RAG – Intelligent Bible Companion

Scripture AI RAG is an intelligent Retrieval-Augmented Generation (RAG) pipeline designed to provide accurate, scripture-based answers and personalized Bible reading plans. Powered by LangChain, Chroma, and an LLM (e.g., OpenAI, Anthropic), it enables deep, contextual interaction with the Bible—without injecting personal interpretation.

---

## Features

- Semantic Search: Retrieve Bible verses using topic, emotion, or question (e.g., “fear”, “forgiveness”, “Why did Jesus weep?”).
- Personalized Bible Reading Plans: Generate daily reading plans based on timeframes (e.g., 3 months), themes (e.g., grace), or study goals.
- Conversational Memory: Context-aware follow-up questions using previous dialogue history.
- Strict Biblical Grounding: All answers are based strictly on scripture passages—no speculation or extra-biblical commentary.

- Context-aware semantic search over Bible passages using Sentence Transformers.
- Google Gemini AI-powered question answering and plan generation strictly grounded in scripture.
- History-aware conversational retrieval for natural dialogue.
- Streamlit-based interactive web UI for Bible Q&A and reading plan generation.

---

## Architecture Overview

```
User Query
    ↓
ChatPromptTemplate (rephrase question in context)
    ↓
History-aware Retriever (Chroma vector store, k=3)
    ↓
LLM Chain (Google Gemini AI + scripture-grounded prompt)
    ↓
Generated Answer with Bible citations
```

---

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Ekuaappiah/bible-assistant-rag.git
cd scripture-ai-rag
```

### 2. Create and activate a virtual/env environment

Using `venv` (Python 3.10+ required):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


Create a `.env` file at the project root with your Google Gemini API credentials:

```env
GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Here's the content converted to **Markdown format**:

---

##  4: Run the Streamlit Web App with Bible File Setup

### 1. Create a `data/` Folder

In your project directory, create a folder called `data`:

```bash
mkdir data
```

### 2. Add Your Bible Text File

Place your Bible file (e.g., `kjv.txt`) into the `data/` folder so your directory looks like this:

```
your_project/
├── app.py
├── data/
│   └── kjv.txt
```

### 3. Update the File Path in Your Code

In `app.py`, update the file loading path to reference the `data` directory:

```python
import os

# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define path to Bible file inside /data
file_path = os.path.join(current_dir, 'data', 'kjv.txt')
```

### 4. **Run the Streamlit web app**

```bash
streamlit run app.py
```

The app provides an interactive UI for Bible Q&A, reading plans, and conversational history.

---

## Prompt Philosophy

- Answers strictly derived from scripture.
- No personal or doctrinal opinions.
- Every answer includes chapter and verse references.
- If no clear scriptural answer exists, respond:  
  `"The Bible does not provide a clear answer to that."`

---

## Future Work

- Add thematic and emotional search capabilities.
- Support multiple Bible translations.
- Develop a user-friendly UI for devotional use.
- Export reading plans as PDF or calendar events.

---

## License

This project is under a license for educational and personal demonstration purposes only.

All third-party libraries (e.g., Hugging Face models) are used under their respective open source licenses.

---

## Author

Built by @EkuaAppiah

---

## Medium Blog (Coming Soon)

A full Medium article will be published soon to explain the architecture, challenges, and design decisions behind this project.

