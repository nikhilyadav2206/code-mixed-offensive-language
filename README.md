# Code-Mixed Offensive Language Detection

A Streamlit-based web application for detecting **offensive / non-offensive content** in **multilingual and code-mixed text** (e.g., Hinglish, Tanglish, Benglish, Telugu-English, etc.).  

The project uses a **transformer-based multilingual model (XLM-R)** to classify user input and tweets as *Offensive* or *Non-Offensive* with confidence scores.


---

## ✨ Features

- 🔤 **Multilingual & Code-Mixed Support**  
  Detects offensive language in:
  - Hinglish (Hindi–English)
  - Tanglish (Tamil–English)
  - Telugu–English
  - Benglish (Bengali–English)
  - Gujarati–English  
  and plain English / Indian languages.

- 🤖 **Transformer-Based Model**  
  Uses a fine-tuned multilingual model ( XLM-RoBERTa) for context-aware classification.

- 🖥️ **Streamlit Web UI**  
  - Simple text box to enter any sentence and get:
    - Predicted label: `Offensive` / `Non-Offensive`
    - Confidence scores for each class
  - Dedicated pages (if enabled) for:
    - Manual prediction
    - Twitter handle analysis / moderation

- 🐦 **Twitter Integration  
  - Fetch recent tweets from a given handle  
  - Run them through the model  
  - Show predictions + option to flag/remove(demo only) 

---

## 🧱 Tech Stack

- **Language:** Python 3.x  
- **Frontend / UI:** [Streamlit](https://streamlit.io/)  
- **NLP / Deep Learning:**  
  - [PyTorch](https://pytorch.org/)  
  - [Hugging Face Transformers](https://huggingface.co/transformers/)  
- **Data / Utils:**  
  - Pandas, NumPy, Scikit-learn  
- **APIs (Optional):**  
  - Twitter API (for fetching tweets)  



---

## 🗂️ Project Structure 


```bash
.
.
├── app/
│   ├── app.py               # Main Streamlit UI
│   ├── utils.py             # Preprocessing, cleaning, tokenization helpers
│   ├── dashboard.py         # Dashboard / analytics components
│   ├── config.py            # All configuration (paths, thresholds, model info)
│   └── predict.py           # CLI-based text prediction script
│
├── data/
│   ├── sample_inputs/           # Raw example text inputs
│   └── sample_inputs_encoded/   # Encoded / preprocessed samples
│
├── train.py                 # Training / fine-tuning script for the model
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
