# Text Similarity Scoring System

This application compares preprocessed correct answers with student answers and calculates similarity scores using two methods:
1. Model-based scoring using sentence transformers
2. AI-powered evaluation using Google's Gemini API

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up the Gemini API key as an environment variable:

   **Windows (PowerShell):**
   ```
   $env:GOOGLE_API_KEY="your-api-key"
   ```

   **Windows (Command Prompt):**
   ```
   set GOOGLE_API_KEY=your-api-key
   ```

   **Linux/Mac:**
   ```
   export GOOGLE_API_KEY="your-api-key"
   ```

3. Run the application:
   ```
   cd src
   streamlit run streamlit_app.py
   ```

## Features

- View preprocessed sentence pairs (stopwords removed, lemmatized, punctuation removed)
- Model-based scoring using:
  - Bi-Encoder similarity
  - Cross-Encoder similarity
  - NLI contradiction detection
- Gemini AI evaluation providing:
  - Numerical score
  - Detailed feedback
  - Analysis of missing concepts

## Data

The application comes with sample preprocessed sentence pairs for concepts including:
- Compiler
- Database
- Operating System
- Algorithm
- Encryption

## Adding New Sentence Pairs

To add new sentence pairs, edit the `sentences_dict` in `src/similarity_scoring.py`. 