import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_answers(answers_dict):
    """Tokenize, remove stopwords, and lemmatize answers."""
    processed_answers = {}
    for q_num, answer in answers_dict.items():
        doc = nlp(answer)
        processed_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        processed_answers[q_num] = " ".join(processed_tokens)
    
    return processed_answers  # Return processed answers dictionary
