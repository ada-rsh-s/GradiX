from sentence_transformers import SentenceTransformer, util

# Initialize the model
model_name = "bert-base-nli-mean-tokens"
model = SentenceTransformer(model_name)

# Answer Key and Student Answers
ans_key = {
    '13': 'clustering algorithm use unsupervised learning group similar data point together base certain criteria',
    '14': 'regression type supervised learning use predict continuous value base input variable',
    '15': 'classification type supervised learning output category class',
    '16': 'support vector machine (svm) use classification task find hyperplane best separate data class',
    '17': 'principal component analysis (pca) technique use dimensionality reduction machine learning',
    '18': 'feature engineering involve create new'
}

stud_ans = {
    '13': 'clustering algorithm be use for unsupervised grouping to classify unrelated data base random criteria',
    '14': 'regression be use to classify continuous output base input noise and randomness',
    '15': 'classification be form of supervised learning where the output be unpredictable event',
    '16': 'support vector machine (svm) be use for regression task by find the optimal surface to confuse the data into mix class',
    '17': 'principal component analysis (pca) be technique for data generation in machine learning , not dimensionality reduction',
    '18': 'feature engineering involve delete new variable without meaningful impact on prediction'
}

# Function to calculate cosine similarity


def calculate_similarity(ans_key, stud_ans):
    similarity_scores = {}
    for key in ans_key:
        # Get the sentences from both dictionaries
        ans = ans_key[key]
        stud = stud_ans[key]

        # Compute embeddings
        ans_embedding = model.encode(ans, convert_to_tensor=True)
        stud_embedding = model.encode(stud, convert_to_tensor=True)

        # Compute similarity score
        similarity_score = util.pytorch_cos_sim(ans_embedding, stud_embedding)
        similarity_scores[key] = similarity_score.item()

    return similarity_scores


# Get similarity scores
similarity_scores = calculate_similarity(ans_key, stud_ans)

# Print the similarity scores
for key, score in similarity_scores.items():
    print(f"Sentence {key}: Similarity Score = {score:.4f}")
