import streamlit as st
import os
import tempfile
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from scanner import process_image
from segregator import segregate_ocr_text
from tokenizer import preprocess_answers
from similarity_scoring import sentences_dict, compute_similarity_and_marks

# Load environment variables from .env file
load_dotenv()

# Get API key and model name from environment variables
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-pro-vision")

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Set up Streamlit page
tab1, tab2 = st.tabs(["Smart Exam Grading", "Similarity Scoring"])

with tab1:
    st.title("📄 Smart Exam Grading")

    # File uploader
    uploaded_files = st.file_uploader("Upload Answer Sheets", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    def process_uploaded_images(files):
        if not files:
            st.warning("⚠️ Please upload at least one image.")
            return

        results = {}  # Store processed results

        for i, file in enumerate(files, start=1):
            with st.spinner(f"📸 Processing Image {i}/{len(files)}: {file.name}..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                # Step 1: Extract text from image
                extracted_text = process_image(temp_file_path)
                print(extracted_text)

                processed_answers = {}
                if extracted_text:
                    # Step 2: Segregate into Q&A format
                    qa_dict = segregate_ocr_text(extracted_text)

                    # Step 3: Tokenize & Lemmatize answers
                    processed_answers = preprocess_answers(qa_dict)

                    # Store results
                    results[file.name] = processed_answers

            # Show final answers only after processing
            st.success(f"✅ Processed: {file.name}")
            st.write("### 📝 Final Processed Answers")
            for q, ans in processed_answers.items():
                st.write(f"**Q{q}:** {ans}")

    # Button to process uploaded images
    if st.button("📥 Submit & Process", key="grading_submit"):
        process_uploaded_images(uploaded_files)

with tab2:
    st.title("Hybrid Sentence Similarity Scoring")
    st.write("This application combines model-based and Gemini AI scoring to evaluate student answers.")

    # Create a function to evaluate similarity with Gemini
    def evaluate_similarity_with_gemini(correct_answer, student_answer):
        try:
            # Set up the model
            model = genai.GenerativeModel(MODEL_NAME)
            
            # Create the prompt
            prompt = f"""
            I need you to evaluate the similarity between these two sentences:

CORRECT ANSWER: "{correct_answer}"

STUDENT ANSWER: "{student_answer}"

Please provide:

A semantic similarity score as a percentage from 0 to 100.

The score should reflect how much of the correct answer's meaning is captured in the student's answer.

The score must increase gradually as the student includes more concepts or ideas from the correct answer — this is partial scoring, not all-or-nothing.

Do not give high marks just because keywords are similar. Only increase marks when the student expresses the correct ideas, even in different words.

If the student partially explains the correct answer, give a medium score (e.g., 30–70%) depending on how much is covered.

If the meaning is mostly accurate, give a high score (e.g., 80–100%).

If the meaning is completely wrong or missing, then give a low score (e.g., 0–20%).

A brief explanation focusing on what parts were correct and what concepts were missing. Explain why the score was given, as if you're providing helpful, constructive feedback to a student.

A list of the key missing concepts from the correct answer that the student did not include or misunderstood.
            """
            
            # Generate response
            response = model.generate_content(prompt)
            
            # Parse the response
            response_text = response.text
            
            # Log the full response for debugging (only during development)
            # st.write("### Full Gemini Response")
            # st.code(response_text)

            # Extract score
            if "SCORE:" in response_text:
                score_text = response_text.split("SCORE:")[1].split("\n")[0].strip()
            elif "**Semantic Similarity Score:**" in response_text:
                score_text = response_text.split("**Semantic Similarity Score:**")[1].split("%")[0].strip()
            elif "Semantic Similarity Score:" in response_text:
                score_text = response_text.split("Semantic Similarity Score:")[1].split("%")[0].strip()
            else:
                score_text = "0"
                
            try:
                score = float(score_text)
            except ValueError as e:
                st.error(f"Error parsing score: {str(e)}")
                score = 0
                
            # Extract explanation
            if "EXPLANATION:" in response_text:
                explanation = response_text.split("EXPLANATION:")[1].split("MISSING CONCEPTS:")[0].strip()
            elif "**Explanation:**" in response_text:
                explanation = response_text.split("**Explanation:**")[1].split("**Missing Concepts:**")[0].strip() if "**Missing Concepts:**" in response_text else response_text.split("**Explanation:**")[1].strip()
            else:
                explanation = "No explanation provided."
            
            # Extract missing concepts
            if "MISSING CONCEPTS:" in response_text:
                missing_concepts = response_text.split("MISSING CONCEPTS:")[1].strip()
            elif "**Missing Concepts:**" in response_text:
                missing_concepts = response_text.split("**Missing Concepts:**")[1].strip()
            else:
                missing_concepts = "None identified."
            
            return {
                "Score": score,
                "Explanation": explanation,
                "Missing Concepts": missing_concepts,
                "Full Response": response_text
            }
        except Exception as e:
            st.error(f"Error in Gemini evaluation: {str(e)}")
            return {
                "Score": 0,
                "Explanation": f"Error: {str(e)}",
                "Missing Concepts": "Evaluation failed",
                "Full Response": "Error occurred"
            }

    # Function that combines both scoring methods
    def hybrid_similarity_scoring(correct_answer, student_answer):
        # Get model-based scoring
        model_results = compute_similarity_and_marks(correct_answer, student_answer)
        model_score = model_results["Marks Percentage"]
        
        # Get Gemini-based scoring
        with st.spinner("Getting Gemini evaluation..."):
            gemini_results = evaluate_similarity_with_gemini(correct_answer, student_answer)
        gemini_score = gemini_results["Score"]
        
        # Calculate average score
        average_score = (model_score + gemini_score) / 2
        
        return {
            "Model Score": model_score,
            "Gemini Score": gemini_score,
            "Average Score": average_score,
            "Model Details": model_results,
            "Gemini Details": gemini_results
        }

    # Display the sentence pairs
    st.subheader("Available Sentence Pairs")
    topic_options = list(sentences_dict.keys())
    topic_display = []

    for topic in topic_options:
        correct, student = sentences_dict[topic]
        topic_display.append({
            "Topic": topic.capitalize(),
            "Correct Answer": correct,
            "Student Answer": student
        })

    df = pd.DataFrame(topic_display)
    st.dataframe(df)

    # Add topic selection and evaluation button
    st.subheader("Evaluate Similarity")
    selected_topic = st.selectbox("Select a topic to evaluate", options=topic_options)

    # Show the selected pair
    if selected_topic:
        correct_answer, student_answer = sentences_dict[selected_topic]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Correct Answer:**\n{correct_answer}")
        with col2:
            st.warning(f"**Student Answer:**\n{student_answer}")

    # Evaluate button    
    if st.button("Evaluate Similarity"):
        if not API_KEY:
            st.error("API Key not found in .env file. Please add your API_KEY to the .env file.")
        else:
            correct_answer, student_answer = sentences_dict[selected_topic]
            
            with st.spinner("Calculating combined similarity scores..."):
                result = hybrid_similarity_scoring(correct_answer, student_answer)
                
                # Display final result only
                st.subheader("Final Result")
                
                # Show final score
                average = result['Average Score']
                if average >= 80:
                    st.success(f"Final Score: {average:.1f}%")
                elif average >= 50:
                    st.warning(f"Final Score: {average:.1f}%")
                else:
                    st.error(f"Final Score: {average:.1f}%")
                
                # Show explanation
                st.markdown("### Explanation")
                st.write(result["Gemini Details"]["Explanation"])

    # Add option to evaluate all topics
    if st.button("Evaluate All Topics"):
        if not API_KEY:
            st.error("API Key not found in .env file. Please add your API_KEY to the .env file.")
        else:
            all_results = {}
            progress = st.progress(0)
            
            for i, topic in enumerate(topic_options):
                correct_answer, student_answer = sentences_dict[topic]
                with st.spinner(f"Evaluating {topic} ({i+1}/{len(topic_options)})..."):
                    all_results[topic] = hybrid_similarity_scoring(correct_answer, student_answer)
                    progress.progress((i+1)/len(topic_options))
            
            # Display all results in a table
            results_data = []
            for topic, result in all_results.items():
                results_data.append({
                    "Topic": topic.capitalize(),
                    "Model Score": f"{result['Model Score']:.1f}%",
                    "Gemini Score": f"{result['Gemini Score']:.1f}%",
                    "Final Score": f"{result['Average Score']:.1f}%",
                    "Analysis": result["Gemini Details"]["Explanation"]
                })
            
            results_df = pd.DataFrame(results_data)
            st.subheader("All Evaluation Results")
            st.dataframe(results_df)
            
            # Visualization of scores
            st.subheader("Score Comparison")
            
            # Prepare data for visualization
            chart_data = pd.DataFrame({
                "Topic": [],
                "Score": [],
                "Method": []
            })
            
            for topic, result in all_results.items():
                # Add model score
                chart_data = pd.concat([chart_data, pd.DataFrame({
                    "Topic": [topic.capitalize()],
                    "Score": [result["Model Score"]],
                    "Method": ["Model"]
                })])
                
                # Add Gemini score
                chart_data = pd.concat([chart_data, pd.DataFrame({
                    "Topic": [topic.capitalize()],
                    "Score": [result["Gemini Score"]],
                    "Method": ["Gemini"]
                })])
                
                # Add Average score
                chart_data = pd.concat([chart_data, pd.DataFrame({
                    "Topic": [topic.capitalize()],
                    "Score": [result["Average Score"]],
                    "Method": ["Average"]
                })])
            
            # Create grouped bar chart
            st.bar_chart(chart_data, x="Topic", y="Score", color="Method")
