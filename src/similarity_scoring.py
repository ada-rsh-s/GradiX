import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load models
bi_encoder_model = SentenceTransformer('sentence-transformers/nli-roberta-base-v2')
cross_encoder_stsb = CrossEncoder('cross-encoder/stsb-roberta-base')
cross_encoder_nli = CrossEncoder('cross-encoder/nli-distilroberta-base')

# Dictionary of sentence pairs (Correct Answer, Student Answer)
sentences_dict = {
    "compiler": (
        "compiler software program translate highlevel programming code machine code allow computer processor execute instruction efficiently",
        "compiler convert highlevel code binary file run computer"  # Partially correct, "binary file" is vague
    ),
    "database": (
        "database organize collection structure data store electronically allow efficient retrieval management modification",
        "database organize data store paper allow fast access"  # Wrong, databases are electronic, not paper-based
    ),
    "operating_system": (
        "operating system system software manage computer hardware software resource provide essential service computer program",
        "operating system system software manage computer hardware software resource provide essential service computer program"  # Ditto, fully correct
    ),
    "algorithm": (
        "algorithm stepbystep procedure formula solve problem use programming data processing",
        "algorithm random step solve problem programming"  # Wrong, algorithms are not random
    ),
    "encryption": (
        "encryption ensure data security convert information unreadable format accessible decryption key",
        "encryption protect data make readable everyone"  # Wrong, encryption does the opposite
    ),
    "machine_learning": (
        "machine learning subfield artificial intelligence focus create algorithm enable system learn data improve performance time without explicit programming model train dataset perform task classification regression clustering find pattern relation input data preprocessing normalization remove outlier feature scaling ensure model accuracy evaluation metric accuracy precision recall validate model performance overfit underfit address regularization crossvalidation apply field finance healthcare ecommerce fraud detection recommendation system diagnosis",
        "machine learning part artificial intelligence use algorithm learn data improve time model train dataset classification preprocessing important accuracy metric evaluate performance overfit issue fix regularization use finance healthcare"  # Partially correct, misses some details but mostly accurate
    ),
    "network": (
        "network system connect computer device share resource data communication protocol enable efficient transfer information",
        "network group computer connect share file slow speed"  # Partially correct, misses protocols and efficiency
    ),
    "firewall": (
        "firewall security system monitor control network traffic base predefined rule protect system unauthorized access",
        "firewall hardware block internet access completely"  # Wrong, firewalls don’t block all access, they filter
    ),
    "virtual_machine": (
        "virtual machine software emulate physical computer allow run multiple operating system instance single hardware",
        "virtual machine software emulate physical computer allow run multiple operating system instance single hardware"  # Ditto, fully correct
    ),
    "cloud_computing": (
        "cloud computing deliver computing service storage database networking internet scalable flexible resource management",
        "cloud computing store data physical server house"  # Wrong, cloud uses remote servers, not physical houses
    ),
    "api": (
        "api application programming interface enable software component communicate share functionality data define rule interaction",
        "api program make software run fast"  # Wrong, APIs are about communication, not speed
    ),
    "blockchain": (
        "blockchain decentralize distribute ledger record transaction secure tamperproof manner use cryptography ensure data integrity",
        "blockchain centralize database store transaction easy modify"  # Wrong, blockchain is decentralized and tamperproof
    ),
    "big_data": (
        "big data large complex dataset exceed traditional processing capability require advance tool analyze pattern trend",
        "big data large dataset process simple spreadsheet find pattern"  # Partially correct, misses complexity and tools
    ),
    "artificial_intelligence": (
        "artificial intelligence field computer science focus create system mimic human intelligence include task learn reason problem solve perceive environment adapt change core component include machine learn deep learn natural language processing computer vision robotics system design process large dataset identify pattern make prediction decision application span industry healthcare finance education transportation entertainment example diagnose disease predict market trend automate customer service selfdriving car game ai challenge include ensure ethical use handle bias data maintain privacy security require significant compute resource advance algorithm neural network reinforcement learn supervise learn unsupervised learn key technique improve performance system over time",
        "artificial intelligence study computer science build machine think human perform task learn reason solve problem process data find pattern predict outcome use machine learn deep learn natural language processing computer vision robotics apply healthcare finance education transportation entertainment example diagnose disease predict trend automate service drive car play game issue include ethic bias privacy security need lot compute power algorithm neural network reinforcement learn supervise learn unsupervised learn help system get better"  # Mostly correct, slightly less detailed but accurate
    ),
    "cybersecurity": (
        "cybersecurity practice protect system network device program digital attack aim prevent unauthorized access data breach theft damage include technique encrypt data implement firewall monitor traffic detect intrusion respond threat key concept include vulnerability assessment penetration test risk management compliance regulation protect asset business individual common threat include malware phishing ransomware ddos attack social engineer require tool antivirus intrusion detection system secure code practice train employee awareness defend system hacker cybercriminal target sensitive information financial record personal data intellectual property field evolve rapidly counter advance attack method",
        "cybersecurity method secure computer network device software physical attack stop hacker access data use encryption firewall monitor detect threat deal vulnerability test risk manage follow rule protect business individual threat malware phishing ransomware ddos attack social engineer tool antivirus intrusion system secure code train employee fight hacker aim steal money data property field change fast keep attack"  # Partially correct, misses some details and introduces "physical attack" inaccurately
    ),
    "data_science": (
        "data science interdisciplinary field combine statistic mathematics computer science domain knowledge extract insight large complex dataset involve process collect clean transform analyze visualize data use tool programming language python r sql statistical method machine learn algorithm deep learn technique goal uncover pattern trend make datadriven decision application include business intelligence predictive analytic scientific research healthcare fraud detection recommendation system ecommerce step include data preprocess feature engineer model build evaluation deployment challenge handle miss data ensure quality scale computation deal highdimensional data ethic privacy concern critical success",
        "data science field study math computer science get info big dataset collect clean analyze data use python r sql statistic machine learn deep learn find pattern trend help business science healthcare fraud detection recommendation system ecommerce step preprocess feature model evaluate deploy issue miss data quality scale ethic privacy"  # Mostly correct, slightly simplified, omits some specifics like visualization
    ),
    "internet_of_things": (
        "internet thing network connect physical device embed sensor software collect exchange data via internet enable automation remote control monitor example smart home device wearable fitness tracker industrial sensor smart city system agriculture monitor healthcare device technology rely protocol wifi bluetooth zigbee cellular network cloud computing process data challenge include secure data privacy manage scale interoperability device energy efficiency application improve efficiency convenience safety field home automation healthcare industry agriculture transportation security concern arise device vulnerability hacking data breach require encryption authentication access control ensure trust system",
        "internet thing group device connect internet sensor software share data automate control monitor smart home wearable industrial sensor smart city agriculture healthcare use wifi bluetooth zigbee cloud computing process data issue secure privacy scale interoperability energy efficiency help home healthcare industry agriculture transportation security problem hacking data breach need encryption authentication access control"  # Ditto, fully correct
    ),
    "quantum_computing": (
        "quantum computing advance computation paradigm leverage quantum mechanic principle superposition entanglement interference perform complex calculation exponentially fast traditional computer use qubit represent data multiple state simultaneously contrast classical bit enable solve problem cryptography optimization simulation drug discovery material science intractable classical system key concept include quantum gate quantum circuit quantum algorithm shor algorithm grover algorithm challenge include maintain qubit stability error correction build scalable hardware operate extreme low temperature application revolutionize field artificial intelligence cryptography secure communication scientific research require deep understand physic mathematics computer science",
        "quantum computing new way compute use quantum mechanic superposition entanglement interference make fast calculation traditional computer use qubit multiple state solve cryptography optimization simulation drug discovery material science hard classical computer concept quantum gate circuit algorithm shor grover issue qubit stability error correction scale hardware cold temperature change artificial intelligence cryptography communication science need physic math computer science"  # Mostly correct, very close to answer key with minor simplification
    )

}


# Define function for similarity scoring
def compute_similarity_and_marks(correct_answer, student_answer):
    # Bi-Encoder Similarity
    vector1 = bi_encoder_model.encode(correct_answer, normalize_embeddings=True)
    vector2 = bi_encoder_model.encode(student_answer, normalize_embeddings=True)
    bi_encoder_score = 1 - cosine(vector1, vector2)

    # Cross-Encoder Similarity
    cross_encoder_score = cross_encoder_stsb.predict([correct_answer, student_answer])

    # NLI Contradiction Score
    nli_scores = cross_encoder_nli.predict([correct_answer, student_answer], apply_softmax=True)
    contradiction_score = nli_scores[0]
    adjusted_opposite_score = 1 - contradiction_score  # Higher means more similar

    # **Updated Weights**
    average_score = (0.3 * bi_encoder_score) + (0.5 * cross_encoder_score) + (0.2 * adjusted_opposite_score)

    # **New Marks Formula**
    def similarity_to_marks(similarity_score):
        if similarity_score < 0.6:
            return 0  # Below 0.6, no marks
        return ((similarity_score - 0.6) / 0.4) * 90 + 10  # Scale from 10% to 100%

    marks_percentage = similarity_to_marks(average_score)

    # **Optional: Apply Length Penalty**
    expected_length = len(correct_answer.split())
    student_length = len(student_answer.split())

    if student_length < 0.8 * expected_length:
        marks_percentage *= 0.85  # Apply 15% penalty

    return {
        "Bi-Encoder Score": bi_encoder_score,
        "Cross-Encoder Score": cross_encoder_score,
        "Adjusted Opposite Score": adjusted_opposite_score,
        "Weighted Average Score": average_score,
        "Marks Percentage": marks_percentage
    }

# Process all sentence pairs
results = {}

for key, (correct, student) in sentences_dict.items():
    results[key] = compute_similarity_and_marks(correct, student)

# Print results for each question
for topic, scores in results.items():
    print(f"\n=== {topic.upper()} ===")
    # print(f"Bi-Encoder Score: {scores['Bi-Encoder Score']:.4f}")
    # print(f"Cross-Encoder STSB Score: {scores['Cross-Encoder Score']:.4f}")
    # print(f"Adjusted Opposite Score: {scores['Adjusted Opposite Score']:.4f}")
    # print(f"Weighted Average Score: {scores['Weighted Average Score']:.4f}")
    print(f"Marks Percentage: {scores['Marks Percentage']:.2f}%")
