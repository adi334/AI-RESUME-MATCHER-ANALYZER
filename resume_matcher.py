import re
import docx2txt
import PyPDF2
import nltk
import spacy
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load spacy English model
nlp = spacy.load('en_core_web_sm')

# Initialize Lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_keywords(text):
    # Basic keyword extraction: words of length >= 3 excluding some stopwords
    words = re.findall(r'\b\w{3,}\b', text.lower())
    custom_stop_words = set(['the', 'and', 'for', 'with', 'are', 'you', 'your'])  # extend if needed
    keywords = [w for w in words if w not in custom_stop_words]
    return set(keywords)

def basic_clean(text):
    # Lowercase
    text = text.lower()

    # Remove special chars and numbers, keep alphabets and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_text(text):
    # Basic cleaning
    text = basic_clean(text)

    # Tokenize and remove stopwords, lemmatize with nltk
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]

    # Join tokens back to string
    filtered_text = ' '.join(tokens)

    # Optionally, use spacy for advanced processing (uncomment if needed)
    # doc = nlp(filtered_text)
    # filtered_text = ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

    return filtered_text

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

def match_resumes(job_description_text, resumes_text_list):
    """
    Preprocess texts then compute semantic similarity scores between job description and resumes.
    """
    # Preprocess job description and resumes
    job_desc_clean = preprocess_text(job_description_text)
    resumes_clean = [preprocess_text(resume) for resume in resumes_text_list]

    # Compute embeddings with semantic model
    job_embedding = model.encode(job_desc_clean, convert_to_tensor=True)
    resume_embeddings = model.encode(resumes_clean, convert_to_tensor=True)

    similarities = util.cos_sim(job_embedding, resume_embeddings)[0]
    return similarities.cpu().numpy()
