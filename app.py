from flask import Flask, request, render_template
import os
import uuid  # to avoid filename collisions
from resume_matcher import extract_text, match_resumes, extract_keywords


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route("/")
def index():
    return render_template('matchresume.html')


@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_files = request.files.getlist('resumes')

        # Validate input
        if not job_description or not resume_files:
            return render_template('matchresume.html', message="Please provide a job description and upload at least one resume.")

        resumes_text = []
        filenames = []

        # Save and extract text from uploaded resumes
        for resume_file in resume_files:
            original_filename = resume_file.filename
            unique_filename = str(uuid.uuid4()) + "_" + original_filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            resume_file.save(file_path)
            text = extract_text(file_path)
            resumes_text.append(text)
            filenames.append(original_filename)  # showing original name in results

        # Calculate similarity scores (cosine similarity)
        similarities = match_resumes(job_description, resumes_text)

        # Extract keywords for missing skills analysis
        job_keywords = extract_keywords(job_description)
        missing_skills_list = []
        for resume_text in resumes_text:
            resume_keywords = extract_keywords(resume_text)
            missing_skills = job_keywords - resume_keywords
            missing_skills_list.append(', '.join(sorted(missing_skills)))

        # Get top 5 matching resumes and their scores
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [filenames[i] for i in top_indices]
        # Convert similarity to percentage score out of 100
        similarity_scores = [round(similarities[i] * 100, 2) for i in top_indices]
        missing_skills_top = [missing_skills_list[i] for i in top_indices]

        return render_template(
            'matchresume.html',
            message="Top matching resumes:",
            top_resumes=top_resumes,
            similarity_scores=similarity_scores,
            missing_skills=missing_skills_top,
            zip=zip
        )

    return render_template('matchresume.html')


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
