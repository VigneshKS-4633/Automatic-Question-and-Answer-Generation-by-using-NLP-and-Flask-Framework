from flask import Flask, request, render_template, flash, redirect, url_for, session, Response, render_template_string, jsonify
from subjective import SubjectiveTest
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.cluster import KMeansClusterer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev')

def analyze_text_difficulty(text):
	"""Analyze text difficulty based on various metrics"""
	sentences = sent_tokenize(text)
	words = word_tokenize(text)
	
	# Calculate average sentence length
	avg_sentence_length = len(words) / len(sentences)
	
	# Calculate word complexity (using wordnet)
	complex_words = 0
	for word in words:
		if len(wordnet.synsets(word)) > 1:  # More meanings = more complex
			complex_words += 1
	
	complexity_ratio = complex_words / len(words)
	
	# Determine difficulty level
	if avg_sentence_length > 20 and complexity_ratio > 0.3:
		return "advanced"
	elif avg_sentence_length > 15 and complexity_ratio > 0.2:
		return "intermediate"
	else:
		return "basic"

def extract_keywords(text):
	"""Extract important keywords from text"""
	words = word_tokenize(text)
	tagged = nltk.pos_tag(words)
	
	# Extract nouns and important words
	keywords = []
	for word, tag in tagged:
		if tag.startswith('NN'):  # Nouns
			keywords.append(word)
		elif tag.startswith('JJ'):  # Adjectives
			keywords.append(word)
	
	return list(set(keywords))  # Remove duplicates

def cluster_topics(text):
	"""Cluster text into topics using K-means"""
	sentences = sent_tokenize(text)
	vectorizer = TfidfVectorizer(stop_words='english')
	tfidf_matrix = vectorizer.fit_transform(sentences)
	
	# Perform K-means clustering
	num_clusters = min(5, len(sentences))  # Maximum 5 clusters
	kmeans = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance)
	clusters = kmeans.cluster(tfidf_matrix.toarray(), assign_clusters=True)
	
	# Get top terms for each cluster
	feature_names = vectorizer.get_feature_names_out()
	topics = []
	for i in range(num_clusters):
		cluster_sentences = [sentences[j] for j in range(len(sentences)) if clusters[j] == i]
		if cluster_sentences:
			cluster_vectors = tfidf_matrix[clusters == i].mean(axis=0).A1
			top_terms = [feature_names[i] for i in cluster_vectors.argsort()[-3:][::-1]]
			topics.append({
				'topic': ' '.join(top_terms),
				'sentences': cluster_sentences
			})
	
	return topics

def analyze_question_variety(questions):
	"""Analyze the variety of questions generated"""
	vectorizer = TfidfVectorizer(stop_words='english')
	tfidf_matrix = vectorizer.fit_transform(questions)
	
	# Calculate cosine similarity between questions
	similarity_matrix = cosine_similarity(tfidf_matrix)
	
	# Calculate average similarity
	avg_similarity = np.mean(similarity_matrix - np.eye(len(questions)))
	
	return {
		'variety_score': 1 - avg_similarity,  # Higher score means more variety
		'similarity_matrix': similarity_matrix.tolist()
	}

def validate_answer(answer, question):
	"""Validate answer quality"""
	# Calculate answer length
	answer_length = len(word_tokenize(answer))
	
	# Check for key terms from question in answer
	question_terms = set(word_tokenize(question.lower()))
	answer_terms = set(word_tokenize(answer.lower()))
	term_coverage = len(question_terms.intersection(answer_terms)) / len(question_terms)
	
	# Calculate answer complexity
	answer_sentences = sent_tokenize(answer)
	avg_sentence_length = len(word_tokenize(answer)) / len(answer_sentences)
	
	return {
		'length_score': min(1, answer_length / 50),  # Normalize to 0-1
		'term_coverage': term_coverage,
		'complexity_score': min(1, avg_sentence_length / 20)
	}

def generate_summary(text):
	"""Generate a concise summary of the text"""
	sentences = sent_tokenize(text)
	if len(sentences) <= 3:
		return text
	
	# Score sentences based on word frequency
	word_freq = {}
	for sentence in sentences:
		words = word_tokenize(sentence.lower())
		for word in words:
			if word not in word_freq:
				word_freq[word] = 1
			else:
				word_freq[word] += 1
	
	# Calculate sentence scores
	sentence_scores = {}
	for sentence in sentences:
		words = word_tokenize(sentence.lower())
		score = sum(word_freq[word] for word in words if word in word_freq)
		sentence_scores[sentence] = score
	
	# Select top sentences
	summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
	summary = ' '.join(sentence for sentence, _ in sorted(summary_sentences, key=lambda x: sentences.index(x[0])))
	
	return summary

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/analyze_text', methods=["POST"])
def analyze_text():
	"""Endpoint to analyze text and provide suggestions"""
	try:
		text = request.json.get('text', '').strip()
		if not text:
			return jsonify({'error': 'No text provided'}), 400
			
		difficulty = analyze_text_difficulty(text)
		keywords = extract_keywords(text)
		topics = cluster_topics(text)
		summary = generate_summary(text)
		
		return jsonify({
			'difficulty': difficulty,
			'keywords': keywords[:5],  # Return top 5 keywords
			'topics': topics,
			'summary': summary,
			'suggestions': {
				'basic': 'Focus on understanding key concepts and definitions',
				'intermediate': 'Explore relationships between concepts and apply knowledge',
				'advanced': 'Analyze complex scenarios and evaluate different perspectives'
			}[difficulty]
		})
	except Exception as e:
		return jsonify({'error': str(e)}), 500

@app.route('/test_generate', methods=["POST"])
def test_generate():
	if request.method == "POST":
		try:
			input_text = request.form.get("itext", "").strip()
			no_of_ques = request.form.get("noq", "").strip()
			difficulty_level = request.form.get("difficulty", "auto")

			# Input validation
			if not input_text:
				flash('Please provide input text')
				return redirect(url_for('index'))
			
			try:
				no_of_ques = int(no_of_ques)
				if no_of_ques <= 0 or no_of_ques > 20:
					flash('Please enter a valid number of questions (1-20)')
					return redirect(url_for('index'))
			except ValueError:
				flash('Please enter a valid number for questions')
				return redirect(url_for('index'))

			# Generate questions
			subjective_generator = SubjectiveTest(input_text, no_of_ques)
			question_list, answer_list = subjective_generator.generate_test()

			if not question_list or not answer_list:
				flash('Could not generate questions from the provided text. Please try with different content.')
				return redirect(url_for('index'))

			# Analyze question variety and validate answers
			variety_analysis = analyze_question_variety(question_list)
			answer_validation = [validate_answer(a, q) for q, a in zip(question_list, answer_list)]
			
			# Generate summary
			summary = generate_summary(input_text)
			
			# Store results in session for later use
			session['questions'] = question_list
			session['answers'] = answer_list
			session['variety_analysis'] = variety_analysis
			session['answer_validation'] = answer_validation
			session['summary'] = summary
			session['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			
			testgenerate = zip(question_list, answer_list)
			return render_template('generatedtestdata.html', 
								 cresults=testgenerate,
								 variety_analysis=variety_analysis,
								 answer_validation=answer_validation,
								 summary=summary)

		except Exception as e:
			flash(f'An error occurred: {str(e)}')
			return redirect(url_for('index'))

@app.route('/export_test', methods=["POST"])
def export_test():
	"""Export test in various formats"""
	try:
		format_type = request.form.get('format', 'pdf')
		questions = session.get('questions', [])
		answers = session.get('answers', [])
		variety_analysis = session.get('variety_analysis', {})
		answer_validation = session.get('answer_validation', [])
		summary = session.get('summary', '')
		timestamp = session.get('timestamp', '')
		
		if not questions or not answers:
			return jsonify({'error': 'No test data available'}), 400
			
		# Format the test data
		test_data = []
		for i, (q, a) in enumerate(zip(questions, answers), 1):
			validation = answer_validation[i-1] if i <= len(answer_validation) else {}
			test_data.append({
				'question_number': i,
				'question': q,
				'answer': a,
				'validation': validation
			})
			
		return jsonify({
			'success': True,
			'data': test_data,
			'format': format_type,
			'variety_analysis': variety_analysis,
			'summary': summary,
			'timestamp': timestamp
		})
	except Exception as e:
		return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5001, debug=True)