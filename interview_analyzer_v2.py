from datetime import datetime
import streamlit as st
import pandas as pd
import sys
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from audio_transcriber import WhisperAudioTranscriber
import gdown
from openai import OpenAI
from moviepy.editor import AudioFileClip
import json
from dotenv import load_dotenv
import re
import logging
import subprocess
import tiktoken
import string
import ast
from jinja2 import Environment, FileSystemLoader
from collections import defaultdict
from html import escape
import traceback
import pytz
import html


### Loading .env file contents
load_dotenv()

# Load environment variables
api_key = os.getenv("OPENAI_KEY")
model_name = os.getenv("OPENAI_MODEL")

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('app.log')
# stream_handler = logging.StreamHandler()

# Create a formatter and attach it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
# logger.addHandler(stream_handler)

## adjusting languages

# Default languages and language aliases
DEFAULT_LANGUAGES = ['javascript', 'c++', 'python', 'sql','reactjs','html','css','c','nodejs','java']
DEFAULT_LANGUAGE_ALIASES = {
    'javascript': ['javascript', 'js'],
    'c++': ['c++', 'cpp', 'cplus'],
    'python': ['python', 'py'],
    'sql': ['sql'],
    'c' : ['c'],
    'reactjs' :['reactjs','react_js','react'],
    'nodejs' : ['nodejs','node_js','node'],
    'html' : ['html'],
    'css' : ['css'],
    'java' : ['java']
}

## Helper Functions

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

def write_content(file_path,content):
    with open(file_path, "w") as file:
            file.write(content)
            file.write('\n') 
## media editing setup

def save_video_section(video_name, output_vide_name, clip_start_time, clip_end_time):
    try:
        logger.info("Saving video section")
        video_name = "./" + video_name.replace("\\", "/")
        clipped_video_name = "./" + output_vide_name.replace("\\", "/")
        command = f"ffmpeg -i {video_name} -ss {str(clip_start_time)} -to {str(clip_end_time)} -r 1 -c:v libx264 -c:a copy {clipped_video_name}"
        # print(command)
        output = subprocess.check_output(command, shell=True)
        logger.info("Video section saved successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error saving video section: {e.output}")
    except Exception as e:
        logger.error(f"Error saving video section: {str(e)}")

def video_to_audio_converter(mp4, mp3):
    try:
        logger.info(f"Converting video to audio: {mp4} -> {mp3}")
        FILETOCONVERT = AudioFileClip(mp4)
        FILETOCONVERT.write_audiofile(mp3)
        FILETOCONVERT.close()
        logger.info(f"Video to audio conversion successful: {mp4} -> {mp3}")
    except Exception as e:
        logger.error(f"Error converting video to audio: {mp4} -> {mp3}, Error: {str(e)}")

def remove_duplicates(df):
    try : 
        for i, row1 in df.iterrows():
            question1 = row1['question_text']
            for j, row2 in df.iterrows():
                if i == j:
                    continue 
                question2 = row2['question_text']
                if question1 in question2 :
                    df.at[i, 'question_text'] = question2
        df.drop_duplicates(subset='question_text', keep='last', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error getting segregated data: {str(e)}")

## Chatgpt Functions Setup

# helping functions

def calculate_tokens(prompt, model_name):
    try:
        # Load tokenizer for the specified model
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(prompt)
        token_count = len(tokens)
        return token_count
    except Exception as e:
        logging.error(f"Error calculating tokens: {e}")
        raise e

# Whisper Setup

def generate_transcript(mp3_file, transcript_file_name):
    try:
        transcriber = WhisperAudioTranscriber(api_key)
        transcriber.load_file(mp3_file)
        transcriber.create_audio_chunks()
        transcriber.start_transcribing(output_filename=transcript_file_name)
    except Exception as e:
        logger.error(f"Error getting segregated data: {str(e)}")

def load_transcript(transcript_file):
    with open(transcript_file, 'r') as file:
        content = file.read()
    return content

def clean_transcript(transcript):
    processed_lines = []
    timestamp = None  # Initialize the timestamp variable
    for line in transcript.splitlines():
        line = line.strip()  # Remove leading/trailing whitespace
        # Check if the line contains a timestamp (format: hh:mm:ss,ms --> hh:mm:ss,ms)
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line):
            timestamp = line
        elif line and not line.isdigit():  # Skip empty lines and line numbers
            if timestamp:  # Ensure timestamp is available
                # Remove punctuation marks using str.translate
                sentence = line.translate(str.maketrans('', '', string.punctuation))
                # Add the processed line (timestamp + sentence)
                processed_lines.append(f"{timestamp} {sentence.strip()}")
    return "\n".join(processed_lines)
# chat completions setup

# ----- Prompts ------------#

def qna_default_prompt():
    file_path = os.path.join("prompts","qna_prompt.txt")
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def questions_scores_prompt():
    file_path = os.path.join("prompts","scores_prompt.txt")
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def interviewer_capability_prompt():
    file_path = os.path.join("prompts","interviewer_capability_prompt.txt")
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def proper_question_generator():
    file_path = os.path.join("prompts","proper_question_generator.txt")
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def topic_wise_feedback():
    file_path = os.path.join("prompts","topic_wise_feedback.txt")
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

# -------Functions -------#

def get_segregated_data(prompt, transcript,user_id,file_id):
    try:
        client = OpenAI(
            api_key=api_key
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt + "Dont give anything else, Just pure json."},
                {"role": "user", "content": transcript}
            ],
            temperature=0.1
        )        
        print(response)

        segregated_text = str(response.choices[0].message.content)
        print(segregated_text)
        # print(segregated_text)
        segregated_text = segregated_text.replace("```json", "").replace("```", "")
        # print(segregated_text)
        usage_text = str(response.usage)
        # Save usage data
        save_usage_data('get_segregated_data', usage_text, 'usage_logs/chatgpt_usage.csv',user_id,file_id)
        return segregated_text
    except Exception as e:
        logger.error(f"Error getting segregated data: {str(e)}")
        write_content("error_file.txt",segregated_text)

def save_usage_data(function_name, usage_text, output_path, user_id, file_id):

    # Default values in case parsing fails
    completion_tokens = 0
    prompt_tokens = 0
    total_tokens = 0
    cached_tokens=0
    raw_usage_text = usage_text # Keep raw usage text even if parsing fails

    try:
        # Use regular expressions to extract values
        completion_tokens_match = re.search(r"completion_tokens=(\d+)", usage_text)
        prompt_tokens_match = re.search(r"prompt_tokens=(\d+)", usage_text)
        total_tokens_match = re.search(r"total_tokens=(\d+)", usage_text)
        cached_tokens_match = re.search(r"cached_tokens=(\d+)", usage_text) # Directly find cached_tokens within PromptTokensDetails

        if completion_tokens_match:
            completion_tokens = int(completion_tokens_match.group(1))
        if prompt_tokens_match:
            prompt_tokens = int(prompt_tokens_match.group(1))
        if total_tokens_match:
            total_tokens = int(total_tokens_match.group(1))
        if cached_tokens_match:
            cached_tokens = int(cached_tokens_match.group(1))


    except Exception as e:
        print(f"Error parsing usage text: {e}")

    # Create a DataFrame with detailed usage information
    usage_df = pd.DataFrame({
        'user_id':[user_id],
        'file_id':[file_id],
        'function_name': [function_name],
        'completion_tokens': [completion_tokens],
        'prompt_tokens': [prompt_tokens],
        'total_tokens': [total_tokens],
        'timestamp': [datetime.now()],
        'raw_usage_text': [raw_usage_text],
        'cached_tokens':[cached_tokens]
    })

    # Check if the file exists
    if os.path.exists(output_path):
        # Append to existing file
        usage_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # Create new file with headers
        usage_df.to_csv(output_path, index=False)

# -------helper functions -------------#

 
def process_transcript_in_chunks(transcript, prompt, chunk_size=8000):
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    final_output = []
    previous_output_list = []
    previous_output = ""
    for i, chunk in enumerate(chunks):
        
        current_output = process_chunk(chunk, previous_output, prompt)
        try:
            chunk_output = json.loads(current_output)
            final_output.extend(chunk_output)
            previous_output_list = [
                    item for item in chunk_output if 'question_text' in item
                ][-2:] 
            previous_output = '\n'.join([str(item) for item in previous_output_list])
        except:
            print(sys.exc_info())
    return final_output


def process_chunk(chunk, previous_output, prompt):
    try:
        final_transcript = "Previous Chunk Output: " + previous_output + "Current Transcript: " + chunk
        output = get_segregated_data(prompt, final_transcript,user_id,file_id)
        return output
    except Exception as e:
        logger.error(f"Error getting segregated data: {str(e)}")

# Function to split large texts based on tech_or_non_tech value
def split_large_text_by_question(qna_data, tech_non_tech, token_limit=25000):
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    for idx, row in qna_data.iterrows():
        question = row['question_text']
        answer = row['answer_text']
        
        # Prepare the text for this Q&A
        qna_text = f"question={question},answer={answer}\n"
        
        # Calculate token count for this Q&A
        token_count = calculate_tokens(qna_text,model_name)
        
        # For non-technical questions, group the entire Q&A together (do not split by individual question_text)
        if tech_non_tech == "non_technical":
            # Add to the current chunk if adding it won't exceed the token limit
            if current_token_count + token_count <= token_limit:
                current_chunk += qna_text
                current_token_count += token_count
            else:
                # If adding this Q&A exceeds the token limit, start a new chunk
                chunks.append(current_chunk.strip())  # Append the current chunk to the list
                current_chunk = qna_text  # Start new chunk with the current Q&A
                current_token_count = token_count  # Reset token count for the new chunk
        else:
            # For technical questions, split by question_text (each question is a separate chunk)
            if current_token_count + token_count <= token_limit:
                current_chunk += qna_text
                current_token_count += token_count
            else:
                # If adding this Q&A exceeds the token limit, start a new chunk
                if current_chunk:  # If there's content in the current chunk, append it
                    chunks.append(current_chunk.strip())
                current_chunk = qna_text  # Start a new chunk with the current Q&A
                current_token_count = token_count  # Reset token count for the new chunk

    # Append the final chunk if it contains any remaining data
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to read prompt from file
def get_prompt_from_file(topic,prompts_folder='prompts'):
    try:
        # Construct the topic-specific prompt file path
        topic_prompt_file = os.path.join(prompts_folder, f'{topic}_skill_prompt.txt')
        
        # Check if the topic-specific prompt file exists
        if os.path.exists(topic_prompt_file):
            with open(topic_prompt_file, 'r') as file:
                prompt = file.read()
            return prompt
        else:
            logging.warning(f"Topic-specific prompt file {topic_prompt_file} not found. Falling back to other_skill_prompt.txt.")
        
        # Fallback to 'other_skill_prompt.txt' for non-technical or other cases
        fallback_prompt_file = os.path.join(prompts_folder, 'other_skill_prompt.txt')
        
        if os.path.exists(fallback_prompt_file):
            with open(fallback_prompt_file, 'r') as file:
                prompt = file.read()
            return prompt
        else:
            logging.error(f"Fallback prompt file {fallback_prompt_file} does not exist.")
            return None
    except Exception as e:
        logging.error(f"Error reading prompt from file: {e}")
        return None


# Function to process Q&A data by topic
def process_qna_data_by_topic(qna_data, user_id, file_id, model_name, prompts_folder='prompts'):
    skills_df = qna_data.copy()

    try:
        if qna_data.empty:
            print("qna_data is empty. Exiting function.")
            return None

        # Ensure 'question_text' is present in the Q&A data
        if 'question_text' not in qna_data.columns:
            logging.error("Missing 'question_text' in Q&A data.")
            return None

        # Initialize 'topic' and 'subtopic' columns in the DataFrame
        skills_df['topic'] = None
        skills_df['subtopic'] = None

        # Get unique topics from the 'question_concept' field
        unique_topics = qna_data['question_concept'].unique()

        for topic in unique_topics:
            try:
                # Filter rows for the current topic
                filtered_qna_data = qna_data[qna_data['question_concept'] == topic]

                # Skip rows without valid question_text
                filtered_qna_data = filtered_qna_data.dropna(subset=['question_text'])
                if filtered_qna_data.empty:
                    continue

                # Get the unique values of 'tech_or_non_tech'
                tech_or_non_tech_values = filtered_qna_data['tech_non_tech'].unique()

                # Get the correct prompt for the topic
                prompt = None
                for tech_non_tech in tech_or_non_tech_values:
                    prompt = get_prompt_from_file(topic, prompts_folder)
                    if prompt is not None:
                        break

                if prompt is None:
                    logging.error(f"No valid prompt found for topic '{topic}' with tech_or_non_tech values: {tech_or_non_tech_values}")
                    continue

                topic_data_scores = []

                # Process Q&A data for both technical and non-technical categories
                for tech_non_tech in tech_or_non_tech_values:
                    tech_qna_data = filtered_qna_data[filtered_qna_data['tech_non_tech'] == tech_non_tech]
                    topic_qna = ""
                    for _, row in tech_qna_data.iterrows():
                        question = row['question_text']
                        answer = row['answer_text']
                        topic_qna += f"question={question},answer={answer}\n"

                    token_count = calculate_tokens(topic_qna, model_name)

                    if token_count > 25000:
                        chunks = split_large_text_by_question(tech_qna_data, tech_non_tech, token_limit=25000)
                        for chunk in chunks:
                            response = get_segregated_data(prompt, chunk, user_id, file_id)
                            topic_data_scores.extend(json.loads(response))
                    else:
                        response = get_segregated_data(prompt, topic_qna, user_id, file_id)
                        topic_data_scores.extend(json.loads(response))

                if topic_data_scores:
                    topic_df = pd.DataFrame(topic_data_scores)
                    topic_df.set_index('question_text', inplace=True)

                    # Align and update the 'topic' and 'subtopic' columns in `skills_df`
                    for question in topic_df.index:
                        if question in skills_df['question_text'].values:
                            skills_df.loc[skills_df['question_text'] == question, 'topic'] = topic_df.at[question, 'topic']
                            skills_df.loc[skills_df['question_text'] == question, 'subtopic'] = topic_df.at[question, 'subtopic']

            except Exception as topic_error:
                logging.error(f"Error processing topic '{topic}': {topic_error}")
                continue

        return skills_df

    except Exception as e:
        logging.error(f"Error processing Q&A data by topic: {e}")
        return None


def score_calculator(prompt,scores_df,user_id,interviewer_id,interview_id,interview_round,score_data_file_destination,model_name,token_limit=25000):
   
    try:

        with st.spinner("Calculating User Scores"):
            combined_text = ""
            current_token_count = 0
            current_batch = []

            for index, row in scores_df.iterrows():
                question = row["question_text"]
                answer = row["answer_text"]
                question_type = row["question_type"]
                question_concept = row["question_concept"]

                row_text = (f"question = {question}, answer = {answer}, "
                            f"question_type = {question_type}, question_concept = {question_concept}\n")
                row_token_count = calculate_tokens(row_text, model_name)

                if current_token_count + row_token_count <= token_limit:
                    combined_text += row_text
                    current_batch.append((index, row))
                    current_token_count += row_token_count
                else:
                    # Process current batch
                    combined_text, current_batch = process_batch(
                        combined_text, current_batch, prompt, user_id, file_id, scores_df
                    )
                    current_token_count = row_token_count
                    combined_text = row_text
                    current_batch = [(index, row)]

            # Process final batch
            if current_batch:
                process_batch(combined_text, current_batch, prompt, user_id, file_id, scores_df)

        # Append metadata
        scores_df["user_id"] = user_id
        scores_df["interviewer_id"] = interviewer_id
        scores_df["interview_id"] = interview_id
        scores_df["interview_round"] = interview_round

        # Save results
        scores_df.to_csv(score_data_file_destination, index=False)
        st.write("Scores saved successfully.")
        return scores_df

    except Exception as e:
        st.write("An error occurred during score calculation.")
        print(f"Unexpected error: {e}")
        return None


def process_batch(combined_text, batch, prompt, user_id, file_id, scores_df):
    try:
        segragated_data_score = get_segregated_data(prompt, combined_text, user_id, file_id)
        scores_data = json.loads(segragated_data_score)

        for batch_index, (row_index, batch_row) in enumerate(batch):
            scores = scores_data[batch_index]
            scores_df.at[row_index, "answer_rating"] = scores["answer_rating"]
            scores_df.at[row_index, "interviewee_communication_rating"] = scores["interviewee_communication_rating"]
            scores_df.at[row_index, "interviewer_communication_rating"] = scores["interviewer_communication_rating"]
    except Exception as e:
        print(f"Error processing batch: {e}")
        st.write("Unable to get score for some rows in this batch.")
    return "", []

# Filter PII data by interview_id (or user_id/interviewer_id)
def get_pii_data(interview_id, user_pii_df, interviewer_pii_df, rating_df):
    rating_row = rating_df[rating_df['interview_id'] == interview_id]
    if rating_row.empty:
        return None, None
    user_id = rating_row['user_id'].iloc[0]
    interviewer_id = rating_row['interviewer_id'].iloc[0]
    user_pii = user_pii_df[user_pii_df['user_id'] == user_id]
    interviewer_pii = interviewer_pii_df[interviewer_pii_df['interviewer_id'] == interviewer_id]
    return user_pii, interviewer_pii


def generate_dynamic_report(coding_df, rating_df, proper_questions_file, user_pii_data, interviewer_pii_data, interview_id, file_id, category_mapping_df, include_recording, topic_feedback_df, include_questions_asked, include_interviewer_details):
    try:
        logger.info(f"Generating report for interview_id: {interview_id}, file_id: {file_id}")
        
        create_directory("reports")
        
        logger.debug("Reading technical languages from CSV")
        technical_languages = set()
        technical_languages_file = os.path.join("data", "technical_languages.csv")
        if os.path.exists(technical_languages_file):
            tech_df = pd.read_csv(technical_languages_file)
            technical_languages = set(tech_df['language'].str.lower().dropna())
        else:
            logger.warning(f"technical_languages.csv not found at {technical_languages_file}, using empty set")
        
        logger.debug("Reading proper_questions.csv")
        proper_questions_df = pd.DataFrame()
        if include_questions_asked and os.path.exists(proper_questions_file):
            proper_questions_df = pd.read_csv(proper_questions_file)
            proper_questions_df = proper_questions_df[proper_questions_df['interview_id'] == interview_id]
            logger.info(f"Proper Questions DataFrame for interview_id {interview_id}:\n{proper_questions_df.to_string()}")
        
        logger.debug(f"Filtering rating_df for interview_id: {interview_id}")
        rating_data = rating_df[rating_df['interview_id'] == interview_id]
        
        if rating_data.empty:
            logger.error(f"No rating data found for interview_id: {interview_id}")
            return None, None, [], []
        
        logger.debug("Extracting metadata")
        metadata = {
            'generated_date': datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%B %d, %Y %H:%M:%S %Z"),
            'interview_datetime': rating_data['interview_datetime'].iloc[0] if 'interview_datetime' in rating_data.columns and not pd.isna(rating_data['interview_datetime'].iloc[0]) else None
        }
        if include_recording and 'interview_recording_link' in rating_data.columns and not pd.isna(rating_data['interview_recording_link'].iloc[0]):
            metadata['interview_recording_link'] = rating_data['interview_recording_link'].iloc[0]
        
        logger.debug("Processing PII data")
        user_pii = user_pii_data[user_pii_data['user_id'] == rating_data['user_id'].iloc[0]].to_dict('records')[0]
        user_pii = {k: escape(str(v)) for k, v in user_pii.items() if pd.notna(v)}

        interviewer_pii = interviewer_pii_data[interviewer_pii_data['interviewer_id'] == rating_data['interviewer_id'].iloc[0]].to_dict('records')[0] if include_interviewer_details else {}
        interviewer_pii = {k: escape(str(v)) for k, v in interviewer_pii.items() if pd.notna(v)}
        
        logger.debug("Categorizing rating columns")
        rating_columns = [col for col in rating_data.columns if col not in ['user_id', 'interviewer_id', 'interview_id', 'interview_track', 'interview_datetime', 'interview_recording_link', 'overall_rating']]
        
        logger.debug("Categorizing ratings based on category mapping CSV")
        language_ratings = {}
        common_topic_ratings = []
        
        if category_mapping_df.empty or 'field_name' not in category_mapping_df.columns or 'category_name' not in category_mapping_df.columns:
            logger.error("Category mapping CSV is missing or invalid. Required columns: 'field_name', 'category_name'")
            st.error("Please upload a valid category mapping CSV with 'field_name' and 'category_name' columns.")
            return None, None, [], []
        
        languages = category_mapping_df['category_name'].str.lower().unique().tolist()
        for lang in languages:
            language_ratings[lang] = []
        
        for col in rating_columns:
            if col == 'code_solving_ability_rating':
                continue
            value = rating_data[col].iloc[0]
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                logger.warning(f"Skipping column '{col}' due to NaN or empty value")
                continue
            try:
                int_value = int(value)
                mapping_row = category_mapping_df[category_mapping_df['field_name'] == col]
                if mapping_row.empty:
                    logger.warning(f"No category mapping found for column '{col}'. Skipping.")
                    continue
                category = mapping_row['category_name'].iloc[0].lower()
                rating_entry = {
                    'name': escape(col.replace('_', ' ').title()),
                    'value': int_value,
                    'display_value': max(0, int_value)
                }
                if category in languages:
                    language_ratings[category].append(rating_entry)
                    logger.debug(f"Assigned column '{col}' to language '{category}': {rating_entry}")
                else:
                    common_topic_ratings.append(rating_entry)
                    logger.debug(f"Assigned column '{col}' to common topic '{category}': {rating_entry}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping column '{col}' due to invalid value: {value}, Error: {str(e)}")
                continue
        
        logger.debug(f"Language ratings: {language_ratings}")
        logger.debug(f"Common topic ratings: {common_topic_ratings}")
        
        logger.debug("Preparing gauge chart data")
        gauge_charts = []
        for idx, rating in enumerate(common_topic_ratings, 1):
            gauge_charts.append({
                'id': f'gauge-{idx}-common',
                'value': rating['display_value']
            })
        for language, ratings in language_ratings.items():
            if ratings:
                sanitized_lang = language.replace('+', '-').replace(' ', '-')
                for idx, rating in enumerate(ratings, 1):
                    gauge_charts.append({
                        'id': f'gauge-{sanitized_lang}-{idx}',
                        'value': rating['display_value']
                    })
        gauge_charts_json = json.dumps(gauge_charts)
        logger.debug(f"gauge_charts_json: {gauge_charts_json}")
        
        logger.debug("Preparing code solving rating")
        code_solving_rating = 'N/A'
        if 'code_solving_ability_rating' in rating_data:
            value = rating_data['code_solving_ability_rating'].iloc[0]
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                code_solving_rating = 'N/A'
            else:
                try:
                    code_solving_rating = escape(str(int(value)))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for code_solving_ability_rating: {value}, Error: {str(e)}")
                    code_solving_rating = 'N/A'
        
        logger.debug("Preparing coding questions")
        coding_data = coding_df[coding_df['interview_id'] == interview_id].to_dict('records')
        coding_questions = [
            {
                'coding_question': escape(str(row.get('coding_question', ''))),
                'coding_language': escape(str(row.get('coding_language', '').lower())),
                'user_answer': escape(str(row.get('user_answer', ''))),
                'remarks': escape(str(row.get('remarks', '')))
            } for row in coding_data
        ]
        
        logger.debug("Grouping questions by topic")
        question_categories = defaultdict(list)
        if include_questions_asked and not proper_questions_df.empty:
            for _, row in proper_questions_df.iterrows():
                topic = row.get('question_concept', 'general').lower()
                category = topic if topic in technical_languages else 'general'
                question_categories[category].append({
                    'question': escape(str(row['question'])),
                    'user_answer_summary': html.unescape(escape(str(row.get('user_answer_summary', 'No summary provided.'))))
                })
        question_categories = dict(question_categories)
        
        logger.debug("Preparing topic feedback data")
        topic_feedback_data = defaultdict(list)
        if not topic_feedback_df.empty:
            logger.debug(f"topic_feedback_df columns: {topic_feedback_df.columns.tolist()}")
            logger.debug(f"topic_feedback_df dtypes: {topic_feedback_df.dtypes}")
            logger.debug(f"topic_feedback_df sample raw: {topic_feedback_df.head().to_dict(orient='records')}")
            for idx, row in topic_feedback_df.iterrows():
                feedback_entry = {}
                logger.debug(f"Processing row {idx} for topic: {row.get('topic', 'unknown')}, raw improvement_points: {row.get('improvement_points')}, raw suggestions: {row.get('suggestions')}")
                try:
                    improvement_points_str = row.get('improvement_points', '[]')
                    improvement_points_str = improvement_points_str.strip().replace("'", '"').replace('""', '"')
                    feedback_entry['improvement_points'] = json.loads(improvement_points_str) if improvement_points_str else []
                    logger.debug(f"Successfully parsed improvement_points: {feedback_entry['improvement_points']}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSONDecodeError parsing improvement_points for topic {row.get('topic', 'unknown')}: {str(e)}, raw data: {improvement_points_str}")
                    feedback_entry['improvement_points'] = []
                try:
                    suggestions_str = row.get('suggestions', '[]')
                    suggestions_str = suggestions_str.strip().replace("'", '"').replace('""', '"')
                    feedback_entry['suggestions'] = json.loads(suggestions_str) if suggestions_str else []
                    logger.debug(f"Successfully parsed suggestions: {feedback_entry['suggestions']}")
                except json.JSONDecodeError as e:
                    logger.error(f"JSONDecodeError parsing suggestions for topic {row.get('topic', 'unknown')}: {str(e)}, raw data: {suggestions_str}")
                    feedback_entry['suggestions'] = []
                
                topic = row.get('topic', '').lower()
                category = topic if topic in technical_languages else 'general'
                feedback_entry['topic'] = topic
                feedback_entry['feedback_summary'] = html.unescape(escape(str(row.get('feedback_summary', ''))))
                topic_feedback_data[category].append(feedback_entry)
        topic_feedback_data = dict(topic_feedback_data)
        logger.debug(f"Final topic_feedback_data: {topic_feedback_data}")
        
        logger.debug("Preparing template data")
        template_data = {
            'user_pii': user_pii,
            'interviewer_pii': interviewer_pii,
            'metadata': metadata,
            'language_ratings': language_ratings,
            'common_topic_ratings': common_topic_ratings,
            'code_solving_rating': code_solving_rating,
            'coding_questions': coding_questions,
            'question_categories': question_categories,
            'gauge_charts_json': gauge_charts_json,
            'include_recording': include_recording,
            'topic_feedback': topic_feedback_data,
            'include_questions_asked': include_questions_asked,
            'include_interviewer_details': include_interviewer_details
        }
        
        logger.debug("Loading Jinja2 template")
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('templates/report_template.html')
        
        logger.debug("Rendering and saving HTML")
        report_html = template.render(**template_data)
        report_path = f"reports/{file_id}_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"Report generated: {report_path}")
        return report_html, report_path, language_ratings, common_topic_ratings
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None, [], []

####  Final Code  ##### 

if "button1" not in st.session_state:
    st.session_state["button1"] = False

st.title('Interviewer Community Interview Analyzer')
st.markdown("""A LLM based Analyzer for Interviewer Community Technical Interviews""")
uploaded_file = st.file_uploader("Choose a CSV file")
uploaded_ratings_file = st.file_uploader("Choose a CSV file for ratings")
uploaded_coding_file = st.file_uploader("Choose a CSV file for coding questions")
uploaded_users_file = st.file_uploader("Choose a CSV file for user details", type="csv")
uploaded_interviewer_file = st.file_uploader("Choose a CSV file for interviewer details", type="csv")
uploaded_category_mapping_file = st.file_uploader("Choose a CSV file for category mapping", type="csv")


input_df = pd.DataFrame()
rating_df = pd.DataFrame()
coding_df = pd.DataFrame()
user_details_df = pd.DataFrame()
interviewer_details_df = pd.DataFrame()
category_mapping_df = pd.DataFrame()

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

if uploaded_ratings_file is not None:
    rating_df = pd.read_csv(uploaded_ratings_file)

if uploaded_coding_file is not None:
    coding_df = pd.read_csv(uploaded_coding_file)

if uploaded_users_file is not None:
    user_details_df = pd.read_csv(uploaded_users_file)

if uploaded_interviewer_file is not None:
    interviewer_details_df = pd.read_csv(uploaded_interviewer_file)

if uploaded_category_mapping_file is not None:
    category_mapping_df = pd.read_csv(uploaded_category_mapping_file)

# Checkbox for including interview recording link
include_recording = st.checkbox("Include interview recording link in report", value=True, help="Uncheck to exclude the recording link from the report.")
# st.divider()

# Checkbox for including questions asked
include_questions_asked = st.checkbox("Include Questions Asked in report", value=True, help="Uncheck to exclude the Questions Asked section from the report.")
# st.divider()

# Checkbox for interviewer details asked
include_interviewer_details = st.checkbox("Include Interviewer Details", value=True, help="Uncheck to exclude the Interviewer Details Section from the report.")
st.divider()


def main(url, file_id, user_id, interviewer_id,interview_id, interview_round, clip_start_time, clip_end_time,category_mapping_df,include_recording,include_questions_asked,include_interviewer_details):
    directory_name = 'videos_and_audio_files'
    transcript_destination = "transcripts"
    all_qna_destination= "final_question_and_answers"
    qna_destination = "question_and_answers"
    skill_destination = "question_wise_skills"
    score_destination = "question_wise_user_scores"
    behaviour_destination = "InterviewAnalysis"
    all_image_destination= "Images"
    all_txt_destination = "Resume&Coding Texts"
    usage_log_directory = 'usage_logs'
    interview_flow_directory = 'interview_flow'
    interviewer_capability_directory = 'interviewer_capability'
    proper_questions_directory = 'proper_questions'
    topic_feedback_directory = 'topic_feedback'
    create_directory(usage_log_directory)
    create_directory(directory_name)
    create_directory(transcript_destination)
    create_directory(qna_destination)
    create_directory(skill_destination)
    create_directory(score_destination)
    create_directory(behaviour_destination)
    create_directory(all_qna_destination)
    create_directory(all_image_destination)
    create_directory(all_txt_destination)
    create_directory(interview_flow_directory)
    create_directory(interviewer_capability_directory)
    create_directory(proper_questions_directory)
    create_directory(topic_feedback_directory)

    video_file_name = f"{file_id}.mp4"
    clipped_video_file_name = f"clipped_{file_id}_{clip_start_time}_{clip_end_time}.mp4"
    audio_file_name = f"{file_id}_{clip_start_time}_{clip_end_time}.mp3"
    transcript_file_name = f"{file_id}_{clip_start_time}_{clip_end_time}.srt"
    date_file_name = f"{file_id}_{clip_start_time}_{clip_end_time}.csv"
    skills_file_name = f"skills_{file_id}_{clip_start_time}_{clip_end_time}.csv"
    interview_analysis_file_name = f"{file_id}_{clip_start_time}_{clip_end_time}.txt"
    image_file_name = f"{file_id}_resume.png"
    resume_txt_file_name = f"{file_id}_resume.txt"
    interview_flow_name = f"{file_id}_interview_flow.csv"
    interviewer_capability_name = f"{file_id}_interviewer_capability.csv"
    proper_questions_name = f"{file_id}_proper_questions.csv"
    topic_feedback_name = f"{file_id}_topic_feedback.csv"
    video_destination = os.path.join(directory_name, video_file_name)
    clipped_video_destination = os.path.join(directory_name, clipped_video_file_name)
    transcript_destination = os.path.join(transcript_destination, transcript_file_name)
    qna_data_file_destination = os.path.join(qna_destination, date_file_name)
    all_qna_data_file_destination = os.path.join(all_qna_destination, date_file_name)
    skill_tag_data_file_destination = os.path.join(skill_destination, skills_file_name)
    score_data_file_destination = os.path.join(score_destination, date_file_name)
    behaviour_data_file_destination = os.path.join(behaviour_destination, date_file_name)
    resume_img_destination=os.path.join(all_image_destination,image_file_name)
    resume_file_destination = os.path.join(all_txt_destination,resume_txt_file_name)
    audio_destination = os.path.join(directory_name, audio_file_name)
    interview_flow_destination = os.path.join(interview_flow_directory,interview_flow_name)
    proper_questions_destination = os.path.join(proper_questions_directory,proper_questions_name)
    topic_feedback_destination = os.path.join(topic_feedback_directory,topic_feedback_name)
    interviewer_capability_destination = os.path.join(interviewer_capability_directory,interviewer_capability_name)

    st.subheader("Video Download")
    logger.info(f"{file_id} - Video Downloading")

    if os.path.exists(video_destination):
        st.write("Video Already Downloaded")
    else:
        with st.spinner("Video Download Started...."):
            # print(url)
            gdown.download(url, video_destination, quiet=False, fuzzy=True)
            st.write("Video Download Completed")
    st.divider()

    logger.info(f"{file_id} - Video Clipping")

    st.subheader("Video Clipping")
    if os.path.exists(clipped_video_destination):
        st.write("Clipped Video Already Present")
    else:
        with st.spinner("Video Clipping Started"):
            save_video_section(video_destination, clipped_video_destination, clip_start_time, clip_end_time)
            # os.remove(video_destination)
            st.write("Clipped Video Ready")
    st.video(f"./{clipped_video_destination}", format="video/mp4")
    st.divider()

    logger.info(f"{file_id} - Audio Extraction")

    st.subheader("Audio Extraction")
    if os.path.exists(audio_destination):
        st.write("Audio Already Present")
    else:
        with st.spinner("Audio Extraction Started"):
            video_to_audio_converter(clipped_video_destination, audio_destination)
            # os.remove(clipped_video_destination)
            st.write("Audio Extraction Completed")
    st.audio(f"./{audio_destination}", format="audio/mp3")
    st.divider()

    logger.info(f"{file_id} - Transcript Generation")

    st.subheader('Transcript Generation')
    if os.path.exists(transcript_destination):
        transcript = load_transcript(transcript_destination)
        cleaned_transcript = clean_transcript(transcript)
        st.write("Transcript Already Present")
    else:
        with st.spinner("Transcript Generation Started...."):
            generate_transcript(audio_destination, transcript_destination)
            # os.remove(audio_destination)
            transcript = load_transcript(transcript_destination)
            cleaned_transcript = clean_transcript(transcript)
            st.write("Transcript Generation Completed")

    transcript_txt = cleaned_transcript
    with st.expander("See Generated Transcript"):
        st.code(transcript_txt, language="python")
    st.divider()

    logger.info(f"{file_id} - Generating Interview QnA")

    st.subheader("Generating Interview QnA")
    prompt = qna_default_prompt()
    if os.path.exists(qna_data_file_destination):
        qna_df = pd.read_csv(qna_data_file_destination)
        # final_qna_df = remove_duplicates(qna_df)
        qna_df.to_csv(qna_data_file_destination, index=False)
        st.write("QnA Already Present")
    else:
        with st.spinner("Getting Questions and Answers"):
            print("test")
            segragated_data_qna_json = process_transcript_in_chunks(transcript_txt, prompt,chunk_size=10000)
            # print(segragated_data_qna)
            # json_data = json.loads(segragated_data_qna)
            qna_df = pd.DataFrame(segragated_data_qna_json)
            qna_df.to_csv(all_qna_data_file_destination,index = False)
            final_qna_df = remove_duplicates(qna_df)

            final_qna_df["user_id"] = user_id
            final_qna_df["interviewer_id"] = interviewer_id
            final_qna_df["interview_id"] = interview_id
            final_qna_df["interview_round"] = interview_round
            final_qna_df["creation_datetime"] = datetime.now()

            final_qna_df.to_csv(qna_data_file_destination, index=False)
            # test_qna_df.to_csv(test_qna_data_file_destination,index=False)
    qna_df_final = pd.read_csv(qna_data_file_destination)
    # qna_df_final = extract_coding_question_from_image(qna_df_final_before_adding_coding)
    qna_df_final.to_csv(qna_data_file_destination, index=False)
    edited_qna_data = st.data_editor(qna_df_final, num_rows="dynamic")
    st.divider()

    # logger.info(f"{file_id} - Generating Question wise Topic and Sub Topics")

    # st.subheader("Generating Question wise Topic and Sub Topics")

    # qna_data = load_transcript(qna_data_file_destination)

    # if os.path.exists(skill_tag_data_file_destination):
    #     st.write("Skill Tags Already Present")
    # else :
    #     with st.spinner("Getting Skill Tags"):
    #         skill_tags_df = process_qna_data_by_topic(qna_df_final, user_id, file_id, model_name, prompts_folder='prompts')

    #         skill_tags_df["user_id"] = user_id
    #         skill_tags_df["interviewer_id"] = interviewer_id
    #         skill_tags_df["interview_id"] = interview_id
    #         skill_tags_df["interview_round"] = interview_round
    #         # skill_tags_df["creation_datetime"] = datetime.now()
            
    #         skill_tags_df.to_csv(skill_tag_data_file_destination, index=False)
    
    # skill_tags_df_final = pd.read_csv(skill_tag_data_file_destination)
    # skills_data = st.data_editor(skill_tags_df_final,num_rows="dynamic")
    # st.divider()

    # logger.info(f"{file_id} - Generating Question wise Scores")

    # st.subheader("Generating Question wise Scores")
    # prompt = questions_scores_prompt()

    # scores_df = skill_tags_df_final.copy()

    # if os.path.exists(score_data_file_destination):
    #     st.write("User Scores Already Present")
    # else:
    #     score_calculator(prompt,scores_df,user_id,interviewer_id,interview_id,interview_round,score_data_file_destination,model_name,token_limit=25000)
    # score_df_final = pd.read_csv(score_data_file_destination)
    # edited_score_data = st.data_editor(score_df_final, num_rows="dynamic")
    # st.divider()

    if include_questions_asked:
        logger.info(f"{file_id} - Proper Questions Framing")
        st.subheader("Generating Proper Questions and user answers summary")
        prompt = proper_question_generator()
        if os.path.exists(proper_questions_destination):
            st.write("Proper Question File already present")
        else:
            with st.spinner("Framing questions"):
                qna_data_filtered = qna_df_final[qna_df_final['question_type'] != 'coding']
                qna_data = qna_data_filtered[["question_text","answer_text","question_concept"]]
                qna_json = qna_data.to_json(orient='records', lines=True)
                segragated_data_qna_json = get_segregated_data(prompt,qna_json, user_id,file_id)
                final_proper_questions_json = json.loads(segragated_data_qna_json)
                proper_questions_df = pd.DataFrame(final_proper_questions_json)
                proper_questions_df["user_id"] = user_id
                proper_questions_df["interviewer_id"] = interviewer_id
                proper_questions_df["interview_id"] = interview_id
                proper_questions_df["interview_round"] = interview_round
                proper_questions_df["creation_datetime"] = datetime.now()
                proper_questions_df.to_csv(proper_questions_destination,index = False)
        proper_questions_df_final = pd.read_csv(proper_questions_destination)
        behaviour_data = st.data_editor(proper_questions_df_final,num_rows="dynamic")
        st.divider()

    st.subheader("Generating Topic wise User Feedback and Improvement Points")

    prompt = topic_wise_feedback()

    if os.path.exists(topic_feedback_destination):
        st.write("Topic wise User Feedback File already present")
    else :
        with st.spinner("Generating Topic wise User Feedback and Improvement Points"):
            qna_data_filtered = qna_df_final[qna_df_final['question_type'] != 'coding']
            qna_data = qna_data_filtered[["question_text","answer_text","question_concept"]]
            qna_json = qna_data.to_json(orient='records', lines=True)
            segragated_data_qna_json = get_segregated_data(prompt,qna_json, user_id,file_id)
            # write_content("error_data.txt",segragated_data_qna_json)
            # Flatten the JSON
            # flattened_data = {**segragated_data_qna_json, "improvement_points": "; ".join(segragated_data_qna_json["improvement_points"])}

            final_topic_feedback_json = json.loads(segragated_data_qna_json)
            topic_feedback_df = pd.DataFrame(final_topic_feedback_json)
            topic_feedback_df["user_id"] = user_id
            topic_feedback_df["interviewer_id"] = interviewer_id
            topic_feedback_df["interview_id"] = interview_id
            topic_feedback_df["interview_round"] = interview_round
            topic_feedback_df["creation_datetime"] = datetime.now()

            topic_feedback_df.to_csv(topic_feedback_destination,index = False)
    topic_feedback_df_final = pd.read_csv(topic_feedback_destination)
    behaviour_data = st.data_editor(topic_feedback_df_final,num_rows="dynamic")
    st.divider()

    # print("Hi")
    user_pii, interviewer_pii = get_pii_data(interview_id, user_details_df, interviewer_details_df, rating_df)
    if user_pii.empty or interviewer_pii.empty:
        st.error("Missing PII data for user or interviewer.")
        print("Hi")
    else:
        # print("hi2")
        report_html, report_path, language_ratings, common_topic_ratings = generate_dynamic_report(
            coding_df, rating_df, proper_questions_destination, 
            user_pii, interviewer_pii, 
            interview_id, file_id,category_mapping_df,include_recording,topic_feedback_df_final,include_questions_asked,include_interviewer_details
        )
        if report_html:
            st.components.v1.html(report_html, height=800, scrolling=True)
            with open(report_path, 'rb') as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name=f"{file_id}_report.html",
                    mime="text/html",
                    key=f"download_{file_id}"
                )
        else:
            st.error("Failed to generate report.")

    logger.info(f"{file_id} - Generating Interviewer Behaviour Analysis")

    st.subheader("Generating Interviewer Behaviour Analysis")

    prompt = interviewer_capability_prompt()

    if os.path.exists(interviewer_capability_destination):
        st.write("Interviewer analysis already present")
    else :
        with st.spinner("Analyzing Interviewer Behaviour"):
            qna_data = qna_df_final[["question_text","answer_text"]]
            qna_json = qna_data.to_json(orient='records', lines=True)
            segragated_data_qna_json = get_segregated_data(prompt,qna_json, user_id,file_id)
            # write_content("error_data.txt",segragated_data_qna_json)
            # Flatten the JSON
            # flattened_data = {**segragated_data_qna_json, "improvement_points": "; ".join(segragated_data_qna_json["improvement_points"])}

            final_interviewer_capability_json = json.loads(segragated_data_qna_json)
            final_interviewer_capability_json = [final_interviewer_capability_json]
            # print(final_interviewer_capability_json)
            final_interviewer_capability_json[0]['improvement_points'] = "\n".join(final_interviewer_capability_json[0]['improvement_points'])

            # Create the DataFrame
            behaviour_df = pd.DataFrame(final_interviewer_capability_json)

            # Add additional fields like user_id, interviewer_id, etc.
            behaviour_df['user_id'] = user_id  # Replace with actual user_id
            behaviour_df['interviewer_id'] = interviewer_id  # Replace with actual interviewer_id
            behaviour_df['interview_id'] = interview_id  # Replace with actual interview_id
            behaviour_df['interview_round'] = interview_round  # Replace with actual round
            behaviour_df['creation_datetime'] = datetime.now()

            # Group by all columns except 'improvement_points' and aggregate improvement points
            group_cols = [col for col in behaviour_df.columns if col != 'improvement_points']
            grouped_df = behaviour_df.groupby(group_cols)['improvement_points'].agg(lambda x: '\n'.join(x)).reset_index()

            grouped_df.to_csv(interviewer_capability_destination,index = False)
    behaviour_df_final = pd.read_csv(interviewer_capability_destination)
    behaviour_data = st.data_editor(behaviour_df_final,num_rows="dynamic")
    st.divider()



if st.button("Generate and Analyze Interview Data"):
    st.subheader("Input df")
    st.dataframe(input_df)
    st.divider()
    for index, row in input_df.iterrows():
        file_id = row['drive_file_id']
        user_id = row['user_id']
        interviewer_id = row['interviewer_id']
        interview_id = row['interview_id']
        interview_track = row['interview_track']
        clip_start_time = int(row['clip_start_time'])
        clip_end_time = int(row['clip_end_time'])
        url = f"https://drive.google.com/u/0/uc?id={file_id}&export=download"
        st.write(
            f"Preparing Interview Intelligence for user: {user_id}, job:{interviewer_id}. Video Used {url}, from time {clip_start_time}(in secs) to time {clip_end_time}(in secs)")
        try:
            main(url, file_id, user_id, interviewer_id,interview_id, interview_track, clip_start_time, clip_end_time,category_mapping_df,include_recording,include_questions_asked,include_interviewer_details)
            st.write(
                f"Prepared Interview Intelligence for user: {user_id}, job:{interviewer_id}. Video Used {url}, from time {clip_start_time}(in secs) to time {clip_end_time}(in secs)")
            logger.info(f"Prepared Interview Intelligence for user: {user_id}, job:{interviewer_id}. Video Used {url}")
        except:
            print(sys.exc_info())
            st.write("Unable to prepare")