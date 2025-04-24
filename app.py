# <<< START: IMPORT STATEMENTS AND HELPER FUNCTIONS (Keep these as they are) >>>
import streamlit as st
import requests
import pandas as pd
import json
import time
import yt_dlp # Import the yt-dlp library
import os
import random
import string
import datetime
import urllib.parse
import textwrap
import tempfile
from io import BytesIO
import anthropic

# Ensure MoviePy is installed: pip install moviepy
# Ensure Pillow is installed: pip install Pillow
# Ensure pydub is installed: pip install pydub
# Ensure numpy is installed: pip install numpy
import numpy as np
# --- Try importing moviepy components with error handling ---
try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips, TextClip
    )
    import moviepy.video.fx.all as vfx
    # This specific import might be less common or part of older versions, handle potential error
    try:
        import moviepy.video.fx.resize as moviepy_resize
    except ImportError:
        print("Note: moviepy.video.fx.resize not found (may be integrated in newer MoviePy).")
        moviepy_resize = None # Define as None if not found

    from pydub import AudioSegment
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    st.error(f"Missing required library: {e}. Please install MoviePy, Pillow, pydub, and numpy.")
    st.stop() # Stop the app if essential libraries are missing

# --- API Clients & Services ---
from openai import OpenAI
# Ensure boto3 is installed: pip install boto3
try:
    import boto3
    from botocore.exceptions import NoCredentialsError
except ImportError:
    st.error("Missing required library: boto3. Please install it.")
    st.stop()

# --- Configuration ---
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3/search"
MAX_RESULTS_PER_QUERY = 3 # Reduce results per query slightly to manage load
YT_DLP_FETCH_TIMEOUT = 30 # Increased timeout for potentially slower connections
DEFAULT_TTS_VOICE = "sage" # Default voice for TTS
# --- Font Path Handling ---
# Use an absolute path or ensure the font is in the same directory as the script
# For Streamlit Cloud, you might need to include the font in your repository
# and reference it relative to the script's location.
# If running locally, ensure the path is correct.
MONTSSERAT_FONT_FILENAME = "Montserrat-Bold.ttf"
# Try to find the font in the current directory or a subdirectory
if os.path.exists(MONTSSERAT_FONT_FILENAME):
    SUBTITLE_FONT_PATH = MONTSSERAT_FONT_FILENAME
elif os.path.exists(os.path.join(os.path.dirname(__file__), MONTSSERAT_FONT_FILENAME)):
     SUBTITLE_FONT_PATH = os.path.join(os.path.dirname(__file__), MONTSSERAT_FONT_FILENAME)
else:
    st.warning(f"Font '{MONTSSERAT_FONT_FILENAME}' not found. Subtitles might use default font.", icon="⚠️")
    SUBTITLE_FONT_PATH = None # Will use default font later if None

SUBTITLE_FONT_SIZE = 50 # Adjust as needed
SUBTITLE_WORDS_PER_GROUP = 2 # Group words for subtitles
SUBTITLE_COLOR = '#FFFF00' # Yellow
SUBTITLE_BG_COLOR = 'rgba(0, 0, 0, 0.6)' # Semi-transparent black
st.set_page_config(layout="wide", page_title="YouTube Video Generator", page_icon="🎥")

# --- Load Secrets ---
try:
    youtube_api_key_secret = st.secrets["YOUTUBE_API_KEY"] # Assuming key name in secrets
    openai_api_key = st.secrets["GPT_API_KEY1"]
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"] # Load Anthropic key
    aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    s3_bucket_name = st.secrets["S3_BUCKET_NAME"]
    s3_region = st.secrets["AWS_REGION"]
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please configure secrets.", icon="🚨")
    st.stop()

# --- Initialize Clients ---
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=openai_api_key)

openai_client = get_openai_client()

@st.cache_resource
def get_anthropic_client():
    # Ensure the key is available before creating the client
    if not anthropic_api_key:
        st.error("Anthropic API Key not found in secrets.", icon="🚨")
        return None
    try:
        return anthropic.Anthropic(api_key=anthropic_api_key)
    except Exception as e:
        st.error(f"Error initializing Anthropic client: {e}", icon="🤖")
        return None

anthropic_client = get_anthropic_client()


@st.cache_resource
def get_s3_client():
    try:
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=s3_region
        )
    except NoCredentialsError:
        st.error("AWS Credentials not found or invalid in secrets.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"Error initializing S3 client: {e}", icon="🚨")
        return None

s3_client = get_s3_client()

# --- Patched Resizer (Optional - uncomment if resize issues occur) ---
# (Keep the patched_resizer code as it was)
try:
    if moviepy_resize: # Check if the module was imported
        def patched_resizer(pilim, newsize):
            if isinstance(newsize, (list, tuple)):
                newsize = tuple(int(dim) for dim in newsize)
            elif isinstance(newsize, (int, float)):
                if hasattr(pilim, "shape"): # numpy array
                    orig_height, orig_width = pilim.shape[:2]
                else: # PIL image
                    orig_width, orig_height = pilim.size
                newsize = (int(orig_width * newsize), int(orig_height * newsize))

            if not isinstance(pilim, Image.Image):
                pilim = Image.fromarray(pilim)

            resized = pilim.resize(newsize, Image.Resampling.LANCZOS) # Updated resampling filter
            return np.array(resized)
        moviepy_resize.resizer = patched_resizer
        print("Applied patched resizer.")
    else:
        print("Skipping resizer patch: moviepy.video.fx.resize not found.")

except Exception as e:
    print(f"Could not apply patched resizer: {e}")
    pass # Continue without patch


# --- Helper Function: YouTube API Search ---
def create_topic_summary_dataframe(selected_videos_dict):
    """
    Creates a DataFrame summarizing generated videos grouped by topic and language.

    Args:
        selected_videos_dict (dict): The session state dictionary
                                      st.session_state.selected_videos.

    Returns:
        pandas.DataFrame: A DataFrame with 'Topic_Language' and 'vidX_url' columns,
                          or an empty DataFrame if no generated videos are found.
    """
    topic_lang_to_generated_urls = {}

    # 1. Collect Generated URLs and Group by Topic-Language
    for video_id, video_data in selected_videos_dict.items():
        topic = video_data.get('Topic')
        generated_urls = video_data.get('Generated URLs', {}) # Get the dictionary of URLs

        # Only proceed if topic exists and some URLs were generated
        if topic and generated_urls:
            for lang, s3_url in generated_urls.items():
                if s3_url: # Ensure URL is not empty/None
                    topic_lang_key = f"{topic}_{lang}" # Create key like 'sofa sale_English'
                    if topic_lang_key not in topic_lang_to_generated_urls:
                        topic_lang_to_generated_urls[topic_lang_key] = []
                    topic_lang_to_generated_urls[topic_lang_key].append(s3_url)

    if not topic_lang_to_generated_urls:
        # Return an empty DataFrame with expected columns if no data
        return pd.DataFrame(columns=['Topic_Language']) # Adjusted column name

    # 2. Determine Max URLs per Topic-Language and Prepare Data for Wide Format
    max_urls = 0
    if topic_lang_to_generated_urls: # Ensure dict is not empty before finding max
        max_urls = max(len(urls) for urls in topic_lang_to_generated_urls.values())

    data_for_df = []
    for topic_lang, urls in topic_lang_to_generated_urls.items():
        row = {'Topic_Language': topic_lang} # Adjusted column name
        # Pad the list of URLs with empty strings up to max_urls
        padded_urls = urls + [''] * (max_urls - len(urls))
        for i, url in enumerate(padded_urls):
            row[f'vid{i+1}_url'] = url
        data_for_df.append(row)

    # 3. Create Final DataFrame
    if data_for_df:
        df_final = pd.DataFrame(data_for_df)
        # Ensure 'Topic_Language' column is first
        topic_col = df_final.pop('Topic_Language') # Adjusted column name
        df_final.insert(0, 'Topic_Language', topic_col) # Adjusted column name
        # Sort columns naturally (vid1, vid2, ... vid10...) if needed
        url_cols = sorted([col for col in df_final.columns if col.startswith('vid')],
                           key=lambda x: int(x.replace('vid','').replace('_url','')))
        final_cols = ['Topic_Language'] + url_cols # Adjusted column name
        df_final = df_final[final_cols]

    else:
        # Should not happen if topic_lang_to_generated_urls was populated, but safe fallback
        df_final = pd.DataFrame(columns=['Topic_Language']) # Adjusted column name

    return df_final


def search_youtube(api_key, query, max_results=40):
    """
    Performs a Youtube using the v3 API.
    Handles multiple terms generated by GPT and splits results among them.
    """
    videos_res = []
    response = None  # Initialize response to None

    # Split query into multiple terms if it contains '|'
    if '|' in query:
        terms = query.split('|')
    else:
        terms = [query]

    # Calculate results per term
    results_per_term = max(1, max_results // len(terms))  # Ensure at least 1 result per term
    st.text(f"Terms: {terms}, Results per term: {results_per_term}")

    for term in terms:
        params = {
            'part': 'snippet',
            'q': term.strip(),
            'key': api_key,
            'type': 'video',
            'maxResults': results_per_term,  # Fetch results for each term
            'videoEmbeddable': 'true'
            # 'order': 'viewCount' # Can add relevance, viewCount, etc.
            # 'relevanceLanguage': lang # Pass language if needed for API
        }
        try:
            response = requests.get(YOUTUBE_API_BASE_URL, params=params, timeout=15)  # Increased timeout
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            results = response.json()

            if 'items' in results:
                for item in results['items']:
                    if item.get('id', {}).get('kind') == 'youtube#video' and 'videoId' in item['id']:
                        video_id = item['id']['videoId']
                        title = item['snippet'].get('title', 'No Title')
                        # Use standard https youtube URL
                        standard_url = f"https://www.youtube.com/watch?v={video_id}"
                        # Use embeddable URL for iframe
                        embed_url = f"https://www.youtube.com/embed/{video_id}"
                        thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg" # Standard definition thumbnail

                        videos_res.append({
                            'title': title,
                            'videoId': video_id,
                            'standard_url': standard_url, # Store standard URL
                            'embed_url': embed_url, # Store embed URL
                            'thumbnail_url': thumbnail_url # Store thumbnail URL
                        })

        except requests.exceptions.Timeout:
            st.error(f"API Request Timeout for query '{term}'.", icon="⏱️")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"API HTTP Error for query '{term}': {http_err}", icon="🔥")
        except requests.exceptions.RequestException as e:
            st.error(f"API Request Error for query '{term}': {e}", icon="🌐")
        except Exception as e:
            st.error(f"An unexpected error occurred during search for '{term}': {e}", icon="💥")

    return videos_res[:max_results]  # Ensure the total results do not exceed max_results


# --- Helper Function: Get Info with yt-dlp ---
def simple_hash(s):
    total = 0
    for i, c in enumerate(s):
        total += (i + 1) * ord(c)
    return str(total % 100000)  # Keep it short

def get_yt_dlp_info(video_url):
    """
    Uses yt-dlp to extract video format information, prioritizing a direct mp4 URL.
    """
    # Prioritize standard mp4 formats often playable directly in browsers/st.video
    # Format 18: mp4 [360p], Format 22: mp4 [720p]
    # Fallback to best mp4, then best overall
    YDL_OPTS = {
        'format': 'bestvideo[ext=mp4]',
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'extract_flat': False, # Need format details
        'socket_timeout': YT_DLP_FETCH_TIMEOUT,
        'retries': 3, # Add retries
    }
    try:
        with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
            info = ydl.extract_info(video_url, download=False)

            # Sanitize info can be large, extract only what's needed
            direct_url = info.get('url')
            format_note = info.get('format_note', 'N/A')
            format_id = info.get('format_id', 'N/A')
            ext = info.get('ext', 'N/A')
            resolution = info.get('resolution', 'N/A')
            filesize = info.get('filesize') or info.get('filesize_approx')

            # Fallback if top-level URL isn't populated (sometimes happens)
            if not direct_url and 'formats' in info:
                 # Check requested formats first
                preferred_formats = ['22', '18']
                found_format = None
                for pf_id in preferred_formats:
                    found_format = next((f for f in info['formats'] if f.get('format_id') == pf_id and f.get('url')), None)
                    if found_format: break
                # If not found, try the format yt-dlp actually selected
                if not found_format:
                    chosen_format_id = info.get('format_id')
                    found_format = next((f for f in info['formats'] if f.get('format_id') == chosen_format_id and f.get('url')), None)
                # If still not found, grab the first format with a URL
                if not found_format:
                    found_format = next((f for f in info['formats'] if f.get('url')), None)

                if found_format:
                    direct_url = found_format.get('url')
                    format_id = found_format.get('format_id', format_id)
                    format_note = found_format.get('format_note', format_note)
                    ext = found_format.get('ext', ext)
                    resolution = found_format.get('resolution', resolution)
                    filesize = found_format.get('filesize') or found_format.get('filesize_approx')

            filesize_str = f"{filesize / (1024*1024):.2f} MB" if filesize else "N/A"
            format_details = f"ID: {format_id}, Res: {resolution}, Ext: {ext}, Size: {filesize_str}"

            if direct_url:
                return {
                    'direct_url': direct_url,
                    'format_details': format_details,
                    'error': None
                }
            else:
                # Log available formats if URL extraction failed unexpectedly
                print(f"Warning: Could not extract direct URL for {video_url}. Info fetched: {info.keys()}") # Log available keys
                return {'error': 'Could not extract direct URL.'}

    except yt_dlp.utils.DownloadError as e:
        # Extract specific error messages if possible
        err_msg = str(e)
        if "confirm your age" in err_msg:
            return {'error': 'Age-restricted video'}
        if "Private video" in err_msg:
            return {'error': 'Private video'}
        if "Video unavailable" in err_msg:
            return {'error': 'Video unavailable'}
        st.warning(f"yt-dlp DownloadError for {video_url}: {e}", icon="🚧")
        return {'error': f"yt-dlp: {err_msg[:100]}"} # Truncate long messages
    except Exception as e:
        st.error(f"Unexpected yt-dlp error for {video_url}: {e}", icon="💥")
        return {'error': f"Unexpected yt-dlp error: {e}"} # Return error dict


# --- Helper Function: Generate Script with ChatGPT ---
def chatGPT(prompt, client, model="gpt-4o", temperature=1):
    """Generates text using OpenAI Chat Completion."""
    if not client:
        st.error("OpenAI client not initialized.", icon="🤖")
        return None
    try:
        # Use the provided client object directly
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{'role': 'user', 'content': prompt}]
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        st.error(f"Error calling OpenAI (ChatGPT - {model}): {e}", icon="🤖")
        return None

# --- Helper Function: Generate Script with Claude ---
def claude(prompt, client, model="claude-3-haiku-20240307", temperature=1, max_retries=5):
    """Generates text using Anthropic Claude."""
    if not client:
        st.error("Anthropic client not initialized.", icon="🤖")
        return None

    tries = 0
    last_error = None
    while tries < max_retries:
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4000, # Adjust as needed, Haiku is cheaper
                temperature=temperature,
                #top_p=0.8, # Optional parameter
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
                # Removed 'thinking' parameter as it's less common and might cause issues
            )
            # Check response structure
            if message.content and isinstance(message.content, list) and len(message.content) > 0:
                 # Assuming the text content is in the first block
                 if hasattr(message.content[0], 'text'):
                      return message.content[0].text.strip()
                 else:
                      st.warning(f"Claude response format unexpected (Block 0 has no 'text'): {message.content[0]}", icon="⚠️")
                      last_error = "Unexpected response format"

            else:
                 st.warning(f"Claude response format unexpected (empty or not list): {message.content}", icon="⚠️")
                 last_error = "Unexpected response format (empty/not list)"

        except anthropic.APIConnectionError as e:
            st.warning(f"Claude APIConnectionError (Attempt {tries+1}/{max_retries}): {e}. Retrying...", icon="🌐")
            last_error = e
            tries += 1
            time.sleep(tries * 2) # Exponential backoff
        except anthropic.RateLimitError as e:
            st.warning(f"Claude RateLimitError (Attempt {tries+1}/{max_retries}): {e}. Retrying...", icon="⏳")
            last_error = e
            tries += 1
            time.sleep(tries * 5) # Longer backoff for rate limits
        except anthropic.APIStatusError as e:
             st.error(f"Claude APIStatusError (Status {e.status_code}): {e.message}", icon="🔥")
             last_error = e
             # Don't retry on persistent status errors like 401, 403, 404
             if e.status_code in [401, 403, 404, 429]: # 429 is rate limit, handled above but good safety
                 break
             tries += 1
             time.sleep(tries * 2)
        except Exception as e:
            st.error(f"Unexpected error calling Claude (Attempt {tries+1}/{max_retries}): {e}", icon="💥")
            last_error = e
            tries += 1
            time.sleep(tries * 2)

    # If loop finishes without success
    st.error(f"Failed to get response from Claude after {max_retries} attempts. Last error: {last_error}", icon="🤖")
    return None


# --- Helper Function: Generate TTS Audio & Timestamps ---
def generate_audio_with_timestamps(text, client, voice_id="sage"):
    """Generates TTS audio using OpenAI, saves it, and gets word timestamps."""
    temp_audio_file = None
    temp_audio_path = None # Initialize path
    if not client:
        st.error("OpenAI client not initialized for TTS.", icon="🤖")
        return None, None
    try:
        # Ensure text is not empty
        if not text or not text.strip():
            raise ValueError("Input text for TTS cannot be empty.")

        # Generate TTS audio
        response = client.audio.speech.create(
                model="tts-1-hd", # Use HD for better quality
                voice=voice_id,
                input=text,
                response_format="mp3", # Use mp3 or opus
                speed=1.0 # Adjust speed if needed (1.0 is normal)
            )

        # Save the generated audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file_obj:
            temp_audio_path = temp_audio_file_obj.name
            temp_audio_file_obj.write(response.content)

        # --- Volume Boost (Optional but often helpful) ---
        try:
            boosted_audio = AudioSegment.from_file(temp_audio_path)
            # Increase volume - adjust dB as needed (e.g., +6 dB doubles perceived loudness)
            boosted_audio = boosted_audio + 8 # Boost by 8 dB
            boosted_audio.export(temp_audio_path, format="mp3")
        except Exception as boost_err:
            st.warning(f"Could not boost audio volume: {boost_err}", icon="🔊")
        # --- End Volume Boost ---

        # Transcribe boosted audio with OpenAI Whisper API for word timestamps
        with open(temp_audio_path, "rb") as audio_file_rb:
            transcribe_response = client.audio.transcriptions.create(
                file=audio_file_rb,
                model="whisper-1",
                response_format="verbose_json", # Necessary for word timestamps
                timestamp_granularities=["word"] # Request word-level timestamps
            )

        # --- Process Transcription Response ---
        word_timings = []
        if hasattr(transcribe_response, 'words') and transcribe_response.words:
            # transcribe_response.words is a list of TranscriptionWord objects
            for word_info in transcribe_response.words:
                word_timings.append({
                    "word": getattr(word_info, 'word', ''),
                    "start": getattr(word_info, 'start', None),
                    "end": getattr(word_info, 'end', None)
                })
        else:
            st.warning("Whisper did not return word timestamps in the expected format.", icon="⏱️")
            # Fallback parsing (keep as is)
            try:
                if isinstance(transcribe_response, str):
                    transcribe_data = json.loads(transcribe_response)
                elif hasattr(transcribe_response, 'to_dict'):
                    transcribe_data = transcribe_response.to_dict()
                elif isinstance(transcribe_response, dict):
                    transcribe_data = transcribe_response
                else:
                    transcribe_data = {}

                if isinstance(transcribe_data, dict) and 'words' in transcribe_data:
                    st.warning("Parsing Whisper response via fallback method.", icon="⚙️")
                    for word_info_dict in transcribe_data['words']:
                        word_timings.append({
                            "word": word_info_dict.get("word", ''),
                            "start": word_info_dict.get("start"),
                            "end": word_info_dict.get("end")
                        })
                else:
                    st.warning("Whisper response parsed, but no 'words' list found.", icon="⏱️")
            except Exception as parse_err:
                st.error(f"Could not parse Whisper transcription response: {parse_err}", icon="🎧")

        # --- Validate Timestamps ---
        valid_timings = []
        for wt in word_timings:
            if wt.get('start') is not None and wt.get('end') is not None:
                try:
                    start_time = float(wt['start'])
                    end_time = float(wt['end'])
                    if end_time >= start_time:
                        valid_timings.append({
                            "word": wt.get('word', ''),
                            "start": start_time,
                            "end": end_time
                        })
                    else:
                        st.warning(f"Skipping invalid time range (end <= start) for word '{wt.get('word')}': start={start_time}, end={end_time}", icon="⏱️")
                except (ValueError, TypeError) as conv_err:
                    st.warning(f"Skipping invalid timestamp format for word '{wt.get('word')}': start={wt['start']}, end={wt['end']} ({conv_err})", icon="⏱️")
            else:
                st.warning(f"Missing start/end time for word '{wt.get('word')}'", icon="⏱️")

        if not valid_timings and word_timings: # If validation removed all timings
             st.warning("No valid word timings could be extracted after validation.", icon="⚠️")

        # If valid timings were found, return the path to the audio file and the timings
        # The caller is responsible for deleting temp_audio_path later
        if valid_timings:
             return temp_audio_path, valid_timings
        else:
             # If no valid timings, still return the audio path (caller might want it)
             # but return None for timings to indicate failure there.
             # Cleanup of the audio file should happen in the caller if timings fail.
             return temp_audio_path, None


    except OpenAI.APIError as api_err:
        st.error(f"OpenAI API Error in TTS/Timestamp: {api_err}", icon="🤖")
    except ValueError as ve: # Catch specific errors like empty text
        st.error(f"Value Error in TTS/Timestamp: {ve}", icon="📄")
    except Exception as e:
        st.error(f"Unexpected error in TTS/Timestamp generation: {repr(e)}", icon="💥")
        import traceback
        traceback.print_exc() # Print full traceback to Streamlit logs
    finally:
        # DO NOT cleanup temp_audio_path here. The caller needs it.
        # Cleanup will happen in the batch processing loop's finally block.
        pass

    # Ensure function returns two values even on error
    return None, None # Return None for both if a major error occurred


# --- Helper Function: Group Word Timings ---
def group_words_with_timing(word_timings, words_per_group=2):
    """Groups words and their timings for subtitles."""
    grouped_timings = []
    if not word_timings: return grouped_timings

    for i in range(0, len(word_timings), words_per_group):
        group_words = word_timings[i:i+words_per_group]
        if group_words:
            combined_word = " ".join(word['word'] for word in group_words).strip() # Ensure no leading/trailing space
            start_time = group_words[0]['start']
            end_time = group_words[-1]['end']
             # Basic validation
            if combined_word and start_time is not None and end_time is not None and end_time > start_time:
                grouped_timings.append({
                    "text": combined_word,
                    "start": start_time,
                    "end": end_time
                })
            else:
                # print(f"Skipping invalid subtitle group: {group_words}")
                pass
    return grouped_timings


# --- Helper Function: Create Text Image for Subtitles ---
def create_text_image(text, fontsize, color, bg_color, font_path, video_width):
    """
    Creates a transparent PNG image with text and rounded background,
    wrapping text to fit video width and centering it.
    """
    try:
        # --- Font Loading ---
        font = None
        if font_path and os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except Exception as font_load_err:
                st.warning(f"Failed to load font {font_path}: {font_load_err}. Using default.", icon="⚠️")

        if not font: # If path was None or loading failed
            try:
                font = ImageFont.load_default(size=fontsize) # Try default with size
            except AttributeError:
                font = ImageFont.load_default() # Fallback for older Pillow
                st.warning(f"Using basic default font (size approx {fontsize}px). Provide TTF font for exact size.", icon="⚠️")


        # --- Configuration ---
        padding_x = 25  # Horizontal padding
        padding_y = 15  # Vertical padding
        bg_radius = 15  # Corner radius
        max_text_width = video_width - (2 * padding_x) - 30 # Max text width
        if max_text_width <= 0: max_text_width = video_width // 2 # Safety net

        # --- Text Wrapping ---
        lines = []
        words = text.split()
        if not words:
            return np.zeros((10, 10, 4), dtype=np.uint8) # Handle empty text

        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            try:
                bbox = font.getbbox(test_line)
                line_width = bbox[2] - bbox[0]
            except AttributeError: # Fallback
                line_width = font.getlength(test_line)

            if line_width <= max_text_width:
                current_line = test_line
            else:
                if current_line: lines.append(current_line)
                # Check if single word is too long
                try:
                     word_bbox = font.getbbox(word)
                     word_width = word_bbox[2] - word_bbox[0]
                except AttributeError:
                     word_width = font.getlength(word)

                if word_width <= max_text_width:
                    current_line = word
                else: # Word too long
                    if current_line and lines[-1] != current_line: pass
                    lines.append(word) # Add long word on its own line
                    current_line = ""

        if current_line: lines.append(current_line)
        wrapped_text = "\n".join(lines)
        if not wrapped_text: wrapped_text = text # Fallback

        # --- Calculate Text Block Dimensions ---
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        try:
            bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center')
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            bbox_y_offset = bbox[1]
            text_width = max(text_width, 1)
            text_height = max(text_height, 1)
        except AttributeError: # Fallback
            st.warning("Using fallback subtitle dimension calculation.", icon="PIL")
            text_width = 0
            for line in lines:
                 try:
                     line_bbox = font.getbbox(line)
                     text_width = max(text_width, line_bbox[2] - line_bbox[0])
                 except AttributeError:
                     text_width = max(text_width, int(font.getlength(line)))
            try:
                ascent, descent = font.getmetrics()
                line_height_metric = ascent + descent
            except AttributeError:
                line_height_metric = fontsize * 1.2 # Estimate
            line_height = line_height_metric + 4 # Add spacing
            text_height = len(lines) * line_height
            bbox_y_offset = -int(fontsize * 0.1) # Rough guess
            text_width = max(text_width, 1)
            text_height = max(text_height, 1)

        # --- Create Final Image ---
        img_width = text_width + 2 * padding_x
        img_height = text_height + 2 * padding_y
        img = Image.new("RGBA", (int(img_width), int(img_height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # --- Draw Background ---
        try:
            # Color parsing logic (keep as is)
            if isinstance(bg_color, str) and bg_color.startswith('rgba'):
                 parts = bg_color.strip('rgba()').split(',')
                 r, g, b = map(int, parts[:3])
                 a = int(float(parts[3]) * 255)
                 fill_color_tuple = (r, g, b, a)
            elif isinstance(bg_color, str):
                 fill_color_tuple = bg_color
            elif isinstance(bg_color, (tuple, list)) and len(bg_color) == 4:
                 fill_color_tuple = tuple(map(int, bg_color))
            else:
                 fill_color_tuple = tuple(map(int, bg_color)) + (int(0.6 * 255),) if len(bg_color) == 3 else (0,0,0, int(0.6*255))

            draw.rounded_rectangle([(0, 0), (img_width, img_height)], radius=bg_radius, fill=fill_color_tuple)
        except Exception as draw_err:
            st.warning(f"Could not draw rounded rect: {draw_err}. Using simple rect.", icon="🎨")
            if 'fill_color_tuple' not in locals(): fill_color_tuple = (0,0,0, int(0.6*255))
            draw.rectangle([(0,0), (img_width, img_height)], fill=fill_color_tuple)

        # --- Draw Text ---
        text_x = padding_x
        text_y = padding_y - bbox_y_offset # Adjust vertical start
        draw.multiline_text(
            (text_x, text_y),
            wrapped_text,
            font=font,
            fill=color,
            align="center",
            spacing=4,
            anchor="la" # Left-ascent anchor
        )

        return np.array(img)

    except Exception as e:
        st.error(f"Error creating text image for '{text[:50]}...': {e}", icon="🎨")
        return np.zeros((10, 10, 4), dtype=np.uint8)


# --- Helper Function: Process Video with TTS and Subtitles ---
def process_video_with_tts(base_video_url, audio_path, word_timings, topic, lang): # Added lang parameter
    """Loads video, adds TTS audio, loops if necessary, adds subtitles centered with wrapping."""
    final_video_clip = None
    temp_output_path = None
    base_video = None
    tts_audio = None
    looped_video = None
    processed_video = None
    subtitle_clips_list = []
    resized_base_video = None # Initialize

    try:
        st.write(f"⏳ Loading base video from URL...")
        try:
            # Let MoviePy handle the URL directly
            # Target 9:16 aspect ratio (e.g., 720x1280) for vertical video
            base_video = VideoFileClip(base_video_url, audio=False, target_resolution=(720, 1280))
        except Exception as load_err:
            st.error(f"Failed to load video using MoviePy: {load_err}", icon="🎬")
            raise # Re-raise the error

        video_duration = base_video.duration
        w = int(base_video.w) if base_video.w else 720
        h = int(base_video.h) if base_video.h else 1280
        st.write(f"✔️ Base video loaded: {w}x{h}, Duration: {video_duration:.2f}s")

        st.write(f"⏳ Loading TTS audio...")
        tts_audio = AudioFileClip(audio_path)
        audio_duration = tts_audio.duration
        st.write(f"✔️ TTS audio loaded: Duration: {audio_duration:.2f}s")

        # --- Video Resizing (Force 9:16 aspect ratio - e.g., 720x1280) ---
        target_w, target_h = 720, 1280
        st.write(f"⏳ Resizing video to {target_w}x{target_h} (if needed)...")
        try:
            # Check if resize is necessary
            if base_video.size != [target_w, target_h]:
                 # Simple resize (might distort if aspect ratio differs significantly)
                 # Consider vfx.crop for cropping to fit if aspect ratio preservation is key
                 resized_base_video = base_video.resize(newsize=(target_w, target_h))
                 st.write(f"✔️ Video resized.")
            else:
                 resized_base_video = base_video # No resize needed
                 st.write(f"✔️ Video already at target size.")
        except Exception as resize_err:
            st.warning(f"Could not resize video: {resize_err}. Using original dimensions.", icon="⚠️")
            resized_base_video = base_video # Fallback
            target_w, target_h = w, h # Use original dimensions

        # --- Video Looping/Trimming ---
        processed_video = resized_base_video # Start with potentially resized video
        if video_duration < audio_duration:
            st.write(f"⏳ Looping video to match audio duration...")
            num_loops = int(np.ceil(audio_duration / video_duration))
            clips_to_loop = [resized_base_video.copy().set_start(i * video_duration) for i in range(num_loops)]
            looped_video = concatenate_videoclips(clips_to_loop)
            processed_video = looped_video.subclip(0, audio_duration)
            st.write(f"✔️ Video looped {num_loops} times and trimmed to {audio_duration:.2f}s")
        elif video_duration > audio_duration:
            st.write(f"⏳ Trimming video to match audio duration...")
            processed_video = resized_base_video.subclip(0, audio_duration)
            st.write(f"✔️ Video trimmed to {audio_duration:.2f}s")
        else:
            st.write("✔️ Video duration matches audio duration.")

        # Set the TTS audio to the processed video
        final_video_clip = processed_video.set_audio(tts_audio).set_duration(audio_duration)
        # Ensure the clip size is correct after processing
        final_video_clip = final_video_clip.resize(newsize=(target_w, target_h))

        # --- Subtitle Generation ---
        st.write(f"⏳ Generating subtitles...")
        grouped_subs = group_words_with_timing(word_timings, words_per_group=SUBTITLE_WORDS_PER_GROUP)

        if grouped_subs:
            total_subs = len(grouped_subs)
            sub_progress_bar = st.progress(0)
            status_text = st.empty()

            for i, sub_data in enumerate(grouped_subs):
                text = sub_data['text']
                start = sub_data['start']
                end = sub_data['end']
                sub_duration = end - start

                if not text.strip() or sub_duration <= 0.05: continue # Skip empty/short subs

                status_text.text(f"Creating subtitle {i+1}/{total_subs}: '{text[:30]}...'")

                # Create subtitle image WITH target video_width
                text_img_array = create_text_image(
                    text.upper(),
                    fontsize=SUBTITLE_FONT_SIZE,
                    color=SUBTITLE_COLOR,
                    bg_color=SUBTITLE_BG_COLOR,
                    font_path=SUBTITLE_FONT_PATH,
                    video_width=target_w # Pass target width
                )

                if text_img_array.shape[0] <= 10 or text_img_array.shape[1] <= 10:
                    st.warning(f"Skipping subtitle due to image creation error for: '{text[:30]}...'")
                    continue

                # Create ImageClip and Position
                subtitle_img_clip = ImageClip(text_img_array)\
                    .set_start(start)\
                    .set_duration(sub_duration)\
                    .set_position(('center', 'center')) # Center position

                subtitle_clips_list.append(subtitle_img_clip)
                sub_progress_bar.progress((i + 1) / total_subs)

            status_text.text(f"✔️ Generated {len(subtitle_clips_list)} subtitle clips.")
            # sub_progress_bar.empty() # Optional: clear progress bar
            # status_text.empty()      # Optional: clear status text
        else:
            st.warning("No valid word timings available to generate subtitles.", icon="⏱️")

        # Composite final video with subtitles
        if subtitle_clips_list:
            st.write("⏳ Compositing video and subtitles...")
            clips_for_composite = [final_video_clip] + subtitle_clips_list
            # Explicitly set the size for CompositeVideoClip using the target dimensions
            final_video_clip = CompositeVideoClip(clips_for_composite, size=(target_w, target_h))
            st.write("✔️ Compositing complete.")
        else:
            st.write("ℹ️ No subtitles added.")

        # --- Export Final Video ---
        st.write("⏳ Exporting final video...")
        timestamp = int(datetime.datetime.now().timestamp())
        safe_topic = urllib.parse.quote(topic.replace(' ', '_')[:30], safe='')
        safe_lang = urllib.parse.quote(lang.replace(' ', '_')[:15], safe='') # Use lang in filename
        temp_output_filename = f"final_{safe_topic}_{safe_lang}_{timestamp}.mp4" # Use safe_lang

        # Use a unique temp file name including language
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=f"final_{safe_topic}_{safe_lang}_") as temp_output_file_obj:
             temp_output_path = temp_output_file_obj.name

        if not isinstance(final_video_clip, (VideoFileClip, CompositeVideoClip)):
             raise TypeError(f"Cannot write final video: Invalid clip object type {type(final_video_clip)}.")

        # Use recommended parameters for web compatibility
        final_video_clip.write_videofile(
            temp_output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=f'temp-audio-{safe_lang}-{timestamp}.m4a', # Unique temp audio file
            remove_temp=True,
            fps=resized_base_video.fps if resized_base_video and resized_base_video.fps else 24, # Use fps from resized video or default
            preset='medium', # 'medium' is a good balance
            threads=os.cpu_count() or 4, # Use available cores
            logger='bar', # Show progress bar in console/logs
            ffmpeg_params=[ # Parameters for faster start/web compatibility
                '-movflags', 'faststart',
                '-profile:v', 'high', # H.264 profile
                '-level', '4.0', # H.264 level
            ]
        )
        st.write(f"✔️ Final video exported to temporary path: {temp_output_path}")

        return temp_output_path, temp_output_filename # Return filename too

    except Exception as e:
        st.error(f"Error during video processing: {e}", icon="🎬")
        import traceback
        traceback.print_exc()
        # Return None to indicate failure
        return None, None

    finally:
        # --- Cleanup ---
        st.write("🧹 Cleaning up video processing resources...")
        try:
             # Close clips safely, checking if they exist and are different objects
             if base_video: base_video.close()
             if tts_audio: tts_audio.close()
             if looped_video and looped_video is not processed_video: looped_video.close()
             if processed_video and processed_video is not final_video_clip and processed_video is not resized_base_video: processed_video.close()
             if resized_base_video and resized_base_video is not base_video: resized_base_video.close()
             if final_video_clip: final_video_clip.close() # Close the final composite clip
             for sub_clip in subtitle_clips_list: sub_clip.close()

        except Exception as cleanup_err:
            st.warning(f"Error during resource cleanup: {cleanup_err}")
        st.write("🧹 Cleanup finished.")


# --- Helper Function: Upload Video to S3 ---
def upload_vid_to_s3(s3_cli, video_path, bucket_name, object_name, region_name):
    """Uploads a video file to S3."""
    if not s3_cli:
        st.error("S3 Client not initialized. Cannot upload.", icon="🚫")
        return None
    if not os.path.exists(video_path):
        st.error(f"Video file not found at path: {video_path}", icon="❌")
        return None

    try:
        with open(video_path, "rb") as video_file:
            s3_cli.put_object(
                Bucket=bucket_name,
                Key=object_name,
                Body=video_file,
                ContentType='video/mp4' # Set appropriate content type
            )

        # URL encode the object name for safety in URLs
        safe_object_name = urllib.parse.quote(object_name)

        # Construct the URL (common format, ensure bucket policy allows public read if needed)
        if region_name == 'us-east-1':
            video_url = f"https://{bucket_name}.s3.amazonaws.com/{safe_object_name}"
        else:
            video_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{safe_object_name}"

        st.success(f"✔️ Video uploaded to S3: {object_name}")
        return video_url
    except NoCredentialsError:
        st.error("AWS Credentials not available for S3 upload.", icon="🔒")
        return None
    except Exception as e:
        st.error(f"S3 Upload Error: {e}", icon="☁️")
        return None
# <<< END: IMPORT STATEMENTS AND HELPER FUNCTIONS >>>


# --- Streamlit App ---

st.title("🎬 YouTube Video Selector & TTS Generator")
st.caption("Search YouTube, select videos, generate TTS script, create subtitled videos, and upload to S3.")

# --- Session State Initialization ---

# Stores data for each *selected* video {videoId: video_data_dict}
if 'selected_videos' not in st.session_state:
    st.session_state.selected_videos = {}

# Stores the results from the last API search {search_term: {'videos': [], 'topic': ''}}
if 'api_search_results' not in st.session_state:
    st.session_state.api_search_results = {}

# Input DataFrame for search terms and topics
if 'search_data' not in st.session_state:
    st.session_state.search_data = pd.DataFrame([
        {'Topic': 'sofa sale', 'Search Term': 'sofa unboxing #shorts' ,'Language' : 'English', 'Video Results': 5},
        {'Topic': 'used cars', 'Search Term': 'my first car #shorts' ,'Language' : 'English,Spanish', 'Video Results': 3} # Example multi-lang
    ])

# Flag to indicate if a search has been run
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False

# --- State for Batch Processing ---
# List of video IDs queued for generation
if 'generation_queue' not in st.session_state:
    st.session_state.generation_queue = []
# Flag indicating batch processing is active
if 'batch_processing_active' not in st.session_state:
    st.session_state.batch_processing_active = False
# Store total count for progress display
if 'batch_total_count' not in st.session_state:
    st.session_state.batch_total_count = 0
# Store count of processed items
if 'batch_processed_count' not in st.session_state:
    st.session_state.batch_processed_count = 0


# --- Input Area (Sidebar) ---
st.sidebar.header("Inputs & Actions")

# API key is handled via secrets

st.sidebar.write("Enter Search Terms and Topics:")
# Use data_editor for input
edited_df = st.sidebar.data_editor(
    st.session_state.search_data,
    num_rows="dynamic",
    use_container_width=True,
    key="search_topic_editor",
    column_config={ # Optional: Improve editor experience
         "Language": st.column_config.TextColumn(
            "Language(s) (comma-sep)",
            help="Enter one or more languages, separated by commas (e.g., English,Spanish)",
            default="English",
        ),
        "Video Results": st.column_config.NumberColumn(
             "Max Results",
             min_value=1,
             max_value=20, # Limit max results per search term
             step=1,
             default=5,
        )
    }
)

# Update session state AFTER the editor widget
st.session_state.search_data = edited_df


# --- Action Buttons (Sidebar) ---
col1, col2 = st.sidebar.columns(2)
search_button = col1.button("🔍 Search Videos", use_container_width=True, disabled=st.session_state.batch_processing_active)
clear_button = col2.button("🧹 Clear All", use_container_width=True, type="secondary", disabled=st.session_state.batch_processing_active)

if clear_button:
    st.session_state.selected_videos = {}
    st.session_state.search_triggered = False
    st.session_state.api_search_results = {}
    st.session_state.generation_queue = [] # Clear queue
    st.session_state.batch_processing_active = False # Reset flag
    st.session_state.batch_total_count = 0
    st.session_state.batch_processed_count = 0
    # Reset the input table to default
    st.session_state.search_data = pd.DataFrame([
        {'Topic': 'sofa sale', 'Search Term': 'sofa unboxing #shorts' ,'Language' : 'English', 'Video Results': 5},
        {'Topic': 'used cars', 'Search Term': 'my first car #shorts' ,'Language' : 'English,Spanish', 'Video Results': 3}
    ])
    st.success("Selections, results, and generation state cleared!", icon="✅")
    st.rerun()

# --- Global Process Button ---
st.sidebar.divider()
# Calculate how many selected videos are ready for processing
ready_videos_count = sum(
    1 for vid_id, data in st.session_state.selected_videos.items()
    # Ready if URL exists, no URL error, and not already fully/partially processed or failed completely
    if data.get('Direct URL') and not data.get('yt_dlp_error') and 'Completed' not in data.get('Status','') and 'Failed' not in data.get('Status','') and 'Skipped' not in data.get('Status','')
)

process_all_button = st.sidebar.button(
    f"🎬 Process {ready_videos_count} Ready Video{'s' if ready_videos_count != 1 else ''}",
    use_container_width=True,
    type="primary",
    disabled=ready_videos_count == 0 or st.session_state.batch_processing_active # Disable if 0 ready or already processing
)

if process_all_button:
    # Find videos that are selected, have a direct URL, and haven't been successfully processed or failed
    ids_to_process = [
        vid_id for vid_id, data in st.session_state.selected_videos.items()
        if data.get('Direct URL') and not data.get('yt_dlp_error') and 'Completed' not in data.get('Status','') and 'Failed' not in data.get('Status','') and 'Skipped' not in data.get('Status','')
    ]
    if ids_to_process:
        st.session_state.generation_queue = ids_to_process
        st.session_state.batch_processing_active = True
        st.session_state.batch_total_count = len(ids_to_process)
        st.session_state.batch_processed_count = 0
        st.sidebar.info(f"Queued {len(ids_to_process)} videos for generation.", icon="⏳")
        # Clear any potential old individual errors before starting batch
        for vid_id in ids_to_process:
             # Reset generation status fields
             st.session_state.selected_videos[vid_id]['Generation Errors'] = {}
             st.session_state.selected_videos[vid_id]['Generated URLs'] = {}
             st.session_state.selected_videos[vid_id]['Status'] = 'Queued' # Set status to Queued
        st.rerun()
    else:
        st.sidebar.warning("No selected videos are ready for processing (need Direct URL and not already processed/failed).", icon="⚠️")


st.sidebar.info("Select videos using '➕'. Processing uses selected videos with valid 'Direct URL'.", icon="ℹ️")
st.sidebar.warning("Video generation can take several minutes per video per language.", icon="⏱️")

# --- Processing Logic ---

# 1. Handle Search Button Click
if search_button:
    # --- Get data directly from the potentially edited DataFrame in session state ---
    search_df_raw = st.session_state.search_data
    valid_input = True

    # --- Data Validation and Cleaning ---
    try:
        if not isinstance(search_df_raw, pd.DataFrame):
             raise ValueError("Input data is not a DataFrame.")

        search_df = search_df_raw.copy()

        # Ensure essential columns exist, provide defaults
        expected_cols = ["Topic", "Search Term", "Language", "Video Results"]
        for col in expected_cols:
            if col not in search_df.columns:
                if col == "Language": search_df[col] = "English"
                elif col == "Video Results": search_df[col] = 5
                else: search_df[col] = "" # Topic, Search Term

        search_df = search_df[expected_cols] # Enforce column order

        # Convert types and handle missing/invalid values
        search_df['Topic'] = search_df['Topic'].fillna('').astype(str).str.strip()
        search_df['Search Term'] = search_df['Search Term'].fillna('').astype(str).str.strip()
        search_df['Language'] = search_df['Language'].fillna('English').astype(str).str.strip()
        # Replace empty language strings with default
        search_df['Language'] = search_df['Language'].apply(lambda x: 'English' if not x else x)

        # Convert 'Video Results', coerce errors to default (5), ensure integer, min value 1
        search_df['Video Results'] = pd.to_numeric(search_df['Video Results'], errors='coerce').fillna(5).astype(int)
        search_df['Video Results'] = search_df['Video Results'].apply(lambda x: max(1, min(x, 20))) # Ensure 1 <= x <= 20

        # Filter out rows where BOTH Topic and Search Term are empty
        is_row_meaningful = (search_df['Topic'] != '') | (search_df['Search Term'] != '')
        search_df_filtered = search_df[is_row_meaningful].copy()

        if search_df_filtered.empty:
            st.sidebar.warning("Please provide at least one row with a 'Topic' or 'Search Term'.", icon="⚠️")
            valid_input = False
        else:
            search_df = search_df_filtered # Use the filtered DataFrame

            # Check for empty required fields in the *filtered* rows
            if search_df['Search Term'].eq('').any():
                st.sidebar.warning("One or more rows are missing a 'Search Term'.", icon="⚠️")
                # Allow auto-search term generation later if 'auto' is used
                # valid_input = False # Decide if this should block the search
            if search_df['Topic'].eq('').any():
                st.sidebar.warning("One or more rows are missing a 'Topic'.", icon="⚠️")
                valid_input = False # Topic is generally required for context

    except Exception as e:
        st.sidebar.error(f"Error processing input table: {e}", icon="🆘")
        valid_input = False

    # --- Proceed if input is valid ---
    if valid_input:
        st.sidebar.success("Input valid, proceeding with search.") # Feedback
        st.session_state.search_triggered = True
        st.session_state.api_search_results = {} # Clear previous results
        # Clear generation state (queue is handled by Process All button)
        st.session_state.batch_processing_active = False
        st.session_state.batch_total_count = 0
        st.session_state.batch_processed_count = 0
        # Store the validated DataFrame for the search step
        st.session_state.current_search_df = search_df

        # Update the editor's base data for the *next* run
        st.session_state.search_data = search_df.copy()

        st.rerun()
    else:
        st.session_state.search_triggered = False # Ensure search doesn't proceed


# 2. Perform API Search if Triggered
if st.session_state.search_triggered and 'current_search_df' in st.session_state and not st.session_state.api_search_results:
    search_df = st.session_state.current_search_df
    search_items = search_df.to_dict('records') # Convert df rows to list of dicts

    st.info(f"Searching API for {len(search_items)} term(s)...", icon="🔍")
    progress_bar = st.progress(0)
    status_text_api = st.empty()

    api_error_occurred = False
    results_cache = {} # Temp cache for this run

    # --- Use LLM for 'auto' search terms ---
    search_terms_to_generate = [item['Topic'] for item in search_items if item['Search Term'].lower() == 'auto']
    generated_search_terms = {} # Dict to store topic -> generated terms

    if search_terms_to_generate:
         with st.spinner(f"Generating 'auto' search terms using Claude for {len(search_terms_to_generate)} topics..."):
              prompts = []
              for topic_to_gen in search_terms_to_generate:
                   prompt = f"""You are a viral video ad expert. I will give you a topic, and you will return the top 3 YouTube Shorts search terms that:
- Are short (2–5 words)
- Clearly describe what viewers will see in the video (visuals only)
- Lead to emotionally engaging, surprising, or curiosity-triggering content
- Are perfect for remixing or using as inspiration for Facebook video ads
- Focus on things like transformations, objects in motion, satisfying actions, luxury aesthetics, clever space-saving, or unexpected reveals
- Avoid abstract or advice-based phrases (like “tips,” “hacks,” or “secrets”)
- Avoid using non visual\\describing words that are not likely to be relevent (like 'On credit', "Financing", etc)
- Add '#shorts' to the end of each search term and separate terms with ' | '

Example:
Input: sofa
Output:
sofa transformation #shorts | hidden bed sofa #shorts | luxury sofa unboxing #shorts

My topic:
{topic_to_gen}"""
                   prompts.append(prompt)

              # Simple sequential generation (can be parallelized if needed)
              for i, topic_to_gen in enumerate(search_terms_to_generate):
                  generated = claude(prompts[i], client=anthropic_client)
                  if generated:
                      generated_search_terms[topic_to_gen] = generated.strip("'\" ") # Store cleaned terms
                  else:
                      generated_search_terms[topic_to_gen] = topic_to_gen # Fallback to topic if generation fails
                      st.warning(f"Could not generate search term for topic '{topic_to_gen}', using topic itself.", icon="⚠️")


    # --- Perform Searches ---
    for i, item in enumerate(search_items):
        term = item['Search Term']
        topic = item['Topic']
        count = int(item['Video Results'])
        lang = item['Language'] # Keep original language string

        # Substitute 'auto' term if generated
        if term.lower() == 'auto':
            term = generated_search_terms.get(topic, topic) # Use generated or fallback to topic

        status_text_api.text(f"Searching ({i+1}/{len(search_items)}): '{term}'...")

        # Use a unique key for caching results based on term AND topic to avoid collision
        # Although term might be unique if generated, topic ensures context
        cache_key = f"{term}_{topic}"

        if cache_key not in results_cache: # Avoid re-searching same term/topic combo
            videos = search_youtube(youtube_api_key_secret, term, count)

            if videos is None: # Indicates critical API error in search_youtube
                st.error(f"Stopping search due to critical API issue (check key/quota) for term: '{term}'.", icon="🚫")
                api_error_occurred = True
                break # Stop processing further terms

            # Store results along with the topic and original language string
            results_cache[cache_key] = {'videos': videos, 'topic': topic , 'lang' : lang, 'search_term_used': term}

            time.sleep(0.1) # Small delay between API calls
        progress_bar.progress((i + 1) / len(search_items))

    status_text_api.text("API Search complete.")
    st.session_state.api_search_results = results_cache # Update main cache
    st.session_state.search_triggered = False # Prevent infinite rerun loop

    if api_error_occurred:
        st.warning("Search halted due to API error.", icon="⚠️")
        # Don't stop the app, allow rerun to display partial results/errors
        st.rerun()
    else:
        st.rerun() # Rerun to display results


# --- Display Area ---
st.divider()

# --- Display Batch Processing Status ---
if st.session_state.batch_processing_active:
    processed = st.session_state.batch_processed_count
    total = st.session_state.batch_total_count
    queue_len = len(st.session_state.generation_queue)
    progress_percent = (processed / total) if total > 0 else 0
    st.info(f"⚙️ Batch Processing: {processed} / {total} Completed ({progress_percent:.0%}). {queue_len} Remaining.", icon="⏳")
    if total > 0:
        st.progress(progress_percent)


# 3. Display Search Results
if st.session_state.api_search_results:
    st.subheader("Search Results & Video Selection (Grid View)")

    # Display results from cache
    for cache_key, result_data in st.session_state.api_search_results.items():
        videos = result_data['videos']
        topic = result_data['topic']
        lang = result_data['lang'] # Original language string for selection state
        search_term_used = result_data['search_term_used']

        # --- Container for the results of THIS search term/topic ---
        term_container = st.container(border=True)
        with term_container:
            st.subheader(f"Results for Topic: \"{topic}\" (Searched: \"{search_term_used}\")")
            if not videos:
                st.write("No videos found via API for this search.")
                continue

            num_videos = len(videos)
            num_cols = 3 # Number of columns/videos per row

            for i in range(0, num_videos, num_cols):
                cols = st.columns(num_cols)

                for j in range(num_cols):
                    video_index = i + j
                    if video_index < num_videos:
                        video = videos[video_index]
                        # --- Place ALL rendering logic for one video inside its column ---
                        with cols[j]:
                            # --- Extract Video Info ---
                            video_id = video['videoId']
                            video_title = video['title']
                            standard_video_url = video['standard_url']
                            embed_url = video['embed_url'] # Use for iframe
                            thumbnail_url = video['thumbnail_url']
                            # Use video_id for unique keys
                            unique_key_base = f"{video_id}_{i+j}" # Simpler key

                            # --- State for controlling video player visibility ---
                            show_video_key = f"show_player_{unique_key_base}"
                            if show_video_key not in st.session_state:
                                st.session_state[show_video_key] = False

                            # --- Get current state for this video ---
                            is_selected = video_id in st.session_state.selected_videos
                            video_state = st.session_state.selected_videos.get(video_id, {})

                            # --- Render Content Vertically ---
                            st.write(f"**{textwrap.shorten(video_title, width=50, placeholder='...')}**")
                            st.caption(f"ID: {video_id}")

                            # --- Conditionally Display Video Player OR Thumbnail ---
                            if st.session_state[show_video_key]:
                                try:
                                    # Use iframe for better embedding control
                                    iframe_code = f"""
                                    <iframe width="100%" height="315"
                                    src="{embed_url}?autoplay=0&modestbranding=1&rel=0"
                                    title="YouTube video player" frameborder="0"
                                    allow="accelerometer; autoplay; clipboard-write; encrypted-media;
                                    gyroscope; picture-in-picture; web-share"
                                    referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
                                    """
                                    st.markdown(iframe_code, unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Preview failed: {e}")
                            else:
                                st.image(thumbnail_url, use_container_width=True, caption="Click 'Show Preview' to load video")

                            # --- Buttons ---
                            # Toggle Video Player Visibility Button
                            toggle_label = "🔼 Hide" if st.session_state[show_video_key] else "▶️ Show"
                            if st.button(
                                f"{toggle_label} Preview",
                                key=f"toggle_vid_btn_{unique_key_base}",
                                help="Show/hide the video preview",
                                use_container_width=True
                            ) :
                                st.session_state[show_video_key] = not st.session_state[show_video_key]
                                st.rerun()

                            # Select / Deselect Button Logic ( Incorporating Change 1 )
                            select_disabled = st.session_state.batch_processing_active or video_state.get('fetching_dlp', False)
                            if is_selected:
                                select_button_label = "✅ Deselect"
                                select_button_type = "secondary"
                            else:
                                select_button_label = "➕ Select"
                                select_button_type = "primary"

                            if video_state.get('fetching_dlp', False):
                                 select_button_label = "⏳ Fetching URL..."
                                 select_button_type = "secondary"


                            if st.button(
                                select_button_label,
                                key=f"select_{unique_key_base}",
                                type=select_button_type,
                                use_container_width=True,
                                disabled=select_disabled
                            ):
                                if is_selected:
                                    # Deselect
                                    if video_id in st.session_state.selected_videos:
                                        del st.session_state.selected_videos[video_id]
                                        st.toast(f"Deselected: {video_title}", icon="➖")
                                    if video_id in st.session_state.generation_queue:
                                        st.session_state.generation_queue.remove(video_id)
                                        st.session_state.batch_total_count = len(st.session_state.generation_queue)
                                else:
                                    # Select (New Logic from Change 1)
                                    st.session_state.selected_videos[video_id] = {
                                        'Search Term Used': search_term_used, # Store the actual term used
                                        'Topic': topic,
                                        'Source Languages': lang,     # Store the original language string
                                        'Video Title': video_title,
                                        'Video ID': video_id,
                                        'Standard URL': standard_video_url, # Store standard URL for fetching
                                        'fetching_dlp': True,         # Trigger fetch ONCE
                                        'Direct URL': None,
                                        'Format Details': None,
                                        'yt_dlp_error': None,
                                        'Generated URLs': {},         # Initialize dict for language-specific URLs
                                        'Generation Errors': {},     # Initialize dict for language-specific errors
                                        'Status': 'Selected, Fetching URL...' # Initial status
                                    }
                                    st.toast(f"Selected: {video_title}. Fetching direct URL...", icon="⏳")
                                st.rerun()

                            # --- Status Display Block (Reflecting Change 1 state structure) ---
                            status_container = st.container()
                            if is_selected: # Check if the key exists
                                video_state = st.session_state.selected_videos[video_id] # Get the single state entry
                                is_fetching_dlp = video_state.get('fetching_dlp', False)
                                dlp_error = video_state.get('yt_dlp_error')
                                has_dlp_info = bool(video_state.get('Direct URL'))
                                status_text = video_state.get('Status', 'Unknown') # Get current status text
                                generated_urls_dict = video_state.get('Generated URLs', {})
                                generation_errors_dict = video_state.get('Generation Errors', {})

                                if is_fetching_dlp:
                                    status_container.info("⏳ Fetching URL...", icon="📡")
                                elif dlp_error:
                                    status_container.error(f"URL Error: {dlp_error}", icon="⚠️")
                                elif not has_dlp_info and not is_fetching_dlp:
                                     status_container.warning("URL fetch failed or pending.", icon="❓")
                                elif has_dlp_info:
                                    # Display detailed status based on the 'Status' field
                                    if 'Processing' in status_text:
                                         status_container.info(f"⚙️ {status_text}", icon="⏳")
                                    elif 'Queued' in status_text:
                                         status_container.info("🕒 Queued", icon="🕒")
                                    elif 'Completed' in status_text:
                                        status_container.success(f"✔️ {status_text}", icon="🎉")
                                        # Show links to generated videos
                                        if generated_urls_dict:
                                             with st.expander("View Generated Video Links"):
                                                  for l, u in generated_urls_dict.items():
                                                       st.link_button(f"View ({l})", url=u) # Use default width
                                    elif 'Failed' in status_text or 'Skipped' in status_text:
                                        status_container.error(f"❌ {status_text}", icon="🔥")
                                        # Show specific errors
                                        if generation_errors_dict:
                                             with st.expander("View Generation Errors"):
                                                  for l, e in generation_errors_dict.items():
                                                      st.caption(f"Error ({l}): {str(e)}")
                                    elif 'Ready' in status_text:
                                         status_container.success("✅ Ready to Process", icon="👍")
                                    else: # Default if status is unexpected
                                         status_container.info(f"Status: {status_text}", icon="ℹ️")
                                else:
                                    # Fallback if state is inconsistent
                                    if not is_fetching_dlp: # Only show if not actively fetching
                                        status_container.warning("Status unavailable.", icon="❓")

# --- yt-dlp Fetching Logic (Incorporating Change 2 - status messages) ---
if not st.session_state.batch_processing_active:
    ids_to_fetch = [
        vid for vid, data in st.session_state.selected_videos.items()
        if data.get('fetching_dlp') # Only fetch if marked
    ]

    if ids_to_fetch:
        fetch_id = ids_to_fetch[0] # Process one video_id at a time per rerun
        video_data = st.session_state.selected_videos.get(fetch_id)

        if video_data:
            standard_url = video_data.get('Standard URL') # Use the standard URL stored
            title = video_data.get('Video Title', fetch_id)

            with st.spinner(f"Fetching direct URL for '{title}'..."):
                 dlp_info = get_yt_dlp_info(standard_url)

            current_state = st.session_state.selected_videos.get(fetch_id)
            if current_state:
                current_state['fetching_dlp'] = False # Mark fetch attempt complete

                if dlp_info and dlp_info.get('direct_url'):
                    current_state['Direct URL'] = dlp_info['direct_url']
                    current_state['Format Details'] = dlp_info['format_details']
                    current_state['yt_dlp_error'] = None
                    current_state['Status'] = 'Ready' # Status: Ready for processing
                    st.toast(f"Direct URL loaded for '{title}'", icon="✅")
                elif dlp_info and dlp_info.get('error'):
                    current_state['Direct URL'] = None
                    current_state['Format Details'] = "Error"
                    current_state['yt_dlp_error'] = dlp_info['error']
                    current_state['Status'] = f"Error (URL Fetch): {dlp_info['error']}" # Status: Error
                    st.toast(f"yt-dlp failed for '{title}': {dlp_info['error']}", icon="⚠️")
                else: # Critical failure
                    current_state['Direct URL'] = None
                    current_state['Format Details'] = "Critical Error"
                    error_msg = dlp_info.get('error', "Critical yt-dlp failure") if dlp_info else "Critical yt-dlp failure"
                    current_state['yt_dlp_error'] = error_msg
                    current_state['Status'] = f"Error (URL Fetch): {error_msg}" # Status: Error
                    st.toast(f"Critical yt-dlp error for '{title}'", icon="🔥")

                st.session_state.selected_videos[fetch_id] = current_state
                st.rerun()

# 4. Video Generation Logic (BATCH PROCESSING - Incorporating Change 4)
if st.session_state.batch_processing_active and st.session_state.generation_queue:
    video_id_to_process = st.session_state.generation_queue[0] # Peek at the next video_id
    video_data = st.session_state.selected_videos.get(video_id_to_process)

    # Initialize lists for cleanup (can have multiple per video_id now)
    temp_audio_paths_this_run = []
    temp_final_video_paths_this_run = []
    generation_success_overall = False # Flag to track if at least one lang succeeded for this video_id

    if video_data and video_data.get('Direct URL') and not video_data.get('yt_dlp_error'):
        processed_count_display = st.session_state.batch_processed_count + 1
        total_count_display = st.session_state.batch_total_count
        st.header(f"⚙️ Processing Video {processed_count_display}/{total_count_display}: {video_data['Video Title']}")
        gen_placeholder = st.container(border=True) # Use a container for logs

        # --- Get Languages ---
        source_langs_str = video_data.get('Source Languages', 'English') # Default to English
        langs_to_process = [lang.strip() for lang in source_langs_str.split(',') if lang.strip()]
        if not langs_to_process: langs_to_process = ['English'] # Ensure at least one

        # --- Store results and errors per language for this run ---
        generated_urls_dict = {}
        generation_errors_dict = {}

        with gen_placeholder:
            st.info(f"Starting video generation process for language(s): {', '.join(langs_to_process)}")

            # --- Loop through each language for this video_id ---
            for index, current_lang in enumerate(langs_to_process):
                lang_status_key = f"{video_id_to_process}_{current_lang}" # Unique key
                st.markdown(f"--- Processing Language: **{current_lang}** ({index+1}/{len(langs_to_process)}) ---")

                # Use st.status for collapsible logs per language
                with st.status(f"Running generation steps for {current_lang}...", state="running", expanded=True) as status_log:
                    audio_path = None # Reset for each language
                    final_video_path = None # Reset for each language
                    try:
                        # Update Status in main state for this video_id
                        st.session_state.selected_videos[video_id_to_process]['Status'] = f'Processing ({current_lang})...'
                        # No rerun here, rely on status_log and final rerun

                        # --- Step 1: Get Topic ---
                        topic = video_data.get('Topic', 'the selected video')
                        status_log.write(f"1/5: Using topic: '{topic}' for language '{current_lang}'...")

                        # --- Step 2: Generate Script ---
                        status_log.write(f"2/5: Generating script using Claude for {current_lang}...")
                        script_prompt = f"Create a short, engaging voiceover script for FB viral video (roughly 15-20 seconds long, maybe 2-3 sentences) about '{topic}' in language {current_lang}. The tone should be informative yet conversational, smooth flow. Just provide the script text, nothing else. create intriguing and engaging script, sell the topic to the audience . be very causal and not 'advertisement' style vibe. end with a call to action 'tap to....' .the text needs to be retentive.NOTE:: DO NOT dont use senetional words and phrasing and DONT make false promises !!! "
                        script_text = claude(script_prompt, client=anthropic_client) # Use Claude

                        if not script_text:
                            raise ValueError(f"Failed to generate script text from Claude for '{current_lang}'.")
                        # Display script inside the status log
                        status_log.text_area(f"Generated Script ({current_lang}):", script_text, height=100, key=f"script_{lang_status_key}")
                        status_log.write(f"3/5: Generating TTS audio and timestamps for {current_lang}...")

                        # --- Step 3: Generate TTS Audio & Timestamps ---
                        audio_path, word_timings = generate_audio_with_timestamps(
                            script_text, client=openai_client, voice_id=DEFAULT_TTS_VOICE
                        )
                        # Check if audio generated but timings failed
                        if audio_path and word_timings is None:
                             status_log.warning("TTS audio generated, but failed to get word timestamps. Proceeding without subtitles.", icon="⏱️")
                             # Allow proceeding, but process_video_with_tts will handle empty timings
                        elif audio_path is None: # Complete TTS failure
                            raise ValueError(f"Failed to generate TTS audio for '{current_lang}'.")

                        if audio_path: temp_audio_paths_this_run.append(audio_path) # Add for cleanup

                        status_log.write(f"4/5: Processing base video and adding audio/subtitles for {current_lang}...")

                        # --- Step 4: Process Video ---
                        final_video_path, final_filename = process_video_with_tts(
                            base_video_url=video_data['Direct URL'],
                            audio_path=audio_path,
                            word_timings=word_timings if word_timings else [], # Pass empty list if timings failed
                            topic=topic,
                            lang=current_lang # Pass current language
                        )
                        if not final_video_path:
                            raise ValueError(f"Video processing (MoviePy) failed for '{current_lang}'.")
                        temp_final_video_paths_this_run.append(final_video_path) # Add for cleanup
                        status_log.write(f"5/5: Uploading final video '{final_filename}' to S3 for {current_lang}...")

                        # --- Step 5: Upload to S3 ---
                        s3_url = upload_vid_to_s3(
                            s3_cli=s3_client,
                            video_path=final_video_path,
                            bucket_name=s3_bucket_name,
                            object_name=final_filename, # Use the language-specific filename
                            region_name=s3_region
                        )
                        if not s3_url:
                            raise ValueError(f"Failed to upload video to S3 for '{current_lang}'.")
                        status_log.write(f"✔️ Generation Complete for {current_lang}! S3 URL ready.")

                        status_log.update(label=f"Generation Complete for {current_lang}!", state="complete", expanded=False)
                        generated_urls_dict[current_lang] = s3_url # Store success URL
                        generation_success_overall = True # Mark success for this video_id

                    except Exception as e:
                        error_message = str(e)[:250] # Truncate error message
                        status_log.error(f"Error for {current_lang}: {error_message}", icon="🔥")
                        status_log.update(label=f"Generation Failed for {current_lang}: {error_message}", state="error", expanded=True)
                        generation_errors_dict[current_lang] = error_message # Store error

            # --- End of language loop ---

            # --- Update Session State for the main video_id entry after processing all languages ---
            # Use .update() to merge results, preserving any previous ones if re-processing
            current_gen_urls = st.session_state.selected_videos[video_id_to_process].get('Generated URLs', {})
            current_gen_urls.update(generated_urls_dict)
            st.session_state.selected_videos[video_id_to_process]['Generated URLs'] = current_gen_urls

            current_gen_errors = st.session_state.selected_videos[video_id_to_process].get('Generation Errors', {})
            current_gen_errors.update(generation_errors_dict)
            st.session_state.selected_videos[video_id_to_process]['Generation Errors'] = current_gen_errors


            # --- Determine Final Status for this video_id ---
            final_status = "Unknown"
            total_langs = len(langs_to_process)
            success_count = len(current_gen_urls) # Count successful URLs in the updated dict
            error_count = len(current_gen_errors)   # Count errors in the updated dict

            if error_count > 0:
                if success_count > 0:
                    final_status = f"Completed with Errors ({success_count}/{total_langs} langs OK)"
                else:
                    final_status = f"Failed ({error_count}/{total_langs} langs)"
            elif success_count == total_langs:
                 final_status = f"Completed ({success_count}/{total_langs} langs)"
            elif success_count > 0: # Some succeeded, none failed explicitly - might happen if some skipped?
                 final_status = f"Partially Completed ({success_count}/{total_langs} langs)"
            else: # No successes, no errors recorded
                 final_status = "Finished (No output recorded)"

            st.session_state.selected_videos[video_id_to_process]['Status'] = final_status

            # Display final status message outside the language loop
            if generation_success_overall:
                st.success(f"✅ Video generation finished for '{video_data['Video Title']}'. Status: {final_status}", icon="🎉")
                # Display generated videos here or rely on the sidebar table
                if current_gen_urls:
                     st.write("Generated Video Links:")
                     for lang, url in current_gen_urls.items():
                          st.link_button(f"View ({lang})", url=url)
            else:
                 st.error(f"❌ Video generation failed for all languages for '{video_data['Video Title']}'. Status: {final_status}", icon="🔥")
                 if current_gen_errors:
                      st.write("Errors:")
                      for lang, err in current_gen_errors.items():
                           st.caption(f"{lang}: {err}")

        finally:
            # --- Cleanup Temporary Files for this run ---
            st.write(f"🧹 Cleaning up temporary files for {video_id_to_process}...")
            for path in temp_audio_paths_this_run + temp_final_video_paths_this_run:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        # st.write(f"Removed: {path}") # Debug log
                    except Exception as rm_err:
                        st.warning(f"Could not delete temp file: {path} ({rm_err})")

            # --- Update Queue and Progress ---
            st.session_state.generation_queue.pop(0) # Remove the processed video_id
            st.session_state.batch_processed_count += 1

            # Check if queue is now empty
            if not st.session_state.generation_queue:
                st.session_state.batch_processing_active = False
                st.balloons()
                st.success("🎉 Batch processing finished!")

            # Rerun to process the next item or update the UI
            # Add a small delay before rerun to allow user to see final status message
            time.sleep(1) # Shorter delay
            st.rerun()

    # --- Handle case where video_id is in queue but data is invalid/missing ---
    elif video_id_to_process in st.session_state.selected_videos:
        st.warning(f"Skipping video {video_id_to_process}. Data missing or invalid Direct URL.", icon="❓")
        st.session_state.selected_videos[video_id_to_process]['Status'] = 'Skipped (Invalid Data)'
        st.session_state.selected_videos[video_id_to_process]['Generation Errors'] = {'Skipped': 'Invalid data or URL'}
        # Update Queue and Progress
        st.session_state.generation_queue.pop(0)
        st.session_state.batch_processed_count += 1
        if not st.session_state.generation_queue:
            st.session_state.batch_processing_active = False
            st.info("Batch processing finished (last item skipped).")
        st.rerun()
    else:
        # Video ID was in queue but somehow removed from selected_videos
        st.error(f"Video ID {video_id_to_process} was in queue but not found in selected videos. Removing.", icon="❓")
        st.session_state.generation_queue.pop(0)
        st.session_state.batch_processed_count += 1 # Increment to avoid potential loops
        if not st.session_state.generation_queue:
            st.session_state.batch_processing_active = False
        st.rerun()


# --- Display Selected Videos Table (Sidebar - Incorporating Change 6) ---
st.sidebar.divider()
st.sidebar.header("Selected & Generated Videos Status")

if st.session_state.selected_videos:
    selected_list_data = []
    for vid_id, data in st.session_state.selected_videos.items():
         list_item = data.copy()
         # Ensure dictionaries are represented, handle potential non-dict values if state gets corrupted
         list_item['Generated URLs'] = str(list_item.get('Generated URLs', {})) if isinstance(list_item.get('Generated URLs'), dict) else str(list_item.get('Generated URLs', 'N/A'))
         list_item['Generation Errors'] = str(list_item.get('Generation Errors', {})) if isinstance(list_item.get('Generation Errors'), dict) else str(list_item.get('Generation Errors', 'N/A'))
         selected_list_data.append(list_item)

    if selected_list_data:
        df_selected = pd.DataFrame(selected_list_data)

        # Define desired columns and order
        display_columns = [
            'Status', 'Video Title', 'Topic', 'Source Languages',
            'Video ID', 'Format Details', 'yt_dlp_error',
            'Generated URLs', 'Generation Errors', # Show the string representations
        ]

        # Ensure all display columns exist, fill missing
        for col in display_columns:
            if col not in df_selected.columns:
                 df_selected[col] = "N/A"

        # Fill specific NaN/None values for better display before reordering
        df_selected['Status'] = df_selected['Status'].fillna('Unknown')
        df_selected['yt_dlp_error'] = df_selected['yt_dlp_error'].fillna('OK')
        # Fill any remaining NAs just in case
        df_selected.fillna("N/A", inplace=True)

        # Reorder DataFrame columns for display
        df_selected_display = df_selected[display_columns]

        st.sidebar.dataframe(
            df_selected_display,
            hide_index=True,
            use_container_width=True
            # Removed column config for URLs/Errors as they are strings now
        )

        # --- Download Button for detailed table ---
        @st.cache_data
        def convert_df_to_csv(df):
            output = BytesIO()
            df.to_csv(output, index=False, encoding='utf-8')
            return output.getvalue()

        try:
            csv_data = convert_df_to_csv(df_selected_display)
            st.sidebar.download_button(
                label="📥 Download Detailed Status (CSV)",
                data=csv_data,
                file_name='selected_videos_detailed_status.csv',
                mime='text/csv',
                use_container_width=True,
                disabled=st.session_state.batch_processing_active
            )
        except Exception as e:
            st.sidebar.warning(f"Could not generate detailed CSV: {e}")

        # --- Topic Summary Section (Using Change 5 function) ---
        st.sidebar.divider()
        st.sidebar.subheader("Generated Video Summary by Topic & Language") # Updated title

        # Generate the summary DataFrame using the updated helper function
        df_topic_summary = create_topic_summary_dataframe(st.session_state.selected_videos)

        if not df_topic_summary.empty:
            st.sidebar.dataframe(
                df_topic_summary,
                use_container_width=True,
                column_config={ # Use LinkColumn config as before
                    col: st.column_config.LinkColumn(f"Video {i+1}")
                    for i, col in enumerate(df_topic_summary.columns) if col.startswith('vid')
                },
                hide_index=True
            )

            # Download Button for the topic summary DataFrame
            try:
                csv_summary_data = convert_df_to_csv(df_topic_summary)
                st.sidebar.download_button(
                    label="📥 Download Topic Summary (CSV)",
                    data=csv_summary_data,
                    file_name='generated_videos_by_topic_language.csv', # Updated filename
                    mime='text/csv',
                    use_container_width=True,
                    disabled=st.session_state.batch_processing_active
                )
            except Exception as e:
                st.sidebar.warning(f"Could not generate summary CSV: {e}")

        else:
            st.sidebar.info("No videos have been successfully generated yet.")

    else: # If selected_list_data is empty
         st.sidebar.info("No videos selected yet.")

else: # If selected_videos dict doesn't exist or is empty
    st.sidebar.info("No videos selected yet.")


# Footer notes
st.sidebar.caption("Select videos, then use 'Process Ready Videos'. Generation creates videos per language.")