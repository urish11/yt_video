# -*- coding: utf-8 -*-
# <<< START: IMPORT STATEMENTS AND HELPER FUNCTIONS >>> 
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
import anthropic # Make sure anthropic client is installed: pip install anthropic
import moviepy.audio.fx.all as afx
# Ensure MoviePy is installed: pip install moviepy
# Ensure Pillow is installed: pip install Pillow
# Ensure pydub is installed: pip install pydub
# Ensure numpy is installed: pip install numpy
import numpy as np

# --- Try importing moviepy components with error handling ---
# Cache for resolved yt-dlp direct URLs to avoid refetching for the same video ID within a session
if 'resolved_vid_urls' not in st.session_state:
  st.session_state['resolved_vid_urls'] = {} # youtube_url: dlp_info_dict


try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips, TextClip,CompositeAudioClip
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
MAX_RESULTS_PER_QUERY = 100 # How many results to fetch *per term* from YouTube API
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
elif __file__ and os.path.exists(os.path.join(os.path.dirname(__file__), MONTSSERAT_FONT_FILENAME)):
     SUBTITLE_FONT_PATH = os.path.join(os.path.dirname(__file__), MONTSSERAT_FONT_FILENAME)
else:
    st.warning(f"Font '{MONTSSERAT_FONT_FILENAME}' not found. Subtitles might use default font.", icon="⚠️")
    SUBTITLE_FONT_PATH = None # Will use default font later if None

SUBTITLE_FONT_SIZE = 50 # Adjust as needed
SUBTITLE_WORDS_PER_GROUP = 2 # Group words for subtitles
SUBTITLE_COLOR = '#FFFF00' # Yellow
SUBTITLE_BG_COLOR = 'rgba(0, 0, 0, 0.6)' # Semi-transparent black
st.set_page_config(layout="wide", page_title="YouTube Video Generator", page_icon="🎥")
SCRIPT_VER_OPTIONS =["default", "default_v2", "1st_person" ,"mix"]

# --- Load Secrets ---
try:
    youtube_api_key_secret = st.secrets["YOUTUBE_API_KEY"]
    openai_api_key = st.secrets["GPT_API_KEY1"]
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"] # Added Anthropic Key
    aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    s3_bucket_name = st.secrets["S3_BUCKET_NAME"]
    s3_region = st.secrets["AWS_REGION"]
    COOKIE_FILE_PATH = st.secrets.get("YOUTUBE_COOKIE_PATH") # Optional cookie path
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please configure secrets.", icon="🚨")
    st.stop()

# --- Initialize Clients ---
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=openai_api_key)

openai_client = get_openai_client()

# Note: Anthropic client doesn't benefit as much from @st.cache_resource
# as it's typically lightweight to initialize. Initialize it where needed or globally.
# client = anthropic.Anthropic(api_key=anthropic_api_key) # Or initialize inside claude function

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

# --- Patched Resizer (Optional) ---
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

            # Use LANCZOS for resizing - generally good quality
            resized = pilim.resize(newsize, Image.Resampling.LANCZOS)
            return np.array(resized)

        moviepy_resize.resizer = patched_resizer
        print("Applied patched resizer.")
    else:
         print("Skipping resizer patch: moviepy.video.fx.resize not found.")
except Exception as e:
    print(f"Could not apply patched resizer: {e}")
    pass # Continue without patch

# --- Helper Function: create_topic_summary_dataframe ---
def create_topic_summary_dataframe(selected_videos_dict):
    """
    Creates a DataFrame summarizing generated videos grouped by a normalized
    'topic_language' key.
    """
    topic_lang_to_generated_urls = {}

    # 1. Collect Generated URLs and Group by Normalized Topic + Language
    for job_key, video_data in selected_videos_dict.items(): # Iterate through job entries
        topic = str(video_data.get('Topic', '')).strip().lower()
        lang = str(video_data.get('Language', '')).strip().lower()
        s3_url = video_data.get('Generated S3 URL')

        if topic and lang and s3_url:
            grouping_key = f"{topic}_{lang}"
            if grouping_key not in topic_lang_to_generated_urls:
                topic_lang_to_generated_urls[grouping_key] = []
            topic_lang_to_generated_urls[grouping_key].append(s3_url)

    if not topic_lang_to_generated_urls:
        return pd.DataFrame(columns=['Topic'])

    # 2. Determine Max URLs per Group and Prepare Data
    max_urls = 0
    if topic_lang_to_generated_urls:
        try:
            max_urls = max(len(urls) for urls in topic_lang_to_generated_urls.values())
        except ValueError:
            max_urls = 0 # Handle case where dict becomes empty after filtering

    data_for_df = []
    for topic_lang_key, urls in topic_lang_to_generated_urls.items():
        row = {'Topic': topic_lang_key}
        # Pad with empty strings if fewer than max_urls for this topic/lang
        padded_urls = urls + [''] * (max_urls - len(urls))
        for i, url in enumerate(padded_urls):
            row[f'vid{i+1}_url'] = url
        data_for_df.append(row)

    # 3. Create Final DataFrame
    if data_for_df:
        df_final = pd.DataFrame(data_for_df)
        # Ensure 'Topic' column is first and URL columns are sorted
        if 'Topic' in df_final.columns:
             topic_col = df_final.pop('Topic')
             df_final.insert(0, 'Topic', topic_col)
             url_cols_present = [col for col in df_final.columns if col.startswith('vid')]
             # Sort URL columns numerically
             url_cols_sorted = sorted(url_cols_present,
                                      key=lambda x: int(x.replace('vid','').replace('_url','')))
             df_final = df_final[['Topic'] + url_cols_sorted]
        else:
             # Fallback if Topic column somehow wasn't created
             df_final = pd.DataFrame(columns=['Topic'])
    else:
         df_final = pd.DataFrame(columns=['Topic'])

    return df_final

# --- Helper Function: YouTube API Search ---
def search_youtube(api_key, query, max_results_per_term=5):
    """
    Performs a Youtube using the v3 API.
    Handles multiple terms generated by GPT or separated by '|'.
    Args:
        api_key (str): YouTube Data API v3 key.
        query (str): The search query string (can contain '|' or be generated).
        max_results_per_term (int): Max results to request for EACH term.
    Returns:
        list: A list of video dictionaries [{'title': '', 'videoId': '', 'url': ''}]
              or None if a critical API error occurs.
    """
    videos_res = []
    # Split query into multiple terms if it contains '|' or handle 'auto' case (which generates '|' separated terms)
    if '|' in query:
        # Clean up potential extra quotes from GPT generation
        query = query.replace('"','').replace("'",'')
        terms = [term.strip() for term in query.split('|') if term.strip()]
        count = max_results_per_term//len(terms)
    else:
        terms = [query.strip()] # Treat as a single term

    st.write(f"Searching for terms: {terms} (Max {max_results_per_term} results per term)")

    total_fetched = 0
    MAX_TOTAL_RESULTS = 100 # Overall limit across all terms for safety

    for term in terms:
        if 1==2:
            st.warning(f"Reached overall result limit ({MAX_TOTAL_RESULTS}). Stopping search.")
            break

        params = {
            'part': 'snippet',
            'q': term,
            'key': api_key,
            'type': 'video',
            'maxResults': count,
            # 'videoEmbeddable': 'true',
            # 'order': 'relevance' # Default is relevance
            # 'regionCode': 'US' # Optional: Bias results towards a region
        }
        try:
            response = requests.get(YOUTUBE_API_BASE_URL, params=params, timeout=15)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            results = response.json()

            processed_ids_this_term = set() # Avoid adding duplicates from the same term search

            if 'items' in results:
                for item in results['items']:
                    if total_fetched >= MAX_TOTAL_RESULTS: break # Check limit again

                    if item.get('id', {}).get('kind') == 'youtube#video' and 'videoId' in item['id']:
                        video_id = item['id']['videoId']
                        # Check if we already added this video ID from this specific term search
                        if video_id in processed_ids_this_term:
                            continue

                        title = item['snippet'].get('title', 'No Title')
                        # Use the standard embeddable URL format
                        standard_url = f"https://www.youtube.com/watch?v={video_id}"
                        videos_res.append({
                            'title': title,
                            'videoId': video_id,
                            'url': standard_url # Store the standard watch URL
                        })
                        processed_ids_this_term.add(video_id)
                        total_fetched += 1

        except requests.exceptions.Timeout:
            st.error(f"API Request Timeout for query '{term}'.", icon="⏱️")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"API HTTP Error for query '{term}': {http_err}", icon="🔥")
            # Check for common quota/key errors
            if response.status_code == 403:
                 st.error("Received 403 Forbidden. Check your YouTube API Key and Quota.", icon="🚫")
                 return None # Signal critical error
            if response.status_code == 400:
                 st.error(f"Received 400 Bad Request. Check API parameters. Details: {response.text}", icon="❓")

        except requests.exceptions.RequestException as e:
            st.error(f"API Request Error for query '{term}': {e}", icon="🌐")
        except Exception as e:
            st.error(f"An unexpected error occurred during search for '{term}': {e}", icon="💥")
            import traceback
            st.error(traceback.format_exc())


    # Return collected results, respecting the overall MAX_TOTAL_RESULTS implicitly
    # No need to slice here as the loop breaks early
    return videos_res

# --- Helper Function: Simple Hash ---
def simple_hash(s):
    """Creates a simple, short hash string from an input string for UI keys."""
    total = 0
    for i, c in enumerate(s):
        total += (i + 1) * ord(c)
    return str(total % 100000) # Keep it relatively short


# --- Helper Function: Get Info with yt-dlp ---
def get_yt_dlp_info(video_url):
    """
    Uses yt-dlp to extract video format information, prioritizing direct mp4 URLs.
    Returns a dictionary: {'direct_url': url, 'format_details': str, 'error': None_or_str}
    """
    # Prioritize standard mp4 formats often playable directly
    # Format 18: mp4 [360p], Format 22: mp4 [720p]
    # Fallback to best mp4, then best overall
    YDL_OPTS = {
        'format': '22/18/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
        'skip_download': True, # Don't download, just get info
        'extract_flat': False, # Need format details if direct URL isn't top-level
        'socket_timeout': YT_DLP_FETCH_TIMEOUT,
        'retries': 3, # Add retries for network issues
        'fragment_retries': 3,
        # Uncomment if using cookies and path is valid
        # 'cookiefile': COOKIE_FILE_PATH if COOKIE_FILE_PATH and os.path.exists(COOKIE_FILE_PATH) else None,
    }
    # Add cookiefile only if path exists
    if COOKIE_FILE_PATH and os.path.exists(COOKIE_FILE_PATH):
        YDL_OPTS['cookiefile'] = COOKIE_FILE_PATH
        print(f"yt-dlp: Using cookie file: {COOKIE_FILE_PATH}")
    elif COOKIE_FILE_PATH:
        print(f"yt-dlp: Cookie file path provided but not found: {COOKIE_FILE_PATH}")


    try:
        with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
            # Extract info. This is the network call.
            info = ydl.extract_info(video_url, download=False)

            # --- Extract relevant details ---
            direct_url = info.get('url') # yt-dlp often puts the best direct URL here based on 'format'
            format_note = info.get('format_note', 'N/A')
            format_id = info.get('format_id', 'N/A') # The ID yt-dlp selected
            ext = info.get('ext', 'N/A')
            resolution = info.get('resolution', info.get('width') and info.get('height') and f"{info['width']}x{info['height']}") or 'N/A'
            filesize = info.get('filesize') or info.get('filesize_approx')

            # --- Fallback: Check 'formats' list if top-level 'url' is missing ---
            # This can happen for some sites or if specific format selection failed slightly differently
            if not direct_url and 'formats' in info:
                print(f"yt-dlp: Top-level URL missing for {video_url}. Checking formats list...")
                selected_format = None
                # Check the format yt-dlp claims it selected first
                if format_id != 'N/A':
                    selected_format = next((f for f in info['formats'] if f.get('format_id') == format_id and f.get('url')), None)

                # If that didn't work, try preferred formats manually
                if not selected_format:
                    preferred_formats = ['22', '18']
                    for pf_id in preferred_formats:
                        selected_format = next((f for f in info['formats'] if f.get('format_id') == pf_id and f.get('url')), None)
                        if selected_format: break

                # If still no URL, grab the *last* format in the list with a URL (often the 'best')
                # Or maybe the first one? Let's try the last one yt-dlp considered.
                if not selected_format:
                     selected_format = next((f for f in reversed(info['formats']) if f.get('url')), None)


                if selected_format:
                    print(f"yt-dlp: Found URL in formats list (Format ID: {selected_format.get('format_id')})")
                    direct_url = selected_format.get('url')
                    # Update details based on the found format
                    format_id = selected_format.get('format_id', format_id)
                    format_note = selected_format.get('format_note', format_note)
                    ext = selected_format.get('ext', ext)
                    resolution = selected_format.get('resolution', resolution)
                    filesize = selected_format.get('filesize') or selected_format.get('filesize_approx') or filesize

            # --- Format details string ---
            filesize_str = f"{filesize / (1024*1024):.2f} MB" if filesize else "N/A"
            format_details = f"ID: {format_id}, Res: {resolution}, Ext: {ext}, Size: {filesize_str}"

            # --- Return result ---
            if direct_url:
                return {
                    'direct_url': direct_url,
                    'format_details': format_details,
                    'error': None
                }
            else:
                # Log available info if URL extraction failed unexpectedly
                print(f"Warning: Could not extract direct URL for {video_url} even after checking formats. Info keys: {info.keys()}")
                # Try to find a reason
                reason = "Unknown reason"
                if info.get('availability'): reason = f"Availability: {info['availability']}"
                if info.get('_type') == 'playlist': reason = "Cannot process playlist URL directly"
                return {'direct_url': None, 'format_details': 'Error', 'error': f'Could not extract direct video URL. ({reason})'}

    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e)
        # Simplify common error messages
        if "confirm your age" in err_msg.lower():
            error_reason = 'Age-restricted video'
        elif "private video" in err_msg.lower():
            error_reason = 'Private video'
        elif "video unavailable" in err_msg.lower():
            error_reason = 'Video unavailable'
        elif "login required" in err_msg.lower():
             error_reason = 'Login required (try using cookies)'
        else:
             error_reason = f"yt-dlp: {err_msg[:150]}" # Truncate long messages
        st.warning(f"yt-dlp DownloadError for {video_url}: {error_reason}", icon="🚧")
        return {'direct_url': None, 'format_details': 'Error', 'error': error_reason}
    except Exception as e:
        st.error(f"Unexpected yt-dlp error for {video_url}: {e}", icon="💥")
        import traceback
        st.error(traceback.format_exc())
        return {'direct_url': None, 'format_details': 'Error', 'error': f"Unexpected yt-dlp error: {e}"}


# --- Helper Function: Generate Script with ChatGPT ---
def chatGPT(prompt, client, model="gpt-4o", temperature=1.0):
    """Generates text using OpenAI Chat Completion."""
    try:
        # Use the passed client object
        response = client.chat.completions.create(
             model=model,
             temperature=temperature,
             messages=[{'role': 'user', 'content': prompt}]
        )
        content = response.choices[0].message.content.strip()
        return content
    except Exception as e:
        st.error(f"Error calling OpenAI (Model: {model}): {e}", icon="🤖")
        # Consider logging the full error: print(f"OpenAI Error: {e}")
        return None


# --- Helper Function: Generate Script with Claude ---
def claude(prompt , model = "claude-3-haiku-20240307", temperature=1.0 , is_thinking = False, max_retries = 5):
    """Generates text using Anthropic Claude."""
    tries = 0
    last_error = None
    # Initialize client here or ensure it's passed/global and valid
    try:
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except Exception as client_err:
         st.error(f"Failed to initialize Anthropic client: {client_err}")
         return None

    while tries < max_retries:
        try:
            # Construct message parameters
            message_params = {
                 "model": model,
                 "max_tokens": 1024, # Adjust token limit as needed
                 "temperature": temperature,
                 "messages": [{"role": "user", "content": prompt}]
            }
            # Add 'thinking' parameter if requested (Note: Check current API support for 'thinking')
            # As of recent checks, 'thinking' might be deprecated or experimental.
            # Let's omit it for broader compatibility unless specifically tested.
            # if is_thinking:
            #    message_params["thinking"] = { "type": "enabled", "budget_tokens": 16000} # Example, check docs

            # Make the API call
            message = client.messages.create(**message_params)

            # Extract content - assumes response structure based on recent docs
            # Response content is a list, typically text is in the first block.
            if message.content and isinstance(message.content, list) and hasattr(message.content[0], 'text'):
                return message.content[0].text.strip()
            else:
                # Handle unexpected response structure
                st.warning(f"Claude response format unexpected: {message.content}")
                last_error = Exception("Unexpected Claude response format")
                tries += 1
                time.sleep(2 ** tries) # Exponential backoff

        except anthropic.APIConnectionError as e:
            st.warning(f"Claude connection error (Attempt {tries+1}/{max_retries}): {e}. Retrying...")
            last_error = e
            tries += 1
            time.sleep(2 ** tries) # Exponential backoff
        except anthropic.RateLimitError as e:
            st.warning(f"Claude rate limit hit (Attempt {tries+1}/{max_retries}). Retrying after delay...")
            last_error = e
            tries += 1
            time.sleep(max(5, 2 ** tries)) # Longer backoff for rate limits
        except anthropic.APIStatusError as e:
            st.error(f"Claude API error (Status {e.status_code}): {e.message}")
            last_error = e
            # Don't retry on persistent API errors like 4xx ?
            break # Or implement specific retry logic based on status code
        except Exception as e:
            st.error(f"Unexpected error calling Claude: {e}")
            last_error = e
            import traceback
            st.error(traceback.format_exc())
            tries += 1 # Retry on general errors too? Maybe limit this.
            time.sleep(2 ** tries)


    st.error(f"Failed to get response from Claude after {max_retries} attempts. Last error: {last_error}")
    return None


# --- Helper Function: Generate TTS Audio & Timestamps ---
def generate_audio_with_timestamps(text, client, voice_id="sage"):
    """Generates TTS audio using OpenAI, saves it, gets word timestamps via Whisper."""
    temp_audio_path = None
    word_timings = [] # Ensure initialized

    try:
        if not text or not text.strip():
            raise ValueError("Input text for TTS cannot be empty.")

        # 1. Generate TTS audio
        response = client.audio.speech.create(
            model="tts-1-hd", # Use HD for better quality
            voice=voice_id,
            input=text,
            response_format="mp3",
            speed=1.0
        )

        # 2. Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file_obj:
            temp_audio_path = temp_audio_file_obj.name
            temp_audio_file_obj.write(response.content)

        # 3. Optional: Volume Boost using pydub
        try:
            boosted_audio = AudioSegment.from_file(temp_audio_path)
            # Adjust dB boost as needed (+6 dB approx doubles loudness)
            boosted_audio = boosted_audio + 8 # Boost by 8 dB
            boosted_audio.export(temp_audio_path, format="mp3")
            # print(f"Audio volume boosted for {temp_audio_path}") # Debug log
        except Exception as boost_err:
            st.warning(f"Could not boost audio volume: {boost_err}", icon="🔊")

        # 4. Transcribe boosted audio with Whisper for timestamps
        with open(temp_audio_path, "rb") as audio_file_rb:
            # Use the client passed to the function
            transcribe_response = client.audio.transcriptions.create(
                file=audio_file_rb,
                model="whisper-1", # Use appropriate Whisper model
                response_format="verbose_json", # Needed for timestamps
                timestamp_granularities=["word"] # Request word-level timestamps
            )

        # 5. Process Transcription Response (Handles SDK v1.x+ response object)
        if hasattr(transcribe_response, 'words') and transcribe_response.words:
            for word_info in transcribe_response.words:
                start_time = getattr(word_info, 'start', None)
                end_time = getattr(word_info, 'end', None)
                # Validate timestamps before adding
                if start_time is not None and end_time is not None:
                    try:
                        start_float = float(start_time)
                        end_float = float(end_time)
                        if end_float >= start_float: # Ensure end is not before start
                            word_timings.append({
                                "word": getattr(word_info, 'word', ''),
                                "start": start_float,
                                "end": end_float
                            })
                        else:
                             st.warning(f"Skipping invalid time range (end <= start) for word '{getattr(word_info, 'word', '')}': start={start_float}, end={end_float}", icon="⏱️")
                    except (ValueError, TypeError):
                         st.warning(f"Skipping invalid timestamp format for word '{getattr(word_info, 'word', '')}': start={start_time}, end={end_time}", icon="⏱️")
                else:
                    st.warning(f"Missing start/end time for word '{getattr(word_info, 'word', '')}'", icon="⏱️")
        else:
             st.warning("Whisper did not return word timestamps in the expected format.", icon="⏱️")
             # Log the response structure for debugging if necessary
             print("Unexpected transcription response structure:", transcribe_response)


        # Return path and timings if successful
        if not word_timings:
             st.warning("No valid word timings extracted after transcription.", icon="⚠️")
             # Return None for timings to indicate failure downstream? Or empty list?
             # Let's return empty list but log the warning.

        # Return path even if timings are empty, but signal timings issue
        return temp_audio_path, word_timings

    except OpenAI.APIError as api_err:
        st.error(f"OpenAI API Error in TTS/Timestamp: {api_err}", icon="🤖")
        last_error = api_err
    except ValueError as ve:
        st.error(f"Value Error in TTS/Timestamp: {ve}", icon="📄")
        last_error = ve
    except Exception as e:
        st.error(f"Unexpected error in TTS/Timestamp generation: {repr(e)}", icon="💥")
        import traceback
        traceback.print_exc()
        last_error = e

    # --- Cleanup on Error ---
    if temp_audio_path and os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
            print(f"Cleaned up temp audio file on error: {temp_audio_path}") # Debug log
        except Exception as rm_err:
            st.warning(f"Could not remove temp audio file {temp_audio_path} during error handling: {rm_err}")

    # Return None, None on failure
    return None, None


# --- Helper Function: Group Word Timings ---
def group_words_with_timing(word_timings, words_per_group=2):
    """Groups words and their timings for subtitles."""
    grouped_timings = []
    if not word_timings: return grouped_timings # Handle empty input

    for i in range(0, len(word_timings), words_per_group):
        group_words_data = word_timings[i:i+words_per_group]
        if group_words_data:
            combined_word = " ".join(word_info['word'] for word_info in group_words_data).strip()
            # Ensure start/end times are valid before using
            if 'start' in group_words_data[0] and 'end' in group_words_data[-1]:
                 start_time = group_words_data[0]['start']
                 end_time = group_words_data[-1]['end']
                 # Basic validation
                 if combined_word and end_time > start_time:
                      grouped_timings.append({
                           "text": combined_word,
                           "start": start_time,
                           "end": end_time
                      })
                 # else: # Optional: Log skipped groups
                 #    print(f"Skipping invalid subtitle group: {group_words_data}")
            # else: # Optional: Log missing start/end
            #    print(f"Skipping group due to missing time data: {group_words_data}")

    return grouped_timings


# --- Helper Function: Create Text Image for Subtitles ---
def create_text_image(text, fontsize, color, bg_color, font_path, video_width):
    """
    Creates a transparent PNG image with text and rounded background,
    wrapping text to fit video width and centering it. Returns a NumPy array.
    """
    try:
        # --- Font Loading ---
        font = None
        if font_path and os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except Exception as font_load_err:
                st.warning(f"Failed to load font {font_path}: {font_load_err}. Using default.", icon="⚠️")
        if not font: # Fallback to default font
             try: font = ImageFont.load_default(size=fontsize) # Newer Pillow
             except AttributeError: font = ImageFont.load_default() # Older Pillow


        # --- Configuration ---
        padding_x = 25
        padding_y = 15
        bg_radius = 15
        # Max width for the text content itself inside the video frame (with margins)
        max_text_width = video_width - (2 * padding_x) - 40 # Added safety margin
        if max_text_width <= 0: max_text_width = video_width // 2 # Safety net

        # --- Text Wrapping ---
        lines = []
        words = text.split()
        if not words: # Handle empty text gracefully
             return np.zeros((10, 10, 4), dtype=np.uint8) # Return tiny transparent array

        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Use getbbox for more accurate width (left, top, right, bottom)
            try: bbox = font.getbbox(test_line) ; line_width = bbox[2] - bbox[0]
            except AttributeError: line_width = font.getlength(test_line) # Fallback

            if line_width <= max_text_width:
                current_line = test_line
            else:
                # Add the previous line if it has content
                if current_line: lines.append(current_line)
                # Start the new line with the current word, but check if word itself is too long
                try: word_bbox = font.getbbox(word); word_width = word_bbox[2] - word_bbox[0]
                except AttributeError: word_width = font.getlength(word)

                if word_width <= max_text_width:
                    current_line = word # Start new line with this word
                else:
                    # Word itself is too long, add it on its own line (might still overflow)
                    lines.append(word)
                    current_line = "" # Reset line after adding the long word

        if current_line: lines.append(current_line) # Add the last line

        wrapped_text = "\n".join(lines)
        if not wrapped_text: wrapped_text = text # Fallback if wrapping resulted empty (unlikely)

        # --- Calculate Text Block Dimensions ---
        # Use a dummy draw object to get accurate multiline bbox
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        try:
            # Use multiline_textbbox for accurate size estimation including line spacing
            # Anchor 'lt' (left-top) is common for bbox calculation origin
            bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center', anchor='lt')
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # Ensure minimum dimensions
            text_width = max(text_width, 1)
            text_height = max(text_height, 1)
            # Get descent for vertical positioning adjustment (negative value usually)
            try: descent = font.getmetrics()[1] if hasattr(font, 'getmetrics') else fontsize * 0.2
            except: descent = fontsize * 0.2 # Fallback descent estimate
        except AttributeError as e:
            st.warning(f"Using fallback subtitle dimension calculation (Pillow upgrade recommended): {e}", icon="PIL")
            # Fallback calculation (less accurate)
            text_width = 0
            for line in lines:
                 try: line_bbox = font.getbbox(line); text_width = max(text_width, line_bbox[2] - line_bbox[0])
                 except AttributeError: text_width = max(text_width, int(font.getlength(line)))
            try: line_height_metric = sum(font.getmetrics()) if hasattr(font, 'getmetrics') else fontsize * 1.2
            except: line_height_metric = fontsize * 1.2
            text_height = len(lines) * (line_height_metric + 4) # Add spacing
            descent = fontsize * 0.2
            text_width = max(text_width, 1)
            text_height = max(text_height, 1)


        # --- Create Final Image ---
        img_width = text_width + 2 * padding_x
        img_height = text_height + 2 * padding_y
        img = Image.new("RGBA", (int(img_width), int(img_height)), (0, 0, 0, 0)) # Transparent background
        draw = ImageDraw.Draw(img)

        # --- Draw Background ---
        # Parse RGBA color string
        try:
            if isinstance(bg_color, str) and bg_color.startswith('rgba'):
                parts = bg_color.strip('rgba()').split(',')
                r, g, b = map(int, parts[:3])
                a = int(float(parts[3]) * 255)
                fill_color_tuple = (r, g, b, a)
            else: # Assume hex or name, let Pillow handle parsing with default alpha if needed
                 from PIL import ImageColor
                 rgb = ImageColor.getrgb(bg_color)
                 fill_color_tuple = rgb + (int(0.6 * 255),) # Add default alpha if not RGBA already? Risky. Better require RGBA string.
                 # Let's stick to requiring rgba format for safety or use a fixed default.
                 # Using fixed default if format is wrong:
                 fill_color_tuple = (0, 0, 0, int(0.6*255))
                 st.warning(f"Subtitle background color '{bg_color}' not in expected rgba format. Using default.")
        except Exception:
             fill_color_tuple = (0, 0, 0, int(0.6*255)) # Default fallback on any parsing error
             st.warning(f"Error parsing subtitle background color '{bg_color}'. Using default.")

        # Draw rounded rectangle
        try:
            draw.rounded_rectangle([(0, 0), (img_width, img_height)], radius=bg_radius, fill=fill_color_tuple)
        except AttributeError: # Fallback for older Pillow without rounded_rectangle
             draw.rectangle([(0, 0), (img_width, img_height)], fill=fill_color_tuple)

        # --- Draw Text ---
        # Calculate text starting position for centering
        text_x = padding_x
        # Adjust y based on Pillow version's anchor behavior and metrics
        text_y = padding_y # Start drawing near the top padding edge

        # Use 'ms' anchor (middle-start) for center alignment with multiline_text
        # Or 'la' (left-ascent) and calculate offset manually? 'ms' is often easier.
        # Let's try 'ma' (middle-ascent) if 'ms' isn't ideal
        anchor_to_try = "ma" # Middle horizontal, ascent vertical alignment

        draw.multiline_text(
            (img_width / 2, text_y), # Position calculation might need adjustment based on anchor
            wrapped_text,
            font=font,
            fill=color,
            align="center",
            spacing=4,
            anchor=anchor_to_try # Experiment with anchors: 'mm', 'ms', 'ma'
        )

        # Convert PIL image to NumPy array for MoviePy
        return np.array(img)

    except Exception as e:
        st.error(f"Error creating text image for '{text[:50]}...': {e}", icon="🎨")
        import traceback
        st.error(traceback.format_exc())
        # Return a small transparent array on error
        return np.zeros((10, 10, 4), dtype=np.uint8)


# --- Helper Function: Download with yt-dlp (Downloads the actual video file) ---
def download_with_ytdlp(video_url, cookie_file_path=None):
    """
    Uses yt-dlp to download a video to a local temp file.
    Args:
        video_url (str): The standard YouTube watch URL (or other yt-dlp supported URL).
        cookie_file_path (str, optional): Path to a Netscape format cookie file.
    Returns:
        str: Path to the downloaded temporary file, or None if download fails.
             Caller is responsible for deleting the file.
    """
    temp_path = None
    st.write(f"ℹ️ Attempting to download video content: {video_url}")
    if cookie_file_path and os.path.exists(cookie_file_path):
        st.write(f"ℹ️ Using cookie file: {cookie_file_path}")
    elif cookie_file_path:
         st.warning(f"⚠️ Cookie file specified but not found: {cookie_file_path}")


    try:
        # Create a temporary file for the download output
        # delete=False means we manage deletion manually
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_obj:
            temp_path = temp_file_obj.name

        # Configure yt-dlp options for downloading
        ydl_opts = {
            'outtmpl': temp_path, # Save download to this specific temp path
            'format': '22/18/bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best', # Prefer reasonable quality mp4
            'quiet': False, # Show yt-dlp output in logs
            'noplaylist': True,
            'merge_output_format': 'mp4', # Ensure output is mp4 if merging is needed
            'overwrites': True, # Overwrite temp file if it somehow exists
            'retries': 3,
            'fragment_retries': 3,
            'socket_timeout': 60, # Longer timeout for download
            # Add cookie option if path is valid and file exists
            'cookiefile': cookie_file_path if cookie_file_path and os.path.exists(cookie_file_path) else None,
            # Add progress hook for potential Streamlit progress bar update (advanced)
            # 'progress_hooks': [lambda d: print(d['status'], d.get('filename'))], # Example hook
        }

        st.write(f"⏳ Starting yt-dlp download to: {temp_path}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url]) # Pass the URL as a list

        # --- Basic Integrity Checks ---
        st.write("🔬 Performing basic checks on downloaded file...")
        if not os.path.exists(temp_path):
            st.error(f"❌ Download Error: File not found after yt-dlp finished: {temp_path}")
            return None # Return None if file doesn't exist
        if os.path.getsize(temp_path) == 0:
            st.error(f"❌ Download Error: File is empty after download: {temp_path}")
            try: os.remove(temp_path) # Clean up empty file
            except OSError: pass
            return None # Return None if file is empty

        file_size_mb = os.path.getsize(temp_path) / (1024*1024)
        st.write(f"✔️ Check Passed: File exists and is not empty (Size: {file_size_mb:.2f} MB).")

        st.success(f"✅ yt-dlp download successful: {temp_path}")
        return temp_path # Return the path to the downloaded file

    except yt_dlp.utils.DownloadError as dl_err:
        # Check for common download errors
        err_str = str(dl_err).lower()
        if "login" in err_str or "authentication" in err_str or "age restricted" in err_str or "private" in err_str:
            st.error(f"❌ yt-dlp DownloadError: Authentication possibly required or video is private/restricted. Cookies might help. Message: {dl_err}", icon="🔒")
        elif "403 forbidden" in err_str:
            st.error(f"❌ yt-dlp DownloadError: Received '403 Forbidden'. YouTube may be blocking the request. Cookies might help, or the video might be unavailable. Message: {dl_err}", icon="🚫")
        elif "video unavailable" in err_str:
             st.error(f"❌ yt-dlp DownloadError: Video unavailable. Message: {dl_err}", icon="❓")
        else:
            st.error(f"❌ yt-dlp DownloadError: {dl_err}")

        # Clean up potentially incomplete file
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError: pass
        return None # Return None on download error
    except Exception as e:
        st.error(f"❌ Unexpected error during download_with_ytdlp: {e}")
        import traceback
        st.error(traceback.format_exc())
        # Clean up temp file on unexpected error
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError: pass
        return None # Return None on unexpected error


# --- Helper Function: Download Direct URL (Less likely needed now, but kept for utility) ---
def download_direct_url(url, suffix=".mp4"):
    """Downloads content from a direct URL to a temporary local file."""
    local_path = None
    try:
        headers = { # Use a common user agent
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            local_path = temp_file.name
            print(f"Attempting to download direct URL '{url}' to temp file: {local_path}")

            # Make the request, stream=True for potentially large files
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                response.raise_for_status() # Check for HTTP errors
                # Write content in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

        print(f"✔️ Direct download successful: {local_path}")
        # Basic check
        if os.path.getsize(local_path) == 0:
            print(f"⚠️ Warning: Direct download resulted in an empty file: {local_path}")
            # Optionally remove empty file?
            # os.remove(local_path)
            # return None
        return local_path

    except requests.exceptions.RequestException as e:
        print(f"❌ Direct Download failed (Network/HTTP Error): {e}")
    except Exception as e:
        print(f"❌ Direct Download failed (Other Error): {e}")

    # Cleanup on failure
    if local_path and os.path.exists(local_path):
        try: os.remove(local_path)
        except OSError: pass
    return None


# --- Helper Function: Process Video with TTS and Subtitles ---
def process_video_with_tts(base_video_url, audio_path, word_timings, topic, lang, copy_num, with_music=False):
    """
    Loads video (using downloaded path), adds TTS audio, loops/trims, adds subtitles.
    Returns the path to the final processed temporary video file, or None on failure.
    """
    # Initialize variables for cleanup
    final_video_clip = None
    temp_output_path = None
    base_video = None
    tts_audio = None
    back_music = None
    combined_audio = None
    looped_video = None
    processed_video = None
    resized_base_video = None
    clips_for_composite = []
    local_vid_path = None # To store the path of the downloaded base video

    try:
        # 1. Download the base video using yt-dlp
        # Note: Using the DIRECT URL fetched earlier by get_yt_dlp_info
        # It's passed as base_video_url here. Moviepy might handle some URLs directly,
        # but downloading first is often more reliable, especially for cloud functions.
        # Let's assume base_video_url *is* the direct URL here.
        # We need to download it.
        st.write(f"⏳ Downloading base video content from direct URL...")
        # Use the download_direct_url helper, or stick with yt-dlp if preferred
        # Using download_direct_url for potentially simpler direct downloads
        local_vid_path = download_direct_url(base_video_url, suffix=".mp4")
        # Alternative: If yt-dlp download is preferred even for direct URLs:
        # local_vid_path = download_with_ytdlp(base_video_url, cookie_file_path=COOKIE_FILE_PATH)

        if not local_vid_path:
            raise ValueError(f"Failed to download base video content from: {base_video_url}")

        # 2. Load downloaded video with MoviePy
        st.write(f"➡️ Loading downloaded video: {local_vid_path}")
        # Ensure target_resolution is set for potential resizing during load
        base_video = VideoFileClip(local_vid_path, audio=False, target_resolution=(720, 1280))

        video_duration = base_video.duration
        w = int(base_video.w) if base_video.w else 720
        h = int(base_video.h) if base_video.h else 1280
        st.write(f"✔️ Base video loaded: {w}x{h}, Duration: {video_duration:.2f}s")

        # 3. Load TTS Audio
        st.write(f"⏳ Loading TTS audio...")
        tts_audio = AudioFileClip(audio_path)
        audio_duration = tts_audio.duration
        if audio_duration <= 0:
             raise ValueError("TTS Audio has zero or negative duration.")

        # 4. Handle Background Music (Optional)
        combined_audio = tts_audio # Default
        if with_music:
            try:
                # Ensure audio file exists relative to the script or provide full path
                music_path = os.path.join('audio', 'Sunrise.mp3') # Assuming 'audio' subfolder
                if os.path.exists(music_path):
                    back_music = AudioFileClip(music_path).fx(afx.volumex, 0.08)
                    # Ensure music duration matches audio duration (loop or cut)
                    music_duration = back_music.duration
                    if music_duration < audio_duration:
                        num_loops = int(np.ceil(audio_duration / music_duration))
                        back_music = concatenate_audioclips([back_music] * num_loops).subclip(0, audio_duration)
                    elif music_duration > audio_duration:
                         back_music = back_music.subclip(0, audio_duration)

                    # Ensure both clips have the same duration before compositing
                    tts_audio = tts_audio.set_duration(audio_duration)
                    back_music = back_music.set_duration(audio_duration)

                    combined_audio = CompositeAudioClip([tts_audio, back_music])
                    st.write("✔️ Background music added.")
                else:
                    st.warning(f"Background music file not found at '{music_path}'. Skipping music.", icon="🎵")
            except Exception as music_err:
                st.warning(f"Could not load or process background music: {music_err}", icon="🎵")
        st.write(f"✔️ Audio loaded/prepared: Duration: {audio_duration:.2f}s")

        # 5. Resize Video Frame (Force 9:16)
        target_w, target_h = 720, 1280
        st.write(f"⏳ Resizing video to {target_w}x{target_h}...")
        try:
            # Use resize method for simplicity. Crop can be used for different framing.
            resized_base_video = base_video.resize(newsize=(target_w, target_h))
            st.write(f"✔️ Video resized.")
        except Exception as resize_err:
            st.warning(f"Could not resize video: {resize_err}. Using original.", icon="⚠️")
            resized_base_video = base_video # Fallback
            # Update target dimensions if fallback is used
            # target_w, target_h = w, h # This might break subtitle positioning/compositing size

        # 6. Loop or Trim Video to Match Audio Duration
        processed_video = resized_base_video # Start with the resized clip
        if video_duration < audio_duration:
            st.write(f"⏳ Looping video ({video_duration:.2f}s) to match audio ({audio_duration:.2f}s)...")
            num_loops = int(np.ceil(audio_duration / video_duration))
            # Create copies for concatenation
            clips_to_loop = [resized_base_video.copy().set_start(i * video_duration).set_duration(video_duration) for i in range(num_loops)]
            looped_video = concatenate_videoclips(clips_to_loop, method="compose")
            processed_video = looped_video.set_duration(audio_duration) # Explicitly set final duration
            st.write(f"✔️ Video looped {num_loops} times.")
        elif video_duration > audio_duration:
            st.write(f"⏳ Trimming video ({video_duration:.2f}s) to match audio ({audio_duration:.2f}s)...")
            processed_video = resized_base_video.subclip(0, audio_duration)
            st.write(f"✔️ Video trimmed.")
        else:
            st.write("✔️ Video duration matches audio duration.")

        # Ensure the processed video has the correct duration set
        processed_video = processed_video.set_duration(audio_duration)

        # 7. Set Audio
        final_video_clip = processed_video.set_audio(combined_audio)


        # 8. Generate and Add Subtitles
        st.write(f"⏳ Generating subtitles...")
        subtitle_clips_list = [] # Renamed to avoid conflict
        grouped_subs = group_words_with_timing(word_timings, words_per_group=SUBTITLE_WORDS_PER_GROUP)
        if grouped_subs:
            # ... (Subtitle generation loop using create_text_image, same as before) ...
            total_subs = len(grouped_subs)
            sub_progress_bar = st.progress(0)
            status_text = st.empty()

            for i, sub_data in enumerate(grouped_subs):
                text = sub_data['text']
                start = sub_data['start']
                end = sub_data['end']
                sub_duration = end - start

                # Ensure start/end are within the final video duration
                if start >= audio_duration: continue # Skip subs starting after video ends
                if end > audio_duration: end = audio_duration # Trim subs ending after video ends
                sub_duration = end - start

                if not text.strip() or sub_duration <= 0.05: continue

                status_text.text(f"Creating subtitle {i+1}/{total_subs}: '{text[:30]}...'")
                text_img_array = create_text_image(
                     text.upper(), fontsize=SUBTITLE_FONT_SIZE, color=SUBTITLE_COLOR,
                     bg_color=SUBTITLE_BG_COLOR, font_path=SUBTITLE_FONT_PATH,
                     video_width=target_w # Use target width for wrapping
                )
                if text_img_array.shape[0] <= 10 or text_img_array.shape[1] <= 10:
                     st.warning(f"Skipping subtitle due to image creation error: '{text[:30]}...'")
                     continue

                subtitle_img_clip = ImageClip(text_img_array)\
                     .set_start(start)\
                     .set_duration(sub_duration)\
                     .set_position(('center', 'center')) # Center position

                subtitle_clips_list.append(subtitle_img_clip)
                sub_progress_bar.progress((i + 1) / total_subs)

            status_text.text(f"✔️ Generated {len(subtitle_clips_list)} subtitle clips.")

            # --- Composite final video with subtitles ---
            st.write("⏳ Compositing video and subtitles...")
            # Ensure the base clip is the first element
            clips_for_composite = [final_video_clip] + subtitle_clips_list
            # Explicitly set the size for CompositeVideoClip using the target dimensions
            final_video_clip = CompositeVideoClip(clips_for_composite, size=(target_w, target_h))
            st.write("✔️ Compositing complete.")
        else:
            st.warning("No valid word timings available to generate subtitles.", icon="⏱️")
            # final_video_clip remains the video with audio but no subtitles


        # 9. Export Final Video to a Temporary File
        st.write("⏳ Exporting final video...")

        # Create a temporary file path for the output video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=f"processed_output_") as temp_output_file_obj:
            temp_output_path = temp_output_file_obj.name

        # Ensure we have a valid clip object
        if not isinstance(final_video_clip, (VideoFileClip, CompositeVideoClip)):
             raise TypeError(f"Cannot write final video: Invalid clip object type {type(final_video_clip)}.")

        # Create a unique temporary audio filename for this specific export run
        # This helps avoid conflicts if multiple processes run concurrently
        temp_audio_filename = f'temp-audio-{os.path.basename(temp_output_path)}.m4a'

        # Define ffmpeg parameters for web compatibility
        ffmpeg_params_list = [
            '-movflags', 'faststart', # Important for web streaming
            '-profile:v', 'high',     # H.264 profile
            '-level', '4.0',          # H.264 level
            # '-preset', 'medium',    # Already handled by MoviePy's preset argument
            # '-crf', '23',           # Optional: Constant Rate Factor for quality/size balance (lower is better quality, larger size)
        ]

        # Determine FPS, default to 24 if invalid
        output_fps = final_video_clip.fps if final_video_clip.fps and final_video_clip.fps > 0 else 24

        # Write the video file
        final_video_clip.write_videofile(
            temp_output_path,
            codec='libx264',          # Common, widely supported codec
            audio_codec='aac',        # Standard audio codec
            temp_audiofile=temp_audio_filename, # Use unique temp audio name
            remove_temp=True,         # Remove the temp audio file after processing
            fps=output_fps,
            preset='medium',          # Good balance of speed and quality
            threads=os.cpu_count() or 4, # Use available CPU cores
            logger='bar',             # Show progress bar in console/logs
            ffmpeg_params=ffmpeg_params_list
        )
        st.write(f"✔️ Final video exported to temporary path: {temp_output_path}")

        # Return the path to the successfully generated temporary file
        return temp_output_path

    except Exception as e:
        st.error(f"Error during video processing: {e}", icon="🎬")
        import traceback
        st.error(traceback.format_exc())
        # Ensure cleanup happens if an error occurred
        # Return None signifies failure
        return None # Explicitly return None on error

    finally:
        # --- Cleanup ---
        st.write("🧹 Cleaning up video processing resources...")
        # Use try-except for each close/remove operation for robustness
        resources_to_close = [
             base_video, tts_audio, back_music, combined_audio, looped_video,
             processed_video, resized_base_video, final_video_clip
        ] + subtitle_clips_list # Add subtitle clips to the list

        for resource in resources_to_close:
             if resource:
                  try: resource.close()
                  except Exception as close_err: pass # Ignore errors during cleanup

        # Delete the downloaded base video temp file
        if local_vid_path and os.path.exists(local_vid_path):
             try:
                  os.remove(local_vid_path)
                  # st.write(f"🧹 Deleted temp base video: {local_vid_path}") # Verbose log
             except Exception as rm_err: st.warning(f"Could not delete temp base video: {local_vid_path} ({rm_err})")

        # Delete the main output temp file only if an error occurred before returning path
        # If function succeeded, the caller is responsible for the returned path.
        # If it failed, temp_output_path might exist but is useless.
        # Let's check if the return value would have been None.
        # This is tricky. Best practice: Caller cleans up the returned path.
        # Here, just ensure MoviePy's internal temp files are cleaned if possible.

        # Attempt to remove the explicitly named temp audio file if it exists
        if 'temp_audio_filename' in locals() and os.path.exists(temp_audio_filename):
             try:
                  os.remove(temp_audio_filename)
                  # st.write(f"🧹 Deleted temp audio file: {temp_audio_filename}") # Verbose log
             except Exception as rm_audio_err:
                  st.warning(f"Could not remove temp audio file {temp_audio_filename}: {rm_audio_err}")

        st.write("🧹 Cleanup finished.")


# --- Helper Function: Upload Video to S3 ---
def upload_vid_to_s3(s3_cli, video_path, bucket_name, object_name, region_name):
    """Uploads a video file to S3."""
    if not s3_cli:
        st.error("S3 Client not initialized. Cannot upload.", icon="🚫")
        return None
    if not video_path or not os.path.exists(video_path): # Check if path is valid and exists
        st.error(f"Video file not found or path invalid: {video_path}", icon="❌")
        return None
    if os.path.getsize(video_path) == 0:
        st.error(f"Video file is empty, cannot upload: {video_path}", icon="⚠️")
        return None

    st.write(f"☁️ Uploading '{object_name}' to S3 bucket '{bucket_name}'...")
    try:
        with open(video_path, "rb") as video_file:
            s3_cli.put_object(
                Bucket=bucket_name,
                Key=object_name,
                Body=video_file,
                ContentType='video/mp4' # Ensure correct MIME type
            )

        # Construct the S3 URL (ensure object_name is URL-encoded)
        safe_object_name = urllib.parse.quote(object_name)
        if region_name == 'us-east-1':
             # us-east-1 URL format might differ (no region needed or optional)
             # Check S3 documentation or common practice. Using the format without region is often safe.
             video_url = f"https://{bucket_name}.s3.amazonaws.com/{safe_object_name}"
             # Alternate: f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{safe_object_name}"
        else:
             video_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{safe_object_name}"

        st.success(f"✔️ Video uploaded to S3: {object_name}")
        st.write(f"🔗 S3 URL: {video_url}")
        return video_url
    except NoCredentialsError:
        st.error("AWS Credentials not available for S3 upload.", icon="🔒")
        return None
    except Exception as e:
        st.error(f"S3 Upload Error: {e}", icon="☁️")
        import traceback
        st.error(traceback.format_exc()) # Log full traceback for debugging
        return None

# --- Helper: Sync Data Editor State ---
# Define sync_search_data function (keep the improved version from previous response)
def sync_search_data():
    """ Safely syncs the data_editor state back to session_state.search_data """
    try:
        editor_state = st.session_state.get("search_topic_editor", {})
        edited_rows = editor_state.get("edited_rows", {})
        added_rows = editor_state.get("added_rows", [])
        deleted_rows = editor_state.get("deleted_rows", [])

        # Use the snapshot from *before* the editor was rendered
        if 'search_data_snapshot' in st.session_state:
            current_df = st.session_state.search_data_snapshot.copy()
        else:
            # Fallback: use current state, but this might contain unsynced changes
            current_df = st.session_state.search_data.copy() if isinstance(st.session_state.search_data, pd.DataFrame) else pd.DataFrame([{'Topic': '', 'Search Term': '', 'Language': 'English', "Script Angle": "default", 'Video Results': 5}])

        # Apply deletions first (indices are based on the snapshot)
        valid_delete_indices = sorted([idx for idx in deleted_rows if idx < len(current_df)], reverse=True)
        if valid_delete_indices:
            current_df = current_df.drop(index=valid_delete_indices).reset_index(drop=True)

        # Apply edits
        for idx_str, changes in edited_rows.items():
            try:
                idx = int(idx_str)
                if 0 <= idx < len(current_df): # Check index validity against potentially shrunk df
                    for col, val in changes.items():
                        if col in current_df.columns:
                            # Handle potential type issues from editor
                            if col == 'Video Results':
                                 try: current_df.loc[idx, col] = int(val) if pd.notna(val) else 5
                                 except (ValueError, TypeError): current_df.loc[idx, col] = 5 # Fallback
                            else:
                                 current_df.loc[idx, col] = str(val) if pd.notna(val) else ""

            except ValueError: pass # Ignore invalid index strings

        # Apply additions
        if added_rows:
            expected_cols = ["Topic", "Search Term", "Language", "Script Angle", "Video Results"]
            processed_adds = []
            for row_dict in added_rows:
                 if isinstance(row_dict, dict):
                      new_row = {}
                      for col in expected_cols:
                           val = row_dict.get(col)
                           # Set defaults or convert type
                           if col == "Language": new_row[col] = str(val).strip() if pd.notna(val) and str(val).strip() else "English"
                           elif col == "Script Angle": new_row[col] = str(val).strip() if pd.notna(val) and str(val).strip() else "default"
                           elif col == "Video Results":
                                try: new_row[col] = int(val) if pd.notna(val) else 5
                                except (ValueError, TypeError): new_row[col] = 5
                           else: new_row[col] = str(val).strip() if pd.notna(val) else ""
                      processed_adds.append(new_row)
            if processed_adds:
                 add_df = pd.DataFrame(processed_adds)
                 current_df = pd.concat([current_df, add_df], ignore_index=True)


        # Final cleanup and validation of the entire DataFrame
        expected_cols = ["Topic", "Search Term", "Language", "Script Angle", "Video Results"]
        for col in expected_cols: # Ensure all columns exist
             if col not in current_df.columns:
                  if col == "Language": current_df[col] = "English"
                  elif col == "Script Angle": current_df[col] = "default"
                  elif col == "Video Results": current_df[col] = 5
                  else: current_df[col] = ""
        current_df = current_df[expected_cols] # Enforce column order

        # Type conversion and validation
        try:
            current_df['Video Results'] = pd.to_numeric(current_df['Video Results'], errors='coerce').fillna(5).astype(int)
            current_df['Video Results'] = current_df['Video Results'].apply(lambda x: max(1, min(x, 50))) # Clamp 1-50
        except Exception: current_df['Video Results'] = 5
        current_df['Topic'] = current_df['Topic'].fillna('').astype(str).str.strip()
        current_df['Search Term'] = current_df['Search Term'].fillna('').astype(str).str.strip()
        current_df['Language'] = current_df['Language'].fillna('English').astype(str).str.strip()
        current_df['Script Angle'] = current_df['Script Angle'].fillna('default').astype(str).str.strip()

        # Ensure 'Script Angle' values are valid options
        current_df['Script Angle'] = current_df['Script Angle'].apply(lambda x: x if x in SCRIPT_VER_OPTIONS else 'default')


        # Remove rows where both Topic and Search Term are empty, unless it's the only row left
        meaningful_rows_filter = (current_df['Topic'] != '') | (current_df['Search Term'] != '')
        if len(current_df) > 1:
            current_df = current_df[meaningful_rows_filter]

        # If empty after filtering, add back a default row
        if current_df.empty:
             current_df = pd.DataFrame([{'Topic': '', 'Search Term': '', 'Language': 'English', "Script Angle": "default", 'Video Results': 5}])

        # Update the main session state and create a fresh snapshot
        st.session_state.search_data = current_df.reset_index(drop=True)
        st.session_state.search_data_snapshot = st.session_state.search_data.copy()
        # print("Data editor sync complete.") # Debug log
    except Exception as sync_err:
         st.error(f"Error syncing data editor state: {sync_err}")
         # Fallback: Keep the previous state? Or reset?
         if 'search_data_snapshot' in st.session_state:
              st.session_state.search_data = st.session_state.search_data_snapshot.copy()
         else: # If snapshot missing too, hard reset
              st.session_state.search_data = pd.DataFrame([{'Topic': '', 'Search Term': '', 'Language': 'English', "Script Angle": "default", 'Video Results': 5}])
              st.session_state.search_data_snapshot = st.session_state.search_data.copy()


# --- Helper: Convert DF to CSV ---
@st.cache_data # Cache the conversion result for a given DataFrame
def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to CSV bytes."""
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8')
    return output.getvalue()

# <<< END: IMPORT STATEMENTS AND HELPER FUNCTIONS >>>


# --- Streamlit App ---

st.title("🎬 YouTube Video Selector & TTS Generator")
st.caption("Search YouTube, select videos (multiple times allowed), generate TTS script, create subtitled videos, and upload to S3.")

# --- Session State Initialization ---
# Stores data for each *selected video job instance* {job_key: video_data_dict}
if 'selected_videos' not in st.session_state:
    st.session_state.selected_videos = {}
# Stores results from the last API search {search_key: result_data}
if 'api_search_results' not in st.session_state:
    st.session_state.api_search_results = {}
# Input DataFrame for search terms and topics
if 'search_data' not in st.session_state:
    st.session_state.search_data = pd.DataFrame([
        {'Topic': 'sofa sale', 'Search Term': 'auto' ,'Language' : 'English',"Script Angle" : "default", 'Video Results': 5}
    ])
# Snapshot for data editor comparison
if 'search_data_snapshot' not in st.session_state:
    st.session_state.search_data_snapshot = st.session_state.search_data.copy()
# Flag to indicate if a search has been run
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False
# --- State for Batch Processing ---
if 'generation_queue' not in st.session_state: st.session_state.generation_queue = []
if 'batch_processing_active' not in st.session_state: st.session_state.batch_processing_active = False
if 'batch_total_count' not in st.session_state: st.session_state.batch_total_count = 0
if 'batch_processed_count' not in st.session_state: st.session_state.batch_processed_count = 0
if 'resolved_vid_urls' not in st.session_state: st.session_state.resolved_vid_urls = {}


# --- Input Area (Sidebar) ---
st.sidebar.header("Inputs & Actions")
st.sidebar.write("Enter Search Terms and Topics:")

# Take snapshot *before* rendering the editor
st.session_state.search_data_snapshot = st.session_state.search_data.copy()

edited_df = st.sidebar.data_editor(
    st.session_state.search_data, # Use the main state data
    column_config={
        "Script Angle": st.column_config.SelectboxColumn("Script Angle", options=SCRIPT_VER_OPTIONS, default="default", required=True),
        "Video Results": st.column_config.NumberColumn("Video Results", min_value=1, max_value=50, step=1, default=5, required=True),
        "Language": st.column_config.TextColumn("Language", default="English", required=True),
        "Topic": st.column_config.TextColumn("Topic"),
        "Search Term": st.column_config.TextColumn("Search Term (or 'auto')")
        },
    num_rows="dynamic",
    use_container_width=True,
    key="search_topic_editor",
    on_change=sync_search_data # Sync *after* edits are registered by Streamlit
)

# --- Action Buttons (Sidebar) ---
col1, col2 = st.sidebar.columns(2)
# Use on_click for search button to ensure sync happens *before* search logic runs
search_button = col1.button(
    "🔍 Search Videos",
    use_container_width=True,
    disabled=st.session_state.batch_processing_active,
    # on_click=sync_search_data # Sync now happens via on_change of editor
)
clear_button = col2.button("🧹 Clear All", use_container_width=True, type="secondary", disabled=st.session_state.batch_processing_active)
with_music = col1.checkbox("With BG music?", value=False)
with_music_rand = col2.checkbox("With BG music randomly?", value=False)

if clear_button:
    # Reset all relevant states
    st.session_state.selected_videos = {}
    st.session_state.search_triggered = False
    st.session_state.api_search_results = {}
    st.session_state.generation_queue = []
    st.session_state.batch_processing_active = False
    st.session_state.batch_total_count = 0
    st.session_state.batch_processed_count = 0
    st.session_state.resolved_vid_urls = {}
    st.session_state.search_data = pd.DataFrame([{'Topic': '', 'Search Term': '','Language' : 'English',"Script Angle" : "default", 'Video Results': 5}])
    st.session_state.search_data_snapshot = st.session_state.search_data.copy()
    # Clear the editor state explicitly if possible (might require widget key manipulation - complex)
    # For now, resetting search_data and rerunning should clear the editor visually.
    st.success("Selections, results, and generation state cleared!", icon="✅")
    st.rerun()

# --- Global Process Button ---
st.sidebar.divider()
ready_jobs_count = sum(
    1 for job_key, data in st.session_state.selected_videos.items()
    if data.get('Direct URL') and not data.get('yt_dlp_error') and not data.get('Generated S3 URL') and not data.get('Generation Error')
)
process_all_button = st.sidebar.button(
    f"🎬 Process {ready_jobs_count} Ready Job{'s' if ready_jobs_count != 1 else ''}",
    use_container_width=True,
    type="primary",
    disabled=ready_jobs_count == 0 or st.session_state.batch_processing_active
)

if process_all_button:
    # Find job keys ready for processing
    job_keys_to_process = [
        job_key for job_key, data in st.session_state.selected_videos.items()
        if data.get('Direct URL') and not data.get('yt_dlp_error') and not data.get('Generated S3 URL') and not data.get('Generation Error')
    ]
    if job_keys_to_process:
        st.session_state.generation_queue = job_keys_to_process
        st.session_state.batch_processing_active = True
        st.session_state.batch_total_count = len(job_keys_to_process)
        st.session_state.batch_processed_count = 0
        st.sidebar.info(f"Queued {len(job_keys_to_process)} generation jobs.", icon="⏳")
        # Update status and clear errors for queued jobs
        for job_key in job_keys_to_process:
            if job_key in st.session_state.selected_videos:
                st.session_state.selected_videos[job_key]['Generation Error'] = None
                st.session_state.selected_videos[job_key]['Generated S3 URL'] = None
                st.session_state.selected_videos[job_key]['Status'] = 'Queued'
        st.rerun()
    else:
        st.sidebar.warning("No selected video jobs are ready for processing (need Direct URL).", icon="⚠️")

st.sidebar.info("Use '➕ Select' to queue a generation job. Each click adds one job.", icon="ℹ️")
st.sidebar.warning("Video generation can take several minutes per job.", icon="⏱️")

# --- Processing Logic ---

# 1. Handle Search Button Click (Logic moved slightly)
if search_button: # This block runs when the button is clicked
    # sync_search_data() # Sync is now handled by on_change of editor

    # Perform validation on the synced data
    search_df = st.session_state.search_data # Use the already synced data
    valid_input = True
    if search_df.empty:
        st.sidebar.warning("Input table is empty.", icon="⚠️")
        valid_input = False
    # Check if required columns have at least one non-empty value after sync/validation in sync_search_data
    elif ((search_df['Topic'] == '') & (search_df['Search Term'] == '')).all():
         if len(search_df) > 1: # Allow single empty row, but not multiple
            st.sidebar.warning("Please provide a 'Topic' or 'Search Term' in at least one row.", icon="⚠️")
            valid_input = False
         # Else: allow the single default empty row.
    # Check 'auto' term requires Topic?
    auto_rows = search_df[search_df['Search Term'].str.lower() == 'auto']
    if not auto_rows.empty and (auto_rows['Topic'] == '').any():
        st.sidebar.warning("Rows with 'auto' in 'Search Term' must have a non-empty 'Topic'.", icon="⚠️")
        valid_input = False

    if valid_input:
        st.sidebar.success("Input valid, proceeding with search.")
        st.session_state.search_triggered = True
        st.session_state.api_search_results = {} # Clear previous results
        st.session_state.current_search_df = search_df.copy() # Store validated DF for search step
        st.rerun() # Rerun to trigger the API search section
    # No else needed, warnings displayed above

# 2. Perform API Search if Triggered
if st.session_state.search_triggered and 'current_search_df' in st.session_state:
    search_df = st.session_state.current_search_df
    search_items = search_df.to_dict('records')

    st.info(f"Searching API for {len(search_items)} topic/term row(s)...", icon="🔍")
    progress_bar = st.progress(0)
    status_text_api = st.empty()

    api_error_occurred = False
    results_cache = {} # Temp cache for *this* search run

    for i, item in enumerate(search_items):
        # Retrieve validated data from the DataFrame row
        term = item['Search Term'] # Already stripped in sync
        topic = item['Topic']     # Already stripped in sync
        count = item['Video Results'] # Already int and clamped
        lang = item['Language']   # Already stripped in sync
        script_ver = item["Script Angle"] # Already stripped in sync

        # Handle 'auto' search term generation
        if term.lower() == 'auto':
            if not topic: # Should be caught by earlier validation, but double-check
                st.warning(f"Skipping row {i+1}: 'auto' search term requires a Topic.", icon="⚠️")
                continue
            status_text_api.text(f"Generating search terms for: '{topic}'...")
            try:
                # --- Use the refined GPT prompt for search terms ---
                generated_term = chatGPT(f"""You are a viral video ad expert. I will give you a topic, and you will return the top 3 YouTube Shorts search terms that:

                                    - Are short (2–5 words)

                                    - Clearly describe what viewers will see in the video (visuals only)

                                    - Lead to emotionally engaging, surprising, or curiosity-triggering content

                                    - Are perfect for remixing or using as inspiration for Facebook video ads

                                    - Focus on things like transformations, objects in motion, satisfying actions, luxury aesthetics, clever space-saving, or unexpected reveals

                                    - Avoid abstract or advice-based phrases (like “tips,” “hacks,” or “secrets”)

                                    - Avoid using non visual\describing words that are not likely to be relevent (like 'On credit', "Financing", etc)

                                    - Add '#shorts' to the end of each search term and separate terms with ' | '

                                    - if the topic is a service (like lawyer) that is intangible, think of something else that can be used (like "Veterans Benefits Lawyer free consultation" give "veteran shares #shorts ") 

                                    Example:  

                                    Input: sofa  

                                    Output:  

                                    'sofa transformation #shorts | hidden bed sofa #shorts | luxury sofa unboxing #shorts'



                                    My topic:

                                 {topic}""", client=openai_client, model="gpt-4") # Use your full validated prompt
                if not generated_term:
                    st.warning(f"Failed to generate search terms for '{topic}'. Skipping.", icon="🤖")
                    continue
                term = generated_term # Replace 'auto' with generated terms
                st.write(f"Generated terms for '{topic}': {term}")
            except Exception as gpt_err:
                st.error(f"Error generating search terms for '{topic}': {gpt_err}", icon="🤖")
                continue

        # Proceed with search using the original or generated term
        status_text_api.text(f"Searching YouTube for: '{term}' (Topic: '{topic}')...")
        # Key for caching results of this specific search combination
        unique_search_key = f"{term}_{topic}_{lang}_{script_ver}"

        if unique_search_key not in results_cache:
            # Pass MAX_RESULTS_PER_QUERY defined earlier
            videos = search_youtube(youtube_api_key_secret, term, MAX_RESULTS_PER_QUERY)

            if videos is None: # Critical API error signalled from search_youtube
                 st.error(f"Stopping search due to critical API issue.", icon="🚫")
                 api_error_occurred = True
                 break

            # Store results
            results_cache[unique_search_key] = {
                'videos': videos,
                'topic': topic,
                'lang': lang,
                "script_ver": script_ver,
                'original_term': term # Store the actual term used for search
            }
            time.sleep(0.1) # Brief pause

        progress_bar.progress((i + 1) / len(search_items))
        if api_error_occurred: break # Exit loop early if critical error

    status_text_api.text("API Search complete." if not api_error_occurred else "API Search halted due to error.")
    st.session_state.api_search_results = results_cache # Update main cache
    st.session_state.search_triggered = False # Reset trigger
    # No st.rerun() here, let results display immediately


# 3. Display Search Results
st.divider() # Separator before results
if st.session_state.api_search_results:
    st.subheader("Search Results & Video Selection (Grid View)")

    if not st.session_state.api_search_results:
         st.info("No search results available. Perform a search first.")

    # Display results grouped by the search cache key
    for search_key, result_data in st.session_state.api_search_results.items():
        videos = result_data['videos']
        topic = result_data['topic']
        lang = result_data['lang']
        script_ver = result_data["script_ver"]
        original_term = result_data['original_term']

        # --- Container for each search result group ---
        term_container = st.container(border=True)
        with term_container:
            st.subheader(f"Results for: \"{original_term}\" (Topic: \"{topic}\", Lang: {lang}, Angle: {script_ver})")
            if not videos:
                st.write("No videos found via API for this search.")
                continue

            num_videos = len(videos)
            num_cols = 3 # Adjust number of columns as desired

            for i in range(0, num_videos, num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    video_index = i + j
                    if video_index < num_videos:
                        video = videos[video_index]
                        # Use the column context manager
                        with cols[j]:
                            # --- Extract Video Info ---
                            video_id = video['videoId']
                            video_title = video['title']
                            # Use the standard watch URL from search results
                            standard_video_url = video.get('url', f"https://www.youtube.com/watch?v={video_id}")

                            # --- Unique key for UI elements in this grid cell ---
                            grid_instance_key = f"{video_id}_{search_key}_{i}_{j}"

                            # --- Video Player/Thumbnail Toggle State ---
                            show_video_key = f"show_player_{grid_instance_key}"
                            if show_video_key not in st.session_state:
                                st.session_state[show_video_key] = False

                            # --- Render Content ---
                            st.write(f"**{textwrap.shorten(video_title, width=50, placeholder='...')}**")
                            st.caption(f"ID: {video_id}")

                            # --- Player / Thumbnail ---
                            if st.session_state[show_video_key]:
                                try: 

                                    iframe_code = f"""
                                          <iframe width="315" height="560"
                                          src="https://www.youtube.com/embed/{video_id}"
                                          title="YouTube video player" frameborder="0"
                                          allow="accelerometer; autoplay; clipboard-write; encrypted-media;
                                          gyroscope; picture-in-picture; web-share"
                                          allowfullscreen></iframe>
                                          """
                                            
                                    st.markdown(iframe_code, unsafe_allow_html=True)
                                  
                                  # st.video(standard_video_url)
                                except Exception as e: st.error(f"Video preview failed: {e}")
                            else:
                                thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"
                                st.image(thumbnail_url, use_container_width=True, caption="Video Thumbnail")

                            # --- Buttons ---
                            # Toggle Preview
                            toggle_label = "🔼 Hide" if st.session_state[show_video_key] else "▶️ Show"
                            if st.button(f"{toggle_label} Preview", key=f"toggle_vid_btn_{grid_instance_key}", help="Show/hide video preview", use_container_width=True):
                                st.session_state[show_video_key] = not st.session_state[show_video_key]
                                st.rerun()

                            # Select Button (Adds ONE job per click)
                            if st.button("➕ Select (Queue Job)", key=f"select_{grid_instance_key}", type="primary", use_container_width=True, disabled=st.session_state.batch_processing_active):
                                # --- ADD JOB LOGIC ---
                                base_video_id = video_id
                                current_lang = lang.strip() # Language from this result group
                                base_key_prefix = f"{base_video_id}_{current_lang}_"
                                # Find existing copy numbers for this specific video+lang combo
                                existing_copy_numbers = [
                                    int(k[len(base_key_prefix):])
                                    for k in st.session_state.selected_videos.keys()
                                    if k.startswith(base_key_prefix) and k[len(base_key_prefix):].isdigit()
                                ]
                                next_copy_number = max(existing_copy_numbers) + 1 if existing_copy_numbers else 1
                                job_key = f"{base_key_prefix}{next_copy_number}"

                                # Add the new job entry
                                st.session_state.selected_videos[job_key] = {
                                    'Job Key': job_key,
                                    'Search Term': original_term,
                                    'Topic': topic,
                                    'Language': current_lang,
                                    'Video Title': video_title,
                                    'Video ID': base_video_id,
                                    'Copy Number': next_copy_number,
                                    'Standard URL': standard_video_url, # Store YT watch URL
                                    'fetching_dlp': True, # Mark for URL fetching
                                    'Direct URL': None, # To be filled by yt-dlp fetch
                                    'Format Details': None,
                                    'yt_dlp_error': None,
                                    'Generated S3 URL': None,
                                    'Generation Error': None,
                                    'Status': 'Selected, Fetching URL...',
                                    'Script Angle': script_ver
                                }
                                st.toast(f"Queued Job #{next_copy_number} ({current_lang}) for: {video_title}", icon="➕")
                                st.rerun() # Update sidebar, trigger fetch

                            # --- Display Status for Existing Jobs ---
                            # Find jobs matching this specific Video ID and Language
                            related_job_keys = [
                                k for k, v in st.session_state.selected_videos.items()
                                if v.get('Video ID') == video_id and v.get('Language') == lang
                            ]
                            if related_job_keys:
                                # Use an expander to avoid cluttering the grid
                                status_expander = st.expander(f"Show Status for {len(related_job_keys)} Queued Job(s)")
                                with status_expander:
                                    # Sort jobs by copy number for consistent display
                                    sorted_job_keys = sorted(related_job_keys, key=lambda k: st.session_state.selected_videos.get(k, {}).get('Copy Number', 0))
                                    for r_job_key in sorted_job_keys:
                                        job_data = st.session_state.selected_videos.get(r_job_key)
                                        if job_data:
                                            copy_num = job_data.get('Copy Number', '?')
                                            status = job_data.get('Status', 'Unknown')
                                            s3_url = job_data.get('Generated S3 URL')
                                            # Combine potential errors for display
                                            error_msg = job_data.get('Generation Error') or job_data.get('yt_dlp_error')

                                            st.markdown(f"**Job #{copy_num}** (`{r_job_key}`)") # Use markdown for bold
                                            if status == 'Processing': st.info("⚙️ Processing...", icon="⏳")
                                            elif status == 'Queued': st.info("🕒 Queued", icon="🕒")
                                            elif status == 'Completed' and s3_url:
                                                st.success("✔️ Generated!", icon="🎉")
                                                st.link_button("View on S3", url=s3_url, type="secondary")
                                            elif status == 'Failed' and error_msg: st.error(f"❌ Failed: {error_msg[:60]}...", icon="🔥")
                                            elif status.startswith('Error:') and error_msg: st.error(f"⚠️ URL Error: {error_msg[:60]}...", icon="⚠️")
                                            elif status == 'Ready': st.success("✅ Ready to Process", icon="👍")
                                            elif status == 'Selected, Fetching URL...': st.info("📡 Fetching URL...", icon="📡")
                                            else: st.write(f"Status: {status}") # Fallback


# 4. yt-dlp Fetching Logic (Runs after UI render if needed)
# (Use the corrected version that works with job_keys and caches results)
if not st.session_state.batch_processing_active:
    job_keys_to_fetch = [
        job_key for job_key, data in st.session_state.selected_videos.items()
        if data.get('fetching_dlp') # Check the flag
    ]

    if job_keys_to_fetch:
        fetch_job_key = job_keys_to_fetch[0] # Process one per rerun cycle
        video_data = st.session_state.selected_videos.get(fetch_job_key)

        if video_data:
            standard_url = video_data.get('Standard URL')
            title = video_data.get('Video Title', fetch_job_key)

            # Only show spinner if fetching is needed
            with st.spinner(f"Fetching video details for '{title}' (Job: {fetch_job_key})..."):
                dlp_info = None
                # Check cache first using the standard watch URL
                if standard_url and standard_url in st.session_state.get('resolved_vid_urls', {}):
                    dlp_info = st.session_state['resolved_vid_urls'][standard_url]
                    print(f"Cache hit for {standard_url}") # Debug
                elif standard_url:
                    print(f"Cache miss - Fetching NEW URL info for {standard_url}") # Debug
                    # Fetch info using yt-dlp helper function
                    dlp_info = get_yt_dlp_info(standard_url)
                    # Update cache ONLY if fetch was successful AND returned a direct URL
                    if dlp_info and dlp_info.get('direct_url'):
                         st.session_state.setdefault('resolved_vid_urls', {})[standard_url] = dlp_info
                else:
                     dlp_info = {'error': 'Missing Standard URL in job data.'} # Cannot fetch without URL

            # Update state for the specific job_key
            current_state = st.session_state.selected_videos.get(fetch_job_key)
            if current_state:
                current_state['fetching_dlp'] = False # Mark fetch attempt as done

                if dlp_info and dlp_info.get('direct_url'):
                    current_state['Direct URL'] = dlp_info['direct_url'] # Store the direct URL
                    current_state['Format Details'] = dlp_info.get('format_details', 'N/A')
                    current_state['yt_dlp_error'] = None
                    current_state['Status'] = 'Ready' # Ready for processing
                    st.toast(f"Direct URL loaded for job '{fetch_job_key}'", icon="✅")
                else: # Handle errors or missing URL from dlp_info
                    error_detail = dlp_info.get('error', "Could not get direct URL") if dlp_info else "yt-dlp fetch failed critically"
                    current_state['Direct URL'] = None
                    current_state['Format Details'] = "Error"
                    current_state['yt_dlp_error'] = error_detail
                    current_state['Status'] = f"Error: {error_detail}" # Update status to reflect error
                    st.toast(f"yt-dlp failed for job '{fetch_job_key}': {error_detail}", icon="⚠️")

                # Save updated state
                st.session_state.selected_videos[fetch_job_key] = current_state
                st.rerun() # Rerun to update UI display


# 5. Video Generation Logic (BATCH PROCESSING)
# (Use the corrected version that processes one job_key at a time)
if st.session_state.batch_processing_active and st.session_state.generation_queue:
    job_key_to_process = st.session_state.generation_queue[0]
    video_data = st.session_state.selected_videos.get(job_key_to_process)
    final_video_path = None # For cleanup
    audio_path = None       # For cleanup

    # Check if job data exists and is ready (has Direct URL, no prior error)
    if video_data and video_data.get('Direct URL') and not video_data.get('yt_dlp_error'):
        processed_count_display = st.session_state.batch_processed_count + 1
        total_count_display = st.session_state.batch_total_count
        st.header(f"⚙️ Processing Job {processed_count_display}/{total_count_display}: {video_data['Video Title']} (Copy #{video_data.get('Copy Number', '?')})")
        gen_placeholder = st.container() # Container for logs of this specific job run

        try:
            # --- Update Status ---
            st.session_state.selected_videos[job_key_to_process]['Status'] = 'Processing'

            # --- Generation Steps (run once per job key) ---
            with gen_placeholder:
                st.info(f"Starting video generation process for job: {job_key_to_process}")
                with st.status("Running generation steps...", state="running", expanded=True) as status_log:
                    try:
                        # --- 1. Get Job Data ---
                        topic = video_data.get('Topic', 'video topic')
                        lang = video_data.get('Language', 'English')
                        script_ver = video_data.get("Script Angle", "default")
                        base_video_direct_url = video_data.get("Direct URL") # Use the fetched direct URL
                        copy_num = video_data.get('Copy Number', 0)

                        if not base_video_direct_url:
                            raise ValueError("Direct video URL missing.")

                        # --- 2. Generate Script ---
                        st.write(f"1/5: Generating script (Angle: {script_ver})...")
                        if script_ver == "mix":
                            script_ver_temp = random.choice([opt for opt in SCRIPT_VER_OPTIONS if opt != 'mix'])
                        else:
                            script_ver_temp = script_ver
                        # --- Construct the full script prompt based on script_ver_temp ---
                        # (Insert your actual prompt logic here, using f-strings)
                        if script_ver_temp == 'default_v2':
                             script_prompt = f"""Generate a short voiceover script (approx. 15-20 seconds, typically 2-3 concise sentences) for a social media video ad about '{topic}' in {lang}.

        

                                                **Goal:** Create an intriguing and engaging script that captures attention quickly, holds it (retentive), and encourages clicks, suitable for platforms like Facebook/Instagram Reels/TikTok.

                                                

                                                **Tone:** Conversational, casual, and authentic. Avoid sounding like a formal advertisement or overly 'salesy'. Speak *to* the viewer directly (use "you" where appropriate).

                                                

                                                **Key Requirements:**

                                                1.  **Strong Hook:** Start immediately with something captivating (e.g., a relatable question, a surprising statement, the core benefit) to grab attention in the first 1-2 seconds.

                                                2.  **Concise Body:** Briefly convey the most interesting or beneficial aspect of the '{topic}'. Focus on clarity and smooth flow.

                                                3.  **Clear Call to Action (CTA):** End the script *only* with the phrase "Tap now to " followed by a simple, clear, non-committal action (e.g., learn more, see details, explore options, find out how).

                                                

                                                **Strict Exclusions (Mandatory):**

                                                * **NO:** "we," "our," or "I."

                                                * **NO:** Sensational language, hype, exaggeration, or false promises. (Be truthful and grounded).

                                                * **NO:** Aggressive or fake urgency/scarcity tactics (e.g., "Act fast!", "Limited spots!").

                                                * **NO:** Geographically suggestive terms (e.g., "Near you," "In your area").

                                                * **NO:** Jargon or overly complex vocabulary.

                                                * **NO:** DONT make false promises.

                                                

                                                

                                                **Output:** Provide ONLY the raw script text, with no extra explanations or formatting.  """


                        elif script_ver_temp == '1st_person':
                             script_prompt = f"""

                                                    Create a brief, captivating first-person voiceover script for a viral FB video about '{topic}' in {lang}. 

                                                    Keep it concise (15-20 seconds when spoken, about 2-3 sentences) with these guidelines:

        

                                                    - Start with an immediate hook in the first 3-5 seconds to grab attention

                                                    - The hook should be intriguing but honest - NO false promises or misleading claims

                                                    - Use first-person perspective throughout

                                                    - Make the tone authentic and conversational, like a friend sharing a discovery

                                                    - Focus on creating genuine interest in the topic with real value

                                                    - Maintain a natural flow that keeps viewers watching

                                                    - End with a simple call to action like "Tap to discover..." or "Tap to learn..."

                                                    - Ensure the content feels genuine, not like an advertisement

        

                                                    IMPORTANT:

                                                    - The opening hook must be attention-grabbing AND truthful

                                                    - Avoid sensational language or exaggerated claims

                                                    - Don't make promises that can't be delivered

                                                    - No urgency phrases like "limited time" or "act now"

                                                    - No geographic claims (e.g., "near you," "in your area")

                                                    - No "we" or "our" language - keep it personal

                                                    - End with "Tap to..." followed by a clear, non-committal action

                                                    * **NO:** DONT make false promises. ('get approved')

                                                    Return only the script text itself, nothing else.

                                                    """
                        else: # Default
                             script_prompt = f"""Create a short, engaging voiceover script for FB viral   video (roughly 15-20 seconds long, maybe 2-3 sentences) about '{topic}' in language {lang}. The tone should be informative yet conversational, '.  smooth flow. Just provide the script text, nothing else. create intriguing and engaging script, sell the topic to the audience . be very causal and not 'advertisement' style vibe. end with a call to action 'tap to....'  .the text needs to be retentive.Don't say 'we' or 'our' .NOTE:: DO NOT dont use senetional words and phrasing and DONT make false promises , use Urgency Language, Avoid geographically suggestive terms (e.g., "Near you," "In your area"). Do not use "we" or "our". in end if video use something "Tap now to.." with a clear, non-committal phrase !!!  """


                        # --- Choose LLM ---
                        # script_text = chatGPT(script_prompt, client=openai_client)
                        script_text = claude(script_prompt) # Assumes claude function uses API key from secrets

                        if not script_text: raise ValueError("Failed to generate script text.")
                        st.text_area("Generated Script:", script_text, height=100, disabled=True, key=f"script_{job_key_to_process}")

                        # --- 3. Generate TTS ---
                        st.write(f"2/5: Generating TTS audio & timestamps...")
                        audio_path, word_timings = generate_audio_with_timestamps(
                            script_text, client=openai_client, voice_id=DEFAULT_TTS_VOICE
                        )
                        if audio_path is None or word_timings is None: # Check both for failure
                            raise ValueError("Failed to generate TTS audio or timestamps.")

                        # --- 4. Process Video ---
                        st.write(f"3/5: Processing base video & adding audio/subtitles...")
                        current_with_music = with_music
                        if with_music_rand: current_with_music = random.choice([True, False])

                        # Pass Direct URL and other necessary data
                        # This function now downloads the direct url, processes, and returns temp output path
                        final_video_path = process_video_with_tts(
                            base_video_url=base_video_direct_url, # Pass the direct URL
                            audio_path=audio_path,
                            word_timings=word_timings,
                            topic=topic,
                            lang=lang,
                            copy_num=copy_num,
                            with_music=current_with_music
                        )
                        if not final_video_path: raise ValueError("Video processing (MoviePy) failed.")

                        # --- 5. Construct Unique S3 Filename ---
                        safe_topic = urllib.parse.quote(topic.replace(' ', '_')[:30], safe='')
                        timestamp = int(datetime.datetime.now().timestamp())
                        final_s3_object_name = f"final_{safe_topic}_{lang}_copy{copy_num}_{timestamp}.mp4"

                        # --- 6. Upload to S3 ---
                        st.write(f"4/5: Uploading '{final_s3_object_name}' to S3...")
                        s3_url = upload_vid_to_s3(
                            s3_cli=s3_client, video_path=final_video_path,
                            bucket_name=s3_bucket_name, object_name=final_s3_object_name,
                            region_name=s3_region
                        )
                        if not s3_url: raise ValueError("Failed to upload video to S3.")

                        # --- 7. Success ---
                        st.write(f"5/5: Generation Complete!")
                        status_log.update(label="Generation Complete!", state="complete", expanded=False)
                        if job_key_to_process in st.session_state.selected_videos:
                            st.session_state.selected_videos[job_key_to_process]['Generated S3 URL'] = s3_url
                            st.session_state.selected_videos[job_key_to_process]['Generation Error'] = None
                            st.session_state.selected_videos[job_key_to_process]['Status'] = 'Completed'
                            st.success(f"✅ Job '{job_key_to_process}' completed!", icon="🎉")
                            st.video(s3_url) # Show the final video
                        else: st.warning(f"Job key {job_key_to_process} missing after completion.")

                    # --- Error Handling within Status Log ---
                    except Exception as e:
                        st.error(f"Error processing job '{job_key_to_process}': {e}", icon="🔥")
                        status_log.update(label=f"Generation Failed: {str(e)[:100]}", state="error", expanded=True)
                        if job_key_to_process in st.session_state.selected_videos:
                            st.session_state.selected_videos[job_key_to_process]['Generation Error'] = str(e)[:200]
                            st.session_state.selected_videos[job_key_to_process]['Generated S3 URL'] = None
                            st.session_state.selected_videos[job_key_to_process]['Status'] = 'Failed'
                        # Allow finally block to handle queue/rerun

        # --- Finally block for cleanup and queue management ---
        finally:
            st.write(f"--- Cleaning up temporary files for job {job_key_to_process} ---")
            # Clean up TTS audio file
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except Exception as rm_err: st.warning(f"Could not delete temp audio: {audio_path} ({rm_err})")
            # Clean up final processed video temp file (returned by process_video_with_tts)
            if final_video_path and os.path.exists(final_video_path):
                try: os.remove(final_video_path)
                except Exception as rm_err: st.warning(f"Could not delete final temp video: {final_video_path} ({rm_err})")
            # Note: process_video_with_tts handles its internal temp files (downloaded base, moviepy temps)

            # Manage Queue
            if st.session_state.generation_queue and st.session_state.generation_queue[0] == job_key_to_process:
                 st.session_state.generation_queue.pop(0)
                 st.session_state.batch_processed_count += 1
            elif st.session_state.generation_queue: # Log unexpected queue state
                 st.warning(f"Queue state unexpected. Expected {job_key_to_process}, found {st.session_state.generation_queue[0]}. Popping front.")
                 st.session_state.generation_queue.pop(0)
                 st.session_state.batch_processed_count += 1

            # Check if batch is finished
            if not st.session_state.generation_queue:
                st.session_state.batch_processing_active = False
                st.balloons()
                st.success("🎉 Batch processing finished!")

            # Rerun to process next item or update UI
            st.rerun()

    # --- Logic for skipping invalid jobs ---
    elif job_key_to_process in st.session_state.selected_videos:
         st.warning(f"Skipping job {job_key_to_process}. Invalid Direct URL or previous error.", icon="❓")
         # Update status to Skipped
         st.session_state.selected_videos[job_key_to_process]['Status'] = 'Skipped (Invalid Data/URL)'
         st.session_state.selected_videos[job_key_to_process]['Generation Error'] = 'Skipped - Invalid data or URL before processing'
         # Manage Queue
         if st.session_state.generation_queue: st.session_state.generation_queue.pop(0)
         st.session_state.batch_processed_count += 1
         if not st.session_state.generation_queue: st.session_state.batch_processing_active = False
         st.rerun()
    else: # Job key somehow disappeared
        st.error(f"Job key {job_key_to_process} in queue but not found. Removing.", icon="❓")
        if st.session_state.generation_queue: st.session_state.generation_queue.pop(0)
        st.session_state.batch_processed_count += 1
        if not st.session_state.generation_queue: st.session_state.batch_processing_active = False
        st.rerun()


# --- Display Selected Jobs Table & Summary (Sidebar) ---
# (Use the corrected display logic from the previous response)
st.sidebar.divider()
st.sidebar.header("Selected & Generated Video Jobs")

if 'selected_videos' in st.session_state and st.session_state.selected_videos:
    selected_list = list(st.session_state.selected_videos.values())
    if selected_list:
        df_selected = pd.DataFrame(selected_list)

        # Define columns for the detailed status table
        display_columns = [
            'Status', 'Video Title', 'Copy Number', 'Topic','Language', 'Script Angle',
            'Video ID', #'Search Term', # Optional
            #'Format Details', # Often too verbose
            'yt_dlp_error', 'Generation Error',
            'Generated S3 URL', 'Job Key'
        ]
        # Ensure required columns exist, fill missing with NA
        for col in display_columns:
             if col not in df_selected.columns:
                  df_selected[col] = pd.NA

        # Clean up display values
        df_selected['Status'] = df_selected['Status'].fillna('Unknown')
        df_selected['Copy Number'] = df_selected['Copy Number'].fillna('?')
        df_selected['yt_dlp_error'] = df_selected['yt_dlp_error'].fillna('OK') # Display OK if no error
        df_selected['Generation Error'] = df_selected['Generation Error'].fillna('OK') # Display OK if no error

        # Create the display DataFrame with selected columns and sorting
        df_selected_display = df_selected[display_columns].copy()
        # Sort by Job Key or other columns as desired
        try: # Sort by Copy Number within Video ID/Lang if possible
             df_selected_display.sort_values(by=['Video ID', 'Language', 'Copy Number'], inplace=True)
        except: # Fallback sort
             df_selected_display.sort_values(by='Job Key', inplace=True)


        # Display the detailed DataFrame
        st.sidebar.dataframe(
            df_selected_display,
            column_config={
                 "Generated S3 URL": st.column_config.LinkColumn("S3 Link", display_text="View"),
                 "yt_dlp_error": st.column_config.TextColumn("URL Fetch Status", width="small", help="Status of fetching direct video URL"),
                 "Generation Error": st.column_config.TextColumn("Generation Status", width="small", help="Status of the video generation process"),
                 "Copy Number": st.column_config.NumberColumn("Copy #", width="small"),
            },
            use_container_width=True,
            hide_index=True
        )

        # Download Button for detailed status
        try:
            csv_data = convert_df_to_csv(df_selected_display) # Use cached conversion
            st.sidebar.download_button(
                 label="📥 Download Job Status (CSV)",
                 data=csv_data,
                 file_name='video_generation_job_status.csv',
                 mime='text/csv',
                 use_container_width=True,
                 disabled=st.session_state.batch_processing_active
            )
        except Exception as e:
            st.sidebar.warning(f"Could not generate detailed CSV: {e}")

        # --- Topic Summary DataFrame Section ---
        st.sidebar.divider()
        st.sidebar.subheader("Generated Video Summary by Topic")
        df_topic_summary = create_topic_summary_dataframe(st.session_state.selected_videos)

        if not df_topic_summary.empty:
            st.sidebar.dataframe(
                 df_topic_summary,
                 use_container_width=True,
                 # Dynamically create LinkColumn config for vidX_url columns
                 column_config={
                     col: st.column_config.LinkColumn(f"Video {i+1}", display_text="View")
                     for i, col in enumerate(df_topic_summary.columns) if col.startswith('vid') and col.endswith('_url')
                 },
                 hide_index=True
            )
            # Download Button for summary
            try:
                csv_summary_data = convert_df_to_csv(df_topic_summary)
                st.sidebar.download_button(
                    label="📥 Download Topic Summary (CSV)",
                    data=csv_summary_data,
                    file_name='generated_videos_by_topic.csv',
                    mime='text/csv',
                    use_container_width=True,
                    disabled=st.session_state.batch_processing_active
                )
            except Exception as e:
                st.sidebar.warning(f"Could not generate summary CSV: {e}")
        else:
            st.sidebar.info("No videos successfully generated yet for summary.")

    else: # If selected_list is empty
         st.sidebar.info("No video jobs selected yet.")
else: # If selected_videos dict is empty or doesn't exist
    st.sidebar.info("No video jobs selected yet.")

# Footer notes
st.sidebar.caption("Each 'Select' click queues one job. URL Fetch after selection. Video Gen uses direct URL.")
