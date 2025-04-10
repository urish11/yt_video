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
    Creates a DataFrame summarizing generated videos grouped by topic.

    Args:
        selected_videos_dict (dict): The session state dictionary
                                     st.session_state.selected_videos.

    Returns:
        pandas.DataFrame: A DataFrame with 'Topic' and 'vidX_url' columns,
                          or an empty DataFrame if no generated videos are found.
    """
    topic_to_generated_urls = {}

    # 1. Collect Generated URLs and Group by Topic
    for video_id, video_data in selected_videos_dict.items():
        topic = video_data.get('Topic')
        s3_url = video_data.get('Generated S3 URL')

        # Only include if topic exists and video was successfully generated
        if topic and s3_url:
            if topic not in topic_to_generated_urls:
                topic_to_generated_urls[topic] = []
            topic_to_generated_urls[topic].append(s3_url)

    if not topic_to_generated_urls:
        # Return an empty DataFrame with expected columns if no data
        return pd.DataFrame(columns=['Topic'])

    # 2. Determine Max URLs per Topic and Prepare Data for Wide Format
    max_urls = 0
    if topic_to_generated_urls: # Ensure dict is not empty before finding max
         max_urls = max(len(urls) for urls in topic_to_generated_urls.values())

    data_for_df = []
    for topic, urls in topic_to_generated_urls.items():
        row = {'Topic': topic}
        # Pad the list of URLs with empty strings up to max_urls
        padded_urls = urls + [''] * (max_urls - len(urls))
        for i, url in enumerate(padded_urls):
            row[f'vid{i+1}_url'] = url
        data_for_df.append(row)

    # 3. Create Final DataFrame
    if data_for_df:
        df_final = pd.DataFrame(data_for_df)
        # Ensure 'Topic' column is first
        topic_col = df_final.pop('Topic')
        df_final.insert(0, 'Topic', topic_col)
        # Sort columns naturally (vid1, vid2, ... vid10...) if needed
        url_cols = sorted([col for col in df_final.columns if col.startswith('vid')],
                           key=lambda x: int(x.replace('vid','').replace('_url','')))
        final_cols = ['Topic'] + url_cols
        df_final = df_final[final_cols]

    else:
         # Should not happen if topic_to_generated_urls was populated, but safe fallback
         df_final = pd.DataFrame(columns=['Topic'])


    return df_final

def search_youtube(api_key, query, max_results=5):
    """
    Performs a Youtube search using the v3 API.
    Handles potential API errors more gracefully.
    """
    params = {
        'part': 'snippet',
        'q': query,
        'key': api_key,
        'type': 'video',
        'maxResults': max_results,
        'videoEmbeddable': 'true'
    }
    videos = []
    response = None # Initialize response to None
    try:
        response = requests.get(YOUTUBE_API_BASE_URL, params=params, timeout=15) # Increased timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        results = response.json()

        if 'items' in results:
            for item in results['items']:
                if item.get('id', {}).get('kind') == 'youtube#video' and 'videoId' in item['id']:
                    video_id = item['id']['videoId']
                    title = item['snippet'].get('title', 'No Title')
                    # Standard watch URL is better for yt-dlp and st.video
                    standard_url = f"https://www.youtube.com/watch?v={video_id}"
                    videos.append({
                        'title': title,
                        'videoId': video_id,
                        'url': standard_url # Store the standard URL
                    })
        return videos

    except requests.exceptions.Timeout:
        st.error(f"API Request Timeout for query '{query}'.", icon="⏱️")
        return [] # Return empty list on timeout
    except requests.exceptions.HTTPError as http_err:
        st.error(f"API HTTP Error for query '{query}': {http_err}", icon="🔥")
        if response is not None:
            if response.status_code == 403:
                st.error("Received status 403 Forbidden. Check API key validity and quota.", icon="🚫")
                # Return None signals a critical API key/quota issue
                return None
            elif response.status_code == 400:
                try:
                    error_details = response.json()
                    st.error(f"Received status 400 Bad Request. Details: {error_details}", icon="👎")
                except json.JSONDecodeError:
                    st.error(f"Received status 400 Bad Request. Response: {response.text}", icon="👎")
            else:
                 st.error(f"API Error Status Code: {response.status_code}. Response: {response.text}", icon="🔥")
        return [] # Return empty list for non-critical HTTP errors
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error for query '{query}': {e}", icon="🌐")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during search for '{query}': {e}", icon="💥")
        return []

# --- Helper Function: Get Info with yt-dlp ---
def get_yt_dlp_info(video_url):
    """
    Uses yt-dlp to extract video format information, prioritizing a direct mp4 URL.
    """
    # Prioritize standard mp4 formats often playable directly in browsers/st.video
    # Format 18: mp4 [360p], Format 22: mp4 [720p]
    # Fallback to best mp4, then best overall
    YDL_OPTS = {
        'format': '22/18/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
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
    try:

        if model == "o1":
            response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{'role': 'user', 'content': prompt}]
        )
            content = response.choices[0].message.content.strip()
            return content


        else:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response.choices[0].message.content.strip()
            return content
    except Exception as e:
        st.error(f"Error calling OpenAI (ChatGPT): {e}", icon="🤖")
        return None


def claude(prompt , model = "claude-3-7-sonnet-20250219", temperature=1 , is_thinking = False, max_retries = 10):
    tries = 0

    while tries < max_retries:
        try:
        
        
        
            client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=st.secrets["ANTHROPIC_API_KEY"])
        
            if is_thinking == False:
                    
                message = client.messages.create(
                    
                model=model,
                max_tokens=20000,
                temperature=temperature,
                
                top_p= 0.8,

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
            )
                return message.content[0].text
            if is_thinking == True:
                message = client.messages.create(
                    
                model=model,
                max_tokens=20000,
                temperature=temperature,
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
                ],
                thinking = { "type": "enabled",
                "budget_tokens": 16000}
            )
                return message.content[1].text
        
        
        
            print(message)
            return message.content[0].text

        except Exception as e:
            st.text(e)
            tries += 1 
            time.sleep(5)

# --- Helper Function: Generate TTS Audio & Timestamps ---
#  --- Helper Function: Generate TTS Audio & Timestamps ---
def generate_audio_with_timestamps(text, client, voice_id="sage"):
    """Generates TTS audio using OpenAI, saves it, and gets word timestamps."""
    temp_audio_file = None
    temp_audio_path = None # Initialize path
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
        # The response object (from OpenAI SDK v1.x+) directly has attributes like 'words'
        # 'words' contains a list of TranscriptionWord objects
        word_timings = []
        if hasattr(transcribe_response, 'words') and transcribe_response.words:
            # transcribe_response.words is a list of TranscriptionWord objects
            # Access attributes directly using dot notation (e.g., word_info.word)
            for word_info in transcribe_response.words:
                word_timings.append({
                    # Safely access attributes, default to None or empty string if needed
                    "word": getattr(word_info, 'word', ''),
                    "start": getattr(word_info, 'start', None),
                    "end": getattr(word_info, 'end', None)
                })
        else:
             # Handle cases where the response might be different or empty
             st.warning("Whisper did not return word timestamps in the expected format.", icon="⏱️")
             # Log the response structure for debugging if necessary
             # print("Unexpected transcription response structure:", transcribe_response)
             # Fallback: could try parsing as JSON if it's an older SDK or different structure
             try:
                 # This fallback might be needed if the response isn't the expected object
                 if isinstance(transcribe_response, str):
                     transcribe_data = json.loads(transcribe_response)
                 elif hasattr(transcribe_response, 'to_dict'): # Some SDK objects have to_dict()
                     transcribe_data = transcribe_response.to_dict()
                 elif isinstance(transcribe_response, dict):
                     transcribe_data = transcribe_response
                 else:
                     transcribe_data = {} # Cannot parse

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
                 # If parsing fails, word_timings remains empty or partially filled

        # --- Validate Timestamps ---
        valid_timings = []
        for wt in word_timings:
            # Check if start and end are not None before attempting float conversion
            if wt.get('start') is not None and wt.get('end') is not None:
                 try:
                     start_time = float(wt['start'])
                     end_time = float(wt['end'])
                     # Ensure end time is after start time
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


        return temp_audio_path, valid_timings # Return the validated list

    except OpenAI.APIError as api_err:
        st.error(f"OpenAI API Error in TTS/Timestamp: {api_err}", icon="🤖")
    except ValueError as ve: # Catch specific errors like empty text
        st.error(f"Value Error in TTS/Timestamp: {ve}", icon="📄")
    except Exception as e:
        # Use repr(e) to get more details potentially, like the exact error type
        st.error(f"Unexpected error in TTS/Timestamp generation: {repr(e)}", icon="💥")
        import traceback
        traceback.print_exc() # Print full traceback to Streamlit logs for debugging
    finally:
        # Cleanup temp file if it exists and an error occurred OR if successful
        # Note: The file needs to exist until MoviePy uses it if processing happens later.
        # Consider moving cleanup *after* video processing.
        # If returning the path, the caller must handle cleanup.
        # For now, let's assume caller will clean up 'temp_audio_path' if it's returned.
        # We only clean up here if the function itself fails and returns None.
        if temp_audio_path and os.path.exists(temp_audio_path) and not valid_timings: # Clean up only on *failure* within this func
            try:
                os.remove(temp_audio_path)
                # print(f"Cleaned up temp audio file on error: {temp_audio_path}") # Debug log
            except Exception as rm_err:
                st.warning(f"Could not remove temp audio file {temp_audio_path} during error handling: {rm_err}")

    # Ensure function returns two values even on error
    return None, None

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
                # Log skipped group?
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
                # Try getting default font with size (newer Pillow)
                font = ImageFont.load_default(size=fontsize)
            except AttributeError:
                # Fallback for older Pillow without size arg
                font = ImageFont.load_default()
                st.warning(f"Using basic default font. Consider providing a TTF font file.", icon="⚠️")


        # --- Configuration ---
        padding_x = 25  # Horizontal padding for the background
        padding_y = 15  # Vertical padding for the background
        bg_radius = 15  # Corner radius for the background
        # Calculate max width for the text itself inside the video frame
        max_text_width = video_width - (2 * padding_x) - 30
        if max_text_width <= 0: max_text_width = video_width // 2 # Safety net

        # --- Text Wrapping ---
        lines = []
        words = text.split()
        if not words:
            return np.zeros((10, 10, 4), dtype=np.uint8) # Handle empty text

        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Use getbbox for more accurate width/height, fallback to getlength
            try:
                 # bbox format: (left, top, right, bottom) relative to (0,0) anchor
                 bbox = font.getbbox(test_line)
                 line_width = bbox[2] - bbox[0]
            except AttributeError: # Fallback for older PIL/Pillow
                 line_width = font.getlength(test_line)


            if line_width <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                # Check if the single word itself is too long
                try:
                    word_bbox = font.getbbox(word)
                    word_width = word_bbox[2] - word_bbox[0]
                except AttributeError:
                    word_width = font.getlength(word)

                if word_width <= max_text_width:
                    current_line = word
                else:
                    # Word too long: Add previous line (if any), add long word on its own line
                    # This might still exceed width if the word is extremely long and cannot be broken
                    if current_line and lines[-1] != current_line: # Avoid double adding
                        pass # Already added above
                    lines.append(word) # Add the long word as its own line
                    current_line = "" # Reset

        if current_line: # Add the last line
            lines.append(current_line)

        wrapped_text = "\n".join(lines)
        if not wrapped_text: wrapped_text = text # Fallback

        # --- Calculate Text Block Dimensions ---
        dummy_img = Image.new("RGBA", (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        try:
            bbox = dummy_draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center') # Use center align here for bbox
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            bbox_y_offset = bbox[1] # Top offset relative to draw origin
            text_width = max(text_width, 1)
            text_height = max(text_height, 1)
        except AttributeError:
            # Fallback
             st.warning("Using fallback subtitle dimension calculation (update Pillow recommended).", icon="PIL")
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
            if isinstance(bg_color, str) and bg_color.startswith('rgba'):
                parts = bg_color.strip('rgba()').split(',')
                r, g, b = map(int, parts[:3])
                a = int(float(parts[3]) * 255)
                fill_color_tuple = (r, g, b, a)
            elif isinstance(bg_color, str): # hex or name
                 fill_color_tuple = bg_color # Pillow handles hex/names directly
            elif isinstance(bg_color, (tuple, list)) and len(bg_color) == 4:
                 fill_color_tuple = tuple(map(int, bg_color)) # Ensure integer tuple
            else: # assume tuple RGB, add alpha
                 fill_color_tuple = tuple(map(int, bg_color)) + (int(0.6 * 255),) if len(bg_color) == 3 else (0,0,0, int(0.6*255))

            draw.rounded_rectangle([(0, 0), (img_width, img_height)], radius=bg_radius, fill=fill_color_tuple)
        except Exception as draw_err:
             st.warning(f"Could not draw rounded rect: {draw_err}. Using simple rect.", icon="🎨")
             # Ensure fill_color_tuple is defined before drawing fallback rectangle
             if 'fill_color_tuple' not in locals():
                 fill_color_tuple = (0,0,0, int(0.6*255)) # Default fallback fill
             draw.rectangle([(0,0), (img_width, img_height)], fill=fill_color_tuple)

        # --- Draw Text ---
        text_x = padding_x
        text_y = padding_y - bbox_y_offset # Adjust vertical start based on calculated bbox top

        draw.multiline_text(
            (text_x, text_y),
            wrapped_text,
            font=font,
            fill=color,
            align="center",
            spacing=4,
            anchor="la" # Anchor at left-ascent of the first line
        )

        return np.array(img)

    except Exception as e:
        st.error(f"Error creating text image for '{text[:50]}...': {e}", icon="🎨")
        return np.zeros((10, 10, 4), dtype=np.uint8)

# --- Helper Function: Process Video with TTS and Subtitles ---
def process_video_with_tts(base_video_url, audio_path, word_timings, topic):
    """Loads video, adds TTS audio, loops if necessary, adds subtitles centered with wrapping."""
    final_video_clip = None
    temp_output_path = None
    base_video = None
    tts_audio = None
    looped_video = None
    processed_video = None
    subtitle_clips_list = []

    try:
        st.write(f"⏳ Loading base video from URL...")
        # Add user agent to potentially help with access issues
        # Note: This might not always work, especially for signed URLs
        # Consider downloading locally first if direct URL access is flaky
        try:
            # Let MoviePy handle the URL directly
            base_video = VideoFileClip(base_video_url, audio=False, target_resolution=(720, 1280)) # Target 720p vertical
            # Or download first if direct URL fails often:
            # with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_vid_file:
            #     response = requests.get(base_video_url, stream=True)
            #     response.raise_for_status()
            #     for chunk in response.iter_content(chunk_size=8192):
            #         temp_vid_file.write(chunk)
            #     temp_video_path = temp_vid_file.name
            # base_video = VideoFileClip(temp_video_path, audio=False)

        except Exception as load_err:
            st.error(f"Failed to load video using MoviePy: {load_err}", icon="🎬")
            # if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            #      os.remove(temp_video_path) # Clean up partial download if any
            raise # Re-raise the error to be caught by the main handler

        video_duration = base_video.duration
        # Ensure dimensions are integers
        w = int(base_video.w) if base_video.w else 720 # Default width
        h = int(base_video.h) if base_video.h else 1280 # Default height
        st.write(f"✔️ Base video loaded: {w}x{h}, Duration: {video_duration:.2f}s")


        st.write(f"⏳ Loading TTS audio...")
        tts_audio = AudioFileClip(audio_path)
        audio_duration = tts_audio.duration
        st.write(f"✔️ TTS audio loaded: Duration: {audio_duration:.2f}s")

        # --- Video Resizing (Force 9:16 aspect ratio - e.g., 720x1280) ---
        target_w, target_h = 720, 1280
        st.write(f"⏳ Resizing video to {target_w}x{target_h}...")
        try:
            # Use crop for resizing while maintaining aspect ratio and centering
            # This will zoom in if the aspect ratio doesn't match
            # resized_base_video = base_video.fx(vfx.crop, width=target_w, height=target_h, x_center=w/2, y_center=h/2)
            # Alternative: Simple resize (might distort if aspect ratio differs)
            resized_base_video = base_video.resize(newsize=(target_w, target_h))
            st.write(f"✔️ Video resized.")
        except Exception as resize_err:
            st.warning(f"Could not resize video: {resize_err}. Using original dimensions.", icon="⚠️")
            resized_base_video = base_video # Fallback to original if resize fails
            target_w, target_h = w, h # Use original dimensions for subtitles etc.


        # --- Video Looping/Trimming ---
        processed_video = resized_base_video # Start with resized_base_video
        if video_duration < audio_duration:
            st.write(f"⏳ Looping video to match audio duration...")
            num_loops = int(np.ceil(audio_duration / video_duration))
            # Use copies of the *resized* clip
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
            # Use st.empty() for progress bar that can be updated
            sub_progress_bar = st.progress(0)
            status_text = st.empty()

            for i, sub_data in enumerate(grouped_subs):
                text = sub_data['text']
                start = sub_data['start']
                end = sub_data['end']
                sub_duration = end - start

                # Skip empty text or very short duration subs
                if not text.strip() or sub_duration <= 0.05:
                    continue

                status_text.text(f"Creating subtitle {i+1}/{total_subs}: '{text[:30]}...'")

                # --- Create subtitle image WITH target video_width ---
                text_img_array = create_text_image(
                    text.upper(),
                    fontsize=SUBTITLE_FONT_SIZE,
                    color=SUBTITLE_COLOR,
                    bg_color=SUBTITLE_BG_COLOR,
                    font_path=SUBTITLE_FONT_PATH,
                    video_width=target_w # <<< Pass target width here
                )

                if text_img_array.shape[0] <= 10 or text_img_array.shape[1] <= 10:
                     st.warning(f"Skipping subtitle due to image creation error for: '{text[:30]}...'")
                     continue

                # --- Create ImageClip and Position ---
                subtitle_img_clip = ImageClip(text_img_array)\
                    .set_start(start)\
                    .set_duration(sub_duration)\
                    .set_position(('center', 'center')) # Position: Center horizontally, 60% down vertically

                subtitle_clips_list.append(subtitle_img_clip)
                sub_progress_bar.progress((i + 1) / total_subs)

            status_text.text(f"✔️ Generated {len(subtitle_clips_list)} subtitle clips.")
            # Optionally clear the progress bar after completion
            # sub_progress_bar.empty()
            # status_text.empty()
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
        temp_output_filename = f"final_{safe_topic}_{timestamp}.mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=f"final_{safe_topic}_") as temp_output_file_obj:
             temp_output_path = temp_output_file_obj.name

        if not isinstance(final_video_clip, (VideoFileClip, CompositeVideoClip)):
             raise TypeError(f"Cannot write final video: Invalid clip object type {type(final_video_clip)}.")

        # Use recommended parameters for web compatibility
        final_video_clip.write_videofile(
            temp_output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=f'temp-audio-{timestamp}.m4a', # Explicit temp audio file
            remove_temp=True,
            fps=resized_base_video.fps or 24, # Use fps from resized video
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

        return temp_output_path, temp_output_filename

    except Exception as e:
        st.error(f"Error during video processing: {e}", icon="🎬")
        # Print traceback for debugging if needed
        import traceback
        traceback.print_exc()
        raise # Re-raise the exception

    finally:
        # --- Cleanup ---
        st.write("🧹 Cleaning up video processing resources...")
        try:
             if base_video: base_video.close()
             if tts_audio: tts_audio.close()
             if looped_video: looped_video.close()
             # Check if processed_video is different from others before closing
             if processed_video and processed_video not in [base_video, looped_video, resized_base_video]:
                  processed_video.close()
             if 'resized_base_video' in locals() and resized_base_video is not base_video:
                  resized_base_video.close()
             if final_video_clip and final_video_clip not in clips_for_composite: # Avoid closing the composite source
                final_video_clip.close()
             for sub_clip in subtitle_clips_list:
                   sub_clip.close()
             # if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
             #      os.remove(temp_video_path) # Remove downloaded video
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
        # Construct the URL (common format, ensure your bucket policy allows public read if needed)
        # Check if region is us-east-1, which often doesn't need the region in the URL hostname
        if region_name == 'us-east-1':
             video_url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        else:
             video_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"

        # URL encode the object name for safety in URLs
        safe_object_name = urllib.parse.quote(object_name)
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
        {'Topic': 'sofa sale', 'Search Term': 'sofa #shorts' ,'Language' : 'English', 'Video Results': 5}
   
        
    ])

# Flag to indicate if a search has been run
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False

# --- NEW: State for Batch Processing ---
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
edited_df = st.sidebar.data_editor(
    st.session_state.search_data,
    num_rows="dynamic",
    # column_config={
    #     "Search Term": st.column_config.TextColumn("YouTube Search Term", required=True),
    #     "Topic": st.column_config.TextColumn("Topic for TTS Script", required=True),
    #     "Language": st.column_config.TextColumn("Language", required=True),
    #     "Video Results": st.column_config.NumberColumn("Vid Results", required=True)
        
    # },
    use_container_width=True,
    key="search_topic_editor",
    # Disable editing while processing? Maybe not necessary for the input table.
    # disabled=st.session_state.batch_processing_active
)
# Update session state with edited data
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
    # Optionally reset the input table
    # st.session_state.search_data = pd.DataFrame([{'Search Term': '', 'Topic': ''}])
    st.success("Selections, results, and generation state cleared!", icon="✅")
    st.rerun()

# --- NEW: Global Process Button ---
st.sidebar.divider()
# Calculate how many selected videos are ready for processing
ready_videos_count = sum(
    1 for vid_id, data in st.session_state.selected_videos.items()
    if data.get('Direct URL') and not data.get('yt_dlp_error') and not data.get('Generated S3 URL') and not data.get('Generation Error') # Only count ready and not already done/failed
)

process_all_button = st.sidebar.button(
    f"🎬 Process {ready_videos_count} Ready Video{'s' if ready_videos_count != 1 else ''}",
    use_container_width=True,
    type="primary",
    disabled=ready_videos_count == 0 or st.session_state.batch_processing_active # Disable if 0 ready or already processing
)

if process_all_button:
    # Find videos that are selected, have a direct URL, and haven't been processed/failed
    ids_to_process = [
        vid_id for vid_id, data in st.session_state.selected_videos.items()
        if data.get('Direct URL') and not data.get('yt_dlp_error') and not data.get('Generated S3 URL') and not data.get('Generation Error')
    ]
    if ids_to_process:
        st.session_state.generation_queue = ids_to_process
        st.session_state.batch_processing_active = True
        st.session_state.batch_total_count = len(ids_to_process)
        st.session_state.batch_processed_count = 0
        st.sidebar.info(f"Queued {len(ids_to_process)} videos for generation.", icon="⏳")
        # Clear any potential old individual errors before starting batch
        for vid_id in ids_to_process:
            st.session_state.selected_videos[vid_id]['Generation Error'] = None
            st.session_state.selected_videos[vid_id]['Generated S3 URL'] = None # Reset S3 URL too
        st.rerun()
    else:
        st.sidebar.warning("No selected videos are ready for processing (need Direct URL).", icon="⚠️")


st.sidebar.info("Select videos using '➕'. Processing uses selected videos with valid 'Direct URL'.", icon="ℹ️")
st.sidebar.warning("Video generation can take several minutes per video.", icon="⏱️")

# --- Processing Logic ---

# 1. Handle Search Button Click
if search_button:
    # Validate input data
    valid_input = True
    if edited_df.empty:
        st.sidebar.warning("Please add at least one Search Term and Topic.", icon="⚠️")
        valid_input = False
    # Use .ne('') for checking empty strings in pandas
    if edited_df['Search Term'].isnull().any() or edited_df['Search Term'].eq('').any():
        st.sidebar.warning("Search Term cannot be empty.", icon="⚠️")
        valid_input = False
    if edited_df['Topic'].isnull().any() or edited_df['Topic'].eq('').any():
        st.sidebar.warning("Topic cannot be empty.", icon="⚠️")
        valid_input = False

    if valid_input:
        st.session_state.search_triggered = True
        st.session_state.api_search_results = {} # Clear previous API results on new search
        # Don't clear selected videos on new search
        # Reset generation state if a new search is performed
        st.session_state.generation_queue = []
        st.session_state.batch_processing_active = False
        st.session_state.batch_total_count = 0
        st.session_state.batch_processed_count = 0
        # Store the dataframe used for this search
        st.session_state.current_search_df = edited_df.copy()
        st.rerun() # Rerun to start the search process below
    else:
        st.session_state.search_triggered = False # Ensure search doesn't proceed


# 2. Perform API Search if Triggered
if st.session_state.search_triggered and 'current_search_df' in st.session_state and not st.session_state.api_search_results:
    search_df = st.session_state.current_search_df
    search_items = search_df.to_dict('records') # Convert df rows to list of dicts

    st.info(f"Searching API for {len(search_items)} term(s)... (Max {MAX_RESULTS_PER_QUERY} results per term)", icon="🔍")
    progress_bar = st.progress(0)
    status_text_api = st.empty()

    api_error_occurred = False
    results_cache = {} # Temp cache for this run

    for i, item in enumerate(search_items):
        term = item['Search Term']
        topic = item['Topic'] # Keep topic associated
        count = int(item['Video Results'])
        lang = item['Language'] # Language for search
        status_text_api.text(f"Searching for: '{term}'...")

        if term not in results_cache: # Avoid re-searching same term in one go
            videos = search_youtube(youtube_api_key_secret, term, count)

            if videos is None: # Critical API error (e.g., 403)
                st.error(f"Stopping search due to critical API issue (check key/quota) for term: '{term}'.", icon="🚫")
                api_error_occurred = True
                break # Stop processing further terms

            # Store results along with the topic
            results_cache[term] = {'videos': videos, 'topic': topic , 'lang' : lang}
            time.sleep(0.1) # Small delay between API calls
        progress_bar.progress((i + 1) / len(search_items))

    status_text_api.text("API Search complete.")
    st.session_state.api_search_results = results_cache # Update main cache

    if api_error_occurred:
        st.session_state.search_triggered = False # Reset trigger on critical error
        st.warning("Search halted due to API error.", icon="⚠️")
        # Don't stop the app, just show the error. Rerun will display current state.
        st.rerun()
    else:
        # Don't necessarily reset trigger, allow results to display
        # st.session_state.search_triggered = False # Reset trigger after successful search display
        st.rerun() # Rerun to display results


# --- Display Area ---
st.divider()

# --- Display Batch Processing Status ---
if st.session_state.batch_processing_active:
    processed = st.session_state.batch_processed_count
    total = st.session_state.batch_total_count
    queue_len = len(st.session_state.generation_queue)
    st.info(f"⚙️ Batch Processing: {processed} / {total} Completed. {queue_len} Remaining in queue.", icon="⏳")
    if total > 0:
        st.progress(processed / total)


# 3. Display Search Results
if st.session_state.api_search_results:
    st.subheader("Search Results & Video Selection")
    # Display results from cache
    for term, result_data in st.session_state.api_search_results.items():
        videos = result_data['videos']
        topic = result_data['topic'] # Retrieve the associated topic
        lang = result_data['lang'] # Retrieve the associated language
        container = st.container(border=True)
        with container:
            st.subheader(f"Results for: \"{term}\" (Topic: \"{topic}\")")
            if not videos:
                st.write("No videos found via API.")
                continue

            for video in videos:
                video_id = video['videoId']
                video_title = video['title']
                standard_video_url = video['url'] # Use the standard URL from search results
                unique_key_base = f"{term}_{video_id}" # More robust key

                # --- Check Video State ---
                is_selected = video_id in st.session_state.selected_videos
                video_state = st.session_state.selected_videos.get(video_id, {})
                has_dlp_info = is_selected and video_state.get('Direct URL') and not video_state.get('yt_dlp_error')
                is_fetching_dlp = is_selected and video_state.get('fetching_dlp', False)
                dlp_error = video_state.get('yt_dlp_error')

                # NEW: Check generation status (part of batch or completed/failed)
                is_in_queue = video_id in st.session_state.generation_queue
                is_currently_processing = st.session_state.batch_processing_active and st.session_state.generation_queue and st.session_state.generation_queue[0] == video_id # Check if it's the *next* item
                generation_error = video_state.get('Generation Error')
                s3_url = video_state.get('Generated S3 URL')
                is_completed = bool(s3_url)
                is_failed = bool(generation_error)

                col_vid, col_actions = st.columns([3, 1])

                with col_vid:
                    st.write(f"**{video_title}**")
                    st.caption(f"ID: {video_id} | [Watch on YouTube]({standard_video_url})")
                    try:
                        # Use st.video with the standard URL - works better generally
                        st.video(standard_video_url)
                    except Exception as e:
                        st.warning(f"Could not embed video player: {standard_video_url}. Error: {e}", icon="🎬")

                with col_actions:
                    # --- Select / Deselect Button ---
                    select_button_label = "???" # Placeholder
                    select_button_type = "secondary"
                    select_disabled = st.session_state.batch_processing_active # Disable selection changes during batch processing

                    if is_selected:
                        select_button_label = "✅ Deselect"
                        select_button_type = "secondary"
                    elif is_fetching_dlp:
                         select_button_label = "⏳ Fetching..."
                         select_button_type = "secondary"
                         select_disabled = True # Also disable if fetching
                    else:
                        select_button_label = "➕ Select"
                        select_button_type = "primary"

                    if st.button(select_button_label, key=f"select_{unique_key_base}", type=select_button_type, use_container_width=True, disabled=select_disabled):
                        if is_selected:
                            del st.session_state.selected_videos[video_id]
                            st.toast(f"Deselected: {video_title}", icon="➖")
                            # Remove from queue if it was there
                            if video_id in st.session_state.generation_queue:
                                st.session_state.generation_queue.remove(video_id)
                                st.session_state.batch_total_count = len(st.session_state.generation_queue) # Adjust total if needed? Or maybe keep original total? Let's keep original for now.
                        else:
                            # Mark as fetching and add basic info
                            st.session_state.selected_videos[video_id] = {
                                'Search Term': term,
                                'Topic': topic, # Store the topic
                                'Language': lang, # Store the language
                                'Video Title': video_title,
                                'Video ID': video_id,
                                'Standard URL': standard_video_url,
                                'fetching_dlp': True, # Mark as fetching
                                'Direct URL': None,
                                'Format Details': None,
                                'yt_dlp_error': None,
                                'Generated S3 URL': None,
                                'Generation Error': None,
                                'Status': 'Selected, Fetching URL...' # Add a general status
                            }
                            st.toast(f"Selected: {video_title}. Fetching direct URL...", icon="⏳")
                        st.rerun()

                    # --- Display Status (yt-dlp and Generation) ---
                    status_container = st.container(border=False) # Use container for status messages
                    if is_selected:
                        if is_fetching_dlp:
                             status_container.info("⏳ Fetching URL...", icon="📡")
                        elif dlp_error:
                             status_container.error(f"URL Error: {dlp_error}", icon="⚠️")
                        elif not has_dlp_info and not is_fetching_dlp:
                             status_container.warning("URL fetch incomplete.", icon="❓")
                        elif has_dlp_info:
                            # Now check generation status if URL is ready
                            if is_currently_processing:
                                status_container.info("⚙️ Processing...", icon="⏳")
                            elif is_in_queue:
                                 status_container.info("🕒 Queued", icon="🕒")
                            elif is_completed:
                                status_container.success("✔️ Generated!", icon="🎉")
                                st.link_button("View on S3", url=s3_url, use_container_width=True)
                            elif is_failed:
                                status_container.error(f"❌ Failed: {generation_error[:50]}...", icon="🔥") # Show truncated error
                            else:
                                # Ready, but not processing/queued/done/failed yet
                                status_container.success("✅ Ready to Process", icon="👍")


                    # REMOVED Individual Generate Button (now handled globally)

            st.divider()

# --- yt-dlp Fetching Logic (runs after initial UI render if needed) ---
# Check if batch processing is NOT active before fetching to avoid conflicts
if not st.session_state.batch_processing_active:
    ids_to_fetch = [
        vid for vid, data in st.session_state.selected_videos.items()
        if data.get('fetching_dlp') # Only fetch if marked
    ]

    if ids_to_fetch:
        fetch_id = ids_to_fetch[0] # Process one at a time per rerun
        video_data = st.session_state.selected_videos.get(fetch_id)

        if video_data: # Ensure data exists
            standard_url = video_data.get('Standard URL')
            title = video_data.get('Video Title', fetch_id)

            # Show spinner only if fetching is actually happening for this ID
            with st.spinner(f"Fetching yt-dlp details for '{title}'..."):
                 dlp_info = get_yt_dlp_info(standard_url)

            # Update state based on dlp_info result
            # Use .get(fetch_id) again in case it was deleted between checks
            current_state = st.session_state.selected_videos.get(fetch_id)
            if current_state: # Check if still exists
                current_state['fetching_dlp'] = False # Mark fetch attempt as complete

                if dlp_info and dlp_info.get('direct_url'):
                    current_state['Direct URL'] = dlp_info['direct_url']
                    current_state['Format Details'] = dlp_info['format_details']
                    current_state['yt_dlp_error'] = None
                    current_state['Status'] = 'Ready'
                    st.toast(f"Direct URL loaded for '{title}'", icon="✅")
                elif dlp_info and dlp_info.get('error'):
                    current_state['Direct URL'] = None
                    current_state['Format Details'] = "Error"
                    current_state['yt_dlp_error'] = dlp_info['error']
                    current_state['Status'] = f"Error: {dlp_info['error']}"
                    st.toast(f"yt-dlp failed for '{title}': {dlp_info['error']}", icon="⚠️")
                else: # Critical failure from get_yt_dlp_info or dlp_info is None
                    current_state['Direct URL'] = None
                    current_state['Format Details'] = "Critical Error"
                    current_state['yt_dlp_error'] = dlp_info.get('error', "Critical yt-dlp failure") if dlp_info else "Critical yt-dlp failure"
                    current_state['Status'] = f"Error: {current_state['yt_dlp_error']}"
                    st.toast(f"Critical yt-dlp error for '{title}'", icon="🔥")

                st.session_state.selected_videos[fetch_id] = current_state # Save updated state
                st.rerun() # Rerun to update UI


# 4. Video Generation Logic (BATCH PROCESSING)
if st.session_state.batch_processing_active and st.session_state.generation_queue:
    video_id_to_process = st.session_state.generation_queue[0] # Peek at the next item
    video_data = st.session_state.selected_videos.get(video_id_to_process)
    final_video_path = None # Initialize paths for cleanup
    audio_path = None

    if video_data and video_data.get('Direct URL') and not video_data.get('yt_dlp_error'):
        processed_count_display = st.session_state.batch_processed_count + 1
        total_count_display = st.session_state.batch_total_count
        st.header(f"⚙️ Processing Video {processed_count_display}/{total_count_display}: {video_data['Video Title']}")
        gen_placeholder = st.container() # Use a container for logs within this specific run

        try:
            # --- Update Status ---
            st.session_state.selected_videos[video_id_to_process]['Status'] = 'Processing'

            with gen_placeholder:
                st.info(f"Starting video generation process...")
                # Use st.status for collapsible logs
                with st.status("Running generation steps...", state="running", expanded=True) as status_log:
                    try:
                        # --- Step 1: Get Topic ---
                        topic = video_data.get('Topic', 'the selected video') # Fallback topic
                        lang = video_data.get('Language', 'English') # Fallback language
                        st.write(f"1/5: Generating script for topic: '{topic}'...")

                        # --- Step 2: Generate Script (Text Only) ---
                        script_prompt = f"Create a short, engaging voiceover script for FB viral   video (roughly 15-20 seconds long, maybe 2-3 sentences) about '{topic}' in language {lang}. The tone should be informative yet conversational, '.  smooth flow. Just provide the script text, nothing else. create intriguing and engaging script, sell the topic to the audience ,make them click in and the topic very attractive. be very causal and not 'advertisement' style vibe. end with a call to action  .the text needs to be retentive and highly engaging. "
                        # script_text = chatGPT(script_prompt,model="o1", client=openai_client)
                        script_text = claude(script_prompt,is_thinking=True)

                        if not script_text:
                            raise ValueError("Failed to generate script text from OpenAI.")
                        st.text_area("Generated Script:", script_text, height=100, disabled=True, key=f"script_{video_id_to_process}")
                        st.write(f"2/5: Generating '{DEFAULT_TTS_VOICE}' TTS audio and timestamps...")


                        # --- Step 3: Generate TTS Audio & Timestamps ---
                        audio_path, word_timings = generate_audio_with_timestamps(
                            script_text, client=openai_client, voice_id=DEFAULT_TTS_VOICE
                        )
                        if audio_path is None or word_timings is None: # Check for None explicitly
                            raise ValueError("Failed to generate TTS audio or timestamps.")
                        st.write(f"3/5: Processing base video and adding audio/subtitles...")


                        # --- Step 4: Process Video (Combine, Loop, Subtitles) ---
                        final_video_path, final_filename = process_video_with_tts(
                            base_video_url=video_data['Direct URL'],
                            audio_path=audio_path,
                            word_timings=word_timings,
                            topic=topic # Pass topic for filename generation
                        )
                        if not final_video_path:
                            raise ValueError("Video processing (MoviePy) failed.")
                        st.write(f"4/5: Uploading final video '{final_filename}' to S3...")


                        # --- Step 5: Upload to S3 ---
                        s3_url = upload_vid_to_s3(
                            s3_cli=s3_client,
                            video_path=final_video_path,
                            bucket_name=s3_bucket_name,
                            object_name=final_filename,
                            region_name=s3_region
                        )
                        if not s3_url:
                            raise ValueError("Failed to upload video to S3.")
                        st.write(f"5/5: Generation Complete! S3 URL ready.")

                        status_log.update(label="Generation Complete!", state="complete", expanded=False)

                        # --- Step 6: Update Session State for this video (Success) ---
                        st.session_state.selected_videos[video_id_to_process]['Generated S3 URL'] = s3_url
                        st.session_state.selected_videos[video_id_to_process]['Generation Error'] = None # Clear error on success
                        st.session_state.selected_videos[video_id_to_process]['Status'] = 'Completed'
                        st.success(f"✅ Video generated and uploaded successfully!", icon="🎉")
                        st.video(s3_url) # Display the final video from S3




                    except Exception as e:
                        st.error(f"Error during video generation step: {e}", icon="🔥")
                        # Update status log to show error
                        status_log.update(label=f"Generation Failed: {e}", state="error", expanded=True)
                        # Update session state for this video (Failure)
                        st.session_state.selected_videos[video_id_to_process]['Generation Error'] = str(e)[:200] # Store truncated error
                        st.session_state.selected_videos[video_id_to_process]['Generated S3 URL'] = None # Clear URL on error
                        st.session_state.selected_videos[video_id_to_process]['Status'] = 'Failed'
                        # Don't raise here, let finally block handle queue and rerun


        finally:
            # --- Cleanup Temporary Files ---
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except Exception as rm_err: st.warning(f"Could not delete temp audio: {audio_path} ({rm_err})")
            if final_video_path and os.path.exists(final_video_path):
                try: os.remove(final_video_path)
                except Exception as rm_err: st.warning(f"Could not delete temp video: {final_video_path} ({rm_err})")

            # --- Update Queue and Progress ---
            # Remove the processed item from the front of the queue
            st.session_state.generation_queue.pop(0)
            st.session_state.batch_processed_count += 1

            # Check if queue is now empty
            if not st.session_state.generation_queue:
                st.session_state.batch_processing_active = False # Mark batch as complete
                st.balloons()
                st.success("🎉 Batch processing finished!")

            # Rerun to process the next item or update the UI if finished
            st.rerun()

    elif video_id_to_process in st.session_state.selected_videos: # Handle case where data is missing or URL invalid for the queued item
        st.warning(f"Skipping video {video_id_to_process}. Data missing or invalid Direct URL.", icon="❓")
        st.session_state.selected_videos[video_id_to_process]['Status'] = 'Skipped (Invalid Data)'
        st.session_state.selected_videos[video_id_to_process]['Generation Error'] = 'Skipped - Invalid data or URL'
        # --- Update Queue and Progress (as it was skipped) ---
        st.session_state.generation_queue.pop(0)
        st.session_state.batch_processed_count += 1
        if not st.session_state.generation_queue:
            st.session_state.batch_processing_active = False
            st.info("Batch processing finished (last item skipped).")
        st.rerun()
    else:
        # Video ID was in queue but somehow removed from selected_videos (shouldn't normally happen)
        st.error(f"Video ID {video_id_to_process} was in queue but not found in selected videos. Removing from queue.")
        st.session_state.generation_queue.pop(0)
        # Don't increment processed count here? Or should we? Let's increment to avoid infinite loops.
        st.session_state.batch_processed_count += 1
        if not st.session_state.generation_queue:
            st.session_state.batch_processing_active = False
        st.rerun()


# --- Display Selected Videos Table (Sidebar) ---
st.sidebar.divider()
st.sidebar.header("Selected & Generated Videos Status")

# Keep the existing logic to display the detailed table first...
if st.session_state.selected_videos:
    selected_list = list(st.session_state.selected_videos.values())
    if selected_list: # Check if list is not empty
        df_selected = pd.DataFrame(selected_list)

        # Define desired columns and order, including the new 'Status'
        display_columns = [
            'Status', 'Video Title', 'Topic', 'Search Term', 'Video ID',
            'Format Details', 'yt_dlp_error', 'Generation Error',
             'Generated S3 URL', #'Direct URL', 'Standard URL' # Keep URLs less prominent maybe
        ]

        # Ensure all display columns exist, fill missing
        for col in display_columns:
            if col not in df_selected.columns:
                df_selected[col] = "N/A" # Use N/A string

        # Fill specific NaN/None values for better display
        df_selected['Status'] = df_selected['Status'].fillna('Unknown')
        df_selected['yt_dlp_error'] = df_selected['yt_dlp_error'].fillna('OK')
        df_selected['Generation Error'] = df_selected['Generation Error'].fillna('OK')
        df_selected.fillna("N/A", inplace=True) # Fill remaining NAs


        # Reorder DataFrame columns for display
        df_selected_display = df_selected[display_columns]

        st.sidebar.dataframe(
            df_selected,
            
            hide_index=True
        )

        # Existing Download Button for detailed table
        @st.cache_data
        def convert_df_to_csv(df):
            # IMPORTANT: Use BytesIO for text output in newer Streamlit/Pandas versions
            output = BytesIO()
            df.to_csv(output, index=False, encoding='utf-8')
            return output.getvalue() # Return bytes

        try:
            csv_data = convert_df_to_csv(df_selected_display)
            st.sidebar.download_button(
                label="📥 Download Detailed Status (CSV)", # Updated label
                data=csv_data,
                file_name='selected_youtube_videos_detailed_status.csv', # Updated filename
                mime='text/csv',
                use_container_width=True,
                disabled=st.session_state.batch_processing_active # Disable download during processing
            )
        except Exception as e:
            st.sidebar.warning(f"Could not generate detailed CSV: {e}")

        # --- NEW SECTION: Topic Summary DataFrame ---
        st.sidebar.divider()
        st.sidebar.subheader("Generated Video Summary by Topic")

        # Generate the summary DataFrame using the helper function
        df_topic_summary = create_topic_summary_dataframe(st.session_state.selected_videos)

        if not df_topic_summary.empty:
            st.sidebar.dataframe(
                df_topic_summary,
                use_container_width=True,
                # Configure columns for link display if desired
                column_config={
                    col: st.column_config.LinkColumn(f"Video {i+1}")
                    for i, col in enumerate(df_topic_summary.columns) if col.startswith('vid')
                },
                 hide_index=True
            )

            # Download Button for the topic summary DataFrame
            try:
                csv_summary_data = convert_df_to_csv(df_topic_summary) # Reuse the conversion function
                st.sidebar.download_button(
                    label="📥 Download Topic Summary (CSV)",
                    data=csv_summary_data,
                    file_name='generated_videos_by_topic.csv',
                    mime='text/csv',
                    use_container_width=True,
                    disabled=st.session_state.batch_processing_active # Disable download during processing
                )
            except Exception as e:
                st.sidebar.warning(f"Could not generate summary CSV: {e}")

        else:
            st.sidebar.info("No videos have been successfully generated yet to create a summary.")
            # Optionally display the empty dataframe structure
            # st.sidebar.dataframe(df_topic_summary, use_container_width=True)

    else: # If selected_videos dict is empty (inner check)
         st.sidebar.info("No videos selected yet. Use '➕ Select' buttons in the main area.")

else: # If selected_videos dict doesn't exist or is empty (outer check)
    st.sidebar.info("No videos selected yet. Use '➕ Select' buttons in the main area.")

# Footer notes
st.sidebar.caption("YT API Search: ~100 quota units/term. URL Fetch on select. Video Gen uses direct URL.")