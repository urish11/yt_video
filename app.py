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
if 'resolved_vid_urls' not in st.session_state:
  st.session_state['resolved_vid_urls'] = {} # vid:url

result = os.popen('pip list').read()
st.code(result, language=None)
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
    (DEBUG VERSION) Creates a DataFrame summarizing generated videos grouped by a normalized
    'topic_language' key. Includes debug output.
    """
    topic_lang_to_generated_urls = {}

    # 1. Collect Generated URLs and Group by Normalized Topic + Language
    for video_key, video_data in selected_videos_dict.items():
        topic = str(video_data.get('Topic', '')).strip().lower()
        lang = str(video_data.get('Language', '')).strip().lower()
        s3_url = video_data.get('Generated S3 URL')

        if topic and lang and s3_url:
            grouping_key = f"{topic}_{lang}"
            if grouping_key not in topic_lang_to_generated_urls:
                topic_lang_to_generated_urls[grouping_key] = []
            topic_lang_to_generated_urls[grouping_key].append(s3_url)

    # --- Add Debugging ---
    # st.sidebar.write("--- Debugging Summary Function ---")
    # st.sidebar.write("`topic_lang_to_generated_urls` (Should show lists with 2 URLs each):")
    # # Use st.json for better dict/list display in sidebar
    # try:
    #     st.sidebar.json(topic_lang_to_generated_urls, expanded=False)
    # except Exception as e:
    #     st.sidebar.write(f"Error displaying json: {e}")
    #     st.sidebar.write(topic_lang_to_generated_urls) # Fallback to plain write
    # --- End Debugging ---

    if not topic_lang_to_generated_urls:
        st.sidebar.write("Debug: No groups found.")
        st.sidebar.write("--- End Debugging Summary Function ---")
        return pd.DataFrame(columns=['Topic'])

    # 2. Determine Max URLs per Group and Prepare Data
    max_urls = 0
    if topic_lang_to_generated_urls:
        try:
            max_urls = max(len(urls) for urls in topic_lang_to_generated_urls.values())
        except ValueError: # Handles case where dictionary is empty after filtering bad data
             max_urls = 0

    # --- Add Debugging ---
    # st.sidebar.write(f"`max_urls` (Should be 2): {max_urls}")
    # --- End Debugging ---

    data_for_df = []
    for topic_lang_key, urls in topic_lang_to_generated_urls.items():
        row = {'Topic': topic_lang_key}
        padded_urls = urls + [''] * (max_urls - len(urls))
        # This loop seems like the most likely place for an error if max_urls is correct
        for i, url in enumerate(padded_urls):
            row[f'vid{i+1}_url'] = url # Creates vid1_url, vid2_url etc. keys in the row dict
        data_for_df.append(row) # Appends the complete row dict

    # --- Add Debugging ---
    # st.sidebar.write("`data_for_df` (Should be list of dicts, each with vid1_url AND vid2_url):")
    # try:
    #     st.sidebar.json(data_for_df, expanded=False)
    # except Exception as e:
    #     st.sidebar.write(f"Error displaying json: {e}")
    #     st.sidebar.write(data_for_df) # Fallback
    # --- End Debugging ---

    # 3. Create Final DataFrame
    if data_for_df:
        # Create DF directly from the list of dictionaries
        df_final = pd.DataFrame(data_for_df)

        # --- Add Debugging ---
        # st.sidebar.write("`df_final` (Immediately after creation, should have vid1/vid2 cols):")
        # st.sidebar.dataframe(df_final)
        # --- End Debugging ---

        # Check if necessary columns were created before trying to reorder
        if 'Topic' in df_final.columns:
             # Ensure 'Topic' column is first
             topic_col = df_final.pop('Topic')
             df_final.insert(0, 'Topic', topic_col)

             # Get existing URL columns and sort them
             url_cols_present = [col for col in df_final.columns if col.startswith('vid')]
             url_cols_sorted = sorted(url_cols_present,
                                      key=lambda x: int(x.replace('vid','').replace('_url','')))
             final_cols = ['Topic'] + url_cols_sorted

             # --- Add Debugging ---
            #  st.sidebar.write("`final_cols` list (Columns expected in final output):")
            #  st.sidebar.write(final_cols)
             # --- End Debugging ---

             # Reindex to ensure columns exist and are in order - might be masking the real issue
             # Let's try returning without reindex first to see raw structure
             # df_final = df_final.reindex(columns=final_cols, fill_value='')
             # Return the dataframe after sorting columns, before reindex
             df_final = df_final[['Topic'] + url_cols_sorted]


        else:
             st.sidebar.warning("Debug: 'Topic' column missing after DataFrame creation.")
             # Fallback if Topic column wasn't created correctly
             if not df_final.empty:
                 # Try to make Topic the first column if it exists under a different name/case? Unlikely.
                 pass # Or handle error
             else:
                 df_final = pd.DataFrame(columns=['Topic'])


    else:
        st.sidebar.write("Debug: `data_for_df` was empty.")
        df_final = pd.DataFrame(columns=['Topic'])

    # --- Add Final Debugging ---
    # st.sidebar.write("`df_final` (Returned value):")
    # st.sidebar.dataframe(df_final)
    # st.sidebar.write("--- End Debugging Summary Function ---")
    # --- End Final Debugging ---

    return df_final


def search_youtube(api_key, query, max_results=40):
    """
    Performs a YouTube search using the v3 API.
    Handles multiple terms generated by GPT and splits results among them.
    """
    videos_res = []
    response = None  # Initialize response to None

    # Split query into multiple terms if it contains '|'
    if '|' in query:
        query = query.replace('"','').replace("'",'')
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
            # 'order': 'viewCount'
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
                        standard_url = f"https://www.youtube.com/watch?v={video_id}"
                        videos_res.append({
                            'title': title,
                            'videoId': video_id,
                            'url': standard_url
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
def download_with_ytdlp(video_url):
    """
    Uses yt-dlp to download a video to a local temp file, performing
    basic existence and size checks after download.
    (Does NOT perform ffmpeg probe for corruption).

    Args:
        video_url (str): The URL of the YouTube video to download.

    Returns:
        str: The path to the downloaded temporary file if successful and passes checks,
             otherwise None. The caller is responsible for deleting the file later.
    """
    temp_path = None # Initialize path
    st.write(f"ℹ️ Attempting to download video: {video_url}")

    try:
        # Set up temp output path ensuring it gets an mp4 extension if possible
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file_obj:
            temp_path = temp_file_obj.name
        # Note: The file is created empty here. yt-dlp will write to this path.

        ydl_opts = {
            'outtmpl': temp_path, # Tell yt-dlp to use this specific path
            'format': '22/18/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', # Best available mp4
            'quiet': False, # Set to False to see yt-dlp output in logs/console
            'noplaylist': True,
            'merge_output_format': 'mp4', # Ensure merged files are mp4
            'overwrites': True, # Overwrite the empty temp file
            # Add options to potentially help with errors
            'retries': 3, # Retry downloads
            'fragment_retries': 3, # Retry fragments
            'socket_timeout': 30, # Increase timeout
            # 'verbose': True, # Uncomment for extremely detailed logs from yt-dlp
        }

        st.write(f"⏳ Starting yt-dlp download to: {temp_path}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # --- Basic Integrity Checks ---
        st.write("🔬 Performing basic checks on downloaded file...")

        # 1. Check if file exists and has size > 0
        if not os.path.exists(temp_path):
            st.error(f"❌ Download Error: File not found after yt-dlp finished: {temp_path}")
            return None
        if os.path.getsize(temp_path) == 0:
            st.error(f"❌ Download Error: File is empty after download: {temp_path}")
            try: os.remove(temp_path) # Clean up empty file
            except OSError: pass
            return None
        st.write(f"✔️ Check 1 Passed: File exists and is not empty (Size: {os.path.getsize(temp_path) / (1024*1024):.2f} MB).")

        # --- FFmpeg probe check is removed as requested ---

        # --- If basic checks passed ---
        st.success(f"✅ yt-dlp download and basic checks successful for: {temp_path}")
        return temp_path

    except yt_dlp.utils.DownloadError as dl_err:
        st.error(f"❌ yt-dlp DownloadError: {dl_err}")
        # Clean up potentially incomplete/bad file
        if temp_path and os.path.exists(temp_path):
             try: os.remove(temp_path)
             except OSError: pass
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error during download_with_ytdlp: {e}")
        import traceback
        st.error(traceback.format_exc()) # Print full traceback to Streamlit logs
        # Clean up potentially incomplete/bad file
        if temp_path and os.path.exists(temp_path):
             try: os.remove(temp_path)
             except OSError: pass
        return None

def download_direct_url(url, suffix=".mp4"):
    """
    Downloads content from a direct URL to a temporary local file.

    Args:
        url (str): The direct URL to the file (e.g., ending in .mp4, .jpg).
        suffix (str): A suggested file extension for the temporary file.

    Returns:
        str: The path to the downloaded temporary file, or None if download fails.
             The caller is responsible for deleting this file when done.
    """
    local_path = None
    try:
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    }

        # Create a temporary file (it gets a random name)
        # delete=False means the file persists after closing, we delete manually
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            local_path = temp_file.name # Get the path
            print(f"Attempting to download direct URL to temp file: {local_path}")

            # Make the request, stream=True handles large files efficiently
            with requests.get(url, stream=True, timeout=60, headers= headers) as response:
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                # Write the content to the temporary file in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

        print(f"✔️ Download successful: {local_path}")
        return local_path

    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed (Network/HTTP Error): {e}")
        # Clean up the potentially incomplete temp file if it exists
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        return None
    except Exception as e:
        print(f"❌ Download failed (Other Error): {e}")
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        return None




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
            local_vid_path = download_with_ytdlp(base_video_url)
            st.text(local_vid_path)
            st.video(local_vid_path)
            base_video = VideoFileClip(local_vid_path, audio=False, target_resolution=(720, 1280)) # Target 720p vertical
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

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=f"final_{safe_topic}_{lang}_") as temp_output_file_obj:
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
        {'Topic': 'sofa sale', 'Search Term': 'sofa unboxing #shorts' ,'Language' : 'English', 'Video Results': 5}
   
        
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
def sync_search_data():
    raw_data = st.session_state.search_data

    # Keep only dicts
    clean_data = [row for row in raw_data if isinstance(row, dict)]
    st.text(clean_data)
    # Remove fully empty rows (all values are empty/None)
    # clean_data = [row for row in clean_data if any(v not in [None, '', []] for v in row.values())]

    # If table is completely cleared, add a default row
    # if not clean_data:
    #     clean_data = [{
    #         "Topic": "",
    #         "Search Term": "",
    #         "Language": "English",
    #         "Video Results": 5
    #     }]

    # Enforce column order and types
    df = pd.DataFrame(clean_data)
    expected_cols = ["Topic", "Search Term", "Language", "Video Results"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    st.session_state.search_data = df[expected_cols]
    st.text(st.session_state.search_data)




# Use data_editor with on_change callback
st.sidebar.data_editor(
    st.session_state.search_data,
    num_rows="dynamic",
    use_container_width=True,
    key="search_topic_editor",
    # on_change=sync_search_data
)

# Update session state with edited data
# st.session_state.search_data = edited_df

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
    # --- Get data from the editor's state (changes) AND the original data ---
    editor_changes = st.session_state.search_topic_editor # Dict with added/edited/deleted
    # Get the DataFrame used to initialize the editor in the *previous* run
    # Ensure it's a DataFrame, provide default if it's the very first run
    if 'search_data' not in st.session_state or not isinstance(st.session_state.search_data, pd.DataFrame):
         st.session_state.search_data = pd.DataFrame(columns=["Topic", "Search Term", "Language", "Video Results"]) # Ensure columns exist
    original_df = st.session_state.search_data

    # --- Add Debugging Output ---
    st.sidebar.write("--- Debugging Input ---")
    st.sidebar.write("Original DataFrame used for Editor (`st.session_state.search_data`):")
    st.sidebar.dataframe(original_df)
    st.sidebar.write("Editor Changes Dict (`st.session_state.search_topic_editor`):")
    st.sidebar.json(editor_changes if editor_changes is not None else "None")
    st.sidebar.write("------------------------")
    # --- End Debugging Output ---

    valid_input = True
    current_rows_list = [] # Initialize list to hold the reconstructed rows

    # --- Reconstruct the current state of the table ---
    try:
        # Start with the original data as a list of dicts
        # Handle potential empty initial DataFrame
        if not original_df.empty:
            current_rows_list = original_df.to_dict('records')
        else:
            current_rows_list = []

        # --- Apply Changes ---
        # It's often easier to work with a DataFrame for applying indices
        temp_df = original_df.copy()

        # Apply deletions (indices are relative to the DataFrame *before* additions/edits of this cycle)
        indices_to_delete = editor_changes.get("deleted_rows", [])
        if indices_to_delete:
             # Ensure indices are valid before dropping
             valid_indices_to_delete = [idx for idx in indices_to_delete if idx < len(temp_df)]
             if len(valid_indices_to_delete) != len(indices_to_delete):
                  st.sidebar.warning("Some deleted row indices were invalid.")
             if valid_indices_to_delete:
                  temp_df = temp_df.drop(index=valid_indices_to_delete).reset_index(drop=True)


        # Apply edits (key is index string, value is dict of changed columns)
        edits = editor_changes.get("edited_rows", {})
        if edits:
             for index_str, changes in edits.items():
                 try:
                      index = int(index_str)
                      if 0 <= index < len(temp_df):
                           for col, value in changes.items():
                               if col in temp_df.columns:
                                    temp_df.loc[index, col] = value
                               else:
                                    st.sidebar.warning(f"Attempted to edit non-existent column '{col}' at index {index}.")
                      else:
                           st.sidebar.warning(f"Attempted to edit non-existent index: {index}.")
                 except ValueError:
                      st.sidebar.warning(f"Invalid index format in edited_rows: {index_str}")


        # Apply additions (list of new row dicts)
        additions = editor_changes.get("added_rows", [])
        if additions:
             # Ensure added rows have expected columns, fill missing if necessary
             processed_additions = []
             expected_cols_set = set(["Topic", "Search Term", "Language", "Video Results"])
             for new_row in additions:
                 # Ensure it's a dict
                 if isinstance(new_row, dict):
                      # Add missing keys with default values
                      for col in expected_cols_set:
                           if col not in new_row:
                               if col == "Language": new_row[col] = "English"
                               elif col == "Video Results": new_row[col] = 5
                               else: new_row[col] = ""
                      processed_additions.append(new_row)
                 else:
                     st.sidebar.warning(f"Skipping invalid item in added_rows: {new_row}")

             if processed_additions:
                  add_df = pd.DataFrame(processed_additions)
                  temp_df = pd.concat([temp_df, add_df], ignore_index=True)

        # The final DataFrame reflecting the current state
        final_current_df = temp_df

    except Exception as e:
        st.sidebar.error(f"Error reconstructing table state from changes: {e}", icon="🆘")
        import traceback
        st.sidebar.text(traceback.format_exc()) # More detailed error
        valid_input = False


    # --- Proceed with validation ONLY if reconstruction succeeded ---
    if valid_input:
        st.sidebar.write("--- Debugging Reconstructed Data ---")
        st.sidebar.write("Reconstructed DataFrame (`final_current_df`):")
        st.sidebar.dataframe(final_current_df)
        st.sidebar.write("-----------------------------------")

        # Check if the *reconstructed* DataFrame is empty
        if final_current_df.empty:
            st.sidebar.warning("The table appears empty after applying edits/additions/deletions.", icon="⚠️")
            valid_input = False
        else:
            # --- Assign the reconstructed DF to search_df for further validation ---
            search_df = final_current_df

            # --- Perform subsequent validation steps (Column handling, Type Conversion, Empty Check, Final Validation) ---
            # (Use the same robust validation block from the previous answer here, operating on 'search_df')
            # Example snippet:
            expected_cols = ["Topic", "Search Term", "Language", "Video Results"]
            for col in expected_cols:
                 if col not in search_df.columns:
                     if col == "Language": search_df[col] = "English"
                     elif col == "Video Results": search_df[col] = 5
                     else: search_df[col] = ""
            search_df = search_df[expected_cols]
            try:
                search_df['Video Results'] = pd.to_numeric(search_df['Video Results'], errors='coerce').fillna(5).astype(int)
                search_df['Video Results'] = search_df['Video Results'].apply(lambda x: max(1, x))
            except Exception as e:
                search_df['Video Results'] = 5
            search_df['Topic'] = search_df['Topic'].fillna('').astype(str)
            search_df['Search Term'] = search_df['Search Term'].fillna('').astype(str)
            is_row_meaningful = (search_df['Topic'].str.strip() != '') | (search_df['Search Term'].str.strip() != '')
            search_df_filtered = search_df[is_row_meaningful].copy()
            if search_df_filtered.empty:
                 st.sidebar.warning("Please ensure at least one row has a non-empty value for 'Topic' or 'Search Term'.", icon="⚠️")
                 valid_input = False
            else:
                 search_df = search_df_filtered # Use filtered DF
                 if search_df['Search Term'].str.strip().eq('').any():
                     st.sidebar.warning("One or more valid rows have an empty Search Term.", icon="⚠️")
                     valid_input = False
                 if search_df['Topic'].str.strip().eq('').any():
                     st.sidebar.warning("One or more valid rows have an empty Topic.", icon="⚠️")
                     valid_input = False
            # --- End example snippet ---

    # --- Proceed if input is valid ---
    if valid_input:
        st.sidebar.success("Input valid, proceeding with search.") # Feedback
        st.session_state.search_triggered = True
        st.session_state.api_search_results = {}
        # ... reset generation state ...
        st.session_state.current_search_df = search_df # Use the final, validated DF

        # **IMPORTANT**: Update the base data for the *next* initialization of the editor
        st.session_state.search_data = search_df.copy() # Use .copy() for safety

        st.rerun()
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
        st.text(f"topic {topic}")
        if term == 'auto':
#             term = chatGPT(f"""I want concise, emotional, and visually-rich YouTube search keywords for a specific topic. These should feel like real titles users would upload — casual, vlog-style, and rooted in personal experiences or moments.

# Avoid anything generic, commercial-sounding, or search-optimized like “best X” or “how to X.”
# No listicles, guides, or reviews.
# Showing the topic in positive light

# Instead, think in terms of reactions,vlog, life moments, surprises, reveals, or storytelling.
# Imagine something someone would upload from their phone right after something big happened.

# Keep each keyword to 2–4 words. 
# Each result should include “#shorts” at the end. In english.
# return as 1 row | delimted
# the main subject of the input must be in the output!!! like the main product

# example:
# Input: Parocīgu automašīnu piedāvājumi bez finansējuma – lūk, kā to izdarīt!
# output: new car reveal #shorts 

# Here’s the topic: 
#             {topic}""",client=openai_client)

                term = chatGPT(f"""You are a viral video ad expert. I will give you a topic, and you will return the top 3 YouTube Shorts search terms that:
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
                                 {topic}""",client=openai_client,model="gpt-4")

        if term not in results_cache: # Avoid re-searching same term in one go
            videos = search_youtube(youtube_api_key_secret, term, count)

            if videos is None: # Critical API error (e.g., 403)
                st.error(f"Stopping search due to critical API issue (check key/quota) for term: '{term}'.", icon="🚫")
                api_error_occurred = True
                break # Stop processing further terms

            # Store results along with the topic

            # if "," in lang:
            #     langs = lang.split(",")
            #     for idx, lang in enumerate(langs):
        
            #         results_cache[f"{term}_{lang}_{idx}"] = {'videos': videos, 'topic': topic , 'lang' : lang}
            # else:
            results_cache[term] = {'videos': videos, 'topic': topic , 'lang' : lang}

            time.sleep(0.1) # Small delay between API calls
        progress_bar.progress((i + 1) / len(search_items))

    status_text_api.text("API Search complete.")
    st.session_state.api_search_results = results_cache # Update main cache
    st.session_state.search_triggered = False  # ✅ Prevent infinite rerun loop

    if api_error_occurred:
        st.session_state.search_triggered = False # Reset trigger on critical error
        st.warning("Search halted due to API error.", icon="⚠️")
        # Don't stop the app, just show the error. Rerun will display current state.
        st.rerun()
    # else:
    #     # Don't necessarily reset trigger, allow results to display
    #     # st.session_state.search_triggered = False # Reset trigger after successful search display
    #     st.rerun() # Rerun to display results


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
    st.subheader("Search Results & Video Selection (Grid View)") # Updated subheader
    # Display results from cache

    # st.text(st.session_state.api_search_results.items())
    for term, result_data in st.session_state.api_search_results.items():
        videos = result_data['videos']
        topic = result_data['topic']
        lang = result_data['lang']

        # --- Container for the results of THIS search term ---
        term_container = st.container(border=True)
        with term_container:
            st.subheader(f"Results for: \"{term}\" (Topic: \"{topic}\")")
            if not videos:
                st.write("No videos found via API.")
                continue

            num_videos = len(videos)
            num_cols = 3 # <<< Number of columns/videos per row

            for i in range(0, num_videos, num_cols):
                # Create a new row of columns for each chunk of videos
                cols = st.columns(num_cols)

                # Populate the columns in this row
                for j in range(num_cols):
                    video_index = i + j
                    if video_index < num_videos:
                        video = videos[video_index]
                        # --- Place ALL rendering logic for one video inside its column ---
                        with cols[j]:
                            # --- Extract Video Info ---
                            video_id = video['videoId']
                            video_title = video['title']
                            standard_video_url = video['url']
                            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/sddefault.jpg"
                            # Use video_id for unique keys within the grid cell
                            search_term_hash = simple_hash(term.strip().lower())
                            unique_key_base = f"{video_id}_{search_term_hash}_{i+j}"

                            # --- State for controlling video player visibility ---
                            show_video_key = f"show_player_{unique_key_base}"
                            if show_video_key not in st.session_state:
                                st.session_state[show_video_key] = False

                            # --- Other states (selection, generation, etc.) ---
                            is_selected = video_id in st.session_state.selected_videos
                            video_state = st.session_state.selected_videos.get(video_id, {})
                            # ... [keep existing state checks: is_fetching_dlp, dlp_error, is_in_queue, etc.] ...

                            # --- Render Content Vertically within the Column ---
                            st.write(f"**{video_title[:50]}...**") # Truncate title if needed
                            st.caption(f"ID: {video_id}") # Keep caption concise
                            # st.link_button("YouTube", standard_video_url) # Optional alternative link

                            # --- Conditionally Display Video Player OR Thumbnail ---
                            if st.session_state[show_video_key]:
                                try:
                                    # Video player will take the width of the column
                                    # st.video(standard_video_url)
                                    iframe_code = f"""
                                <iframe width="315" height="560"
                                src="https://www.youtube.com/embed/{video_id}"
                                title="YouTube video player" frameborder="0"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media;
                                gyroscope; picture-in-picture; web-share"
                                allowfullscreen></iframe>
                                """
                                    st.markdown(iframe_code, unsafe_allow_html=True)

                              
                                except:
                                    st.error("Video failed to load.") # Placeholder error
                            else:
                                st.image(thumbnail_url, use_container_width=True) # Make image fill column width

                            # --- Buttons (Stacked Vertically Below Player/Thumb) ---
                            # Button to Toggle Video Player Visibility
                            toggle_label = "🔼 Hide" if st.session_state[show_video_key] else "▶️ Show"
                            if st.button(
                                f"{toggle_label} Preview",
                                key=f"toggle_vid_btn_{unique_key_base}",
                                help="Show/hide the video preview",
                                use_container_width=True
                            ) :
                                st.session_state[show_video_key] = not st.session_state[show_video_key]
                                st.rerun()

                            # Select / Deselect Button
                           # States for yt-dlp and generation
                            is_fetching_dlp = video_state.get('fetching_dlp', False)
                            dlp_error = video_state.get('yt_dlp_error')
                            has_dlp_info = bool(video_state.get('Direct URL'))
                            is_in_queue = video_id in st.session_state.generation_queue
                            is_completed = bool(video_state.get('Generated S3 URL'))
                            is_failed = bool(video_state.get('Generation Error'))
                            is_currently_processing = video_state.get('Status') == 'Processing'
                            s3_url = video_state.get('Generated S3 URL')
                            generation_error = video_state.get('Generation Error', '')
                            
                            # Determine button label and type
                            if is_selected:
                                select_button_label = "✅ Deselect"
                                select_button_type = "secondary"
                                select_disabled = False
                            elif is_fetching_dlp:
                                select_button_label = "⏳ Fetching..."
                                select_button_type = "secondary"
                                select_disabled = True
                            else:
                                select_button_label = "➕ Select"
                                select_button_type = "primary"
                                select_disabled = st.session_state.batch_processing_active
                            
                            # Select/Deselect Button
                            if st.button(
                                select_button_label,
                                key=f"select_{unique_key_base}",
                                type=select_button_type,
                                use_container_width=True,
                                disabled=select_disabled
                            ):
                                if is_selected:
                                    del st.session_state.selected_videos[video_id]
                                    st.toast(f"Deselected: {video_title}", icon="➖")
                                    if video_id in st.session_state.generation_queue:
                                        st.session_state.generation_queue.remove(video_id)
                                        st.session_state.batch_total_count = len(st.session_state.generation_queue)
                                else:
                                    if "," in lang:
                                        langs = lang.split(",")
                                        for idx, lang in enumerate(langs):
                                            st.session_state.selected_videos[f'{video_id}_{lang}'] = {
                                                'Search Term': term,
                                                'Topic': topic,
                                                'Language': lang,
                                                'Video Title': video_title,
                                                'Video ID': video_id,
                                                'Standard URL': standard_video_url,
                                                'fetching_dlp': True,
                                                'Direct URL': None,
                                                'Format Details': None,
                                                'yt_dlp_error': None,
                                                'Generated S3 URL': None,
                                                'Generation Error': None,
                                                'Status': 'Selected, Fetching URL...'
                                            }
                                            st.toast(f"Selected: {video_title}. Fetching direct URL...", icon="⏳")
                                    else:
                                        st.session_state.selected_videos[video_id] = {
                                        'Search Term': term,
                                        'Topic': topic,
                                        'Language': lang,
                                        'Video Title': video_title,
                                        'Video ID': video_id,
                                        'Standard URL': standard_video_url,
                                        'fetching_dlp': True,
                                        'Direct URL': None,
                                        'Format Details': None,
                                        'yt_dlp_error': None,
                                        'Generated S3 URL': None,
                                        'Generation Error': None,
                                        'Status': 'Selected, Fetching URL...'
                                    }


                                    st.toast(f"Selected: {video_title}. Fetching direct URL...", icon="⏳")
                                st.rerun()
                            
                            # Status Display Block
                            status_container = st.container()
                            if is_selected:
                                if is_fetching_dlp:
                                    status_container.info("⏳ Fetching URL...", icon="📡")
                                elif dlp_error:
                                    status_container.error(f"URL Error: {dlp_error}", icon="⚠️")
                                elif not has_dlp_info:
                                    status_container.warning("URL fetch incomplete.", icon="❓")
                                elif has_dlp_info:
                                    if is_currently_processing:
                                        status_container.info("⚙️ Processing...", icon="⏳")
                                    elif is_in_queue:
                                        status_container.info("🕒 Queued", icon="🕒")
                                    elif is_completed:
                                        status_container.success("✔️ Generated!", icon="🎉")
                                        st.link_button("View on S3", url=s3_url, use_container_width=True)
                                    elif is_failed:
                                        status_container.error(f"❌ Failed: {generation_error[:50]}...", icon="🔥")
                                    else:
                                        status_container.success("✅ Ready to Process", icon="👍")

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
              if standard_url in st.session_state['resolved_vid_urls']:
                dlp_info = st.session_state['resolved_vid_urls'][standard_url]
              else:
                # dlp_info = get_yt_dlp_info(standard_url)
                dlp_info ={'direct_url': "dummy",
                    'format_details': "dummy",
                    'error': None}
                
                st.session_state['resolved_vid_urls'][standard_url] = dlp_info

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
                        script_prompt = f"""Create a short, engaging voiceover script for FB viral   video (roughly 15-20 seconds long, maybe 2-3 sentences) about '{topic}' in language {lang}. The tone should be informative yet conversational, '.  smooth flow. Just provide the script text, nothing else. create intriguing and engaging script, sell the topic to the audience . be very causal and not 'advertisement' style vibe. end with a call to action 'tap to....'  .the text needs to be retentive.Don't say 'we' or 'our' .NOTE:: DO NOT dont use senetional words and phrasing and DONT make false promises , use Urgency Language, Avoid geographically suggestive terms (e.g., "Near you," "In your area"). Do not use "we" or "our". in end if video use something "Tap now to.." with a clear, non-committal phrase !!!  """
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
                            base_video_url=video_data["Standard URL"],
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
