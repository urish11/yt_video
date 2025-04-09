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
import tempfile
from io import BytesIO
import moviepy
import numpy as np
from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips, TextClip
)
import moviepy.video.fx.all as vfx
# import moviepy.video.fx.resize as resize # Using patched version below
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont

# --- API Clients & Services ---
from openai import OpenAI
import boto3
from botocore.exceptions import NoCredentialsError

# --- Configuration ---
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3/search"
MAX_RESULTS_PER_QUERY = 3 # Reduce results per query slightly to manage load
YT_DLP_FETCH_TIMEOUT = 30 # Increased timeout for potentially slower connections
DEFAULT_TTS_VOICE = "sage" # Default voice for TTS
SUBTITLE_FONT_PATH = "./Montserrat-Bold.ttf" # Ensure this font file exists
SUBTITLE_FONT_SIZE = 50 # Adjust as needed
SUBTITLE_WORDS_PER_GROUP = 2 # Group words for subtitles
SUBTITLE_COLOR = '#FFFF00' # Yellow
SUBTITLE_BG_COLOR = 'rgba(0, 0, 0, 0.6)' # Semi-transparent black

# --- Load Secrets ---
try:
    youtube_api_key_secret = st.secrets["YOUTUBE_API_KEY"] # Assuming key name in secrets
    openai_api_key = st.secrets["openai_api_key"]
    aws_access_key = st.secrets["aws_access_key"]
    aws_secret_key = st.secrets["aws_secret_key"]
    s3_bucket_name = st.secrets["s3_bucket_name"]
    s3_region = st.secrets["s3_region"]

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

except KeyError as e:
    st.error(f"Missing secret key: {e}. Please configure it in Streamlit secrets.", icon="🚨")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during initialization: {e}", icon="🚨")
    st.stop()

# --- Patched Resizer (from second script - if needed, MoviePy versions vary) ---
# This attempts to fix potential issues with MoviePy's default resizer
# If you encounter resize errors, uncomment and test this. Otherwise, keep it commented.
# try:
#     from moviepy.video.fx import resize as moviepy_resize
#     def patched_resizer(pilim, newsize):
#         if isinstance(newsize, (list, tuple)):
#             newsize = tuple(int(dim) for dim in newsize)
#         elif isinstance(newsize, (int, float)):
#             if hasattr(pilim, "shape"):
#                 orig_height, orig_width = pilim.shape[:2]
#             else:
#                 orig_width, orig_height = pilim.size
#             newsize = (int(orig_width * newsize), int(orig_height * newsize))

#         if not isinstance(pilim, Image.Image):
#             pilim = Image.fromarray(pilim)

#         resized = pilim.resize(newsize, Image.Resampling.LANCZOS) # Updated resampling filter
#         return np.array(resized)
#     moviepy_resize.resizer = patched_resizer
#     print("Applied patched resizer.")
# except Exception as e:
#     print(f"Could not apply patched resizer: {e}")
#     pass # Continue without patch

# --- Helper Function: YouTube API Search ---
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
        return None # Indicate critical failure

# --- Helper Function: Generate Script with ChatGPT ---
def chatGPT(prompt, client, model="gpt-4o", temperature=0.7):
    """Generates text using OpenAI Chat Completion."""
    try:
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

# --- Helper Function: Generate TTS Audio & Timestamps ---
def generate_audio_with_timestamps(text, client, voice_id="sage"):
    """Generates TTS audio using OpenAI, saves it, and gets word timestamps."""
    temp_audio_file = None
    try:
        # Ensure text is not empty
        if not text or not text.strip():
             raise ValueError("Input text for TTS cannot be empty.")

        response = client.audio.speech.create(
            model="tts-1-hd", # Use HD for better quality
            voice=voice_id,
            input=text,
            response_format="mp3", # Use mp3 or opus
            speed=1.0 # Adjust speed if needed (1.0 is normal)
        )

        # Save the generated audio to a temporary file
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio_file.name
        temp_audio_file.write(response.content)
        temp_audio_file.close() # Close the file handle

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
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )

        transcribe_data = json.loads(transcribe_response.to_json())

        word_timings = []
        if 'words' in transcribe_data:
            for word_info in transcribe_data['words']:
                word_timings.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"]
                })
        else:
             st.warning("Whisper did not return word timestamps.", icon="⏱️")
             # Fallback: maybe generate sentence-level timestamps if needed later

        return temp_audio_path, word_timings

    except Exception as e:
        st.error(f"Error in TTS/Timestamp generation: {e}", icon="🔊")
        # Cleanup temp file if it exists and an error occurred
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            os.remove(temp_audio_file.name)
        return None, None

# --- Helper Function: Group Word Timings ---
def group_words_with_timing(word_timings, words_per_group=2):
    """Groups words and their timings for subtitles."""
    grouped_timings = []
    if not word_timings: return grouped_timings

    for i in range(0, len(word_timings), words_per_group):
        group_words = word_timings[i:i+words_per_group]
        if group_words:
            combined_word = " ".join(word['word'] for word in group_words)
            start_time = group_words[0]['start']
            end_time = group_words[-1]['end']
            grouped_timings.append({
                "text": combined_word,
                "start": start_time,
                "end": end_time
            })
    return grouped_timings

# --- Helper Function: Create Text Image for Subtitles ---
def create_text_image(text, fontsize, color, bg_color, font_path):
    """Creates a transparent PNG image with text and rounded background."""
    try:
        # Check if font file exists
        if not os.path.exists(font_path):
             st.warning(f"Subtitle font not found at {font_path}. Using default.", icon="⚠️")
             # Fallback to a basic PIL font if custom font is missing
             try:
                 font = ImageFont.load_default(size=fontsize) # Request size
             except AttributeError: # Older PIL/Pillow might not support size
                 font = ImageFont.load_default()
        else:
             font = ImageFont.truetype(font_path, fontsize)

        # Simple approximation for bounding box if getbbox is not available or buggy
        # Use textlength which gives width in pixels. Height is approx fontsize.
        try:
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            ascent, descent = font.getmetrics()
            text_actual_height = ascent + descent # More accurate height
            bbox_y_offset = bbox[1]
        except AttributeError:
             # Fallback for older PIL/Pillow or if getbbox fails
             text_width = int(font.getlength(text))
             text_height = fontsize # Approximate height
             text_actual_height = fontsize
             bbox_y_offset = -int(fontsize * 0.2) # Rough estimate for baseline offset

        padding_x = 25 # Horizontal padding
        padding_y = 10 # Vertical padding

        img_width = text_width + 2 * padding_x
        img_height = text_actual_height + 2 * padding_y
        radius = 15 # Corner radius

        # Create image with transparent background
        img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw rounded rectangle background using the provided RGBA color string
        try:
            # PIL expects a tuple for color, convert rgba string
            # Example: 'rgba(0, 0, 0, 0.6)' -> (0, 0, 0, int(0.6*255))
            if isinstance(bg_color, str) and bg_color.startswith('rgba'):
                parts = bg_color.strip('rgba()').split(',')
                r, g, b = map(int, parts[:3])
                a = int(float(parts[3]) * 255)
                fill_color_tuple = (r, g, b, a)
            elif isinstance(bg_color, str): # Assume hex or name, let PIL handle
                 fill_color_tuple = bg_color
            else: # Assume it's already a tuple
                 fill_color_tuple = bg_color

            draw.rounded_rectangle([(0, 0), (img_width, img_height)], radius=radius, fill=fill_color_tuple)
        except Exception as draw_err:
             st.warning(f"Could not draw rounded rect: {draw_err}. Using simple rect.", icon="🎨")
             draw.rectangle([(0,0), (img_width, img_height)], fill=fill_color_tuple)


        # Calculate text position (centered)
        text_x = padding_x
        # Adjust y position based on PIL version / bbox availability
        text_y = padding_y - bbox_y_offset # Try to align based on bbox baseline

        draw.text((text_x, text_y), text, font=font, fill=color)

        return np.array(img)

    except Exception as e:
        st.error(f"Error creating text image for '{text}': {e}", icon="🎨")
        # Return a small transparent pixel as fallback
        return np.zeros((10, 10, 4), dtype=np.uint8)


# --- Helper Function: Process Video with TTS and Subtitles ---
def process_video_with_tts(base_video_url, audio_path, word_timings, topic):
    """Loads video, adds TTS audio, loops if necessary, adds subtitles."""
    final_video_clip = None
    temp_output_path = None

    try:
        # Load base video
        st.write(f"⏳ Loading base video from URL...")
        base_video = VideoFileClip(base_video_url, audio=False) # Load without original audio
        video_duration = base_video.duration
        w, h = base_video.size
        st.write(f"✔️ Base video loaded: {w}x{h}, Duration: {video_duration:.2f}s")

        # Load TTS audio
        st.write(f"⏳ Loading TTS audio...")
        tts_audio = AudioFileClip(audio_path)
        audio_duration = tts_audio.duration
        st.write(f"✔️ TTS audio loaded: Duration: {audio_duration:.2f}s")

        # --- Video Looping/Trimming ---
        processed_video = base_video
        if video_duration < audio_duration:
            st.write(f"⏳ Looping video to match audio duration...")
            num_loops = int(np.ceil(audio_duration / video_duration))
            # Use concatenate_videoclips for looping
            clips_to_loop = [base_video] * num_loops
            looped_video = concatenate_videoclips(clips_to_loop)
            # Trim the looped video to the exact audio duration
            processed_video = looped_video.subclip(0, audio_duration)
            st.write(f"✔️ Video looped {num_loops} times and trimmed to {audio_duration:.2f}s")
        elif video_duration > audio_duration:
            st.write(f"⏳ Trimming video to match audio duration...")
            processed_video = base_video.subclip(0, audio_duration)
            st.write(f"✔️ Video trimmed to {audio_duration:.2f}s")
        else:
            st.write("✔️ Video duration matches audio duration.")

        # Set the TTS audio to the processed video
        final_video_clip = processed_video.set_audio(tts_audio)

        # --- Subtitle Generation ---
        st.write(f"⏳ Generating subtitles...")
        subtitle_clips = []
        grouped_subs = group_words_with_timing(word_timings, words_per_group=SUBTITLE_WORDS_PER_GROUP)

        if grouped_subs:
            total_subs = len(grouped_subs)
            sub_progress = st.progress(0)
            for i, sub_data in enumerate(grouped_subs):
                text = sub_data['text']
                start = sub_data['start']
                end = sub_data['end']
                sub_duration = end - start

                # Skip empty text or excessively short durations
                if not text.strip() or sub_duration <= 0.05:
                    continue

                # Create subtitle image
                text_img_array = create_text_image(
                    text.upper(), # Make subtitles uppercase
                    fontsize=SUBTITLE_FONT_SIZE,
                    color=SUBTITLE_COLOR,
                    bg_color=SUBTITLE_BG_COLOR,
                    font_path=SUBTITLE_FONT_PATH
                )

                # Create ImageClip
                subtitle_img_clip = ImageClip(text_img_array)\
                    .set_start(start)\
                    .set_duration(sub_duration)\
                    .set_position(('center', 'bottom'), relative=True) # Position relative to video
                    #.margin(bottom=int(h * 0.08), opacity=0) # Add margin from bottom (8%)

                subtitle_clips.append(subtitle_img_clip)
                sub_progress.progress((i + 1) / total_subs)
            st.write(f"✔️ Generated {len(subtitle_clips)} subtitle clips.")
        else:
            st.warning("No word timings available to generate subtitles.", icon="⏱️")

        # Composite final video with subtitles
        if subtitle_clips:
            st.write("⏳ Compositing video and subtitles...")
            final_video_clip = CompositeVideoClip([final_video_clip] + subtitle_clips, size=final_video_clip.size)
            st.write("✔️ Compositing complete.")
        else:
            st.write("ℹ️ No subtitles added.")


        # --- Export Final Video ---
        st.write("⏳ Exporting final video...")
        # Create a unique temporary filename
        timestamp = int(datetime.datetime.now().timestamp())
        safe_topic = urllib.parse.quote(topic.replace(' ', '_')[:30], safe='')
        temp_output_filename = f"final_{safe_topic}_{timestamp}.mp4"
        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=f"final_{safe_topic}_")
        temp_output_path = temp_output_file.name
        temp_output_file.close() # Close handle before moviepy writes to it

        # Write video file (use standard codecs)
        # Consider presets: 'ultrafast', 'fast', 'medium', 'slow'
        # threads=4 might speed up encoding on multi-core CPUs
        final_video_clip.write_videofile(
            temp_output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=f'temp-audio-{timestamp}.m4a', # Explicit temp audio file
            remove_temp=True, # Remove temp audio file after processing
            fps=base_video.fps or 24, # Use original fps or default to 24
            preset='medium', # Balance speed and quality
            threads=4, # Use multiple threads if available
            logger='bar' # Show progress bar in console/logs
        )
        st.write(f"✔️ Final video exported to temporary path: {temp_output_path}")

        # Clean up loaded clips to free memory
        base_video.close()
        tts_audio.close()
        if 'looped_video' in locals(): looped_video.close()
        processed_video.close()
        final_video_clip.close() # Close the final composite clip
        for sub_clip in subtitle_clips: sub_clip.close()


        return temp_output_path, temp_output_filename # Return path and suggested S3 filename

    except Exception as e:
        st.error(f"Error during video processing: {e}", icon="🎬")
        # Clean up clips if they exist
        if 'base_video' in locals() and base_video: base_video.close()
        if 'tts_audio' in locals() and tts_audio: tts_audio.close()
        if 'looped_video' in locals() and 'looped_video' in locals(): looped_video.close()
        if 'processed_video' in locals() and processed_video: processed_video.close()
        if 'final_video_clip' in locals() and final_video_clip: final_video_clip.close()
        if 'subtitle_clips' in locals():
            for sub_clip in subtitle_clips: sub_clip.close()
        # Attempt to remove temp output file if it exists on error
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except Exception as rm_err:
                st.warning(f"Could not remove temp video file on error: {rm_err}")
        return None, None

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
        # Construct the URL (common format)
        video_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
        st.success(f"✔️ Video uploaded to S3: {object_name}")
        return video_url
    except NoCredentialsError:
        st.error("AWS Credentials not available for S3 upload.", icon="🔒")
        return None
    except Exception as e:
        st.error(f"S3 Upload Error: {e}", icon="☁️")
        return None


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="YouTube Select & Generate")
st.title("🎬 YouTube Video Selector & TTS Generator")
st.caption("Search YouTube, select a video, generate TTS script, create subtitled video, and upload to S3.")

# --- Session State Initialization ---
if 'selected_videos' not in st.session_state:
    st.session_state.selected_videos = {} # {videoId: video_data}
# if 'last_api_key' not in st.session_state: # API Key from secrets now
#     st.session_state.last_api_key = ""
if 'search_data' not in st.session_state:
     # Use a DataFrame for structured input
     st.session_state.search_data = pd.DataFrame([
         {'Search Term': 'nature relaxation', 'Topic': 'the calming effect of nature sounds'},
         {'Search Term': 'study music', 'Topic': 'how background music improves focus'}
     ])
if 'search_triggered' not in st.session_state:
     st.session_state.search_triggered = False
if 'api_search_results' not in st.session_state:
     st.session_state.api_search_results = {} # Cache API results {search_term: [video_list]}
if 'video_to_generate' not in st.session_state:
     st.session_state.video_to_generate = None # ID of the video being processed

# --- Input Area (Sidebar) ---
st.sidebar.header("Inputs")
# API key is now handled via secrets, no input needed.
# st.sidebar.text_input("Enter YouTube Data API v3 Key:", type="password", value=st.session_state.last_api_key)

st.sidebar.write("Enter Search Terms and Topics:")
edited_df = st.sidebar.data_editor(
     st.session_state.search_data,
     num_rows="dynamic",
     column_config={
         "Search Term": st.column_config.TextColumn("YouTube Search Term", required=True),
         "Topic": st.column_config.TextColumn("Topic for TTS Script", required=True)
     },
     use_container_width=True,
     key="search_topic_editor"
)
# Update session state with edited data
st.session_state.search_data = edited_df

st.sidebar.info("`yt-dlp` fetches details after you **Select** a video. "
                "Video generation happens after you click **Generate**.", icon="ℹ️")
st.sidebar.warning("Direct URLs from `yt-dlp` can be **temporary**.", icon="⏳")
st.sidebar.warning("Video generation can take **several minutes**.", icon="⏱️")

# --- Action Buttons (Sidebar) ---
col1, col2 = st.sidebar.columns(2)
search_button = col1.button("🔍 Search Videos", use_container_width=True)
clear_button = col2.button("🧹 Clear All", use_container_width=True, type="secondary")

if clear_button:
    st.session_state.selected_videos = {}
    st.session_state.search_triggered = False
    st.session_state.api_search_results = {}
    st.session_state.video_to_generate = None
    # Optionally reset the input table
    # st.session_state.search_data = pd.DataFrame([{'Search Term': '', 'Topic': ''}])
    st.success("Selections, results, and generation state cleared!", icon="✅")
    st.rerun()

# --- Processing Logic ---

# 1. Handle Search Button Click
if search_button:
     # Validate input data
     valid_input = True
     if edited_df.empty:
          st.sidebar.warning("Please add at least one Search Term and Topic.", icon="⚠️")
          valid_input = False
     if edited_df['Search Term'].isnull().any() or (edited_df['Search Term'] == '').any():
          st.sidebar.warning("Search Term cannot be empty.", icon="⚠️")
          valid_input = False
     if edited_df['Topic'].isnull().any() or (edited_df['Topic'] == '').any():
          st.sidebar.warning("Topic cannot be empty.", icon="⚠️")
          valid_input = False

     if valid_input:
          st.session_state.search_triggered = True
          st.session_state.api_search_results = {} # Clear previous API results on new search
          st.session_state.video_to_generate = None # Reset generation state
          # Store the dataframe used for this search
          st.session_state.current_search_df = edited_df.copy()
          st.rerun() # Rerun to start the search process below
     else:
         st.session_state.search_triggered = False # Ensure search doesn't proceed

# 2. Perform API Search if Triggered
if st.session_state.search_triggered and 'current_search_df' in st.session_state:
    search_df = st.session_state.current_search_df
    search_items = search_df.to_dict('records') # Convert df rows to list of dicts

    st.info(f"Searching API for {len(search_items)} term(s)... (Max {MAX_RESULTS_PER_QUERY} results per term)", icon="🔍")
    st.divider()

    api_error_occurred = False
    with st.spinner("Performing API searches..."):
        for item in search_items:
            term = item['Search Term']
            topic = item['Topic'] # Keep topic associated
            if term not in st.session_state.api_search_results:
                videos = search_youtube(youtube_api_key_secret, term, MAX_RESULTS_PER_QUERY)

                if videos is None: # Critical API error (e.g., 403)
                    st.error(f"Stopping search due to critical API issue (check key/quota) for term: '{term}'.", icon="🚫")
                    api_error_occurred = True
                    break # Stop processing further terms

                # Store results along with the topic
                st.session_state.api_search_results[term] = {'videos': videos, 'topic': topic}
                time.sleep(0.2) # Small delay between API calls

    if api_error_occurred:
        st.session_state.search_triggered = False # Reset trigger on critical error
        st.warning("Search halted due to API error.", icon="⚠️")
        st.stop()

# 3. Display Search Results (if search was triggered and completed)
if st.session_state.search_triggered and st.session_state.api_search_results:
     st.subheader("Search Results & Video Selection")
     # Display results from cache
     for term, result_data in st.session_state.api_search_results.items():
         videos = result_data['videos']
         topic = result_data['topic'] # Retrieve the associated topic
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
                 unique_key_base = f"{term}_{video_id}"

                 # --- Check Video State ---
                 is_selected = video_id in st.session_state.selected_videos
                 video_state = st.session_state.selected_videos.get(video_id, {})
                 has_dlp_info = is_selected and video_state.get('Direct URL') and not video_state.get('yt_dlp_error')
                 is_fetching_dlp = is_selected and video_state.get('fetching_dlp', False)
                 dlp_error = video_state.get('yt_dlp_error')
                 is_generating = st.session_state.video_to_generate == video_id
                 generation_error = video_state.get('Generation Error')
                 s3_url = video_state.get('Generated S3 URL')


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
                     if is_selected:
                         select_button_label = "✅ Deselect"
                         select_button_type = "secondary"
                     elif is_fetching_dlp:
                          select_button_label = "⏳ Fetching URL..."
                          select_button_type = "secondary"
                     else:
                         select_button_label = "➕ Select"
                         select_button_type = "primary"

                     if st.button(select_button_label, key=f"select_{unique_key_base}", type=select_button_type, use_container_width=True, disabled=is_fetching_dlp or is_generating):
                         if is_selected:
                             del st.session_state.selected_videos[video_id]
                             st.toast(f"Deselected: {video_title}", icon="➖")
                             if st.session_state.video_to_generate == video_id:
                                 st.session_state.video_to_generate = None # Stop generation if deselected
                         else:
                             # Mark as fetching and add basic info
                             st.session_state.selected_videos[video_id] = {
                                 'Search Term': term,
                                 'Topic': topic, # Store the topic
                                 'Video Title': video_title,
                                 'Video ID': video_id,
                                 'Standard URL': standard_video_url,
                                 'fetching_dlp': True, # Mark as fetching
                                 'Direct URL': None,
                                 'Format Details': None,
                                 'yt_dlp_error': None,
                                 'Generated S3 URL': None,
                                 'Generation Error': None
                             }
                             st.toast(f"Selected: {video_title}. Fetching direct URL...", icon="⏳")
                         st.rerun()

                     # --- Display yt-dlp Status ---
                     if is_selected and not is_fetching_dlp: # Show status only after fetch attempt
                         if dlp_error:
                             st.error(f"yt-dlp Error: {dlp_error}", icon="⚠️")
                         elif has_dlp_info:
                             st.success(f"URL Ready: {video_state.get('Format Details', 'N/A')}", icon="✔️")
                         # If selected but no info and no error, something went wrong internally
                         elif not video_state.get('Direct URL'):
                              st.warning("yt-dlp info missing.", icon="❓")


                     # --- Generate Button ---
                     st.markdown("---") # Separator
                     if is_selected and has_dlp_info and not dlp_error:
                         generate_key = f"generate_{unique_key_base}"
                         gen_button_label = "⏳ Generating..." if is_generating else "🎬 Generate Video w/ TTS"
                         gen_button_type = "secondary" if is_generating else "primary"

                         if st.button(gen_button_label, key=generate_key, type=gen_button_type, use_container_width=True, disabled=is_generating):
                             # Set the video ID to generate and rerun
                             st.session_state.video_to_generate = video_id
                             # Clear previous generation errors/results for this video
                             st.session_state.selected_videos[video_id]['Generation Error'] = None
                             st.session_state.selected_videos[video_id]['Generated S3 URL'] = None
                             st.rerun()
                     elif is_selected and dlp_error:
                         st.button("Cannot Generate (yt-dlp Error)", key=f"gen_disabled_{unique_key_base}", disabled=True, use_container_width=True)
                     elif is_selected and not has_dlp_info and not is_fetching_dlp and not dlp_error:
                          st.button("Cannot Generate (URL Missing)", key=f"gen_disabled_{unique_key_base}", disabled=True, use_container_width=True)
                     elif is_fetching_dlp:
                          st.button("Wait for URL Fetch...", key=f"gen_wait_{unique_key_base}", disabled=True, use_container_width=True)


                     # --- Display Generation Status ---
                     if generation_error:
                         st.error(f"Gen Error: {generation_error}", icon="🔥")
                     if s3_url:
                         st.success("Video Generated!", icon="🎉")
                         st.link_button("View on S3", url=s3_url, use_container_width=True)
                         # st.video(s3_url) # Embedding S3 might require public access / CORS config

             st.divider()

     # --- yt-dlp Fetching Logic (runs after initial UI render if needed) ---
     ids_to_fetch = [
         vid for vid, data in st.session_state.selected_videos.items()
         if data.get('fetching_dlp')
     ]

     if ids_to_fetch:
         fetch_id = ids_to_fetch[0] # Process one at a time per rerun
         video_data = st.session_state.selected_videos[fetch_id]
         standard_url = video_data['Standard URL']
         title = video_data['Video Title']

         with st.spinner(f"Fetching yt-dlp details for '{title}'..."):
             dlp_info = get_yt_dlp_info(standard_url)

         # Update state based on dlp_info result
         current_state = st.session_state.selected_videos[fetch_id]
         current_state['fetching_dlp'] = False # Mark fetch attempt as complete

         if dlp_info and dlp_info.get('direct_url'):
             current_state['Direct URL'] = dlp_info['direct_url']
             current_state['Format Details'] = dlp_info['format_details']
             current_state['yt_dlp_error'] = None
             st.toast(f"Direct URL loaded for '{title}'", icon="✅")
         elif dlp_info and dlp_info.get('error'):
             current_state['Direct URL'] = None
             current_state['Format Details'] = "Error"
             current_state['yt_dlp_error'] = dlp_info['error']
             st.toast(f"yt-dlp failed for '{title}': {dlp_info['error']}", icon="⚠️")
         else: # Critical failure from get_yt_dlp_info
             current_state['Direct URL'] = None
             current_state['Format Details'] = "Critical Error"
             current_state['yt_dlp_error'] = "Critical yt-dlp failure"
             st.toast(f"Critical yt-dlp error for '{title}'", icon="🔥")

         st.session_state.selected_videos[fetch_id] = current_state # Save updated state
         st.rerun() # Rerun to update UI (show status, enable generate button)

# 4. Video Generation Logic (if triggered)
if st.session_state.video_to_generate:
    video_id_to_process = st.session_state.video_to_generate
    video_data = st.session_state.selected_videos.get(video_id_to_process)
    final_video_path = None # Initialize paths for cleanup
    audio_path = None

    if video_data and video_data.get('Direct URL') and not video_data.get('yt_dlp_error'):
        st.info(f"🚀 Starting video generation process for: {video_data['Video Title']}", icon="🤖")
        gen_placeholder = st.empty() # Placeholder for status updates
        gen_placeholder.status("Initializing generation...", state="running", expanded=True)

        try:
            # --- Step 1: Get Topic ---
            topic = video_data.get('Topic', 'the selected video') # Fallback topic
            gen_placeholder.status(f"1/5: Generating script for topic: '{topic}'...", state="running", expanded=True)

            # --- Step 2: Generate Script (Text Only) ---
            script_prompt = f"Create a short, engaging voiceover script (roughly 15-45 seconds long, maybe 3-6 sentences) about '{topic}'. The tone should be informative yet conversational, suitable for a voice like 'sage'. Focus on clarity and smooth flow. Just provide the script text, nothing else."
            script_text = chatGPT(script_prompt, client=openai_client)
            if not script_text:
                 raise ValueError("Failed to generate script text from OpenAI.")
            st.write("📝 Generated Script:")
            st.text_area("Script Text", script_text, height=100, disabled=True)
            gen_placeholder.status(f"2/5: Generating '{DEFAULT_TTS_VOICE}' TTS audio and timestamps...", state="running", expanded=True)


            # --- Step 3: Generate TTS Audio & Timestamps ---
            audio_path, word_timings = generate_audio_with_timestamps(
                script_text, client=openai_client, voice_id=DEFAULT_TTS_VOICE
            )
            if not audio_path or word_timings is None: # Allow empty timings, but not None
                raise ValueError("Failed to generate TTS audio or timestamps.")
            gen_placeholder.status(f"3/5: Processing base video and adding audio/subtitles...", state="running", expanded=True)


            # --- Step 4: Process Video (Combine, Loop, Subtitles) ---
            final_video_path, final_filename = process_video_with_tts(
                base_video_url=video_data['Direct URL'],
                audio_path=audio_path,
                word_timings=word_timings,
                topic=topic # Pass topic for filename generation
            )
            if not final_video_path:
                raise ValueError("Video processing (MoviePy) failed.")
            gen_placeholder.status(f"4/5: Uploading final video '{final_filename}' to S3...", state="running", expanded=True)


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
            gen_placeholder.status(f"5/5: Generation Complete! S3 URL ready.", state="complete", expanded=True)


            # --- Step 6: Update Session State & Display ---
            st.session_state.selected_videos[video_id_to_process]['Generated S3 URL'] = s3_url
            st.session_state.selected_videos[video_id_to_process]['Generation Error'] = None # Clear error on success
            st.success(f"✅ Video generated and uploaded successfully!", icon="🎉")
            st.video(s3_url) # Display the final video from S3


        except Exception as e:
            st.error(f"Error during video generation: {e}", icon="🔥")
            st.session_state.selected_videos[video_id_to_process]['Generation Error'] = str(e)
            st.session_state.selected_videos[video_id_to_process]['Generated S3 URL'] = None # Clear URL on error
            if 'gen_placeholder' in locals():
                 gen_placeholder.status(f"Generation Failed: {e}", state="error", expanded=True)

        finally:
            # --- Cleanup Temporary Files ---
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: st.warning(f"Could not delete temp audio: {audio_path}")
            if final_video_path and os.path.exists(final_video_path):
                try: os.remove(final_video_path)
                except: st.warning(f"Could not delete temp video: {final_video_path}")

            # Reset the generation trigger and rerun to update UI
            st.session_state.video_to_generate = None
            st.rerun()

    elif st.session_state.video_to_generate: # Handle case where data is missing but trigger is set
         st.warning(f"Cannot generate video for ID {st.session_state.video_to_generate}. Data missing or yt-dlp error.", icon="❓")
         st.session_state.video_to_generate = None # Reset trigger
         st.rerun()


# --- Display Selected Videos Table (Sidebar) ---
st.sidebar.divider()
st.sidebar.header("Selected & Generated Videos")

if st.session_state.selected_videos:
    selected_list = list(st.session_state.selected_videos.values())

    # Create DataFrame, handling potentially missing columns gracefully
    df_selected = pd.DataFrame(selected_list)

    # Define desired columns and order
    display_columns = [
        'Search Term', 'Topic', 'Video Title', 'Video ID',
        'Direct URL', 'Format Details', 'Standard URL', 'yt_dlp_error',
        'Generated S3 URL', 'Generation Error'
    ]

    # Ensure all display columns exist in the DataFrame, fill missing with None or "N/A"
    for col in display_columns:
        if col not in df_selected.columns:
            df_selected[col] = None # Use None as default placeholder

    # Fill NaN values specifically for display where None isn't ideal
    df_selected.fillna({'yt_dlp_error': "OK", 'Generation Error': "OK"}, inplace=True)
    df_selected.fillna("N/A", inplace=True) # Fill remaining NAs


    # Reorder DataFrame columns for display
    df_selected_display = df_selected[display_columns]

    st.sidebar.dataframe(
        df_selected_display,
        use_container_width=True,
        column_config={
            "Video Title": st.column_config.TextColumn(width="medium"),
            "Topic": st.column_config.TextColumn(width="medium"),
            "Direct URL": st.column_config.LinkColumn("Direct URL", display_text="Link", width="small"),
            "Standard URL": st.column_config.LinkColumn("Watch Page", display_text="Link", width="small"),
            "Generated S3 URL": st.column_config.LinkColumn("Generated Video", display_text="S3 Link", width="small"),
            "yt_dlp_error": st.column_config.TextColumn("yt-dlp Status", width="small"),
            "Generation Error": st.column_config.TextColumn("Gen Status", width="small"),
            "Format Details": st.column_config.TextColumn(width="medium"),
        },
        hide_index=True
    )

    # Option to download selected videos info as CSV
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df_selected_display)

    st.sidebar.download_button(
        label="📥 Download Table as CSV",
        data=csv_data,
        file_name='selected_youtube_videos_generated.csv',
        mime='text/csv',
        use_container_width=True
    )

else:
    st.sidebar.info("No videos selected yet. Use '➕ Select' buttons in the main area.")

# Footer notes
st.sidebar.caption("YT API Search: ~100 quota units/term. yt-dlp fetch on select. Video Gen on generate.")