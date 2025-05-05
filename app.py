# <<< START: IMPORT STATEMENTS AND HELPER FUNCTIONS (Keep these as they are - minor changes noted below) >>>
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
import moviepy.audio.fx.all as afx
# Ensure MoviePy is installed: pip install moviepy
# Ensure Pillow is installed: pip install Pillow
# Ensure pydub is installed: pip install pydub
# Ensure numpy is installed: pip install numpy
import numpy as np
# --- Try importing moviepy components with error handling ---
if 'resolved_vid_urls' not in st.session_state:
  st.session_state['resolved_vid_urls'] = {} # vid:url


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
MAX_RESULTS_PER_QUERY = 3 # Reduce results per query slightly to manage load
YT_DLP_FETCH_TIMEOUT = 30 # Increased timeout for potentially slower connections
DEFAULT_TTS_VOICE = "sage" # Default voice for TTS
# --- Font Path Handling ---
MONTSSERAT_FONT_FILENAME = "Montserrat-Bold.ttf"
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
SCRIPT_VER_OPTIONS =["default", "default_v2", "1st_person" ,"mix"]
# --- Load Secrets ---
try:
    youtube_api_key_secret = st.secrets["YOUTUBE_API_KEY"] # Assuming key name in secrets
    openai_api_key = st.secrets["GPT_API_KEY1"]
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

# --- Helper Functions (Keep sync_search_data, search_youtube, simple_hash, get_yt_dlp_info, chatGPT, claude, generate_audio_with_timestamps, group_words_with_timing, create_text_image, download_with_ytdlp, download_direct_url, upload_vid_to_s3 as they were) ---

# --- Helper Function: create_topic_summary_dataframe (No change needed, it aggregates based on Topic/Language from job data) ---
def create_topic_summary_dataframe(selected_videos_dict):
    """
    Creates a DataFrame summarizing generated videos grouped by a normalized
    'topic_language' key. Includes debug output.
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

    # (Rest of the function remains the same - calculates max_urls, pads, creates DF)
    if not topic_lang_to_generated_urls:
        return pd.DataFrame(columns=['Topic'])

    max_urls = 0
    if topic_lang_to_generated_urls:
        try:
            max_urls = max(len(urls) for urls in topic_lang_to_generated_urls.values())
        except ValueError:
            max_urls = 0

    data_for_df = []
    for topic_lang_key, urls in topic_lang_to_generated_urls.items():
        row = {'Topic': topic_lang_key}
        # Pad with empty strings if fewer than max_urls for this topic/lang
        padded_urls = urls + [''] * (max_urls - len(urls))
        for i, url in enumerate(padded_urls):
            row[f'vid{i+1}_url'] = url
        data_for_df.append(row)

    if data_for_df:
        df_final = pd.DataFrame(data_for_df)
        if 'Topic' in df_final.columns:
             topic_col = df_final.pop('Topic')
             df_final.insert(0, 'Topic', topic_col)
             url_cols_present = [col for col in df_final.columns if col.startswith('vid')]
             url_cols_sorted = sorted(url_cols_present,
                                      key=lambda x: int(x.replace('vid','').replace('_url','')))
             df_final = df_final[['Topic'] + url_cols_sorted]
        else:
             df_final = pd.DataFrame(columns=['Topic']) # Fallback
    else:
         df_final = pd.DataFrame(columns=['Topic'])

    return df_final


# (Keep search_youtube, simple_hash, get_yt_dlp_info, chatGPT, claude, generate_audio_with_timestamps, group_words_with_timing, create_text_image, download_with_ytdlp, download_direct_url functions as they were)

# --- Helper Function: process_video_with_tts (Minor change to accept base_video_url and generate unique output filename base) ---
def process_video_with_tts(base_video_url, audio_path, word_timings, topic, lang, copy_num, with_music=False): # Added lang, copy_num
    """Loads video, adds TTS audio, loops if necessary, adds subtitles centered with wrapping."""
    final_video_clip = None
    temp_output_path = None
    base_video = None
    tts_audio = None
    looped_video = None
    processed_video = None
    subtitle_clips_list = []
    local_vid_path = None # For cleanup

    try:
        st.write(f"⏳ Downloading/Loading base video from URL for processing...")
        # Download the base video locally first using yt-dlp for reliability
        local_vid_path = download_with_ytdlp(base_video_url, cookie_file_path=COOKIE_FILE_PATH)
        if not local_vid_path:
            raise ValueError(f"Failed to download base video: {base_video_url}")

        st.write(f"➡️ Loading downloaded video: {local_vid_path}")
        # Load the local file with MoviePy
        base_video = VideoFileClip(local_vid_path, audio=False, target_resolution=(720, 1280)) # Target 720p vertical

        video_duration = base_video.duration
        w = int(base_video.w) if base_video.w else 720
        h = int(base_video.h) if base_video.h else 1280
        st.write(f"✔️ Base video loaded: {w}x{h}, Duration: {video_duration:.2f}s")

        st.write(f"⏳ Loading TTS audio...")
        tts_audio = AudioFileClip(audio_path)
        audio_duration = tts_audio.duration

        combined_audio = tts_audio # Default
        if with_music:
            try:
                # Make sure Sunrise.mp3 is in an 'audio' subfolder or adjust path
                music_path = os.path.join('audio', 'Sunrise.mp3')
                if os.path.exists(music_path):
                    back_music = AudioFileClip(music_path).fx(afx.volumex, 0.08).set_duration(audio_duration) # Ensure music matches audio duration
                    # Pad or loop music if shorter? For now, just set duration.
                    # If music is shorter than TTS, consider looping:
                    # music_duration = back_music.duration
                    # if music_duration < audio_duration:
                    #    num_loops = int(np.ceil(audio_duration / music_duration))
                    #    back_music = concatenate_audioclips([back_music] * num_loops).subclip(0, audio_duration)

                    combined_audio = CompositeAudioClip([tts_audio.set_duration(audio_duration), back_music]) # Composite
                    st.write("✔️ Background music added.")
                else:
                    st.warning(f"Background music file not found at '{music_path}'. Skipping music.", icon="🎵")
            except Exception as music_err:
                st.warning(f"Could not load or process background music: {music_err}", icon="🎵")
        st.write(f"✔️ TTS audio loaded: Duration: {audio_duration:.2f}s")


        # --- Video Resizing (Force 9:16 aspect ratio - e.g., 720x1280) ---
        target_w, target_h = 720, 1280
        st.write(f"⏳ Resizing video to {target_w}x{target_h}...")
        try:
            resized_base_video = base_video.resize(newsize=(target_w, target_h))
            st.write(f"✔️ Video resized.")
        except Exception as resize_err:
            st.warning(f"Could not resize video: {resize_err}. Using original dimensions.", icon="⚠️")
            resized_base_video = base_video
            target_w, target_h = w, h

        # --- Video Looping/Trimming ---
        processed_video = resized_base_video
        if video_duration < audio_duration:
            st.write(f"⏳ Looping video to match audio duration...")
            num_loops = int(np.ceil(audio_duration / video_duration))
            clips_to_loop = [resized_base_video.copy().set_start(i * video_duration).set_duration(video_duration) for i in range(num_loops)] # ensure each loop has correct duration
            looped_video = concatenate_videoclips(clips_to_loop, method="compose") # Use compose for potential overlaps
            processed_video = looped_video.subclip(0, audio_duration) # Trim precisely
            st.write(f"✔️ Video looped {num_loops} times and trimmed to {audio_duration:.2f}s")
        elif video_duration > audio_duration:
            st.write(f"⏳ Trimming video to match audio duration...")
            processed_video = resized_base_video.subclip(0, audio_duration)
            st.write(f"✔️ Video trimmed to {audio_duration:.2f}s")
        else:
             st.write("✔️ Video duration matches audio duration.")


        # Set the potentially combined audio to the processed video
        final_video_clip = processed_video.set_audio(combined_audio).set_duration(audio_duration)
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

                if not text.strip() or sub_duration <= 0.05:
                    continue

                status_text.text(f"Creating subtitle {i+1}/{total_subs}: '{text[:30]}...'")

                text_img_array = create_text_image(
                     text.upper(),
                     fontsize=SUBTITLE_FONT_SIZE,
                     color=SUBTITLE_COLOR,
                     bg_color=SUBTITLE_BG_COLOR,
                     font_path=SUBTITLE_FONT_PATH,
                     video_width=target_w
                )

                if text_img_array.shape[0] <= 10 or text_img_array.shape[1] <= 10:
                     st.warning(f"Skipping subtitle due to image creation error for: '{text[:30]}...'")
                     continue

                subtitle_img_clip = ImageClip(text_img_array)\
                     .set_start(start)\
                     .set_duration(sub_duration)\
                     .set_position(('center', 'center'))

                subtitle_clips_list.append(subtitle_img_clip)
                sub_progress_bar.progress((i + 1) / total_subs)

            status_text.text(f"✔️ Generated {len(subtitle_clips_list)} subtitle clips.")
            # sub_progress_bar.empty() # Optional: Clear progress bar
            # status_text.empty() # Optional: Clear status text
        else:
             st.warning("No valid word timings available to generate subtitles.", icon="⏱️")


        # Composite final video with subtitles
        if subtitle_clips_list:
            st.write("⏳ Compositing video and subtitles...")
            clips_for_composite = [final_video_clip] + subtitle_clips_list
            final_video_clip = CompositeVideoClip(clips_for_composite, size=(target_w, target_h))
            st.write("✔️ Compositing complete.")
        else:
             st.write("ℹ️ No subtitles added.")


        # --- Export Final Video ---
        st.write("⏳ Exporting final video...")
        # Generate a base filename - S3 upload function will add timestamp/copy number
        safe_topic = urllib.parse.quote(topic.replace(' ', '_')[:30], safe='')
        # Timestamp and copy number will be added in the main loop before S3 upload
        # For the local temp file, a unique name is generated automatically

        # Use NamedTemporaryFile for the output path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=f"processed_{safe_topic}_") as temp_output_file_obj:
            temp_output_path = temp_output_file_obj.name

        if not isinstance(final_video_clip, (VideoFileClip, CompositeVideoClip)):
             raise TypeError(f"Cannot write final video: Invalid clip object type {type(final_video_clip)}.")

        # Create a unique temporary audio filename for this process run
        temp_audio_filename = f'temp-audio-{os.path.basename(temp_output_path)}.m4a'

        final_video_clip.write_videofile(
            temp_output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=temp_audio_filename, # Use unique temp audio file
            remove_temp=True,
            fps=resized_base_video.fps if resized_base_video.fps and resized_base_video.fps > 0 else 24, # Use fps or default
            preset='medium',
            threads=os.cpu_count() or 4,
            logger='bar',
            ffmpeg_params=[
                 '-movflags', 'faststart',
                 '-profile:v', 'high',
                 '-level', '4.0',
            ]
        )
        st.write(f"✔️ Final video exported to temporary path: {temp_output_path}")

        # Return the path to the generated temp file
        # The unique S3 object name will be constructed *outside* this function
        return temp_output_path # Return only the path

    except Exception as e:
        st.error(f"Error during video processing: {e}", icon="🎬")
        import traceback
        traceback.print_exc()
        # Ensure path is returned as None on failure
        temp_output_path = None # Set explicitly to None on error
        raise # Re-raise the exception to be caught by the main loop

    finally:
        # --- Cleanup ---
        st.write("🧹 Cleaning up video processing resources...")
        try:
            # Close MoviePy clips
            if base_video: base_video.close()
            if tts_audio: tts_audio.close()
            if 'back_music' in locals() and back_music: back_music.close() # Close music if loaded
            if combined_audio and combined_audio not in [tts_audio]: combined_audio.close() # Close composite if different
            if looped_video: looped_video.close()
            if 'resized_base_video' in locals() and resized_base_video is not base_video: resized_base_video.close()
            if processed_video and processed_video not in [base_video, looped_video, resized_base_video]: processed_video.close()
            # Don't close final_video_clip here if it's the one being written? Moviepy handles it.
            # But close composite sources if they exist
            if 'clips_for_composite' in locals():
                 # final_video_clip is the first element, skip it
                 for clip in clips_for_composite[1:]: # Close subtitle clips
                     if clip: clip.close()

            # Delete the downloaded base video temp file
            if local_vid_path and os.path.exists(local_vid_path):
                 os.remove(local_vid_path)
                 st.write(f"🧹 Deleted temp base video: {local_vid_path}")

            # Delete the temp audio file used by MoviePy if it wasn't removed automatically
            if 'temp_audio_filename' in locals() and os.path.exists(temp_audio_filename):
                try:
                    os.remove(temp_audio_filename)
                    st.write(f"🧹 Deleted temp audio file: {temp_audio_filename}")
                except Exception as rm_audio_err:
                    st.warning(f"Could not remove temp audio file {temp_audio_filename}: {rm_audio_err}")


        except Exception as cleanup_err:
            st.warning(f"Error during resource cleanup: {cleanup_err}")
        st.write("🧹 Cleanup finished.")
        # Return None if path wasn't successfully set before finally block
        # This happens if an error occurred before the return statement
        if 'temp_output_path' not in locals(): temp_output_path = None
        # IMPORTANT: Don't return the path from finally, let the try block do it or return None implicitly
        # We only return the path if the try block completes successfully before the finally block runs.


# (Keep upload_vid_to_s3 function as it was)
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
st.caption("Search YouTube, select videos (multiple times allowed), generate TTS script, create subtitled videos, and upload to S3.")

# --- Session State Initialization ---

# Stores data for each *selected video job instance* {job_key: video_data_dict}
# job_key format: f"{videoId}_{language}_{copyNumber}"
if 'selected_videos' not in st.session_state:
    st.session_state.selected_videos = {}

# Stores the results from the last API search {search_term: {'videos': [], 'topic': '', 'lang': '', 'script_ver': ''}}
if 'api_search_results' not in st.session_state:
    st.session_state.api_search_results = {}

# Input DataFrame for search terms and topics
if 'search_data' not in st.session_state:
    st.session_state.search_data = pd.DataFrame([
        {'Topic': 'sofa sale', 'Search Term': 'auto' ,'Language' : 'English',"Script Angle" : "default", 'Video Results': 5}
        # Add more default rows if needed
    ])

# Flag to indicate if a search has been run
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False

# --- State for Batch Processing ---
# List of job_keys queued for generation
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

st.sidebar.write("Enter Search Terms and Topics:")
# (Keep sync_search_data function as it was)
def sync_search_data():
    # This function seems complex and might need review depending on data_editor behavior
    # Assuming it correctly reconstructs the DataFrame from the editor state
    raw_data = st.session_state.search_data # This might be the issue source if editor modifies inplace

    # Let's try getting data directly from editor state if possible, might be more reliable
    editor_state = st.session_state.get("search_topic_editor", {})
    edited_rows = editor_state.get("edited_rows", {})
    added_rows = editor_state.get("added_rows", [])
    deleted_rows = editor_state.get("deleted_rows", [])

    # Start with the original data used to populate the editor in the *last* run
    if 'search_data_snapshot' not in st.session_state:
         # If no snapshot, use the potentially modified session_state.search_data
         current_df = st.session_state.search_data.copy() if isinstance(st.session_state.search_data, pd.DataFrame) else pd.DataFrame([{'Topic': '', 'Search Term': '', 'Language': 'English', "Script Angle": "default", 'Video Results': 5}])
    else:
         current_df = st.session_state.search_data_snapshot.copy()

    # Apply deletions first (indices are based on the state *before* edits/adds)
    valid_delete_indices = [idx for idx in deleted_rows if idx < len(current_df)]
    if valid_delete_indices:
        current_df = current_df.drop(index=valid_delete_indices).reset_index(drop=True)

    # Apply edits
    for idx_str, changes in edited_rows.items():
        try:
            idx = int(idx_str)
            if 0 <= idx < len(current_df):
                for col, val in changes.items():
                    if col in current_df.columns:
                        current_df.loc[idx, col] = val
        except ValueError:
            pass # Ignore invalid index string

    # Apply additions
    if added_rows:
        # Ensure added rows have the expected columns
        expected_cols = ["Topic", "Search Term", "Language", "Script Angle", "Video Results"]
        processed_adds = []
        for row_dict in added_rows:
             if isinstance(row_dict, dict):
                  # Add default values for missing columns
                  for col in expected_cols:
                       if col not in row_dict:
                            if col == "Language": row_dict[col] = "English"
                            elif col == "Script Angle": row_dict[col] = "default"
                            elif col == "Video Results": row_dict[col] = 5
                            else: row_dict[col] = ""
                  processed_adds.append(row_dict)
        if processed_adds:
             add_df = pd.DataFrame(processed_adds)
             current_df = pd.concat([current_df, add_df], ignore_index=True)


    # Ensure final DataFrame has the right columns and types
    expected_cols = ["Topic", "Search Term", "Language", "Script Angle", "Video Results"]
    for col in expected_cols:
        if col not in current_df.columns:
            if col == "Language": current_df[col] = "English"
            elif col == "Script Angle": current_df[col] = "default"
            elif col == "Video Results": current_df[col] = 5
            else: current_df[col] = ""

    current_df = current_df[expected_cols] # Reorder and select columns

    # Type conversion and validation
    try:
        current_df['Video Results'] = pd.to_numeric(current_df['Video Results'], errors='coerce').fillna(5).astype(int)
        current_df['Video Results'] = current_df['Video Results'].apply(lambda x: max(1, min(x, 50))) # Clamp results (e.g., 1-50)
    except Exception:
        current_df['Video Results'] = 5 # Fallback
    current_df['Topic'] = current_df['Topic'].fillna('').astype(str)
    current_df['Search Term'] = current_df['Search Term'].fillna('').astype(str)
    current_df['Language'] = current_df['Language'].fillna('English').astype(str)
    current_df['Script Angle'] = current_df['Script Angle'].fillna('default').astype(str)

    # Remove rows where both Topic and Search Term are empty (unless it's the only row)
    meaningful_rows = (current_df['Topic'].str.strip() != '') | (current_df['Search Term'].str.strip() != '')
    if len(current_df) > 1:
        current_df = current_df[meaningful_rows]

    # If empty after filtering, add back a default row
    if current_df.empty:
         current_df = pd.DataFrame([{'Topic': '', 'Search Term': '', 'Language': 'English', "Script Angle": "default", 'Video Results': 5}])

    # Update the main session state and create a snapshot for the next run
    st.session_state.search_data = current_df.reset_index(drop=True)
    st.session_state.search_data_snapshot = st.session_state.search_data.copy()
    #st.text("Sync completed.") # Debug message


# Use data_editor with on_change callback
# Make a snapshot before rendering the editor
if 'search_data_snapshot' not in st.session_state:
     st.session_state.search_data_snapshot = st.session_state.search_data.copy()

edited_df = st.sidebar.data_editor(
    st.session_state.search_data, # Use the main state data
    column_config={
        "Script Angle": st.column_config.SelectboxColumn("Script Angle",options=SCRIPT_VER_OPTIONS, default="default"),
        "Video Results": st.column_config.NumberColumn("Video Results", min_value=1, max_value=50, step=1, default=5),
        "Language": st.column_config.TextColumn("Language", default="English")
        },
    num_rows="dynamic",
    use_container_width=True,
    key="search_topic_editor",
    # on_change=sync_search_data # Call sync *after* edits are registered
)

# --- Action Buttons (Sidebar) ---
col1, col2 = st.sidebar.columns(2)
search_button = col1.button("🔍 Search Videos", use_container_width=True, disabled=st.session_state.batch_processing_active, on_click=sync_search_data) # Sync before search logic runs
clear_button = col2.button("🧹 Clear All", use_container_width=True, type="secondary", disabled=st.session_state.batch_processing_active)
with_music = col1.checkbox("With BG music?", value=False) # Default to False
with_music_rand = col2.checkbox("With BG music randomly?", value=False) # Default to False

if clear_button:
    st.session_state.selected_videos = {}
    st.session_state.search_triggered = False
    st.session_state.api_search_results = {}
    st.session_state.generation_queue = [] # Clear queue
    st.session_state.batch_processing_active = False # Reset flag
    st.session_state.batch_total_count = 0
    st.session_state.batch_processed_count = 0
    st.session_state.resolved_vid_urls = {} # Clear resolved URL cache
    # Optionally reset the input table to a single default row
    st.session_state.search_data = pd.DataFrame([{'Topic': '', 'Search Term': '','Language' : 'English',"Script Angle" : "default", 'Video Results': 5}])
    st.session_state.search_data_snapshot = st.session_state.search_data.copy() # Reset snapshot too
    st.success("Selections, results, and generation state cleared!", icon="✅")
    st.rerun()

# --- Global Process Button ---
st.sidebar.divider()
# Calculate how many selected jobs are ready for processing
ready_jobs_count = sum(
    1 for job_key, data in st.session_state.selected_videos.items()
    if data.get('Direct URL') and not data.get('yt_dlp_error') and not data.get('Generated S3 URL') and not data.get('Generation Error')
)

process_all_button = st.sidebar.button(
    f"🎬 Process {ready_jobs_count} Ready Job{'s' if ready_jobs_count != 1 else ''}",
    use_container_width=True,
    type="primary",
    disabled=ready_jobs_count == 0 or st.session_state.batch_processing_active # Disable if 0 ready or already processing
)

# --- MODIFIED: Process All Button Logic ---
if process_all_button:
    # Find job keys that are ready for processing
    job_keys_to_process = [
        job_key for job_key, data in st.session_state.selected_videos.items()
        if data.get('Direct URL') and not data.get('yt_dlp_error') and not data.get('Generated S3 URL') and not data.get('Generation Error')
    ]
    if job_keys_to_process:
        st.session_state.generation_queue = job_keys_to_process # Queue the unique job keys
        st.session_state.batch_processing_active = True
        st.session_state.batch_total_count = len(job_keys_to_process)
        st.session_state.batch_processed_count = 0
        st.sidebar.info(f"Queued {len(job_keys_to_process)} generation jobs.", icon="⏳")
        # Clear any potential old errors and set status to Queued for these specific jobs
        for job_key in job_keys_to_process:
            if job_key in st.session_state.selected_videos: # Check if key exists before updating
                st.session_state.selected_videos[job_key]['Generation Error'] = None
                st.session_state.selected_videos[job_key]['Generated S3 URL'] = None # Reset S3 URL too
                st.session_state.selected_videos[job_key]['Status'] = 'Queued' # Update status
        st.rerun()
    else:
        st.sidebar.warning("No selected video jobs are ready for processing (need Direct URL).", icon="⚠️")


st.sidebar.info("Select videos using '➕'. Each click queues one generation job. Processing uses jobs with valid 'Direct URL'.", icon="ℹ️")
st.sidebar.warning("Video generation can take several minutes per job.", icon="⏱️")

# --- Processing Logic ---

# 1. Handle Search Button Click
# The search button now uses on_click=sync_search_data, so sync happens first.
# We directly use the synced st.session_state.search_data here.
if search_button: # This block runs *after* on_click sync
    search_df = st.session_state.search_data # Use the synced data

    # Basic validation after sync
    valid_input = True
    if search_df.empty:
        st.sidebar.warning("Input table is empty.", icon="⚠️")
        valid_input = False
    # Check if required columns have at least one non-empty value after sync/validation in sync_search_data
    elif search_df[['Topic', 'Search Term']].replace('', pd.NA).isnull().all().all():
         # Allow if it's the single default row maybe? Or enforce input.
         if len(search_df) > 1 or (search_df.iloc[0]['Topic'] != '' or search_df.iloc[0]['Search Term'] != ''):
            st.sidebar.warning("Please provide a 'Topic' or 'Search Term' in at least one row.", icon="⚠️")
            valid_input = False
         # Else: allow the default empty row if it's the only one.

    # Check for specifically empty Search Term if Topic is also empty (or vice versa)? - Handled above.
    # Check 'auto' term requires Topic?
    if 'auto' in search_df['Search Term'].values and search_df.loc[search_df['Search Term'] == 'auto', 'Topic'].str.strip().eq('').any():
        st.sidebar.warning("Rows with 'auto' in 'Search Term' must have a non-empty 'Topic'.", icon="⚠️")
        valid_input = False


    if valid_input:
        st.sidebar.success("Input valid, proceeding with search.") # Feedback
        st.session_state.search_triggered = True
        st.session_state.api_search_results = {} # Clear previous results
        # Reset generation states if needed? Usually done via Clear button.
        st.session_state.current_search_df = search_df.copy() # Store the validated DF for the search step

        # Don't necessarily need to update search_data again here if sync handles it well
        # st.session_state.search_data = search_df.copy()
        # st.session_state.search_data_snapshot = search_df.copy()

        st.rerun()
    else:
        st.session_state.search_triggered = False # Ensure search doesn't proceed


# 2. Perform API Search if Triggered
if st.session_state.search_triggered and 'current_search_df' in st.session_state: # Removed check for empty api_search_results here
    search_df = st.session_state.current_search_df
    search_items = search_df.to_dict('records')

    st.info(f"Searching API for {len(search_items)} topic/term row(s)...", icon="🔍")
    progress_bar = st.progress(0)
    status_text_api = st.empty()

    api_error_occurred = False
    results_cache = {} # Temp cache for *this* search run

    for i, item in enumerate(search_items):
        term = item['Search Term'].strip()
        topic = item['Topic'].strip()
        count = int(item['Video Results']) # Already validated int
        lang = item['Language'].strip()
        script_ver = item["Script Angle"].strip()

        # Handle 'auto' search term generation
        if term.lower() == 'auto':
            if not topic:
                st.warning(f"Skipping row {i+1}: 'auto' search term requires a Topic.", icon="⚠️")
                continue # Skip this row
            status_text_api.text(f"Generating search terms for: '{topic}'...")
            try:
                # --- Use the refined GPT prompt for search terms ---
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
                               {topic}""",client=openai_client,model="gpt-4") # Ensure model is specified if needed
                if not term:
                    st.warning(f"Failed to generate search terms for '{topic}'. Skipping.", icon="🤖")
                    continue
                st.write(f"Generated terms for '{topic}': {term}") # Show generated terms
            except Exception as gpt_err:
                st.error(f"Error generating search terms for '{topic}': {gpt_err}", icon="🤖")
                continue # Skip this row on error

        # Proceed with search using the original or generated term
        status_text_api.text(f"Searching YouTube for: '{term}' (Topic: '{topic}')...")
        unique_search_key = f"{term}_{topic}_{lang}_{script_ver}" # Use a key combining factors

        if unique_search_key not in results_cache: # Avoid re-searching exact same combo in one go
            videos = search_youtube(youtube_api_key_secret, term, count) # Pass validated count

            if videos is None: # Indicates potential critical API error from search_youtube
                 st.error(f"Stopping search due to critical API issue (check key/quota) for term: '{term}'.", icon="🚫")
                 api_error_occurred = True
                 break # Stop processing further terms

            # Store results along with topic, lang, script_ver
            results_cache[unique_search_key] = {'videos': videos, 'topic': topic, 'lang': lang, "script_ver": script_ver, 'original_term': term}

            time.sleep(0.1) # Small delay between API calls
        progress_bar.progress((i + 1) / len(search_items))

    status_text_api.text("API Search complete." if not api_error_occurred else "API Search halted due to error.")
    st.session_state.api_search_results = results_cache # Update main cache with results from *this* run
    st.session_state.search_triggered = False # Reset trigger AFTER processing search logic

    # No automatic rerun here - let the results display in the current run
    # st.rerun()


# --- Display Area ---
st.divider()

# --- Display Batch Processing Status ---
if st.session_state.batch_processing_active:
    processed = st.session_state.batch_processed_count
    total = st.session_state.batch_total_count
    queue_len = len(st.session_state.generation_queue)
    st.info(f"⚙️ Batch Processing: {processed} / {total} Completed. {queue_len} Remaining in queue.", icon="⏳")
    if total > 0:
        st.progress(min(1.0, processed / total)) # Ensure progress doesn't exceed 1.0


# 3. Display Search Results
if st.session_state.api_search_results:
    st.subheader("Search Results & Video Selection (Grid View)")

    if not st.session_state.api_search_results:
         st.info("No search results available. Perform a search first.")

    # Display results grouped by the search cache key (term_topic_lang_scriptver)
    for search_key, result_data in st.session_state.api_search_results.items():
        videos = result_data['videos']
        topic = result_data['topic']
        lang = result_data['lang']
        script_ver = result_data["script_ver"]
        original_term = result_data['original_term'] # The term used for the actual search

        # --- Container for the results of THIS search ---
        term_container = st.container(border=True)
        with term_container:
            st.subheader(f"Results for: \"{original_term}\" (Topic: \"{topic}\", Lang: {lang}, Angle: {script_ver})")
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
                        with cols[j]:
                            # --- Extract Video Info ---
                            video_id = video['videoId']
                            video_title = video['title']
                            standard_video_url = f"https://www.youtube.com/watch?v={video_id}" # Correct YT URL
                            # Construct unique base key for UI elements related to this video instance in the grid
                            grid_instance_key = f"{video_id}_{search_key}_{i}_{j}" # Unique key for this display instance


                            # --- State for controlling video player visibility ---
                            show_video_key = f"show_player_{grid_instance_key}"
                            if show_video_key not in st.session_state:
                                st.session_state[show_video_key] = False

                            # --- Render Content Vertically ---
                            st.write(f"**{textwrap.shorten(video_title, width=50, placeholder='...')}**")
                            st.caption(f"ID: {video_id}")

                            # --- Conditionally Display Video Player OR Thumbnail ---
                            if st.session_state[show_video_key]:
                                try:
                                    # Using st.video for better compatibility
                                    st.video(standard_video_url)
                                except Exception as e:
                                    st.error(f"Video preview failed to load: {e}")
                            else:
                                thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg" # Medium quality thumbnail
                                st.image(thumbnail_url, use_container_width=True, caption="Video Thumbnail")

                            # --- Buttons ---
                            # Toggle Preview Button
                            toggle_label = "🔼 Hide" if st.session_state[show_video_key] else "▶️ Show"
                            if st.button(f"{toggle_label} Preview", key=f"toggle_vid_btn_{grid_instance_key}", help="Show/hide the video preview", use_container_width=True):
                                st.session_state[show_video_key] = not st.session_state[show_video_key]
                                st.rerun()

                            # Select Button (queues ONE job per click)
                            # This button doesn't need complex state checks anymore, just adds a job.
                            if st.button("➕ Select (Queue Job)", key=f"select_{grid_instance_key}", type="primary", use_container_width=True, disabled=st.session_state.batch_processing_active):
                                # --- ADD JOB LOGIC (Copied & adapted from previous correction) ---
                                base_video_id = video_id
                                # Use the language associated with *this search result*
                                current_lang = lang.strip()

                                # Calculate the next copy number
                                base_key_prefix = f"{base_video_id}_{current_lang}_"
                                existing_copy_numbers = [
                                    int(k[len(base_key_prefix):])
                                    for k in st.session_state.selected_videos.keys()
                                    if k.startswith(base_key_prefix) and k[len(base_key_prefix):].isdigit()
                                ]
                                next_copy_number = max(existing_copy_numbers) + 1 if existing_copy_numbers else 1
                                job_key = f"{base_key_prefix}{next_copy_number}"

                                # Add the new job
                                st.session_state.selected_videos[job_key] = {
                                    'Job Key': job_key,
                                    'Search Term': original_term, # Store the term used for search
                                    'Topic': topic,
                                    'Language': current_lang,
                                    'Video Title': video_title,
                                    'Video ID': base_video_id,
                                    'Copy Number': next_copy_number,
                                    'Standard URL': standard_video_url, # Use correct URL
                                    'fetching_dlp': True,
                                    'Direct URL': None,
                                    'Format Details': None,
                                    'yt_dlp_error': None,
                                    'Generated S3 URL': None,
                                    'Generation Error': None,
                                    'Status': 'Selected, Fetching URL...',
                                    'Script Angle': script_ver
                                }
                                st.toast(f"Queued Job #{next_copy_number} ({current_lang}) for: {video_title}", icon="➕")
                                st.rerun() # Rerun to update sidebar and trigger fetching

                            # --- Display Status for EXISTING jobs related to THIS video_id ---
                            # Find jobs in selected_videos that match the current video_id and lang
                            related_job_keys = [
                                k for k, v in st.session_state.selected_videos.items()
                                if v.get('Video ID') == video_id and v.get('Language') == lang
                            ]
                            if related_job_keys:
                                status_expander = st.expander(f"Show Status for {len(related_job_keys)} Queued Job(s)")
                                with status_expander:
                                    for r_job_key in sorted(related_job_keys): # Sort for consistency
                                        job_data = st.session_state.selected_videos.get(r_job_key)
                                        if job_data:
                                            copy_num = job_data.get('Copy Number', '?')
                                            status = job_data.get('Status', 'Unknown')
                                            s3_url = job_data.get('Generated S3 URL')
                                            error_msg = job_data.get('Generation Error') or job_data.get('yt_dlp_error')

                                            st.caption(f"Job #{copy_num} ({r_job_key})")
                                            if status == 'Processing':
                                                st.info(f"⚙️ Processing...", icon="⏳")
                                            elif status == 'Queued':
                                                 st.info(f"🕒 Queued", icon="🕒")
                                            elif status == 'Completed' and s3_url:
                                                 st.success(f"✔️ Generated!", icon="🎉")
                                                 st.link_button("View on S3", url=s3_url)
                                            elif status == 'Failed' and error_msg:
                                                 st.error(f"❌ Failed: {error_msg[:60]}...", icon="🔥")
                                            elif status == 'Error' and error_msg: # yt-dlp error usually sets status to Error: msg
                                                 st.error(f"⚠️ URL Error: {error_msg[:60]}...", icon="⚠️")
                                            elif status == 'Ready':
                                                st.success(f"✅ Ready to Process", icon="👍")
                                            elif status == 'Selected, Fetching URL...':
                                                 st.info(f"📡 Fetching URL...", icon="📡")
                                            else:
                                                 st.write(f"Status: {status}") # Fallback display


# --- MODIFIED: yt-dlp Fetching Logic ---
# (Use the corrected version from the previous response that works with job_keys)
if not st.session_state.batch_processing_active:
    job_keys_to_fetch = [
        job_key for job_key, data in st.session_state.selected_videos.items()
        if data.get('fetching_dlp')
    ]

    if job_keys_to_fetch:
        fetch_job_key = job_keys_to_fetch[0]
        video_data = st.session_state.selected_videos.get(fetch_job_key)

        if video_data:
            standard_url = video_data.get('Standard URL')
            title = video_data.get('Video Title', fetch_job_key)

            with st.spinner(f"Fetching yt-dlp details for '{title}' (Job: {fetch_job_key})..."):
                if standard_url and standard_url in st.session_state.get('resolved_vid_urls', {}):
                    dlp_info = st.session_state['resolved_vid_urls'][standard_url]
                    # print(f"Using cached URL for {standard_url}") # Debug
                elif standard_url:
                    # print(f"Fetching NEW URL for {standard_url}") # Debug
                    # Use actual call here
                    dlp_info = get_yt_dlp_info(standard_url) # Fetches direct URL if possible
                    # Update cache ONLY if successful
                    if dlp_info and dlp_info.get('direct_url'):
                         st.session_state.setdefault('resolved_vid_urls', {})[standard_url] = dlp_info
                else:
                     dlp_info = {'error': 'Missing Standard URL in job data.'}


            current_state = st.session_state.selected_videos.get(fetch_job_key)
            if current_state:
                current_state['fetching_dlp'] = False

                if dlp_info and dlp_info.get('direct_url'):
                    # IMPORTANT: Store the DIRECT URL needed for MoviePy, not the standard YT page URL
                    current_state['Direct URL'] = dlp_info['direct_url']
                    current_state['Format Details'] = dlp_info.get('format_details', 'N/A')
                    current_state['yt_dlp_error'] = None
                    current_state['Status'] = 'Ready'
                    st.toast(f"Direct URL loaded for job '{fetch_job_key}'", icon="✅")
                else: # Handle errors or missing URL
                    error_detail = dlp_info.get('error', "Could not get direct URL") if dlp_info else "yt-dlp fetch failed"
                    current_state['Direct URL'] = None
                    current_state['Format Details'] = "Error"
                    current_state['yt_dlp_error'] = error_detail
                    current_state['Status'] = f"Error: {error_detail}"
                    st.toast(f"yt-dlp failed for job '{fetch_job_key}': {error_detail}", icon="⚠️")

                st.session_state.selected_videos[fetch_job_key] = current_state
                st.rerun()


# --- MODIFIED: Video Generation Logic (BATCH PROCESSING) ---
# (Use the corrected version from the previous response that works with job_keys and removes the inner loop)
if st.session_state.batch_processing_active and st.session_state.generation_queue:
    job_key_to_process = st.session_state.generation_queue[0] # Peek at the next job key
    video_data = st.session_state.selected_videos.get(job_key_to_process)
    # Initialize paths for cleanup in this scope
    final_video_path = None
    audio_path = None

    # Check if the job data is valid and ready (has Direct URL from yt-dlp)
    if video_data and video_data.get('Direct URL') and not video_data.get('yt_dlp_error'):
        processed_count_display = st.session_state.batch_processed_count + 1
        total_count_display = st.session_state.batch_total_count
        st.header(f"⚙️ Processing Job {processed_count_display}/{total_count_display}: {video_data['Video Title']} (Copy #{video_data.get('Copy Number', '?')})")
        gen_placeholder = st.container()

        try:
            # Update Status for this specific job
            st.session_state.selected_videos[job_key_to_process]['Status'] = 'Processing'

            # --- Start Generation Steps (run once per job key) ---
            with gen_placeholder:
                st.info(f"Starting video generation process for job: {job_key_to_process}")
                with st.status("Running generation steps...", state="running", expanded=True) as status_log:
                    try:
                        # --- Step 1: Get data from video_data ---
                        topic = video_data.get('Topic', 'video topic')
                        lang = video_data.get('Language', 'English')
                        script_ver = video_data.get("Script Angle", "default")
                        # CRITICAL: Use the Direct URL fetched by yt-dlp for processing
                        base_video_direct_url = video_data.get("Direct URL")
                        copy_num = video_data.get('Copy Number', 0)

                        if not base_video_direct_url:
                             raise ValueError("Direct video URL is missing for processing.")

                        st.write(f"1/5: Generating script for topic: '{topic}'...")
                        # --- Script Generation ---
                        if script_ver == "mix":
                            script_ver_temp = random.choice([opt for opt in SCRIPT_VER_OPTIONS if opt != 'mix']) # Pick a concrete version
                        else:
                            script_ver_temp = script_ver
                        # (Include the full script prompt logic here based on script_ver_temp)
                        script_prompt = f"Create a script for {topic} in {lang} using angle {script_ver_temp}" # Simplified placeholder
                        # Choose your LLM call:
                        # script_text = chatGPT(script_prompt, client=openai_client, model="gpt-4")
                        script_text = claude(script_prompt, is_thinking=True) # Assuming claude function is defined and working
                        # ---

                        if not script_text:
                            raise ValueError("Failed to generate script text.")
                        st.text_area("Generated Script:", script_text, height=100, disabled=True, key=f"script_{job_key_to_process}")

                        st.write(f"2/5: Generating TTS audio and timestamps...")
                        # --- Step 2: Generate TTS ---
                        audio_path, word_timings = generate_audio_with_timestamps(
                            script_text, client=openai_client, voice_id=DEFAULT_TTS_VOICE
                        )
                        if audio_path is None or word_timings is None:
                            raise ValueError("Failed to generate TTS audio or timestamps.")

                        st.write(f"3/5: Processing base video and adding audio/subtitles...")
                        # --- Step 3: Process Video ---
                        current_with_music = with_music
                        if with_music_rand:
                            current_with_music = random.choice([True, False])

                        # Pass the DIRECT URL to the processing function
                        # Also pass lang and copy_num for potential use in filename generation inside process_video_with_tts if needed, though we do it outside now.
                        temp_processed_video_path = process_video_with_tts(
                            base_video_url=base_video_direct_url, # <<< Use the DIRECT URL
                            audio_path=audio_path,
                            word_timings=word_timings,
                            topic=topic,
                            lang=lang,
                            copy_num=copy_num,
                            with_music=current_with_music
                        )
                        # process_video_with_tts now only returns the path to the temp file
                        final_video_path = temp_processed_video_path # Assign for cleanup

                        if not final_video_path:
                            raise ValueError("Video processing (MoviePy) failed.")

                        # --- Step 4: Construct Unique S3 Filename ---
                        safe_topic = urllib.parse.quote(topic.replace(' ', '_')[:30], safe='')
                        timestamp = int(datetime.datetime.now().timestamp())
                        # Use lang and copy_num for a unique name
                        final_s3_object_name = f"final_{safe_topic}_{lang}_copy{copy_num}_{timestamp}.mp4"

                        st.write(f"4/5: Uploading final video '{final_s3_object_name}' to S3...")
                        # --- Step 5: Upload to S3 ---
                        s3_url = upload_vid_to_s3(
                            s3_cli=s3_client,
                            video_path=final_video_path, # Path from process_video_with_tts
                            bucket_name=s3_bucket_name,
                            object_name=final_s3_object_name,
                            region_name=s3_region
                        )
                        if not s3_url:
                            raise ValueError("Failed to upload video to S3.")

                        st.write(f"5/5: Generation Complete! S3 URL ready.")
                        status_log.update(label="Generation Complete!", state="complete", expanded=False)

                        # --- Step 6: Update Session State (Success) ---
                        # Use the specific job_key_to_process
                        if job_key_to_process in st.session_state.selected_videos:
                            st.session_state.selected_videos[job_key_to_process]['Generated S3 URL'] = s3_url
                            st.session_state.selected_videos[job_key_to_process]['Generation Error'] = None
                            st.session_state.selected_videos[job_key_to_process]['Status'] = 'Completed'
                            st.success(f"✅ Job '{job_key_to_process}' generated successfully!", icon="🎉")
                            st.video(s3_url) # Display result
                        else:
                             st.warning(f"Job key {job_key_to_process} disappeared during processing.")


                    except Exception as e:
                        st.error(f"Error during video generation step for job '{job_key_to_process}': {e}", icon="🔥")
                        status_log.update(label=f"Generation Failed: {str(e)[:100]}", state="error", expanded=True)
                        # Update session state (Failure)
                        if job_key_to_process in st.session_state.selected_videos:
                            st.session_state.selected_videos[job_key_to_process]['Generation Error'] = str(e)[:200]
                            st.session_state.selected_videos[job_key_to_process]['Generated S3 URL'] = None
                            st.session_state.selected_videos[job_key_to_process]['Status'] = 'Failed'
                        # Allow finally block to handle queue/rerun

        finally:
            # --- Cleanup, Queue Management, and Rerun ---
            st.write(f"--- Cleaning up for job {job_key_to_process} ---") # Debug message
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    # st.write(f"Cleaned up temp audio: {audio_path}") # Verbose log
                except Exception as rm_err: st.warning(f"Could not delete temp audio: {audio_path} ({rm_err})")
            if final_video_path and os.path.exists(final_video_path):
                try:
                    os.remove(final_video_path)
                    # st.write(f"Cleaned up temp video: {final_video_path}") # Verbose log
                except Exception as rm_err: st.warning(f"Could not delete temp video: {final_video_path} ({rm_err})")

            # Remove the processed item from the queue
            if st.session_state.generation_queue and st.session_state.generation_queue[0] == job_key_to_process:
                 st.session_state.generation_queue.pop(0)
                 st.session_state.batch_processed_count += 1
            elif st.session_state.generation_queue : # If queue changed unexpectedly, log it
                 st.warning(f"Queue mismatch: Expected {job_key_to_process}, found {st.session_state.generation_queue[0]}")
                 # Decide how to handle: pop anyway? stop? For now, pop to avoid infinite loop.
                 st.session_state.generation_queue.pop(0)
                 st.session_state.batch_processed_count += 1


            # Check if queue is now empty
            if not st.session_state.generation_queue:
                st.session_state.batch_processing_active = False
                st.balloons()
                st.success("🎉 Batch processing finished!")

            # Rerun to process the next item or update UI
            st.rerun()

    # --- Logic for skipping invalid jobs (remains the same, uses job_key_to_process) ---
    elif job_key_to_process in st.session_state.selected_videos:
         st.warning(f"Skipping job {job_key_to_process}. Data missing or invalid Direct URL.", icon="❓")
         st.session_state.selected_videos[job_key_to_process]['Status'] = 'Skipped (Invalid Data)'
         st.session_state.selected_videos[job_key_to_process]['Generation Error'] = 'Skipped - Invalid data or URL'
         # Update Queue and Progress
         if st.session_state.generation_queue: st.session_state.generation_queue.pop(0)
         st.session_state.batch_processed_count += 1
         if not st.session_state.generation_queue:
             st.session_state.batch_processing_active = False
             st.info("Batch processing finished (last item skipped).")
         st.rerun()
    else: # Job key not found
        st.error(f"Job key {job_key_to_process} was in queue but not found in selected videos. Removing from queue.")
        if st.session_state.generation_queue: st.session_state.generation_queue.pop(0)
        st.session_state.batch_processed_count += 1
        if not st.session_state.generation_queue:
            st.session_state.batch_processing_active = False
        st.rerun()


# --- MODIFIED: Display Selected Videos Table (Sidebar) ---
st.sidebar.divider()
st.sidebar.header("Selected & Generated Video Jobs")

if st.session_state.selected_videos:
    selected_list = list(st.session_state.selected_videos.values())
    if selected_list:
        df_selected = pd.DataFrame(selected_list)

        # Define desired columns and order, including Copy Number and Job Key
        display_columns = [
            'Status', 'Video Title', 'Copy Number', 'Topic','Language', 'Script Angle', #'Search Term',
            'Video ID', #'Format Details', # Maybe hide less important details by default
            'yt_dlp_error', 'Generation Error',
            'Generated S3 URL', 'Job Key'
        ]
        # Ensure required columns exist
        for col in display_columns:
             if col not in df_selected.columns:
                  df_selected[col] = pd.NA # Use pandas NA for better handling

        # Fill specific NaN/None values for better display
        df_selected['Status'] = df_selected['Status'].fillna('Unknown')
        df_selected['Copy Number'] = df_selected['Copy Number'].fillna('?')
        # Keep errors as NA if None/NaN, or display the error string
        df_selected['yt_dlp_error'] = df_selected['yt_dlp_error'].fillna('OK')
        df_selected['Generation Error'] = df_selected['Generation Error'].fillna('OK')
        # Fill remaining NAs with a placeholder like "N/A" string for display if needed
        # df_selected.fillna("N/A", inplace=True) # Optional general fill

        # Reorder DataFrame columns for display
        df_selected_display = df_selected[display_columns].copy() # Create copy for display modification

        # Sort by Job Key potentially for better order
        df_selected_display.sort_values(by='Job Key', inplace=True)


        st.sidebar.dataframe(
            df_selected_display, # Display the prepared dataframe
            column_config={
                 "Generated S3 URL": st.column_config.LinkColumn("S3 Link", display_text="View"),
                 "yt_dlp_error": st.column_config.TextColumn("URL Fetch Status", width="small"),
                 "Generation Error": st.column_config.TextColumn("Generation Status", width="small"),
            },
            use_container_width=True,
            hide_index=True
        )

        # Download Button for detailed table (using the display df)
        @st.cache_data
        def convert_df_to_csv(df):
            output = BytesIO()
            df.to_csv(output, index=False, encoding='utf-8')
            return output.getvalue()

        try:
            # Use df_selected_display which includes the ordering and selected columns
            csv_data = convert_df_to_csv(df_selected_display)
            st.sidebar.download_button(
                 label="📥 Download Job Status (CSV)",
                 data=csv_data,
                 file_name='video_generation_job_status.csv', # Updated filename
                 mime='text/csv',
                 use_container_width=True,
                 disabled=st.session_state.batch_processing_active
            )
        except Exception as e:
            st.sidebar.warning(f"Could not generate detailed CSV: {e}")

        # --- Topic Summary DataFrame Section ---
        st.sidebar.divider()
        st.sidebar.subheader("Generated Video Summary by Topic")

        # Generate the summary DataFrame using the helper function
        # This function still works as it aggregates by Topic/Language found in job data
        df_topic_summary = create_topic_summary_dataframe(st.session_state.selected_videos)

        if not df_topic_summary.empty:
            st.sidebar.dataframe(
                 df_topic_summary,
                 use_container_width=True,
                 column_config={
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
                    file_name='generated_videos_by_topic.csv',
                    mime='text/csv',
                    use_container_width=True,
                    disabled=st.session_state.batch_processing_active
                )
            except Exception as e:
                st.sidebar.warning(f"Could not generate summary CSV: {e}")
        else:
            st.sidebar.info("No videos have been successfully generated yet to create a summary.")

    else: # If selected_list is empty
         st.sidebar.info("No video jobs selected yet. Use '➕ Select' buttons in the main area.")

else: # If selected_videos dict is empty
    st.sidebar.info("No video jobs selected yet. Use '➕ Select' buttons in the main area.")

# Footer notes
st.sidebar.caption("Each 'Select' click queues one job. URL Fetch happens after selection. Video Gen uses direct URL.")
