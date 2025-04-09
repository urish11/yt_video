import streamlit as st
import requests
import pandas as pd
import time
import yt_dlp # Import the yt-dlp library

# --- Configuration ---
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3/search"
MAX_RESULTS_PER_QUERY = 3 # Reduce results per query slightly to manage load
YT_DLP_FETCH_TIMEOUT = 20 # Seconds to wait for yt-dlp info extraction

# --- Helper Function: YouTube API Search (Unchanged) ---
def search_youtube(api_key, query, max_results=5):
    """
    Performs a Youtube using the v3 API. (Same as before)
    """
    params = {
        'part': 'snippet',
        'q': query,
        'key': api_key,
        'type': 'video',
        'maxResults': max_results,
        'videoEmbeddable': 'true'
    }
    try:
        response = requests.get(YOUTUBE_API_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        videos = []
        if 'items' in results:
            for item in results['items']:
                if 'videoId' in item['id']:
                    video_id = item['id']['videoId']
                    title = item['snippet']['title']
                    # Use the standard watch URL - yt-dlp and st.video work well with this
                    video_url = f"{video_id}"
                    videos.append({
                        'title': title,
                        'videoId': video_id,
                        'url': video_url # Standard watch/embed URL
                    })
            return videos
        else:
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error for query '{query}': {e}")
        if response is not None:
            if response.status_code == 403:
                 st.error("Received status 403 Forbidden. Check API key/quota.")
                 return None
            elif response.status_code == 400:
                 st.error(f"Received status 400 Bad Request. Details: {response.text}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during search for '{query}': {e}")
        return []

# --- Helper Function: Get Info with yt-dlp ---
def get_yt_dlp_info(video_url):
    """
    Uses yt-dlp to extract video format information, including a direct URL.

    Args:
        video_url (str): The standard YouTube watch URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID).

    Returns:
        dict: A dictionary containing 'direct_url', 'format_details', 'error',
              or None if a critical error occurs.
              Returns {'error': 'message'} if info extraction fails gracefully.
    """
    # Define desired format - try for a decent quality single mp4 file first
    # Adjust 'height<=?720' or 'height<=?1080' as needed
    YDL_OPTS = {
        'format': 'bestvideo[ext=mp4][height<=?720]+bestaudio[ext=m4a]/best[ext=mp4][height<=?720]/best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
        'skip_download': True, # VERY IMPORTANT
        'extract_flat': False, # We need format details for the single video
        # 'forcejson': True, # May help sometimes
        'socket_timeout': YT_DLP_FETCH_TIMEOUT, # Add timeout
    }
    try:
        with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
            info = ydl.extract_info(video_url, download=False)

            direct_url = info.get('url') # The direct URL for the chosen format
            format_note = info.get('format_note')
            format_id = info.get('format_id')
            ext = info.get('ext')
            resolution = info.get('resolution')

            # Sometimes 'url' might not be populated if only format dictionary is returned
            # Fallback by iterating formats if top-level 'url' is missing
            if not direct_url and 'formats' in info:
                 # Try finding the specific format ID yt-dlp chose
                 chosen_format_id = info.get('format_id')
                 found_format = next((f for f in info['formats'] if f.get('format_id') == chosen_format_id), None)
                 if found_format:
                      direct_url = found_format.get('url')


            format_details = f"ID: {format_id}, Res: {resolution}, Note: {format_note}, Ext: {ext}"

            if direct_url:
                 return {
                     'direct_url': direct_url,
                     'format_details': format_details,
                     'error': None
                 }
            else:
                 # Log available formats if URL extraction failed unexpectedly
                 # print("yt-dlp info:", info.get('formats')) # for debugging
                 return {'error': 'Could not extract direct URL for the best format.'}

    except yt_dlp.utils.DownloadError as e:
        st.warning(f"yt-dlp failed for {video_url}: {e}")
        return {'error': str(e)}
    except Exception as e:
        st.error(f"Unexpected yt-dlp error for {video_url}: {e}")
        # Indicate a more severe failure
        return None # Or return {'error': 'Unexpected critical error'}


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("YouTube Video Selector (with yt-dlp info)")
st.caption("Uses YouTube API for search and `yt-dlp` to fetch direct URLs for *selected* videos.")

# --- Session State Initialization ---
if 'selected_videos' not in st.session_state:
    st.session_state.selected_videos = {} # {videoId: video_data}
if 'last_api_key' not in st.session_state:
    st.session_state.last_api_key = ""
if 'last_search_terms' not in st.session_state:
     st.session_state.last_search_terms = ""
if 'search_triggered' not in st.session_state:
     st.session_state.search_triggered = False
if 'api_search_results' not in st.session_state:
     st.session_state.api_search_results = {} # Cache API results per term

# --- Input Area ---
st.sidebar.header("Inputs")
api_key = st.sidebar.text_input("Enter YouTube Data API v3 Key:", type="password", value=st.session_state.last_api_key)
search_terms_input = st.sidebar.text_area(
    "Enter Search Terms (one per line):",
    height=150,
    value=st.session_state.last_search_terms,
    help="Each line triggers a YouTube API search."
)
st.sidebar.info("`yt-dlp` will fetch details *after* you select a video. This may take a few seconds per selection.")
st.sidebar.warning("Direct URLs from `yt-dlp` are often **temporary** and may expire.")

# --- Action Buttons ---
col1, col2 = st.sidebar.columns(2)
search_button = col1.button("Search Videos", use_container_width=True)
clear_button = col2.button("Clear All", use_container_width=True, type="secondary")

if clear_button:
    st.session_state.selected_videos = {}
    st.session_state.search_triggered = False
    st.session_state.api_search_results = {}
    st.success("Selections and search results cleared!")
    st.rerun()

# Store current inputs
st.session_state.last_api_key = api_key
st.session_state.last_search_terms = search_terms_input

# --- Processing and Display Area ---
if search_button:
     st.session_state.search_triggered = True
     st.session_state.api_search_results = {} # Clear previous API results on new search

if st.session_state.search_triggered:
    if not api_key:
        st.sidebar.warning("Please enter your YouTube API Key.")
    elif not search_terms_input:
        st.sidebar.warning("Please enter at least one search term.")
    else:
        search_terms = [term.strip() for term in search_terms_input.split('\n') if term.strip()]
        if not search_terms:
             st.sidebar.warning("Please enter valid search terms.")
        else:
            st.info(f"Searching API for {len(search_terms)} term(s)... (Max {MAX_RESULTS_PER_QUERY} results per term)")
            st.divider()

            api_error_occurred = False
            with st.spinner("Performing API searches..."):
                 for term in search_terms:
                      # Only search API if results not already cached in this session run
                      if term not in st.session_state.api_search_results:
                           videos = search_youtube(api_key, term, MAX_RESULTS_PER_QUERY)
                           if videos is None: # Critical API error
                                st.error(f"Stopping search due to API key/quota issue for term: '{term}'.")
                                api_error_occurred = True
                                break # Stop processing further terms
                           st.session_state.api_search_results[term] = videos
                           time.sleep(0.1) # Small delay between API calls

            if api_error_occurred:
                 st.session_state.search_triggered = False # Reset trigger on critical error
                 st.stop()

            # Display results from cache
            for term, videos in st.session_state.api_search_results.items():
                container = st.container(border=True)
                with container:
                    st.subheader(f"Results for: \"{term}\"")
                    if not videos:
                        st.write("No videos found via API.")
                        continue

                    for video in videos:
                        video_id = video['videoId']
                        video_title = video['title']
                        # Use standard URL for display and yt-dlp input
                        standard_video_url = "https://www.youtube.com/watch?v=" + video['url']
                        unique_key = f"select_{term}_{video_id}"

                        col_vid, col_btn = st.columns([4, 1])

                        with col_vid:
                            st.write(f"**{video_title}**")
                            try:
                                st.video(standard_video_url)
                            except Exception as e:
                                st.warning(f"Could not embed video player: {standard_video_url}. Error: {e}")
                                st.write(f"Link: {standard_video_url}")

                        with col_btn:
                            is_selected = video_id in st.session_state.selected_videos
                            has_dlp_info = is_selected and 'direct_url' in st.session_state.selected_videos[video_id]
                            dlp_error = is_selected and st.session_state.selected_videos[video_id].get('yt_dlp_error')

                            select_button_label = "⏳ Fetching..." if is_selected and not has_dlp_info and not dlp_error else ("✅ Deselect" if is_selected else "➕ Select")
                            select_button_type = "secondary" if is_selected else "primary"
                            select_button_disabled = is_selected and not has_dlp_info and not dlp_error # Disable while fetching

                            if st.button(select_button_label, key=unique_key, type=select_button_type, use_container_width=True, disabled=select_button_disabled):
                                if is_selected:
                                    # Deselect
                                    del st.session_state.selected_videos[video_id]
                                    st.toast(f"Deselected: {video_title}")
                                else:
                                    # Select: Add basic info first
                                    st.session_state.selected_videos[video_id] = {
                                        'Search Term': term,
                                        'Video Title': video_title,
                                        'Video ID': video_id,
                                        'Standard URL': standard_video_url,
                                        'Direct URL': None, # Placeholder
                                        'Format Details': None, # Placeholder
                                        'yt_dlp_error': None # Placeholder
                                    }
                                    st.toast(f"Selected: {video_title}. Fetching details...")
                                    # Trigger rerun to show "Fetching..." and disable button
                                    st.rerun()

                                # Rerun needed to update button states anyway
                                st.rerun()

                            # Show yt-dlp status if selected
                            if is_selected and dlp_error:
                                 st.error(f"yt-dlp Error: {dlp_error}", icon="⚠️")
                            elif has_dlp_info:
                                 st.success("yt-dlp info loaded.", icon="✔️")


                        st.divider()

            # --- yt-dlp Fetching Logic (after rendering initial UI) ---
            # Check if any selected videos need yt-dlp info fetched
            ids_to_fetch = [
                vid for vid, data in st.session_state.selected_videos.items()
                if data.get('Direct URL') is None and data.get('yt_dlp_error') is None
            ]

            if ids_to_fetch:
                fetch_id = ids_to_fetch[0] # Process one at a time per rerun
                video_data = st.session_state.selected_videos[fetch_id]
                standard_url = video_data['Standard URL']
                title = video_data['Video Title']

                with st.spinner(f"Fetching yt-dlp details for '{title}'..."):
                    dlp_info = get_yt_dlp_info(standard_url)

                if dlp_info and dlp_info.get('error') is None:
                    # Update successful fetch
                    st.session_state.selected_videos[fetch_id]['Direct URL'] = dlp_info['direct_url']
                    st.session_state.selected_videos[fetch_id]['Format Details'] = dlp_info['format_details']
                    st.session_state.selected_videos[fetch_id]['yt_dlp_error'] = None # Clear any previous error just in case
                    st.toast(f"yt-dlp details loaded for '{title}'", icon="✅")
                elif dlp_info and dlp_info.get('error'):
                    # Update graceful fetch error
                    st.session_state.selected_videos[fetch_id]['Direct URL'] = "Error"
                    st.session_state.selected_videos[fetch_id]['Format Details'] = "Error"
                    st.session_state.selected_videos[fetch_id]['yt_dlp_error'] = dlp_info['error']
                    st.toast(f"yt-dlp failed for '{title}'", icon="⚠️")
                else:
                    # Update critical fetch error (e.g., unexpected exception)
                    st.session_state.selected_videos[fetch_id]['Direct URL'] = "Error"
                    st.session_state.selected_videos[fetch_id]['Format Details'] = "Error"
                    st.session_state.selected_videos[fetch_id]['yt_dlp_error'] = "Critical yt-dlp failure"
                    st.toast(f"Critical yt-dlp error for '{title}'", icon="🔥")

                # Rerun to process next item or update UI
                st.rerun()


# --- Display Selected Videos Table ---
st.sidebar.divider()
st.sidebar.header("Selected Videos")

if st.session_state.selected_videos:
    selected_list = list(st.session_state.selected_videos.values())

    # Create DataFrame, handling potentially missing dlp columns gracefully
    df_selected = pd.DataFrame(selected_list)
    df_display_columns = [
        'Search Term', 'Video Title', 'Video ID',
        'Direct URL', 'Format Details', 'Standard URL', 'yt_dlp_error'
        ]
    # Ensure all columns exist, fill missing with appropriate placeholder if needed
    for col in df_display_columns:
        if col not in df_selected.columns:
             # Decide placeholder based on column
             df_selected[col] = "N/A" if col not in ['Direct URL', 'Format Details', 'yt_dlp_error'] else None

    # Reorder for display
    df_selected = df_selected[df_display_columns]

    st.sidebar.dataframe(
        df_selected,
        use_container_width=True,
        # Optional: Configure column widths or hiding
        column_config={
            "Direct URL": st.column_config.LinkColumn("Direct URL", display_text="Link"),
            "Standard URL": st.column_config.LinkColumn("Standard URL", display_text="Watch Page"),
            "yt_dlp_error": st.column_config.TextColumn("yt-dlp Status", help="Error message if yt-dlp failed")
        }
        )

    # Option to download selected videos as CSV
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df_selected)

    st.sidebar.download_button(
        label="Download Selected as CSV",
        data=csv,
        file_name='selected_youtube_videos_with_yt-dlp.csv',
        mime='text/csv',
        use_container_width=True
    )

else:
    st.sidebar.info("No videos selected yet. Use '➕ Select' buttons.")

# Footer notes
st.sidebar.caption("API search: 100 quota units per term. yt-dlp fetch happens on select.")
