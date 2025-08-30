from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import tempfile
import shutil
import uuid
from typing import List, Optional
import asyncio
from pathlib import Path
import logging
import aiohttp
import aiofiles
from urllib.parse import urlparse
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FFmpeg Video Generator API",
    description="Create videos from image URLs with optional background music",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
MAX_IMAGES = 10
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}
MAX_URL_LENGTH = 2048

# Assets directory and helper to locate a Poppins font file. We prioritize:
# 1) POPPINS_FONT_PATH env var (absolute path recommended)
# 2) Common filenames in the repo under assets/ or assets/fonts/
ASSETS_DIR = Path("assets")

def get_poppins_font_path() -> Optional[Path]:
    env_path = os.environ.get("POPPINS_FONT_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        ASSETS_DIR / "poppins_semibold.ttf",
        ASSETS_DIR / "Poppins-SemiBold.ttf",
        ASSETS_DIR / "Poppins-SemiBold.otf",
        ASSETS_DIR / "fonts" / "poppins_semibold.ttf",
        ASSETS_DIR / "fonts" / "Poppins-SemiBold.ttf",
        ASSETS_DIR / "fonts" / "Poppins-Regular.ttf",
    ]

    for c in candidates:
        if c.exists():
            return c

    # As a last attempt, scan assets for any file containing 'poppins'
    if ASSETS_DIR.exists():
        for ext in ("*.ttf", "*.otf"):
            for fp in ASSETS_DIR.rglob(ext):
                if "poppins" in fp.name.lower():
                    return fp
    return None

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"FFmpeg check failed: {e}")
        return False

def validate_file_size(file: UploadFile) -> bool:
    """Validate file size"""
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        return False
    return True

def validate_image_format(filename: str) -> bool:
    """Validate image format"""
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_FORMATS

def validate_audio_format(filename: str) -> bool:
    """Validate audio format"""
    return Path(filename).suffix.lower() in SUPPORTED_AUDIO_FORMATS

async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file to destination"""
    try:
        with open(destination, "wb") as buffer:
            content = await upload_file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="File too large")
            buffer.write(content)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

def validate_image_url(url: str) -> bool:
    """Validate image URL format"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False
        if len(url) > MAX_URL_LENGTH:
            return False
        return True
    except Exception:
        return False

async def download_image_from_url(session: aiohttp.ClientSession, url: str, destination: Path) -> bool:
    """Download image from URL and save to destination"""
    try:
        logger.info(f"Downloading image from: {url}")

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                logger.error(f"Failed to download image: HTTP {response.status}")
                return False

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp']):
                logger.error(f"Invalid content type: {content_type}")
                return False

            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_FILE_SIZE:
                logger.error(f"Image too large: {content_length} bytes")
                return False

            # Download and save
            async with aiofiles.open(destination, 'wb') as f:
                total_size = 0
                async for chunk in response.content.iter_chunked(8192):
                    total_size += len(chunk)
                    if total_size > MAX_FILE_SIZE:
                        logger.error(f"Image too large during download: {total_size} bytes")
                        return False
                    await f.write(chunk)

            logger.info(f"Downloaded image: {destination} ({total_size} bytes)")
            return True

    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return False

def get_image_extension_from_url(url: str, content_type: str = None) -> str:
    """Get appropriate image extension from URL or content type"""
    # Try to get extension from URL
    parsed_url = urlparse(url)
    path_ext = Path(parsed_url.path).suffix.lower()

    if path_ext in SUPPORTED_IMAGE_FORMATS:
        return path_ext

    # Fallback to content type
    if content_type:
        extension = mimetypes.guess_extension(content_type)
        if extension and extension.lower() in SUPPORTED_IMAGE_FORMATS:
            return extension.lower()

    # Default fallback
    return '.jpg'

def validate_audio_url(url: str) -> bool:
    """Validate audio URL format"""
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False
        if len(url) > MAX_URL_LENGTH:
            return False
        return True
    except Exception:
        return False

async def download_audio_from_url(session: aiohttp.ClientSession, url: str, destination: Path) -> bool:
    """Download audio from URL and save to destination"""
    try:
        logger.info(f"Downloading audio from: {url}")

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status != 200:
                logger.error(f"Failed to download audio: HTTP {response.status}")
                return False

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(audio_type in content_type for audio_type in ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/m4a', 'audio/aac', 'audio/ogg']):
                logger.error(f"Invalid audio content type: {content_type}")
                return False

            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_FILE_SIZE:
                logger.error(f"Audio too large: {content_length} bytes")
                return False

            # Download and save
            async with aiofiles.open(destination, 'wb') as f:
                total_size = 0
                async for chunk in response.content.iter_chunked(8192):
                    total_size += len(chunk)
                    if total_size > MAX_FILE_SIZE:
                        logger.error(f"Audio too large during download: {total_size} bytes")
                        return False
                    await f.write(chunk)

            logger.info(f"Downloaded audio: {destination} ({total_size} bytes)")
            return True

    except Exception as e:
        logger.error(f"Error downloading audio from {url}: {e}")
        return False

def get_audio_extension_from_url(url: str, content_type: str = None) -> str:
    """Get appropriate audio extension from URL or content type"""
    # Try to get extension from URL
    parsed_url = urlparse(url)
    path_ext = Path(parsed_url.path).suffix.lower()

    if path_ext in SUPPORTED_AUDIO_FORMATS:
        return path_ext

    # Fallback to content type
    if content_type:
        extension = mimetypes.guess_extension(content_type)
        if extension and extension.lower() in SUPPORTED_AUDIO_FORMATS:
            return extension.lower()

    # Default fallback
    return '.mp3'

async def generate_elevenlabs_tts(
    session: aiohttp.ClientSession,
    api_key: str,
    text: str,
    destination: Path,
    voice_id: str = "LcfcDJNUP1GQjkzn1xUU",
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.5,
    similarity_boost: float = 0.7,
    timeout_sec: int = 60
) -> bool:
    """Generate speech using ElevenLabs TTS and save as MP3 to destination.
    Returns True on success, False on failure. Logs detailed info.
    """
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost
            }
        }
        logger.info(f"Requesting TTS from ElevenLabs: voice_id={voice_id}, model={model_id}")
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
            if resp.status != 200:
                # Try to read error text safely
                try:
                    err_text = await resp.text()
                except Exception:
                    err_text = "<no body>"
                logger.error(f"ElevenLabs TTS failed: HTTP {resp.status} - {err_text}")
                return False

            # Save binary audio
            async with aiofiles.open(destination, 'wb') as f:
                total = 0
                async for chunk in resp.content.iter_chunked(8192):
                    total += len(chunk)
                    if total > MAX_FILE_SIZE:
                        logger.error(f"Generated audio too large: {total} bytes")
                        return False
                    await f.write(chunk)

        if not destination.exists() or destination.stat().st_size == 0:
            logger.error("TTS audio file not created or empty")
            return False

        logger.info(f"ElevenLabs TTS saved: {destination} ({destination.stat().st_size} bytes)")
        return True
    except Exception as e:
        logger.error(f"Error generating TTS via ElevenLabs: {e}")
        return False

def create_video_from_images(
    image_paths: List[Path],
    output_path: Path,
    duration_per_image: float = 2.0,
    transition_duration: float = 1.0,
    fps: int = 25,
    text_content: Optional[str] = None,
    second_text_content: Optional[str] = None
) -> bool:
    """Create video from images using FFmpeg"""
    try:
        logger.info(f"Creating video from {len(image_paths)} images")
        logger.info(f"Output path: {output_path}")
        if text_content:
            logger.info(f"Adding first text overlay: {text_content}")
        if second_text_content:
            logger.info(f"Adding second text overlay: {second_text_content}")

        # Verify all input images exist
        for i, img_path in enumerate(image_paths):
            if not img_path.exists():
                logger.error(f"Image {i} does not exist: {img_path}")
                return False
            logger.info(f"Image {i}: {img_path} (size: {img_path.stat().st_size} bytes)")

        # Create a temporary directory for processed images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Using temp directory: {temp_path}")

            if len(image_paths) == 1:
                # Single image - create a portrait video with fade in only
                logger.info("Creating portrait video from single image with fade effects")

                # Build video filter with optional text overlay
                video_filter = (
                    f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                    f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
                    f"fade=t=in:st=0:d={transition_duration}"
                )

                # Add first text overlay if provided
                if text_content:
                    text_filter = (
                        f"drawtext=text='{text_content}':fontsize=72:fontcolor=white:"
                        f"x=(w-text_w)/2:y=h-text_h-100:"
                        f"box=1:boxcolor=black@0.8:boxborderw=25:"
                        f"shadowcolor=black:shadowx=2:shadowy=2:"
                        f"enable='between(t,0,3)'"
                    )
                    video_filter += f",{text_filter}"

                # Add second text overlay if provided
                if second_text_content:
                    video_duration = duration_per_image + transition_duration
                    second_text_filter = (
                        f"drawtext=text='{second_text_content}':fontsize=72:fontcolor=white:"
                        f"x=(w-text_w)/2:y=h-text_h-100:"
                        f"box=1:boxcolor=black@0.8:boxborderw=25:"
                        f"shadowcolor=black:shadowx=2:shadowy=2:"
                        f"enable='between(t,3,{video_duration})'"
                    )
                    video_filter += f",{second_text_filter}"

                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-t", str(duration_per_image + transition_duration),  # Add time for fade in
                    "-i", str(image_paths[0]),
                    "-vf", video_filter,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "ultrafast",
                    "-crf", "28",
                    "-r", str(fps),
                    "-movflags", "+faststart",
                    "-an",  # No audio stream
                    str(output_path)
                ]

                logger.info(f"FFmpeg command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

                if result.returncode != 0:
                    logger.error(f"FFmpeg error (return code {result.returncode}): {result.stderr}")
                    logger.error(f"FFmpeg stdout: {result.stdout}")
                    return False

                # Check output file size
                if not output_path.exists() or output_path.stat().st_size < 1000:
                    logger.error("Output video file is empty or too small.")
                    return False

                logger.info(f"Single image portrait video created successfully: {output_path.stat().st_size} bytes")
                return True

            else:
                # Multiple images - use simpler fade approach instead of complex xfade
                logger.info("Creating portrait slideshow with fade transitions")

                # Calculate total duration correctly for overlapping transitions
                if len(image_paths) == 1:
                    total_duration = duration_per_image + transition_duration
                else:
                    # For multiple images: first image full duration + (remaining images - transition overlaps) + last image fade out
                    total_duration = duration_per_image + ((len(image_paths) - 1) * (duration_per_image - transition_duration)) + transition_duration
                logger.info(f"Expected total video duration: {total_duration} seconds")

                # Create individual videos with fade effects
                temp_videos = []
                for i, img_path in enumerate(image_paths):
                    temp_video = temp_path / f"video_{i:04d}.mp4"
                    temp_videos.append(temp_video)

                    # Create fade effects based on position - fixed timing
                    fade_filters = []

                    if i == 0:  # First image - fade in + fade out
                        fade_filters.append(f"fade=t=in:st=0:d={transition_duration}")
                        if len(image_paths) > 1:
                            fade_filters.append(f"fade=t=out:st={duration_per_image - transition_duration}:d={transition_duration}")
                        video_duration = duration_per_image
                    elif i == len(image_paths) - 1:  # Last image - fade in + extended fade out
                        fade_filters.append(f"fade=t=in:st=0:d={transition_duration}")
                        fade_filters.append(f"fade=t=out:st={duration_per_image - transition_duration}:d={transition_duration}")
                        video_duration = duration_per_image  # No extension needed
                    else:  # Middle images - overlapping fades
                        fade_filters.append(f"fade=t=in:st=0:d={transition_duration}")
                        fade_filters.append(f"fade=t=out:st={duration_per_image - transition_duration}:d={transition_duration}")
                        video_duration = duration_per_image - transition_duration  # Shorter duration to prevent gaps

                    # Build video filter
                    video_filter = f"scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
                    if fade_filters:
                        video_filter += "," + ",".join(fade_filters)

                    # Add text overlay to first video only (first 3 seconds of entire video)
                    if i == 0 and text_content:
                        text_filter = (
                            f"drawtext=text='{text_content}':fontsize=72:fontcolor=white:"
                            f"x=(w-text_w)/2:y=h-text_h-100:"
                            f"box=1:boxcolor=black@0.8:boxborderw=25:"
                            f"shadowcolor=black:shadowx=2:shadowy=2:"
                            f"enable='between(t,0,3)'"
                        )
                        video_filter += f",{text_filter}"

                    # Add second text overlay to last video only (from 3 seconds to end of video)
                    if i == len(image_paths) - 1 and second_text_content:
                        # Calculate when second text should start appearing in the final concatenated video
                        second_text_start = max(3.0, 0.0)  # Start after first text ends
                        second_text_filter = (
                            f"drawtext=text='{second_text_content}':fontsize=72:fontcolor=white:"
                            f"x=(w-text_w)/2:y=h-text_h-100:"
                            f"box=1:boxcolor=black@0.8:boxborderw=25:"
                            f"shadowcolor=black:shadowx=2:shadowy=2:"
                            f"enable='gte(t,0)'"  # Show throughout the last video
                        )
                        video_filter += f",{second_text_filter}"

                    single_cmd = [
                        "ffmpeg", "-y",
                        "-loop", "1",
                        "-t", str(video_duration),
                        "-i", str(img_path),
                        "-vf", video_filter,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-preset", "ultrafast",
                        "-crf", "28",
                        "-r", str(fps),
                        "-an",
                        str(temp_video)
                    ]

                    logger.info(f"Creating video {i+1}/{len(image_paths)} with fade effects (duration: {video_duration}s)")
                    logger.info(f"Filter: {video_filter}")
                    result = subprocess.run(single_cmd, capture_output=True, text=True, timeout=90)
                    if result.returncode != 0:
                        logger.error(f"Error creating video for image {i}: {result.stderr}")
                        return False

                    if not temp_video.exists() or temp_video.stat().st_size < 1000:
                        logger.error(f"Temp video was not created or is too small: {temp_video}")
                        return False

                    logger.info(f"Created temp video: {temp_video} (size: {temp_video.stat().st_size} bytes)")

                # Create concat file for final merge
                concat_file = temp_path / "concat.txt"
                with open(concat_file, 'w') as f:
                    for video in temp_videos:
                        video_path = str(video).replace('\\', '/')
                        f.write(f"file '{video_path}'\n")

                logger.info(f"Created concat file with {len(temp_videos)} videos")

                # Concatenate videos with stream copy for faster processing
                concat_cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_file),
                    "-c", "copy",
                    "-movflags", "+faststart",
                    str(output_path)
                ]

                logger.info(f"Concatenating videos with fade transitions")
                result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=180)
                if result.returncode != 0:
                    logger.error(f"Error concatenating videos: {result.stderr}")
                    return False

                # Check output file size
                if not output_path.exists() or output_path.stat().st_size < 1000:
                    logger.error("Output video file is empty or too small after processing.")
                    return False

                logger.info(f"Portrait slideshow with fade transitions created successfully: {output_path.stat().st_size} bytes")
                return True

    except subprocess.TimeoutExpired as e:
        logger.error(f"FFmpeg command timed out: {e}")
        return False
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def add_audio_to_video(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Add audio track to video"""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-stream_loop", "-1",  # Loop audio indefinitely - moved before audio input
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",  # End when video (longest stream) ends
            "-map", "0:v:0",  # Video from first input
            "-map", "1:a:0",  # Audio from second input
            "-movflags", "+faststart",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Error adding audio: {result.stderr}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error adding audio: {e}")
        return False

def create_video_with_audio_and_text(
    background_image_path: Path,
    main_audio_path: Optional[Path],
    background_music_path: Path,
    output_path: Path,
    text_content: str,
    video_duration: float = 15.0,
    fps: int = 25,
    audio_delay: float = 0.0,
    tail_after_audio: float = 2.0
) -> bool:
    """Create video with background image, mixed audio, and centered text overlay"""
    try:
        logger.info(f"Creating video with background image: {background_image_path}")
        logger.info(f"Main audio: {main_audio_path}")
        logger.info(f"Background music: {background_music_path}")
        logger.info(f"Text content: {text_content}")
        logger.info(
            f"Requested video duration (fallback): {video_duration}s, "
            f"Audio delay: {audio_delay}s, Tail after audio: {tail_after_audio}s"
        )

        # Verify all input files exist
        if not background_image_path.exists():
            logger.error(f"Background image does not exist: {background_image_path}")
            return False
        if main_audio_path is not None and not main_audio_path.exists():
            logger.error(f"Main audio does not exist: {main_audio_path}")
            return False
        if not background_music_path.exists():
            logger.error(f"Background music does not exist: {background_music_path}")
            return False

        # Determine target duration: end 2 seconds after main audio finishes
        def _probe_audio_duration(p: Path) -> Optional[float]:
            try:
                r = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(p)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if r.returncode == 0:
                    val = r.stdout.strip()
                    return float(val) if val else None
            except Exception as e:
                logger.warning(f"ffprobe duration check failed for {p}: {e}")
            return None

        if main_audio_path is None:
            target_duration = 15.0 if video_duration is None else float(video_duration)
            _main_dur = None
        else:
            target_duration = video_duration
            _main_dur = _probe_audio_duration(main_audio_path)
            if _main_dur and _main_dur > 0:
                target_duration = max(0.1, _main_dur + audio_delay + tail_after_audio)
        logger.info(
            f"Computed target video duration: {target_duration:.3f}s "
            f"(main_audio={_main_dur})"
        )

        # Format text with greedy line breaks based on an estimated character width
        # so we can fit as many words as possible while respecting left/right margins.
        words = text_content.split()
        lines = []
        current_line = []

        # Rendering parameters (also used below in drawtext filter)
        font_size = 56
        box_margin = 60  # padding around text inside the box (left+right each)
        min_lr_margin = 2 * box_margin  # total internal margin across width

        # Estimate max characters per line using average glyph width ~= 0.55 * font_size
        # Allowed pixel width for text content
        allowed_px = 1080 - (2 * box_margin)
        approx_char_px = max(1.0, 0.55 * font_size)
        max_chars_per_line = max(10, int(allowed_px / approx_char_px))

        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) <= max_chars_per_line:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Very long word: put it alone on its own line
                    lines.append(word)
        
        if current_line:  # Add remaining words
            lines.append(' '.join(current_line))
        
        # Join lines using real newlines. We will pass this via a text file to avoid
        # complex escaping issues that can cause stray characters like "n" to appear.
        formatted_text = '\n'.join(lines)
        
        logger.info(f"Formatted text: {formatted_text}")

        # Write formatted text to a temporary file and use drawtext=textfile=... .
        # This guarantees proper newlines and spaces rendering.
        tmp_txt = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        try:
            tmp_txt.write(formatted_text)
            tmp_txt.flush()
        finally:
            tmp_txt.close()

        textfile_path = Path(tmp_txt.name).as_posix()

        # Choose Poppins font if available, otherwise fall back to system font by family name
        poppins_fp = get_poppins_font_path()
        if poppins_fp is not None and Path(poppins_fp).exists():
            font_spec = f"fontfile='{Path(poppins_fp).resolve().as_posix()}'"
            logger.info(f"Using Poppins fontfile: {Path(poppins_fp).resolve().as_posix()}")
        else:
            font_spec = "font=Poppins"
            logger.warning("Poppins font not found in assets or POPPINS_FONT_PATH. Falling back to 'Poppins' family name.")

        # Build video filter with background image, scaling, and text overlay (using textfile)
        video_filter = (
            f"scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
            f"drawtext=textfile='{textfile_path}':{font_spec}:"
            f"fontsize={font_size}:"
            f"fontcolor=black:"
            f"x=(w-text_w)/2:"
            f"y=(h-text_h)/2:"
            f"box=1:boxcolor=white@0.45:boxborderw={box_margin}:"  # 50% more transparent
            f"shadowcolor=gray:shadowx=2:shadowy=2:"
            f"line_spacing=10:"
            f"text_align=center:"
            f"text_shaping=1"  # Better rendering for complex scripts and spacing
        )

        # Build audio filter: if no main audio, use background music only at 150% of previous 0.08 -> 0.12
        if main_audio_path is None:
            audio_filter = (
                f"[1:a]volume=0.12[aout]"  # Background music at 12% volume
            )
        else:
            audio_delay_ms = int(audio_delay * 1000)
            audio_filter = (
                f"[1:a]volume=0.12[bg];"  # Increased background music volume to 12%
                f"[2:a]volume=1.2,adelay={audio_delay_ms}[delayed];"
                f"[bg][delayed]amix=inputs=2:duration=longest:dropout_transition=2[mixed]"
            )

        # FFmpeg command with three inputs: background image, background music, main audio
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-t", str(target_duration),
            "-r", str(fps),  # Set input framerate before input
            "-i", str(background_image_path),  # Input 0: Background image
            "-stream_loop", "-1",
            "-i", str(background_music_path),  # Input 1: Background music (looped)
        ]
        if main_audio_path is not None:
            cmd += ["-i", str(main_audio_path)]
        cmd += [
            "-vf", video_filter,
            "-filter_complex", audio_filter,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",  # Faster encoding
            "-crf", "28",  # Slightly lower quality for faster processing
            "-r", str(fps),  # Output framerate
            "-c:a", "aac",
            "-b:a", "128k",  # Standard audio bitrate
            "-ar", "44100",  # Standard audio sample rate
            "-t", str(target_duration),
            "-map", "0:v:0",
            "-map", "[mixed]" if main_audio_path is not None else "[aout]",
            "-movflags", "+faststart",
            str(output_path)
        ]

        logger.info(f"FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Clean up temp text file
        try:
            Path(tmp_txt.name).unlink(missing_ok=True)
        except Exception:
            pass

        if result.returncode != 0:
            logger.error(f"FFmpeg error (return code {result.returncode}): {result.stderr}")
            logger.error(f"FFmpeg stdout: {result.stdout}")
            return False

        # Check output file size
        if not output_path.exists() or output_path.stat().st_size < 1000:
            logger.error("Output video file is empty or too small.")
            return False

        logger.info(f"Video with audio and text created successfully: {output_path.stat().st_size} bytes")
        return True

    except subprocess.TimeoutExpired as e:
        logger.error(f"FFmpeg command timed out: {e}")
        return False
    except Exception as e:
        logger.error(f"Error creating video with audio and text: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

@app.get("/")
async def root():
    """API health check"""
    ffmpeg_available = check_ffmpeg()
    return {
        "message": "FFmpeg Video Generator API",
        "status": "running",
        "ffmpeg_available": ffmpeg_available,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "ffmpeg": check_ffmpeg(),
        "upload_dir": UPLOAD_DIR.exists(),
        "output_dir": OUTPUT_DIR.exists()
    }

@app.post("/create-video")
async def create_video(
    image_urls: List[str] = Form(..., description="List of image URLs"),
    audio: Optional[UploadFile] = File(None, description="Optional audio file"),
    audio_url: Optional[str] = Form(None, description="Optional audio URL"),
    text_content: Optional[str] = Form(None, description="First text to display for first 3 seconds"),
    second_text_content: Optional[str] = Form(None, description="Second text to display from 3 seconds to end"),
    duration_per_image: float = Form(3.0, description="Duration per image in seconds"),
    transition_duration: float = Form(1.0, description="Transition duration in seconds"),
    fps: int = Form(25, description="Output video FPS")
):
    """Create video from image URLs with optional audio and text overlay"""

    # Check FFmpeg availability
    if not check_ffmpeg():
        raise HTTPException(status_code=503, detail="FFmpeg not available")

    # Validate inputs
    if not image_urls or len(image_urls) == 0:
        raise HTTPException(status_code=400, detail="At least one image URL is required")

    if len(image_urls) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_IMAGES} images allowed")

    # Validate that only one audio source is provided
    if audio and audio.filename and audio_url:
        raise HTTPException(status_code=400, detail="Provide either audio file or audio URL, not both")

    # Validate URLs
    for i, url in enumerate(image_urls):
        if not validate_image_url(url):
            raise HTTPException(status_code=400, detail=f"Invalid URL format for image {i+1}")

    if audio_url and not validate_audio_url(audio_url):
        raise HTTPException(status_code=400, detail="Invalid audio URL format")

    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    request_dir = UPLOAD_DIR / request_id
    request_dir.mkdir(exist_ok=True)

    try:
        # Download images from URLs
        image_paths = []
        async with aiohttp.ClientSession() as session:
            for i, url in enumerate(image_urls):
                # Get appropriate extension
                extension = get_image_extension_from_url(url)
                image_path = request_dir / f"image_{i:04d}{extension}"

                # Download image
                success = await download_image_from_url(session, url, image_path)
                if not success:
                    raise HTTPException(status_code=400, detail=f"Failed to download image {i+1} from URL: {url}")

                # Validate downloaded file exists and has content
                if not image_path.exists() or image_path.stat().st_size == 0:
                    raise HTTPException(status_code=400, detail=f"Downloaded image {i+1} is empty or corrupted")

                image_paths.append(image_path)

            # Handle audio if provided via URL
            audio_path = None
            if audio_url:
                # Get appropriate extension
                extension = get_audio_extension_from_url(audio_url)
                audio_path = request_dir / f"audio{extension}"

                # Download audio
                success = await download_audio_from_url(session, audio_url, audio_path)
                if not success:
                    raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {audio_url}")

                # Validate downloaded file exists and has content
                if not audio_path.exists() or audio_path.stat().st_size == 0:
                    raise HTTPException(status_code=400, detail="Downloaded audio is empty or corrupted")

        # Handle audio if provided as uploaded file
        if audio and audio.filename and not audio_url:
            if not validate_file_size(audio):
                raise HTTPException(status_code=413, detail="Audio file is too large")

            if not validate_audio_format(audio.filename):
                raise HTTPException(status_code=400, detail="Audio file has unsupported format")

            audio_path = request_dir / f"audio{Path(audio.filename).suffix}"
            await save_upload_file(audio, audio_path)

        # Create output video
        output_filename = f"video_{request_id}.mp4"
        temp_video_path = OUTPUT_DIR / f"temp_{output_filename}"
        final_video_path = OUTPUT_DIR / output_filename

        # Generate video from images with text overlay
        success = create_video_from_images(
            image_paths,
            temp_video_path if audio_path else final_video_path,
            duration_per_image,
            transition_duration,
            fps,
            text_content,
            second_text_content
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create video from images")

        # Add audio if provided
        if audio_path:
            success = add_audio_to_video(temp_video_path, audio_path, final_video_path)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to add audio to video")

            # Clean up temp video
            if temp_video_path.exists():
                temp_video_path.unlink()

        # Clean up uploaded files
        shutil.rmtree(request_dir, ignore_errors=True)

        if not final_video_path.exists():
            raise HTTPException(status_code=500, detail="Video file was not created")

        # Return success response
        return {
            "message": "Video created successfully",
            "video_id": request_id,
            "download_url": f"/download/{output_filename}",
            "file_size": final_video_path.stat().st_size,
            "images_processed": len(image_paths),
            "audio_added": audio_path is not None,
            "audio_source": "url" if audio_url else ("file" if audio and audio.filename else None),
            "text_added": text_content is not None,
            "second_text_added": second_text_content is not None
        }

    except HTTPException:
        # Clean up on error
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
    except Exception as e:
        # Clean up on error
        shutil.rmtree(request_dir, ignore_errors=True)
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/create-audio-video")
async def create_audio_video(
    background_image_url: str = Form(..., description="Background image URL"),
    background_music: Optional[UploadFile] = File(None, description="Background music file"),
    background_music_url: Optional[str] = Form(None, description="Background music URL"),
    text_content: str = Form(..., description="Text content to display (5-20 words)"),
    video_duration: float = Form(15.0, description="Video duration in seconds (default 15)"),
    fps: int = Form(25, description="Output video FPS")
):
    """Create video with background image, background music only (no main audio), and centered text overlay."""
    
    # Check FFmpeg availability
    if not check_ffmpeg():
        raise HTTPException(status_code=503, detail="FFmpeg not available")

    # Validate inputs
    if not background_image_url:
        raise HTTPException(status_code=400, detail="Background image URL is required")
    
    if not validate_image_url(background_image_url):
        raise HTTPException(status_code=400, detail="Invalid background image URL format")

    if not text_content or len(text_content.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text content is required")

    # Validate text length (should be 5-20 words)
    word_count = len(text_content.strip().split())
    if word_count < 5 or word_count > 20:
        raise HTTPException(status_code=400, detail="Text content should be between 5-20 words")

    # Background music required
    if not ((background_music and background_music.filename) or background_music_url):
        raise HTTPException(status_code=400, detail="Background music (file or URL) is required")

    # Validate that only one source per audio type is provided
    if (background_music and background_music.filename) and background_music_url:
        raise HTTPException(status_code=400, detail="Provide either background music file or URL, not both")

    # Validate audio URLs if provided
    if background_music_url and not validate_audio_url(background_music_url):
        raise HTTPException(status_code=400, detail="Invalid background music URL format")

    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    request_dir = UPLOAD_DIR / request_id
    request_dir.mkdir(exist_ok=True)

    try:
        # Download background image
        async with aiohttp.ClientSession() as session:
            # Get background image
            bg_extension = get_image_extension_from_url(background_image_url)
            bg_image_path = request_dir / f"background{bg_extension}"
            
            success = await download_image_from_url(session, background_image_url, bg_image_path)
            if not success:
                raise HTTPException(status_code=400, detail=f"Failed to download background image from URL: {background_image_url}")

            if not bg_image_path.exists() or bg_image_path.stat().st_size == 0:
                raise HTTPException(status_code=400, detail="Downloaded background image is empty or corrupted")

            # Handle background music
            bg_music_path = None
            if background_music_url:
                # Download from URL
                music_extension = get_audio_extension_from_url(background_music_url)
                bg_music_path = request_dir / f"background_music{music_extension}"
                
                success = await download_audio_from_url(session, background_music_url, bg_music_path)
                if not success:
                    raise HTTPException(status_code=400, detail=f"Failed to download background music from URL: {background_music_url}")

                if not bg_music_path.exists() or bg_music_path.stat().st_size == 0:
                    raise HTTPException(status_code=400, detail="Downloaded background music is empty or corrupted")

        # No main audio: always use background music only

        # Handle uploaded background music file
        if background_music and background_music.filename and not background_music_url:
            if not validate_file_size(background_music):
                raise HTTPException(status_code=413, detail="Background music file is too large")

            if not validate_audio_format(background_music.filename):
                raise HTTPException(status_code=400, detail="Background music file has unsupported format")

            bg_music_path = request_dir / f"background_music{Path(background_music.filename).suffix}"
            await save_upload_file(background_music, bg_music_path)

        # Create output video
        output_filename = f"audio_video_{request_id}.mp4"
        final_video_path = OUTPUT_DIR / output_filename

        # Generate video with background image, mixed audio, and text overlay
        success = create_video_with_audio_and_text(
            bg_image_path,
            None,
            bg_music_path,
            final_video_path,
            text_content,
            15.0,
            fps,
            0.0
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create video with audio and text")

        # Clean up uploaded files
        shutil.rmtree(request_dir, ignore_errors=True)

        if not final_video_path.exists():
            raise HTTPException(status_code=500, detail="Video file was not created")

        # Return success response
        return {
            "message": "Audio video created successfully",
            "video_id": request_id,
            "download_url": f"/download/{output_filename}",
            "file_size": final_video_path.stat().st_size,
            "video_duration": 15.0,
            "audio_delay": 0.0,
            "text_content": text_content,
            "background_image_source": "url",
            "main_audio_source": None,
            "background_music_source": "url" if background_music_url else "file"
        }

    except HTTPException:
        # Clean up on error
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
    except Exception as e:
        # Clean up on error
        shutil.rmtree(request_dir, ignore_errors=True)
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download generated video"""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    )

@app.delete("/cleanup/{video_id}")
async def cleanup_video(video_id: str):
    """Clean up generated video file"""
    file_path = OUTPUT_DIR / f"video_{video_id}.mp4"

    if file_path.exists():
        file_path.unlink()
        return {"message": "Video deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Video not found")

@app.get("/list-videos")
async def list_videos():
    """List all generated videos"""
    videos = []
    for file_path in OUTPUT_DIR.glob("video_*.mp4"):
        videos.append({
            "filename": file_path.name,
            "size": file_path.stat().st_size,
            "created": file_path.stat().st_ctime,
            "download_url": f"/download/{file_path.name}"
        })

    return {"videos": videos, "count": len(videos)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)