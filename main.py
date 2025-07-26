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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FFmpeg Video Generator API",
    description="Create videos from images with optional background music",
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

def create_video_from_images(
    image_paths: List[Path],
    output_path: Path,
    duration_per_image: float = 2.0,
    transition_duration: float = 0.5,
    fps: int = 25
) -> bool:
    """Create video from images using FFmpeg"""
    try:
        logger.info(f"Creating video from {len(image_paths)} images")
        logger.info(f"Output path: {output_path}")

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
                # Single image - create a portrait video with fade in/out effects
                logger.info("Creating portrait video from single image with fade effects")
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-t", str(duration_per_image + transition_duration * 2),  # Add time for fade in/out
                    "-i", str(image_paths[0]),
                    "-vf", (
                        f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                        f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
                        f"fade=in:0:{int(fps * transition_duration)},"
                        f"fade=out:{int(fps * duration_per_image)}:{int(fps * transition_duration)}"
                    ),
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
                # Multiple images - create portrait slideshow with crossfade transitions
                logger.info("Creating portrait slideshow with crossfade transitions")

                # Create individual videos for each image with proper duration for crossfading
                temp_videos = []
                for i, img_path in enumerate(image_paths):
                    temp_video = temp_path / f"video_{i:04d}.mp4"
                    temp_videos.append(temp_video)

                    # Calculate duration: base duration + transition time
                    video_duration = duration_per_image
                    if i == 0:  # First image gets fade in
                        video_duration += transition_duration
                    if i == len(image_paths) - 1:  # Last image gets fade out
                        video_duration += transition_duration

                    single_cmd = [
                        "ffmpeg", "-y",
                        "-loop", "1",
                        "-t", str(video_duration),
                        "-i", str(img_path),
                        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-preset", "ultrafast",
                        "-crf", "28",
                        "-r", str(fps),
                        "-an",
                        str(temp_video)
                    ]

                    logger.info(f"Creating video {i+1}/{len(image_paths)} for crossfade")
                    result = subprocess.run(single_cmd, capture_output=True, text=True, timeout=90)
                    if result.returncode != 0:
                        logger.error(f"Error creating video for image {i}: {result.stderr}")
                        return False

                    if not temp_video.exists() or temp_video.stat().st_size < 1000:
                        logger.error(f"Temp video was not created or is too small: {temp_video}")
                        return False

                    logger.info(f"Created temp video: {temp_video} (size: {temp_video.stat().st_size} bytes)")

                # Create complex FFmpeg filter for crossfade transitions
                if len(temp_videos) == 2:
                    # Simple case: two videos with crossfade
                    crossfade_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(temp_videos[0]),
                        "-i", str(temp_videos[1]),
                        "-filter_complex", (
                            f"[0]fade=out:{int(fps * duration_per_image)}:{int(fps * transition_duration)}[v0];"
                            f"[1]fade=in:0:{int(fps * transition_duration)}[v1];"
                            f"[v0][v1]overlay=0:0"
                        ),
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-preset", "ultrafast",
                        "-crf", "28",
                        "-movflags", "+faststart",
                        str(output_path)
                    ]
                else:
                    # Multiple videos: create crossfade chain
                    filter_complex = []
                    overlay_inputs = []

                    # Add fade effects to first and last videos
                    filter_complex.append(f"[0]fade=in:0:{int(fps * transition_duration)}[v0fade]")
                    filter_complex.append(f"[{len(temp_videos)-1}]fade=out:{int(fps * duration_per_image)}:{int(fps * transition_duration)}[v{len(temp_videos)-1}fade]")

                    # Create crossfade transitions between consecutive videos
                    for i in range(len(temp_videos) - 1):
                        if i == 0:
                            input1 = "v0fade"
                        else:
                            input1 = f"v{i}"

                        if i == len(temp_videos) - 2:  # Last transition
                            input2 = f"v{i+1}fade"
                        else:
                            input2 = f"{i+1}"

                        # Calculate crossfade timing
                        fade_start = int(fps * (duration_per_image - transition_duration))

                        filter_complex.append(
                            f"[{input1}][{input2}]xfade=transition=fade:duration={transition_duration}:offset={duration_per_image - transition_duration}[v{i+1}]"
                        )

                    crossfade_cmd = [
                        "ffmpeg", "-y"
                    ]

                    # Add all input videos
                    for video in temp_videos:
                        crossfade_cmd.extend(["-i", str(video)])

                    crossfade_cmd.extend([
                        "-filter_complex", ";".join(filter_complex),
                        "-map", f"[v{len(temp_videos)-1}]",
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-preset", "ultrafast",
                        "-crf", "28",
                        "-movflags", "+faststart",
                        str(output_path)
                    ])

                logger.info(f"Creating video with crossfade transitions")
                result = subprocess.run(crossfade_cmd, capture_output=True, text=True, timeout=300)

                if result.returncode != 0:
                    logger.error(f"Crossfade creation failed: {result.stderr}")

                    # Fallback to simple concatenation without transitions
                    logger.info("Falling back to simple concatenation")
                    concat_file = temp_path / "concat.txt"
                    with open(concat_file, 'w') as f:
                        for video in temp_videos:
                            video_path = str(video).replace('\\', '/')
                            f.write(f"file '{video_path}'\n")

                    concat_cmd = [
                        "ffmpeg", "-y",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", str(concat_file),
                        "-c", "copy",
                        "-movflags", "+faststart",
                        str(output_path)
                    ]

                    logger.info(f"Concatenating portrait videos")
                    result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=180)
                    if result.returncode != 0:
                        logger.error(f"Error concatenating videos: {result.stderr}")
                        return False

                # Check output file size
                if not output_path.exists() or output_path.stat().st_size < 1000:
                    logger.error("Output video file is empty or too small after processing.")
                    return False

                logger.info(f"Portrait slideshow with transitions created successfully: {output_path.stat().st_size} bytes")
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
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",  # End when shortest stream ends
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
    images: List[UploadFile] = File(..., description="List of image files"),
    audio: Optional[UploadFile] = File(None, description="Optional audio file"),
    duration_per_image: float = Form(2.0, description="Duration per image in seconds"),
    transition_duration: float = Form(0.5, description="Transition duration in seconds"),
    fps: int = Form(25, description="Output video FPS")
):
    """Create video from uploaded images with optional audio"""

    # Check FFmpeg availability
    if not check_ffmpeg():
        raise HTTPException(status_code=503, detail="FFmpeg not available")

    # Validate inputs
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required")

    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_IMAGES} images allowed")

    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    request_dir = UPLOAD_DIR / request_id
    request_dir.mkdir(exist_ok=True)

    try:
        # Validate and save images
        image_paths = []
        for i, image in enumerate(images):
            # Validate file
            if not validate_file_size(image):
                raise HTTPException(status_code=413, detail=f"Image {i+1} is too large")

            if not validate_image_format(image.filename):
                raise HTTPException(status_code=400, detail=f"Image {i+1} has unsupported format")

            # Save image
            image_path = request_dir / f"image_{i:04d}{Path(image.filename).suffix}"
            await save_upload_file(image, image_path)
            image_paths.append(image_path)

        # Handle audio if provided
        audio_path = None
        if audio and audio.filename:
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

        # Generate video from images
        success = create_video_from_images(
            image_paths,
            temp_video_path if audio_path else final_video_path,
            duration_per_image,
            transition_duration,
            fps
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
            "audio_added": audio_path is not None
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