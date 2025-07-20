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
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Video format enum
class VideoFormat(str, Enum):
    LANDSCAPE = "landscape"  # 16:9 - 1280x720
    SQUARE = "square"        # 1:1 - 1080x1080
    REEL = "reel"           # 9:16 - 1080x1920
    STORY = "story"         # 9:16 - 1080x1920 (same as reel)

app = FastAPI(
    title="FFmpeg Video Generator API",
    description="Create videos from images with optional background music - supports multiple formats including Reels",
    version="2.0.0"
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

# Video format configurations
VIDEO_FORMATS = {
    VideoFormat.LANDSCAPE: {
        "width": 1280,
        "height": 720,
        "description": "Landscape 16:9 format"
    },
    VideoFormat.SQUARE: {
        "width": 1080,
        "height": 1080,
        "description": "Square 1:1 format for Instagram posts"
    },
    VideoFormat.REEL: {
        "width": 1080,
        "height": 1920,
        "description": "Vertical 9:16 format for Instagram/TikTok Reels"
    },
    VideoFormat.STORY: {
        "width": 1080,
        "height": 1920,
        "description": "Vertical 9:16 format for Stories"
    }
}

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

def get_video_filter(format_type: VideoFormat, blur_background: bool = False) -> str:
    """Get FFmpeg video filter string based on format"""
    config = VIDEO_FORMATS[format_type]
    width = config["width"]
    height = config["height"]
    
    if format_type == VideoFormat.REEL or format_type == VideoFormat.STORY:
        if blur_background:
            # For reel format with blurred background
            return (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease:eval=frame,"
                   f"boxblur=10:1,setsar=1[bg];"
                   f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease:eval=frame,"
                   f"setsar=1[fg];"
                   f"[bg][fg]overlay=(W-w)/2:(H-h)/2")
        else:
            # Standard reel format with padding
            return (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")
    elif format_type == VideoFormat.SQUARE:
        if blur_background:
            return (f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease:eval=frame,"
                   f"boxblur=10:1,setsar=1[bg];"
                   f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease:eval=frame,"
                   f"setsar=1[fg];"
                   f"[bg][fg]overlay=(W-w)/2:(H-h)/2")
        else:
            return (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")
    else:  # LANDSCAPE
        return (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")

def create_video_from_images(
    image_paths: List[Path], 
    output_path: Path, 
    duration_per_image: float = 2.0,
    transition_duration: float = 0.5,
    fps: int = 25,
    format_type: VideoFormat = VideoFormat.LANDSCAPE,
    blur_background: bool = False,
    background_color: str = "black"
) -> bool:
    """Create video from images using FFmpeg with support for different formats"""
    try:
        # Create a temporary directory for processed images
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if len(image_paths) == 1:
                # Single image - create a short video
                video_filter = get_video_filter(format_type, blur_background).replace("color=black", f"color={background_color}")
                video_filter += f",fps={fps}"
                
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", str(image_paths[0]),
                    "-t", str(duration_per_image),
                    "-vf", video_filter,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "medium",  # Better quality for reels
                    "-crf", "18",  # Higher quality for mobile viewing
                    "-movflags", "+faststart",  # Optimize for web streaming
                    str(output_path)
                ]
            else:
                # Multiple images - create slideshow with transitions
                temp_videos = []
                
                for i, img_path in enumerate(image_paths):
                    temp_video = temp_path / f"video_{i:04d}.mp4"
                    temp_videos.append(temp_video)
                    
                    video_filter = get_video_filter(format_type, blur_background).replace("color=black", f"color={background_color}")
                    video_filter += f",fps={fps}"
                    
                    single_cmd = [
                        "ffmpeg", "-y",
                        "-loop", "1",
                        "-i", str(img_path),
                        "-t", str(duration_per_image),
                        "-vf", video_filter,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-preset", "medium",
                        "-crf", "18",
                        "-movflags", "+faststart",
                        str(temp_video)
                    ]
                    
                    result = subprocess.run(single_cmd, capture_output=True, text=True, timeout=120)
                    if result.returncode != 0:
                        logger.error(f"Error creating video for image {i}: {result.stderr}")
                        return False
                
                # Create smooth transitions between videos if requested
                if transition_duration > 0 and len(temp_videos) > 1:
                    # Create videos with crossfade transitions
                    filter_complex = ""
                    inputs = []
                    
                    for i, video in enumerate(temp_videos):
                        inputs.extend(["-i", str(video)])
                    
                    # Build filter for crossfade transitions
                    filter_parts = []
                    for i in range(len(temp_videos) - 1):
                        if i == 0:
                            filter_parts.append(f"[0][1]xfade=transition=fade:duration={transition_duration}:offset={duration_per_image-transition_duration}[v01]")
                        else:
                            filter_parts.append(f"[v{i-1:02d}{i:02d}][{i+1}]xfade=transition=fade:duration={transition_duration}:offset={i*(duration_per_image-transition_duration)+duration_per_image-transition_duration}[v{i:02d}{i+1:02d}]")
                    
                    if len(filter_parts) > 0:
                        filter_complex = ";".join(filter_parts)
                        
                        concat_cmd = [
                            "ffmpeg", "-y"
                        ] + inputs + [
                            "-filter_complex", filter_complex,
                            "-map", f"[v{len(temp_videos)-2:02d}{len(temp_videos)-1:02d}]",
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-preset", "medium",
                            "-crf", "18",
                            "-movflags", "+faststart",
                            str(output_path)
                        ]
                    else:
                        # Fallback to simple concatenation
                        concat_file = temp_path / "concat.txt"
                        with open(concat_file, 'w') as f:
                            for video in temp_videos:
                                f.write(f"file '{video}'\n")
                        
                        concat_cmd = [
                            "ffmpeg", "-y",
                            "-f", "concat",
                            "-safe", "0",
                            "-i", str(concat_file),
                            "-c", "copy",
                            str(output_path)
                        ]
                else:
                    # Simple concatenation without transitions
                    concat_file = temp_path / "concat.txt"
                    with open(concat_file, 'w') as f:
                        for video in temp_videos:
                            f.write(f"file '{video}'\n")
                    
                    concat_cmd = [
                        "ffmpeg", "-y",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", str(concat_file),
                        "-c", "copy",
                        str(output_path)
                    ]
                
                result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    logger.error(f"Error concatenating videos: {result.stderr}")
                    return False
                
                return True
            
            # Execute the command for single image
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
            
            return True
            
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        return False

def add_audio_to_video(video_path: Path, audio_path: Path, output_path: Path, fade_in: float = 0.5, fade_out: float = 0.5) -> bool:
    """Add audio track to video with optional fade effects"""
    try:
        audio_filter = f"afade=t=in:st=0:d={fade_in},afade=t=out:st=-{fade_out}:d={fade_out}"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",  # Good audio bitrate for mobile
            "-af", audio_filter,
            "-shortest",  # End when shortest stream ends
            "-map", "0:v:0",  # Video from first input
            "-map", "1:a:0",  # Audio from second input
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
        "message": "FFmpeg Video Generator API - Now with Reel Support!",
        "status": "running",
        "ffmpeg_available": ffmpeg_available,
        "version": "2.0.0",
        "supported_formats": list(VIDEO_FORMATS.keys())
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "ffmpeg": check_ffmpeg(),
        "upload_dir": UPLOAD_DIR.exists(),
        "output_dir": OUTPUT_DIR.exists(),
        "supported_formats": VIDEO_FORMATS
    }

@app.get("/formats")
async def get_formats():
    """Get available video formats"""
    return {
        "formats": {
            format_type: {
                **config,
                "aspect_ratio": f"{config['width']}:{config['height']}"
            }
            for format_type, config in VIDEO_FORMATS.items()
        }
    }

@app.post("/create-video")
async def create_video(
    images: List[UploadFile] = File(..., description="List of image files"),
    audio: Optional[UploadFile] = File(None, description="Optional audio file"),
    duration_per_image: float = Form(2.0, description="Duration per image in seconds"),
    transition_duration: float = Form(0.0, description="Transition duration in seconds (0 for no transitions)"),
    fps: int = Form(25, description="Output video FPS"),
    format_type: VideoFormat = Form(VideoFormat.REEL, description="Video format type"),
    blur_background: bool = Form(False, description="Use blurred background for better fit"),
    background_color: str = Form("black", description="Background color (black, white, etc.)"),
    audio_fade_in: float = Form(0.5, description="Audio fade in duration"),
    audio_fade_out: float = Form(0.5, description="Audio fade out duration")
):
    """Create video from uploaded images with optional audio - Now supports Reel format!"""
    
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
        format_suffix = format_type.value
        output_filename = f"video_{format_suffix}_{request_id}.mp4"
        temp_video_path = OUTPUT_DIR / f"temp_{output_filename}"
        final_video_path = OUTPUT_DIR / output_filename
        
        # Generate video from images
        success = create_video_from_images(
            image_paths, 
            temp_video_path if audio_path else final_video_path,
            duration_per_image,
            transition_duration,
            fps,
            format_type,
            blur_background,
            background_color
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create video from images")
        
        # Add audio if provided
        if audio_path:
            success = add_audio_to_video(temp_video_path, audio_path, final_video_path, audio_fade_in, audio_fade_out)
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
        format_config = VIDEO_FORMATS[format_type]
        return {
            "message": "Video created successfully",
            "video_id": request_id,
            "download_url": f"/download/{output_filename}",
            "file_size": final_video_path.stat().st_size,
            "images_processed": len(image_paths),
            "audio_added": audio_path is not None,
            "format": {
                "type": format_type,
                "dimensions": f"{format_config['width']}x{format_config['height']}",
                "description": format_config['description']
            },
            "settings": {
                "duration_per_image": duration_per_image,
                "transition_duration": transition_duration,
                "fps": fps,
                "blur_background": blur_background,
                "background_color": background_color
            }
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
    # Handle both old and new filename formats
    patterns = [
        f"video_{video_id}.mp4",  # Old format
        f"video_*_{video_id}.mp4"  # New format with format type
    ]
    
    deleted = False
    for pattern in patterns:
        for file_path in OUTPUT_DIR.glob(pattern):
            file_path.unlink()
            deleted = True
    
    if deleted:
        return {"message": "Video deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Video not found")

@app.get("/list-videos")
async def list_videos():
    """List all generated videos"""
    videos = []
    for file_path in OUTPUT_DIR.glob("video_*.mp4"):
        filename_parts = file_path.stem.split("_")
        format_type = "unknown"
        if len(filename_parts) >= 3 and filename_parts[1] in [f.value for f in VideoFormat]:
            format_type = filename_parts[1]
        
        videos.append({
            "filename": file_path.name,
            "format": format_type,
            "size": file_path.stat().st_size,
            "created": file_path.stat().st_ctime,
            "download_url": f"/download/{file_path.name}"
        })
    
    return {"videos": videos, "count": len(videos)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
