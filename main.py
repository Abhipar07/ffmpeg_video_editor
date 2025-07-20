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
    title="FFmpeg Reel Video Generator API",
    description="Create reel format videos from images with smooth transitions and optional background music",
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
MAX_IMAGES = 30  # Increased from 10 to 30
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}

# Reel format dimensions (9:16 aspect ratio)
REEL_WIDTH = 1080
REEL_HEIGHT = 1920

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

def create_reel_video_with_transitions(
    image_paths: List[Path], 
    output_path: Path, 
    duration_per_image: float = 3.0,
    transition_duration: float = 1.0,
    transition_type: str = "crossfade",
    fps: int = 30
) -> bool:
    """Create reel format video from images with smooth transitions using FFmpeg"""
    try:
        logger.info(f"Creating reel video with {len(image_paths)} images")
        
        if len(image_paths) == 1:
            # Single image - create a short video
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(image_paths[0]),
                "-t", str(duration_per_image),
                "-vf", f"scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=decrease,pad={REEL_WIDTH}:{REEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,fps={fps}",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",  # Changed from medium to fast for better compatibility
                "-crf", "23",       # Changed from 20 to 23 for smaller file size
                str(output_path)
            ]
            
            logger.info(f"Single image command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"FFmpeg error for single image: {result.stderr}")
                return False
            return True
            
        else:
            # Multiple images - use simpler approach that's more reliable
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                logger.info(f"Using temp directory: {temp_path}")
                
                if transition_type == "crossfade" and len(image_paths) <= 10:
                    # Use direct crossfade for smaller number of images
                    return create_crossfade_video(image_paths, output_path, duration_per_image, transition_duration, fps, temp_path)
                else:
                    # Use concatenation method for larger number of images or other transitions
                    return create_concat_video(image_paths, output_path, duration_per_image, fps, temp_path)
            
    except Exception as e:
        logger.error(f"Error creating reel video: {e}")
        return False

def create_simple_reel_video(
    image_paths: List[Path], 
    output_path: Path, 
    duration_per_image: float,
    fps: int
) -> bool:
    """Fallback method: Create simple reel video without transitions"""
    try:
        logger.info("Using simple fallback method for video creation")
        
        if len(image_paths) == 1:
            # Single image
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(image_paths[0]),
                "-t", str(duration_per_image),
                "-vf", f"scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=decrease,pad={REEL_WIDTH}:{REEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black",
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "28",
                str(output_path)
            ]
        else:
            # Multiple images using image2 demuxer
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy images with sequential naming
                for i, img_path in enumerate(image_paths):
                    ext = img_path.suffix
                    new_path = temp_path / f"img{i:05d}{ext}"
                    shutil.copy2(img_path, new_path)
                
                # Create image list file
                img_list_file = temp_path / "images.txt"
                with open(img_list_file, 'w') as f:
                    for i in range(len(image_paths)):
                        ext = image_paths[i].suffix
                        f.write(f"file 'img{i:05d}{ext}'\n")
                        f.write(f"duration {duration_per_image}\n")
                    # Duplicate last image for proper duration
                    if len(image_paths) > 0:
                        ext = image_paths[-1].suffix
                        f.write(f"file 'img{len(image_paths)-1:05d}{ext}'\n")
                
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(img_list_file),
                    "-vf", f"scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=decrease,pad={REEL_WIDTH}:{REEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black",
                    "-r", str(fps),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "ultrafast",
                    "-crf", "28",
                    str(output_path)
                ]
        
        logger.info(f"Fallback command: {' '.join(cmd[:10])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"Fallback method also failed: {result.stderr}")
            return False
        
        logger.info("Fallback method succeeded")
        return True
        
    except Exception as e:
        logger.error(f"Error in simple reel video creation: {e}")
        return False

def create_crossfade_video(
    image_paths: List[Path], 
    output_path: Path, 
    duration_per_image: float,
    transition_duration: float,
    fps: int,
    temp_path: Path
) -> bool:
    """Create video with crossfade transitions (for smaller number of images)"""
    try:
        # Create input arguments
        input_args = []
        for img_path in image_paths:
            input_args.extend([
                "-loop", "1",
                "-t", str(duration_per_image),
                "-i", str(img_path)
            ])
        
        # Build filter complex for crossfade
        filter_parts = []
        
        # Scale all inputs first
        for i in range(len(image_paths)):
            filter_parts.append(f"[{i}:v]scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=decrease,pad={REEL_WIDTH}:{REEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,fps={fps},settb=1/30[v{i}]")
        
        # Create crossfade chain
        if len(image_paths) == 2:
            filter_parts.append(f"[v0][v1]xfade=transition=fade:duration={transition_duration}:offset={duration_per_image-transition_duration}[out]")
        else:
            # Chain multiple crossfades
            current_label = "v0"
            for i in range(1, len(image_paths)):
                offset = duration_per_image * i - transition_duration * i
                if i == 1:
                    filter_parts.append(f"[{current_label}][v{i}]xfade=transition=fade:duration={transition_duration}:offset={offset}[tmp{i}]")
                    current_label = f"tmp{i}"
                elif i == len(image_paths) - 1:
                    filter_parts.append(f"[{current_label}][v{i}]xfade=transition=fade:duration={transition_duration}:offset={offset}[out]")
                else:
                    filter_parts.append(f"[{current_label}][v{i}]xfade=transition=fade:duration={transition_duration}:offset={offset}[tmp{i}]")
                    current_label = f"tmp{i}"
        
        filter_complex = ";".join(filter_parts)
        
        cmd = [
            "ffmpeg", "-y"
        ] + input_args + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            str(output_path)
        ]
        
        logger.info(f"Crossfade command: {' '.join(cmd[:10])}...")  # Log first few parts
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f"Crossfade error: {result.stderr}")
            return False
        return True
        
    except Exception as e:
        logger.error(f"Error in crossfade video creation: {e}")
        return False

def create_concat_video(
    image_paths: List[Path], 
    output_path: Path, 
    duration_per_image: float,
    fps: int,
    temp_path: Path
) -> bool:
    """Create video using concatenation method (more reliable for many images)"""
    try:
        # Create individual videos for each image
        temp_videos = []
        for i, img_path in enumerate(image_paths):
            temp_video = temp_path / f"video_{i:04d}.mp4"
            temp_videos.append(temp_video)
            
            # Create individual video clip
            single_cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(img_path),
                "-t", str(duration_per_image),
                "-vf", f"scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=decrease,pad={REEL_WIDTH}:{REEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,fps={fps}",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
                str(temp_video)
            ]
            
            logger.info(f"Creating video {i+1}/{len(image_paths)}")
            result = subprocess.run(single_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"Error creating video for image {i}: {result.stderr}")
                return False
        
        # Create concat file
        concat_file = temp_path / "concat.txt"
        with open(concat_file, 'w') as f:
            for video in temp_videos:
                f.write(f"file '{video.absolute()}'\n")
        
        # Concatenate videos
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path)
        ]
        
        logger.info("Concatenating videos")
        result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Error concatenating videos: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in concat video creation: {e}")
        return False

def add_audio_to_video(video_path: Path, audio_path: Path, output_path: Path, fade_audio: bool = True) -> bool:
    """Add audio track to video with optional fade effects"""
    try:
        # Build audio filter for smooth fade in/out
        audio_filter = "volume=0.8"  # Slightly lower volume
        if fade_audio:
            audio_filter += ",afade=t=in:ss=0:d=1,afade=t=out:st=-1:d=1"  # 1 second fade in/out
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
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
        "message": "FFmpeg Reel Video Generator API",
        "status": "running",
        "ffmpeg_available": ffmpeg_available,
        "version": "2.0.0",
        "max_images": MAX_IMAGES,
        "output_format": f"{REEL_WIDTH}x{REEL_HEIGHT} (9:16 reel format)"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "ffmpeg": check_ffmpeg(),
        "upload_dir": UPLOAD_DIR.exists(),
        "output_dir": OUTPUT_DIR.exists(),
        "max_images_supported": MAX_IMAGES,
        "reel_dimensions": f"{REEL_WIDTH}x{REEL_HEIGHT}"
    }

@app.post("/create-reel")
async def create_reel_video(
    images: List[UploadFile] = File(..., description="List of image files (max 30)"),
    audio: Optional[UploadFile] = File(None, description="Optional audio file"),
    duration_per_image: float = Form(3.0, description="Duration per image in seconds"),
    transition_duration: float = Form(1.0, description="Transition duration in seconds"),
    transition_type: str = Form("crossfade", description="Transition type (crossfade, fade, etc.)"),
    fps: int = Form(30, description="Output video FPS"),
    fade_audio: bool = Form(True, description="Apply fade in/out to audio")
):
    """Create reel format video from uploaded images with smooth transitions and optional audio"""
    
    # Check FFmpeg availability
    if not check_ffmpeg():
        raise HTTPException(status_code=503, detail="FFmpeg not available")
    
    # Validate inputs
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_IMAGES} images allowed")
    
    if duration_per_image < 1.0:
        raise HTTPException(status_code=400, detail="Duration per image must be at least 1 second")
    
    if transition_duration < 0.1 or transition_duration > duration_per_image:
        raise HTTPException(status_code=400, detail="Transition duration must be between 0.1s and duration_per_image")
    
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
        output_filename = f"reel_{request_id}.mp4"
        temp_video_path = OUTPUT_DIR / f"temp_{output_filename}"
        final_video_path = OUTPUT_DIR / output_filename
        
        # Generate reel video from images
        success = create_reel_video_with_transitions(
            image_paths, 
            temp_video_path if audio_path else final_video_path,
            duration_per_image,
            transition_duration,
            transition_type,
            fps
        )
        
        if not success:
            logger.warning("Primary video creation failed, trying fallback method")
            # Try fallback method with simpler approach
            success = create_simple_reel_video(
                image_paths,
                temp_video_path if audio_path else final_video_path,
                duration_per_image,
                fps
            )
            if not success:
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to create reel video from images. Check server logs for details."
                )
        
        # Add audio if provided
        if audio_path:
            success = add_audio_to_video(temp_video_path, audio_path, final_video_path, fade_audio)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to add audio to reel video")
            
            # Clean up temp video
            if temp_video_path.exists():
                temp_video_path.unlink()
        
        # Clean up uploaded files
        shutil.rmtree(request_dir, ignore_errors=True)
        
        if not final_video_path.exists():
            raise HTTPException(status_code=500, detail="Reel video file was not created")
        
        # Calculate total video duration
        total_duration = len(image_paths) * duration_per_image - (len(image_paths) - 1) * transition_duration if len(image_paths) > 1 else duration_per_image
        
        # Return success response
        return {
            "message": "Reel video created successfully",
            "video_id": request_id,
            "download_url": f"/download/{output_filename}",
            "file_size": final_video_path.stat().st_size,
            "images_processed": len(image_paths),
            "audio_added": audio_path is not None,
            "video_specs": {
                "format": "MP4",
                "resolution": f"{REEL_WIDTH}x{REEL_HEIGHT}",
                "aspect_ratio": "9:16",
                "fps": fps,
                "duration": f"{total_duration:.1f}s",
                "transition_type": transition_type,
                "transition_duration": transition_duration
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
    """Download generated reel video"""
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
    """Clean up generated reel video file"""
    file_path = OUTPUT_DIR / f"reel_{video_id}.mp4"
    
    if file_path.exists():
        file_path.unlink()
        return {"message": "Reel video deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Reel video not found")

@app.get("/list-reels")
async def list_reel_videos():
    """List all generated reel videos"""
    videos = []
    for file_path in OUTPUT_DIR.glob("reel_*.mp4"):
        videos.append({
            "filename": file_path.name,
            "size": file_path.stat().st_size,
            "created": file_path.stat().st_ctime,
            "download_url": f"/download/{file_path.name}",
            "format": "9:16 Reel"
        })
    
    return {"reel_videos": videos, "count": len(videos)}

@app.get("/supported-transitions")
async def get_supported_transitions():
    """Get list of supported transition types"""
    return {
        "transitions": [
            {"name": "crossfade", "description": "Smooth crossfade between images"},
            {"name": "fade", "description": "Fade to black between images"},
            {"name": "none", "description": "No transition (direct cut)"}
        ],
        "default": "crossfade",
        "recommended_duration": "0.5-2.0 seconds"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
