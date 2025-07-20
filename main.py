from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import tempfile
import shutil
import uuid
from typing import List, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FFmpeg Video Generator API",
    description="Create vertical (reel) videos from images with transitions and music",
    version="1.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGES = 30
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return True
    except:
        return False

def validate_file_size(file: UploadFile):
    return not hasattr(file, 'size') or file.size <= MAX_FILE_SIZE

def validate_image_format(filename: str):
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_FORMATS

def validate_audio_format(filename: str):
    return Path(filename).suffix.lower() in SUPPORTED_AUDIO_FORMATS

async def save_upload_file(upload_file: UploadFile, destination: Path):
    content = await upload_file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    with open(destination, "wb") as buffer:
        buffer.write(content)

def create_video_with_transitions(images: List[Path], output_path: Path, duration: float, transition: float, fps: int):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        filter_complex = ""
        inputs = []
        overlay_idx = 0

        # Convert each image to video segment
        for i, img in enumerate(images):
            vid = temp_dir / f"img_{i}.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-loop", "1", "-i", str(img),
                "-t", str(duration + (transition if i != len(images) - 1 else 0)),
                "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
                "-r", str(fps), "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", str(vid)
            ], check=True)
            inputs.append(f"-i {vid}")

        input_cmds = sum(([arg, str(temp_dir / f"img_{i}.mp4")] for i, arg in enumerate(["-i"] * len(images))), [])

        # Build filter_complex
        for i in range(len(images) - 1):
            a = f"[v{i}]" if i > 0 else f"[0:v][1:v]"
            b = f"[v{i+1}]" if i > 0 else ""
            out = f"[v{i+1}]"
            xfade_type = "fade" if i % 2 == 0 else "circleopen"
            start_time = duration * (i + 1) + transition * i
            if i == 0:
                filter_complex += f"[0:v][1:v]xfade=transition={xfade_type}:duration={transition}:offset={duration}[v1];"
            else:
                filter_complex += f"[v{i}][{i+2}:v]xfade=transition={xfade_type}:duration={transition}:offset={start_time}[v{i+1}];"

        final_map = f"[v{len(images)-1}]"
        cmd = [
            "ffmpeg", "-y", *input_cmds,
            "-filter_complex", filter_complex.rstrip(";"),
            "-map", final_map, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            str(output_path)
        ]

        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(result.stderr)
            return False
        return True

def add_audio(video: Path, audio: Path, output: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(output)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

@app.post("/create-video")
async def create_video(
    images: List[UploadFile] = File(...),
    audio: Optional[UploadFile] = File(None),
    duration: float = Form(2.0),
    transition: float = Form(0.5),
    fps: int = Form(25)
):
    if not check_ffmpeg():
        raise HTTPException(status_code=503, detail="FFmpeg not available")
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Max {MAX_IMAGES} images allowed")

    request_id = str(uuid.uuid4())
    request_dir = UPLOAD_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    try:
        for i, img in enumerate(images):
            if not validate_file_size(img) or not validate_image_format(img.filename):
                raise HTTPException(status_code=400, detail="Invalid image")
            path = request_dir / f"img_{i}{Path(img.filename).suffix}"
            await save_upload_file(img, path)
            image_paths.append(path)

        audio_path = None
        if audio:
            if not validate_file_size(audio) or not validate_audio_format(audio.filename):
                raise HTTPException(status_code=400, detail="Invalid audio")
            audio_path = request_dir / f"audio{Path(audio.filename).suffix}"
            await save_upload_file(audio, audio_path)

        temp_video = OUTPUT_DIR / f"temp_{request_id}.mp4"
        final_video = OUTPUT_DIR / f"video_{request_id}.mp4"

        success = create_video_with_transitions(image_paths, temp_video if audio_path else final_video, duration, transition, fps)
        if not success:
            raise HTTPException(status_code=500, detail="Video generation failed")

        if audio_path:
            if not add_audio(temp_video, audio_path, final_video):
                raise HTTPException(status_code=500, detail="Failed to add audio")
            temp_video.unlink(missing_ok=True)

        shutil.rmtree(request_dir, ignore_errors=True)

        return {
            "message": "Video created successfully",
            "video_id": request_id,
            "download_url": f"/download/{final_video.name}",
            "file_size": final_video.stat().st_size,
            "image_count": len(image_paths),
            "audio_added": audio_path is not None
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        shutil.rmtree(request_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Internal error")

@app.get("/download/{filename}")
async def download_video(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path=file_path, filename=filename, media_type="video/mp4")

@app.get("/")
def root():
    return {"message": "FFmpeg Reel Video API", "status": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
