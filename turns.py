import whisper
from pyannote.audio import Pipeline
import torch

def transcribe_multi_speaker(audio_file):
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Transcribe audio
    result = model.transcribe(audio_file)
    
    # Load speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", 
                                        use_auth_token="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")
    
    # Perform speaker diarization
    diarization = pipeline(audio_file)
    
    # Create a list to store speaker segments with timestamps
    speaker_segments = []
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        end = turn.end
        speaker_segments.append((start, end, speaker))
    
    # Sort speaker segments by start time
    speaker_segments.sort(key=lambda x: x[0])
    
    # Process each segment
    for start, end, speaker in speaker_segments:
        # Find words that fall within this segment
        words = [w for w in result["segments"] if start <= w["start"] < end]
        text = " ".join([w["text"] for w in words])
        
        if text.strip():  # Only print if there's actual text
            print(f"[{start:.2f} - {end:.2f}] Speaker {speaker}: {text.strip()}")

# Usage
audio_file = "./input.wav"
transcribe_multi_speaker(audio_file)
