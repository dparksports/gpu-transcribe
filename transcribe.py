import whisper
from pyannote.audio import Pipeline
import torch

def transcribe_multi_speaker(audio_file):
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Transcribe audio
    result = model.transcribe(audio_file)
    transcription = result["text"]
    
    # Load speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", 
                                        use_auth_token="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")
    
    # Perform speaker diarization
    diarization = pipeline(audio_file)
    
    # Combine transcription with speaker labels
    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        end = turn.end
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append((start, end))
    
    # Split transcription by speaker
    speaker_transcripts = {}
    for speaker, times in speakers.items():
        speaker_text = ""
        for start, end in times:
            words = [w for w in result["segments"] if start <= w["start"] < end]
            speaker_text += " ".join([w["text"] for w in words])
        speaker_transcripts[speaker] = speaker_text.strip()
        print(f"Speaker {speaker}: {speaker_text.strip()}")

    
    return speaker_transcripts

# Usage
audio_file = "./input.wav"
transcripts = transcribe_multi_speaker(audio_file)

print(f"===================================")
for speaker, text in transcripts.items():
    print(f"Speaker {speaker}: {text}")
