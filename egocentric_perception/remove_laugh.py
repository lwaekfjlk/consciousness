from datetime import timedelta
import os
import whisper
import librosa

import whisperx
import gc 

def transcribe_audio(path):
    model = whisper.load_model("base") # Change this to your desired model
    print("Whisper model loaded.")
    transcribe = model.transcribe(audio=path)
    segments = transcribe['segments']

    import pdb; pdb.set_trace()
    for segment in segments:
        startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
        endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
        text = segment['text']
        segmentId = segment['id']+1
        segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] is ' ' else text}\n\n"

        srtFilename = "VIDEO_FILENAME.srt"
        with open(srtFilename, 'a', encoding='utf-8') as srtFile:
            srtFile.write(segment)
    print(srtFilename)
    return srtFilename

if __name__ == '__main__':
    device = "cuda" 
    audio_file = "./data/audios/utterances_final/1_70.aac"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    #transcribe_audio("./data/audios/utterances_final/1_90.aac")

    from scipy.io import wavfile

    duration = librosa.get_duration(filename=audio_file)
    start = result["segments"][0]['start']
    end = result["segments"][0]['end']
    # the timestamp to split at (in seconds)

    # read the file and get the sample rate and data
    data, rate = librosa.load(audio_file, res_type='kaiser_fast', duration=1000, sr=22050*2, offset=0)

    # get the frame to split at
    start_frame = len(data) * (start / duration)
    end_frame = len(data) * (end / duration)

    filtered_data = data[int(start_frame):int(end_frame)]
    # save the result
    wavfile.write('filtered_data.wav', rate,filtered_data)