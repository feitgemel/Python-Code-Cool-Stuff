from faster_whisper import WhisperModel

model_size = "large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="float16")

file = "Python-Code-Cool-Stuff/Fast-Whisper/a.mp3"

segments , info = model.transcribe(file , beam_size=5)

print("detected language '%s' with probability %f" % (info.language, info.language_probability))


for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start , segment.end, segment.text ))

