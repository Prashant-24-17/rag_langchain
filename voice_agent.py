import whisper

model = whisper.load_model('base')
result = model.transcribe('what.mp3' , fp16=False)
print(result.keys())

