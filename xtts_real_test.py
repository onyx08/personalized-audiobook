import os

# Windows DLL search paths required by TorchCodec.
os.add_dll_directory(r"C:\ffmpeg\bin")
os.add_dll_directory(
    r"C:\kanu\college\sem7\NNDL\personalized-audiobook"
    r"\.venv-xtts\Lib\site-packages\torch\lib"
)

from TTS.api import TTS

output = r"C:\Users\KANISH~1\AppData\Local\Temp\xtts-real-test\kk_xtts_test.wav"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="This is a real voice generation test for my personalized audiobook project.",
    speaker_wav=r"reference_audio\kk_referenceaudio.wav",
    language="en",
    file_path=output,
)

print("DONE:", output)