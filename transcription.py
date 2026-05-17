import sounddevice as sd
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

# record audio from microphone
SAMPLE_RATE = 16000
DURATION = 5  # seconds


def get_transription(audio):
    
    try:

        if audio is None:
            return None

        input_features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features

        # generate token ids
        predicted_ids = model.generate(input_features)

        # decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription
    
    except Exception as e:
        print("Exception in get_transription as" ,str(e) )
        return None