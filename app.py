import streamlit as st
from TTS.api import TTS
import torch

# Function to get TTS model
@st.cache_resource
def get_model(model_name):
    return TTS(model_name)

# Main app
def main():
    st.title("Text-to-Speech with Multiple Models")

    st.markdown("""
    <style>
    body {
        background-color: #e0f2f7; /* A light, cool blue */
    }
    .custom-heading {
        color: #4CAF50;
        font-family: 'Arial', sans-serif;
        text-align: center;
        padding: 10px;
        background-color: #e8f5e9;
        border-radius: 8px;
    }
    </style>
    <h1 class="custom-heading">Welcome to the TTS App!</h1>
    """, unsafe_allow_html=True)

    # Model selection
    model_map = {
        "Tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
        "Fastspeech2": "tts_models/en/ljspeech/fastspeech2",
        "VITS": "tts_models/en/ljspeech/vits",
        "TransformerTTS": "tts_models/en/ljspeech/transformer-tts"
    }
    display_model_names = list(model_map.keys())

    selected_display_name = st.selectbox(
        "Choose a TTS model",
        display_model_names
    )
    model_choice = model_map[selected_display_name]

    # Text input
    text_input = st.text_area("Enter text to synthesize:", "Hello, this is a test.")

    if st.button("Synthesize"):
        if text_input:
            try:
                if selected_display_name in ["Fastspeech2", "TransformerTTS"]:
                    from transformers import pipeline
                    import soundfile as sf
                    import os
                    import uuid

                    @st.cache_resource
                    def get_huggingface_model():
                        return pipeline("text-to-speech", model="saheedniyi/YarnGPT")

                    pipe = get_huggingface_model()
                    output = pipe(text_input)

                    AUDIO_DIR = "audio"
                    if not os.path.exists(AUDIO_DIR):
                        os.makedirs(AUDIO_DIR)
                    
                    audio_filename = f"{uuid.uuid4()}.wav"
                    audio_filepath = os.path.join(AUDIO_DIR, audio_filename)

                    sf.write(audio_filepath, output["audio"], samplerate=output["sampling_rate"])
                    st.audio(audio_filepath, format="audio/wav")
                else:
                    # Get the model
                    tts = get_model(model_choice)

                    # Save the audio to a file
                    audio_file = "output.wav"
                    tts.tts_to_file(text=text_input, speaker=tts.speakers[0] if tts.speakers else None, file_path=audio_file)

                    # Display the audio
                    st.audio(audio_file, format="audio/wav")

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
