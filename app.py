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

    # Model selection
    model_choice = st.selectbox(
        "Choose a TTS model",
        ["tts_models/en/ljspeech/tacotron2-DDC", "tts_models/en/ljspeech/fastspeech2", "tts_models/en/ljspeech/vits", "tts_models/en/ljspeech/transformer-tts"]
    )

    # Text input
    text_input = st.text_area("Enter text to synthesize:", "Hello, this is a test.")

    if st.button("Synthesize"):
        if text_input:
            try:
                # Get the model
                tts = get_model(model_choice)

                # Synthesize the text
                wav = tts.tts(text=text_input, speaker=tts.speakers[0] if tts.speakers else None)

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
