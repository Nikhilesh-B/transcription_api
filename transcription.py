from flask import Flask, jsonify
from flask_restful import reqparse, Resource, Api
from base64 import b64decode
from whisper_jax import FlaxWhisperPipline 
from gradio_client import Client

API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/"


app = Flask(__name__)
api = Api(app)

class Transcription(Resource):
    def __init__(self) -> None:
        super().__init__()
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('audio_byte_string')
        self.count = 0
        self.pipeline = FlaxWhisperPipline("openai/whisper-tiny",batch_size=16)
        self.client = Client(API_URL)
    def transcribe_audio(self, audio_path, return_timestamps=False):
        """Function to transcribe an audio file using the Whisper JAX endpoint."""
        task = "transcribe"
        text, runtime = self.client.predict(audio_path, 
                                       task,
                                       return_timestamps,
                                       api_name="/predict_1")
    
        return text

    def post(self):
        print("Hello up here")
        args = self.parser.parse_args()
        audio_byte_string = args['audio_byte_string']
        audio_data_raw_bytes = b64decode(audio_byte_string)
        file_name = 'conversation'+str(self.count)+'.mp3'
        with open(file_name, 'wb') as audio_file:
            audio_file.write(audio_data_raw_bytes)
        
        text = self.transcribe_audio(file_name)
        print(text)

        return jsonify({'transcription':text})

api.add_resource(Transcription, '/transcription')

if __name__ == "__main__":
    app.run(debug=True)