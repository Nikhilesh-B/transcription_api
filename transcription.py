from flask import Flask, jsonify
from flask_restful import reqparse, Resource, Api
from base64 import b64decode
from whisper_jax import FlaxWhisperPipline 

app = Flask(__name__)
api = Api(app)

class Transcription(Resource):
    def __init__(self) -> None:
        super().__init__()
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('audio_byte_string')
        self.count = 0
        self.pipeline = FlaxWhisperPipline("openai/whisper-tiny",batch_size=16)

    def post(self):
        print("Hello up here")
        args = self.parser.parse_args()
        audio_byte_string = args['audio_byte_string']
        audio_data_raw_bytes = b64decode(audio_byte_string)
        file_name = 'conversation'+str(self.count)+'.mp3'
        with open(file_name, 'wb') as audio_file:
            audio_file.write(audio_data_raw_bytes)

        ##then we take this and use the whisper jax configuration to devleop the rest of this part 
        # translate
        text = self.pipeline(file_name, task="translate")
        self.count+=1

        return jsonify({'transcription':text})

api.add_resource(Transcription, '/transcription')

if __name__ == "__main__":
    app.run(debug=True)