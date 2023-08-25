from flask import Flask, jsonify, request, render_template, make_response
from flask_restful import reqparse, Resource, Api
from pybase64 import b64decode
# from whisper_jax import FlaxWhisperPipline 

app = Flask(__name__)
api = Api(app)

class Transcription(Resource):
    def __init__(self) -> None:
        super().__init__()
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('audio_byte_string')
        self.count = 0
        # self.pipeline = FlaxWhisperPipline("openai/whisper-large-v2",batch_size=16)

    def post(self):
        data = request.get_json()

        decoded = b64decode(data["data"]).decode("utf-8")

        return decoded, 200

    # def post(self):
        # args = self.parser.parse_args()
    #     audio_byte_string = args['audio_byte_string']
    #     audio_data_raw_bytes = b64decode(audio_byte_string) #! will be able to get to here
    #     file_name = 'conversation'+str(self.count)+'.mp3'
    #     with open(file_name, 'wb') as audio_file: #!need to write to GCP Cloud storage not local file
    #         audio_file.write(audio_data_raw_bytes)

    #     ##then we take this and use the whisper jax configuration to devleop the rest of this part 
    #     # translate
    #     text = self.pipeline("audio.mp3", task="translate")
    #     self.count+=1

    #     return jsonify({'transcription':text})

class Echo(Resource):
    def get(self):
        headers = {"content-type":"text/html"}
        resp = make_response(render_template('index.html'), 200, headers)
        return resp

    def post(self):
        data = request.get_json()
        return data, 200

api.add_resource(Echo, '/echo')
api.add_resource(Transcription, '/transcription/')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)