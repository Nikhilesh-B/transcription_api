from flask import Flask, jsonify, request, render_template, make_response
from flask_restful import reqparse, Resource, Api
from pybase64 import b64decode
import os
from google.cloud import storage
from gradio_client import Client

API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/"

app = Flask(__name__)
api = Api(app)

class Transcription(Resource):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.client = Client(API_URL)
        self.bucket_name = "supportify-394619.appspot.com"

    def transcribe_audio(self, audio_path, return_timestamps=False):
        """Function to transcribe an audio file using the Whisper JAX endpoint."""
        task = "transcribe"
        text, runtime = self.client.predict(audio_path, 
                                       task,
                                       return_timestamps,
                                       api_name="/predict_1")
    
        return text

    def writeb_to_bucket(self, bucket_name, blob_name, bites):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with blob.open("wb") as f:
            f.write(bites)
    
    def readb_from_bucket(self, bucket_name, blob_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with blob.open("rb") as f:
            bites = f.read()
        
        return bites

    def list_bucket_contents(self, bucket_name):
        storage_client = storage.Client()

        blobs = storage_client.list_blobs(bucket_name)
        blobnames = [blob.name for blob in blobs]

        return blobnames

    def post(self):
        data = request.get_json()

        filename = data["filename"]
        decoded = b64decode(data["audio_byte_string"]) #bytes format
        
        storage_client = storage.Client()

        self.writeb_to_bucket(self.bucket_name, filename, decoded)

        ##TODO: implement deletion of audio file from bucket

        tmp_filename = "/tmp/audio.mp3" #will overwrite last-transcripted audio (file does not persist)
        with open(tmp_filename, 'wb') as audio_file:
            bucket = storage_client.get_bucket(self.bucket_name)
            blob = bucket.blob(filename)
            blob.download_to_file(audio_file)
        
        text = self.transcribe_audio(tmp_filename)

        contents = self.list_bucket_contents(self.bucket_name)

        return {"transcription":text,"bucket-contents":contents}, 200

    def get(self):
        return self.list_bucket_contents(self.bucket_name)


class Echo(Resource):
    """
    A dummy resource. Useful to test in-browser whether cloud deployment was successful.
    """
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