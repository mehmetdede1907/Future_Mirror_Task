from flask import Flask, render_template, request, jsonify, send_from_directory
from mirror import future_mirror, future_mirror_dalle
import tempfile
import os
import json
from vosk import Model, KaldiRecognizer
import wave
from pydub import AudioSegment
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Initialize Vosk model
try:
    model = Model(lang="en-us")
except Exception as e:
    app.logger.error(f"Failed to load Vosk model: {str(e)}")
    model = None

def recognize_speech(audio_path):
    if model is None:
        raise Exception("Vosk model not loaded")
    
    # Convert to WAV if it's not already
    if not audio_path.lower().endswith('.wav'):
        wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
    else:
        wav_path = audio_path

    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    
    result = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result += json.loads(rec.Result())["text"] + " "
    
    result += json.loads(rec.FinalResult())["text"]
    return result.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        model = request.form['model']
        
        if model == 'stable_diffusion':
            description, image_path, metadata = future_mirror(user_input)
        elif model == 'dalle':
            description, image_path, metadata = future_mirror_dalle(user_input)
        else:
            return jsonify({'error': 'Invalid model selection'}), 400
        
        image_filename = os.path.basename(image_path)
        
        return jsonify({
            'description': description,
            'image_path': f'/output/{image_filename}',
            'metadata': metadata
        })
    return render_template('index.html')

@app.route('/output/<path:filename>')
def serve_image(filename):
    return send_from_directory('output', filename)

@app.route('/voice_input', methods=['POST'])
def voice_input():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_audio:
        audio_file.save(temp_audio.name)
        temp_audio_path = temp_audio.name

    try:
        text = recognize_speech(temp_audio_path)

        if not text:
            return jsonify({'error': 'Speech not understood'}), 400

        return jsonify({'text': text})
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    app.run(debug=True)