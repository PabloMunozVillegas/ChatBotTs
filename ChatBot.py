from flask import Flask, request, jsonify
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

app = Flask(__name__)

# Configuración global
max_length = 50
tokenizer = None
model = None
data_file = 'datos.json'
model_file = 'chatbot_model.keras'
tokenizer_file = 'tokenizer.json'

def load_data():
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_tokenizer():
    global tokenizer
    with open(tokenizer_file, 'r', encoding='utf-8') as f:
        tokenizer = Tokenizer.from_json(json.load(f))

def load_chatbot_model():
    global model
    model = load_model(model_file)

def preprocess_data(questions, responses):
    tokenizer.fit_on_texts(questions + responses)
    question_sequences = tokenizer.texts_to_sequences(questions)
    response_sequences = tokenizer.texts_to_sequences(responses)
    padded_questions = pad_sequences(question_sequences, maxlen=max_length, padding='post')
    response_sequences = pad_sequences(response_sequences, maxlen=max_length, padding='post')
    padded_responses = to_categorical(response_sequences, num_classes=len(tokenizer.word_index) + 1)
    return padded_questions, padded_responses

@app.route('/preguntar', methods=['GET'])
def preguntar():
    data = load_data()
    return jsonify(data)

@app.route('/respuesta', methods=['GET'])
def respuesta():
    data = load_data()
    return jsonify(data)

@app.route('/preguntar_entrenamiento', methods=['POST'])
def preguntar_entrenamiento():
    global model
    input_text = request.json.get('input')
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq, maxlen=max_length, padding='post')
    predicted_probs = model.predict(input_padded)[0]
    predicted_indices = np.argmax(predicted_probs, axis=-1)
    predicted_words = [tokenizer.index_word.get(idx, '<OOV>') for idx in predicted_indices.flatten()]
    response = ' '.join(predicted_words).strip()
    return jsonify({'response': response})

@app.route('/respuesta_entrenamiento', methods=['POST'])
def respuesta_entrenamiento():
    global model
    global tokenizer
    
    data = load_data()
    user_input = request.json.get('question')
    correct_response = request.json.get('response')

    # Actualizar datos
    for item in data:
        if item['question'] == user_input:
            item['response'] = correct_response
            break
    else:
        data.append({"question": user_input, "response": correct_response})

    save_data(data)

    # Actualizar modelo
    questions = [item['question'] for item in data]
    responses = [item['response'] for item in data]
    padded_questions, padded_responses = preprocess_data(questions, responses)

    model.fit(padded_questions, padded_responses, epochs=5, batch_size=1)
    model.save(model_file)

    return jsonify({'status': 'success', 'message': 'Modelo actualizado'})

@app.route('/respuesta_correcta', methods=['POST'])
def respuesta_correcta():
    user_input = request.json.get('question')
    feedback = request.json.get('feedback')
    
    if feedback.lower() == 'n':
        correct_response = request.json.get('response')
        data = load_data()
        for item in data:
            if item['question'] == user_input:
                item['response'] = correct_response
                break
        else:
            data.append({"question": user_input, "response": correct_response})

        save_data(data)
        
        # Actualizar el modelo con el nuevo dato
        questions = [item['question'] for item in data]
        responses = [item['response'] for item in data]
        padded_questions, padded_responses = preprocess_data(questions, responses)
        
        model.fit(padded_questions, padded_responses, epochs=5, batch_size=1)
        model.save(model_file)

    return jsonify({'status': 'success', 'message': 'Respuesta actualizada'})

if __name__ == '__main__':
    load_tokenizer()
    load_chatbot_model()
    app.run(debug=True)
