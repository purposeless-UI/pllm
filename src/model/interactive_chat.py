import os
import sys
import torch
import threading
import time
from flask import Flask, render_template_string, request, jsonify

# Ensure the root directory is visible to Python
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.model import PaniniConfig
from src.model.generate import PaniniGenerator

# Initialize Flask App
app = Flask(__name__)

# --- Setup configuration paths safely ---
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "panini_sanskrit_final.pt")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer.json")

# Global placeholder for the generator
generator = None

# --- BACKGROUND FLASK SERVER THREAD ---
def run_flask_server():
    print("🌐 [FLASK] Background web server thread engaged on port 5001.")
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)

# --- Beautiful, Responsive HTML Layout ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="sa">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Panini Custom Sanskrit AI Chat</title>
    <style>
        body {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            background-color: #f4f6f9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .chat-container {
            width: 100%;
            max-width: 700px;
            height: 85vh;
            background: #ffffff;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #ff7e5f, #feb47b);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        .chat-box {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #fafafa;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message {
            max-width: 75%;
            padding: 12px 18px;
            border-radius: 14px;
            font-size: 1.15rem;
            line-height: 1.6;
            word-wrap: break-word;
        }
        .message strong {
            display: block;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
            opacity: 0.85;
        }
        .message.user {
            background-color: #007aff;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 2px;
        }
        .message.bot {
            background-color: #e9e9eb;
            color: #1c1c1e;
            align-self: flex-start;
            border-bottom-left-radius: 2px;
        }
        .chat-input-area {
            display: flex;
            padding: 15px;
            background: #ffffff;
            border-top: 1px solid #eee;
            gap: 10px;
        }
        .chat-input-area input {
            flex: 1;
            padding: 14px;
            border: 1px solid #ddd;
            border-radius: 30px;
            font-size: 1.1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        .chat-input-area input:focus {
            border-color: #ff7e5f;
        }
        .chat-input-area button {
            background: #ff7e5f;
            color: white;
            border: none;
            padding: 0 25px;
            border-radius: 30px;
            font-size: 1.05rem;
            cursor: pointer;
            transition: background 0.2s;
        }
        .chat-input-area button:hover {
            background: #e66e50;
        }
    </style>
</head>
<body>

<div class="chat-container">
    <div class="chat-header">🧠 Panini Sanskrit Language Model Chat</div>
    <div class="chat-box" id="chatBox">
        <div class="message bot"><strong>🤖 Panini AI:</strong>नमस्ते! Please type your Sanskrit prompt below.</div>
    </div>
    <div class="chat-input-area">
        <input type="text" id="userInput" placeholder="Type Sanskrit text here (e.g., कृषकः क्षेत्रं कृषयात्)..." onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
    function handleKeyPress(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    }

    function sendMessage() {
        const inputField = document.getElementById('userInput');
        const text = inputField.value.trim();
        if (!text) return;

        appendMessage(text, 'user');
        inputField.value = '';

        const typingId = appendMessage('Computing responses...', 'bot');

        fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt: text })
        })
        .then(res => res.json())
        .then(data => {
            document.getElementById(typingId).remove();
            appendMessage(data.response, 'bot');
        })
        .catch(err => {
            document.getElementById(typingId).remove();
            appendMessage('❌ Engine Error processing prompt matrix.', 'bot');
        });
    }

    function appendMessage(text, sender) {
        const chatBox = document.getElementById('chatBox');
        const msgDiv = document.createElement('div');
        const uniqueId = 'msg-' + Date.now() + Math.random().toString(36).substr(2, 4);
        
        msgDiv.id = uniqueId;
        msgDiv.className = 'message ' + sender;
        
        // Add marked names based on user or bot
        if (sender === 'user') {
            msgDiv.innerHTML = '<strong>👤 User:</strong>' + text;
        } else {
            msgDiv.innerHTML = '<strong>🤖 Panini AI:</strong>' + text;
        }
        
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return uniqueId;
    }
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    global generator
    data = request.json or {}
    user_prompt = data.get("prompt", "")
    
    if not user_prompt:
        return jsonify({"response": "⚠️ Prompt cannot be empty."})
        
    try:
        model_output = generator.generate(prompt=user_prompt, max_new_tokens=60)
        return jsonify({"response": model_output})
    except Exception as e:
        return jsonify({"response": f"❌ Error: {str(e)}"})

if __name__ == '__main__':
    # 1. Start the Flask server in a background thread first
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()

    # 2. FIXED CONFIGURATION MAP: Synchronized perfectly to match local 3-layer parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config = PaniniConfig(
        vocab_size=64000,
        dim=256,
        n_layers=3,
        n_heads=4,
        n_kv_heads=1,
        max_seq_len=64
    )
    
    print(f"🚀 Initializing Panini Generator on engine: {device}")
    print(f"📂 Targeting checkpoint file path: {MODEL_PATH}")
    generator = PaniniGenerator(MODEL_PATH, TOKENIZER_PATH, config, device)

    # 3. Keep the main thread alive updating the visualizer canvas loop natively
    print("🎨 [VISUALIZER] Main thread active rendering loop engaged.")
    while True:
        try:
            if hasattr(generator, 'visualizer') and generator.visualizer.active:
                generator.visualizer.update_and_render()
            else:
                print("⚠️ [VISUALIZER] Window closed by user.")
                break
        except KeyboardInterrupt:
            print("🛑 Server shutting down.")
            break
        except Exception:
            pass
        time.sleep(0.01) # Prevents CPU usage spikes
