from flask import Flask, render_template, request, jsonify
import os
import sys
import time

# Try to import Chatbot class
try:
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chatbot import Chatbot
except ImportError:
    # If import fails, 
    class Chatbot:
        def __init__(self, documents_dir):
            self.documents_dir = documents_dir
            self.initialized = False
            print(f"Creating chatbot with documents directory: {documents_dir}")
        
        def initialize_document_qa(self, directory):
            print(f"Initializing document QA with directory: {directory}")
            self.initialized = True
            #delay
            time.sleep(2)
        
        def process_message(self, message):
            if not self.initialized:
                return "Chatbot is not initialized."
            return f"You asked: '{message}'. This is a placeholder response since the actual Chatbot class wasn't imported."

app = Flask(__name__)

# Global chatbot instance
chatbot = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    try:
        data = request.json
        documents_dir = data.get('directory', '')
        
        if not documents_dir:
            return jsonify({'status': 'error', 'message': 'No directory provided'})
        
        if not os.path.exists(documents_dir):
            return jsonify({'status': 'error', 'message': 'Directory does not exist'})
        
        global chatbot
        chatbot = Chatbot(documents_dir)
        chatbot.initialize_document_qa(documents_dir)
        
        return jsonify({'status': 'success', 'message': 'Chatbot initialized successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'status': 'error', 'message': 'No message provided'})
        
        global chatbot
        if not chatbot:
            return jsonify({'status': 'error', 'message': 'Chatbot not initialized'})
        
        response = chatbot.process_message(message)
        return jsonify({'status': 'success', 'response': response})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Create templates directory and HTML file
def create_templates():
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            display: flex;
            flex-direction: column;
            height: 95vh;
        }
        .setup-panel {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chat-container {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chat-messages {
            flex-grow: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f9f9f9;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
            max-width: 80%;
        }
        .user-message {
            background-color: #e1f5fe;
            margin-left: auto;
            color: #01579b;
        }
        .bot-message {
            background-color: #f0f0f0;
            margin-right: auto;
            color: #333;
        }
        .system-message {
            background-color: #fff3e0;
            margin: 0 auto;
            color: #e65100;
            text-align: center;
            font-style: italic;
        }
        .message-header {
            font-size: 0.8em;
            margin-bottom: 5px;
            opacity: 0.7;
        }
        .message-content {
            word-wrap: break-word;
        }
        .input-area {
            display: flex;
            padding: 15px;
            background-color: white;
            border-top: 1px solid #eee;
        }
        input[type="text"] {
            flex-grow: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            outline: none;
        }
        button {
            padding: 10px 15px;
            margin-left: 10px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0b7dda;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .status {
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .status.error {
            background-color: #ffebee;
            color: #c62828;
        }
        .status.success {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .status.info {
            background-color: #e3f2fd;
            color: #1565c0;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-left: 5px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="setup-panel">
            <h2>Document Chatbot</h2>
            <div>
                <label for="directory">Documents Directory:</label>
                <input type="text" id="directory" placeholder="/path/to/your/documents" style="width: 50%; margin: 0 10px;">
                <button id="initButton">Initialize Chatbot</button>
                <span id="initLoading" class="loading" style="display: none;"></span>
            </div>
            <div id="status" class="status info">Status: Not initialized</div>
        </div>
        
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="message system-message">
                    <div class="message-content">Welcome to Document Chatbot! Please provide a documents directory and initialize the chatbot.</div>
                </div>
            </div>
            
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Type your message here..." disabled>
                <button id="sendButton" disabled>Send</button>
                <span id="sendLoading" class="loading" style="display: none;"></span>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const directoryInput = document.getElementById('directory');
            const initButton = document.getElementById('initButton');
            const initLoading = document.getElementById('initLoading');
            const statusDiv = document.getElementById('status');
            const chatMessages = document.getElementById('chatMessages');
            const userInput = document.getElementById('userInput');
            const sendButton = document.getElementById('sendButton');
            const sendLoading = document.getElementById('sendLoading');
            
            let chatbotInitialized = false;
            
            // Initialize chatbot
            initButton.addEventListener('click', function() {
                const directory = directoryInput.value.trim();
                if (!directory) {
                    updateStatus('Please enter a documents directory', 'error');
                    return;
                }
                
                // Disable button and show loading indicator
                initButton.disabled = true;
                initLoading.style.display = 'inline-block';
                updateStatus('Initializing chatbot...', 'info');
                
                // Add system message
                addMessage('System', 'Initializing chatbot... This may take a moment.', 'system');
                
                // Send initialization request
                fetch('/initialize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ directory: directory })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        chatbotInitialized = true;
                        updateStatus('Chatbot initialized successfully', 'success');
                        userInput.disabled = false;
                        sendButton.disabled = false;
                        addMessage('System', 'Chatbot initialized successfully! You can now start chatting.', 'system');
                    } else {
                        updateStatus(`Error: ${data.message}`, 'error');
                        addMessage('System', `Failed to initialize chatbot: ${data.message}`, 'system');
                    }
                })
                .catch(error => {
                    updateStatus(`Error: ${error.message}`, 'error');
                    addMessage('System', `An error occurred: ${error.message}`, 'system');
                })
                .finally(() => {
                    initButton.disabled = false;
                    initLoading.style.display = 'none';
                });
            });
            
            // Send message
            function sendMessage() {
                const message = userInput.value.trim();
                if (!message) return;
                if (!chatbotInitialized) {
                    updateStatus('Please initialize the chatbot first', 'error');
                    return;
                }
                
                // Add user message to chat
                addMessage('You', message, 'user');
                
                // Clear input and disable
                userInput.value = '';
                userInput.disabled = true;
                sendButton.disabled = true;
                sendLoading.style.display = 'inline-block';
                
                // Send chat request
                fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        addMessage('Bot', data.response, 'bot');
                    } else {
                        updateStatus(`Error: ${data.message}`, 'error');
                        addMessage('System', `Error: ${data.message}`, 'system');
                    }
                })
                .catch(error => {
                    updateStatus(`Error: ${error.message}`, 'error');
                    addMessage('System', `An error occurred: ${error.message}`, 'system');
                })
                .finally(() => {
                    userInput.disabled = false;
                    sendButton.disabled = false;
                    sendLoading.style.display = 'none';
                    userInput.focus();
                });
            }
            
            // Send message when Send button is clicked
            sendButton.addEventListener('click', sendMessage);
            
            // Send message when Enter key is pressed
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Helper function to add messages to the chat
            function addMessage(sender, content, type) {
                const timestamp = new Date().toLocaleTimeString();
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                
                const messageHeader = document.createElement('div');
                messageHeader.className = 'message-header';
                messageHeader.textContent = `${timestamp} - ${sender}`;
                
                const messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContent.textContent = content;
                
                messageDiv.appendChild(messageHeader);
                messageDiv.appendChild(messageContent);
                chatMessages.appendChild(messageDiv);
                
                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Helper function to update status
            function updateStatus(message, type) {
                statusDiv.textContent = message;
                statusDiv.className = `status ${type}`;
            }
        });
    </script>
</body>
</html>
    """
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)

if __name__ == '__main__':
    # Create templates directory and HTML file
    create_templates()
    
    # Run the Flask app
    print("Starting Flask web server for chatbot...")
    app.run(host='127.0.0.1', port=5000, debug=True)