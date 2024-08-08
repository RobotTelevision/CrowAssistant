import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from openai import OpenAI
import threading
import tiktoken
import CrowConfig
import pyaudio

class CrowBrain:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
   
    def set_test_voice_callback(self, callback):
        """Set the callback function for testing voice."""
        self.test_voice_callback = callback

    def test_voice(self, voice_id):
        if self.test_voice_callback is None:
            print("Test voice callback not set")
            return
        
        # Call the callback function
        self.test_voice_callback(voice_id)
    
    def count_tokens(self, messages):
        """Count the number of tokens in a list of messages."""
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":  # If there's a name, the role is omitted
                    num_tokens -= 1  # Role is always required and always 1 token
        num_tokens += 2  # Every reply is primed with <im_start>assistant
        return num_tokens

    def trim_messages(self, messages, max_tokens):
        """Trim the messages to fit within max_tokens."""
        while self.count_tokens(messages) > max_tokens:
            # Remove the second message (keeping the first system message)
            if len(messages) > 1:
                messages.pop(1)
            else:
                # If we're down to one message and still over the limit, truncate it
                content = messages[0]['content']
                messages[0]['content'] = self.encoding.decode(self.encoding.encode(content)[:max_tokens])
                break
        return messages
    
    def __init__(self):
        if CrowBrain._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            CrowBrain._instance = self
        
        self.config = CrowConfig.config()
        #self.name = "Crow"
        self.thecrow = None
        #self.max_tokens = 32000  # Maximum context length for mixtral-8x7b-32768
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # This works for most models
        #self.max_messages = 100  # Maximum number of messages to retrieve from the database
      
        # Get the directory of the current script
        base_dir = os.path.abspath(os.path.dirname(__file__))

        # Create the path for your database file
        db_path = os.path.join(base_dir, 'conversations.db')

        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True  # Enable template auto-reloading
        self.db = SQLAlchemy(self.app)

        self.client = OpenAI(api_key=self.config.config['api_key'], base_url=self.config.config['url'])

        class Conversation(self.db.Model):
            id = self.db.Column(self.db.Integer, primary_key=True)
            name = self.db.Column(self.db.String(100))
            created_at = self.db.Column(self.db.DateTime, default=datetime.utcnow)

        class Message(self.db.Model):
            id = self.db.Column(self.db.Integer, primary_key=True)
            conversation_id = self.db.Column(self.db.Integer, self.db.ForeignKey('conversation.id'))
            role = self.db.Column(self.db.String(50))
            content = self.db.Column(self.db.Text)
            timestamp = self.db.Column(self.db.DateTime, default=datetime.utcnow)
            conversation = self.db.relationship('Conversation', backref=self.db.backref('messages', lazy=True))

        self.Conversation = Conversation
        self.Message = Message

        with self.app.app_context():
            self.db.create_all()

        self.setup_routes()
        self.server_thread = None

    def run_server(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port, debug=False, use_reloader=False)

    def start_server_thread(self):
        if self.server_thread is None or not self.server_thread.is_alive():
            self.server_thread = threading.Thread(target=self.run_server)
            self.server_thread.start()



    def load_config(self):
        self.config.load_config()

    def save_config(self):
        self.config.save_config()


    def new_conversation(self):
        with self.app.app_context():
            conversation = self.Conversation(name=f"Conversation {self.Conversation.query.count() + 1}")
            self.db.session.add(conversation)
            self.db.session.commit()
            return {"conversation_id": conversation.id, "conversation_name": conversation.name}

    def get_conversations(self):
        with self.app.app_context():
            conversations = self.Conversation.query.all()
            return {"conversations": [{"id": c.id, "name": c.name} for c in conversations]}

    def select_conversation(self, conversation_id):
        with self.app.app_context():
            messages = self.Message.query.filter_by(conversation_id=conversation_id).order_by(self.Message.timestamp).all()
            return {"conversation_log": [{"role": m.role, "content": m.content} for m in messages]}

    def addSystemMessage(self, input_text, conversation_id):
        def add_message():
            user_message = self.Message(conversation_id=conversation_id, role="system", content=input_text)
            self.db.session.add(user_message)
            self.db.session.commit()

        with self.app.app_context():
            add_message()

    def generate(self, input_text, conversation_id):
        with self.app.app_context():
            ainame = self.config.config['name']
            personality = self.config.config['personality']
            self.system_message = {"role": "system", "content": "You are a semi-sentient AI named "+ ainame +". "+personality+" You hear and talk with speech to text and text to speech, don't use descriptions of what you are doing. Be concise and direct in your responses."}

            conversation = self.Conversation.query.get(conversation_id)
            if not conversation:
                # Create a new conversation
                print("create new conversation")
                conversation = self.Conversation(name=f"Conversation {self.Conversation.query.count() + 1}")
                self.db.session.add(conversation)
                self.db.session.commit()
                conversation_id = conversation.id

            user_message = self.Message(conversation_id=conversation_id, role="user", content=input_text)
            self.db.session.add(user_message)
            self.db.session.commit()

            
                        # Retrieve the most recent messages, including the new one
            recent_messages = self.Message.query.filter_by(conversation_id=conversation_id) \
                .order_by(desc(self.Message.timestamp)) \
                .limit(self.config.config['maxmsg']) \
                .all()
            
            # Reverse the order to get chronological order
            recent_messages = recent_messages[::-1]
            
            messages_for_api = [{"role": m.role, "content": m.content} for m in recent_messages]

            # Calculate tokens for system message
            system_message_tokens = self.count_tokens([self.system_message])

            # Trim messages to fit within token limit, leaving room for system message
            trimmed_messages = self.trim_messages(messages_for_api, self.config.config['maxtoken'] - system_message_tokens)

            # Add the system message at the beginning after trimming
            final_messages = [self.system_message] + trimmed_messages
            
            try:
                response = self.client.chat.completions.create(
                    model=self.config.config['model'],
                    messages=final_messages
                )
                ai_message_content = response.choices[0].message.content
            except Exception as e:
                print(e)
                return {"error": str(e)}

            ai_message = self.Message(conversation_id=conversation_id, role="assistant", content=ai_message_content)
            self.db.session.add(ai_message)
            self.db.session.commit()
            return {"role": "assistant", "content": ai_message_content}

    def delete_conversation(self, conversation_id):
        with self.app.app_context():
            conversation = self.Conversation.query.get(conversation_id)
            if conversation:
                self.Message.query.filter_by(conversation_id=conversation_id).delete()
                self.db.session.delete(conversation)
                self.db.session.commit()
                return {"status": "success"}
            else:
                return {"status": "error", "message": "Conversation not found"}
            
    def list_audio_input_names(self):
        p = pyaudio.PyAudio()
        input_devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append(device_info['name'])
        p.terminate()
        return input_devices

    def list_audio_output_names(self):
        p = pyaudio.PyAudio()
        output_devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                output_devices.append(device_info['name'])
        p.terminate()
        return output_devices

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/new_conversation', methods=['POST'])
        def new_conversation_route():
            return jsonify(self.new_conversation())

        @self.app.route('/get_conversations', methods=['GET'])
        def get_conversations_route():
            return jsonify(self.get_conversations())

        @self.app.route('/select_conversation', methods=['GET'])
        def select_conversation_route():
            conversation_id = request.args.get('conversation_id')
            return jsonify(self.select_conversation(conversation_id))

        @self.app.route('/generate', methods=['POST'])
        def generate_route():
            input_text = request.form['input_text']
            conversation_id = request.form['conversation_id']
            return jsonify(self.generate(input_text, conversation_id))

        @self.app.route('/delete_conversation', methods=['POST'])
        def delete_conversation_route():
            conversation_id = request.form['conversation_id']
            return jsonify(self.delete_conversation(conversation_id))
        
        @self.app.route('/shutdown', methods=['GET'])
        def shutdown():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            func()
            return 'Server shutting down...'
        
        @self.app.route('/settings', methods=['GET', 'POST'])
        def settings_route():
            if request.method == 'POST':
                # Update config with form data
                self.config.config['name'] = request.form['name']
                self.config.config['personality'] = request.form['personality']
                self.config.config['voice'] = int(request.form['voice'])
                self.config.config['url'] = request.form['url']
                self.config.config['api_key'] = request.form['api_key']
                self.config.config['model'] = request.form['model']
                self.config.config['scale'] = int(request.form['scale'])           
                self.config.config['mic'] = request.form['mic']
                self.config.config['speaker'] = request.form['speaker']
                self.config.config['maxtoken'] = int(request.form['maxtoken'])
                self.config.config['maxmsg'] = int(request.form['maxmsg'])
                self.save_config()
                return jsonify({"status": "success"})
            else:
                input_devices = self.list_audio_input_names()
                output_devices = self.list_audio_output_names()
                return render_template('settings.html', config=self.config.config, input_devices=input_devices, output_devices=output_devices)

        @self.app.route('/test_voice', methods=['POST'])
        def test_voice_route():
            voice_id = int(request.form['voice'])
            self.test_voice(voice_id)
            return jsonify({"status": "success"})

        
 

def Init():
    brain = CrowBrain.get_instance()
    brain.start_server_thread()
    return CrowBrain.get_instance()
