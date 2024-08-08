import os
import json

class config:

    def __init__(self):
        self.config = {
            "name": "Crow",
            "personality": "",
            "voice": 1,
            "url": "https://api.groq.com/openai/v1",
            "api_key": os.environ.get("API_KEY", ""),
            "model": "mixtral-8x7b-32768",
            "mic":"default",
            "speaker":"default",
            "scale":3,
            "port":5000,
            "maxtoken":32000,
            "maxmsg":100,
        }
        self.CONFIG_FILE = 'config.json'
        self.load_config()

    def load_config(self):
        print("Loading Config")
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
        else:
            print("no config")
            #we need to launch the settings window
      
    def save_config(self):
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
    

