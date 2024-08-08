
import os
from queue import Queue
import pyaudio
from CrowSTT import AudioToTextRecorder
import random
import string
import time
import keyboard
import threading
import Crow
import CrowBrain
import wave
import numpy as np
import re
import sys
import shlex
import requests
import ctypes
import Volume
import CrowConfig
# import zipfile
# from urllib.parse import urlparse
# import shutil


##At some point I want to make it so it downloads the things it needs for TTS etc... but for now we'll do it manually
# def download_and_extract(target_file, url):
#     # Determine the filename from the URL
#     parsed_url = urlparse(url)
#     download_filename = os.path.basename(parsed_url.path) or 'downloaded_file'
    
#     # If the target file exists, we'll still proceed with the download
#     # as we want to update all files in the zip
#     if os.path.exists(target_file):
#         print(f"Note: {target_file} already exists, but we'll proceed with the download and update.")

#     print(f"Downloading from {url}")
#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()  # Raises an HTTPError for bad requests

#         # Save the downloaded content
#         with open(download_filename, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)
#         print(f"File {download_filename} has been downloaded successfully.")

#         # Process the downloaded file
#         if download_filename.lower().endswith('.zip'):
#             print(f"Extracting all files from {download_filename}")
#             try:
#                 with zipfile.ZipFile(download_filename, 'r') as zip_ref:
#                     # Extract all contents, overwriting existing files
#                     zip_ref.extractall(path=".", members=None)
#                 print("All files have been extracted successfully.")
                
#                 # Verify if the target file was part of the extracted files
#                 if os.path.exists(target_file):
#                     print(f"Successfully obtained {target_file}")
#                 else:
#                     print(f"Warning: {target_file} was not found in the extracted files.")
#             except zipfile.BadZipFile:
#                 print("Error: The downloaded file is not a valid zip file.")
#                 return
#         else:
#             # If it's not a zip, just rename it to the target file
#             os.replace(download_filename, target_file)
#             print(f"Downloaded file renamed to {target_file}")

#     except requests.RequestException as e:
#         print(f"Error downloading file: {e}")
#     finally:
#         # Clean up the downloaded zip file if it exists
#         if os.path.exists(download_filename) and download_filename != target_file:
#             os.remove(download_filename)
#             print(f"Cleaned up {download_filename}")





def delete_wav_files():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the 'wav' folder
    wav_folder = os.path.join(script_dir, 'wav')
    
    # Check if the 'wav' folder exists
    if not os.path.exists(wav_folder):
        print(f"The folder {wav_folder} does not exist.")
        return
    
    try:
        # Iterate over all files in the 'wav' folder
        for filename in os.listdir(wav_folder):
            file_path = os.path.join(wav_folder, filename)
            
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {filename}")
        
        print("All files in the 'wav' folder have been deleted.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Fetch the absolute path of the script
script_path = os.path.abspath(__file__)

# Extract the directory from the absolute path
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


#download_and_extract('piper.exe', 'https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip')


# Global queue for TTS files
tts_queue = Queue()
inputque = Queue()

is_playing = False
is_talking = False
outputdev = 12

# Initialize PyAudio
audio = pyaudio.PyAudio()

voicenum = 113# config.config['voice']



# Initialize CrowBrain
brain = None# CrowBrain.Init()
current_conversation_id = 1

saidname = False

def handle_completed_sentence(sentence):
    global current_conversation_id
    global brain
    global voicenum
    if not is_playing:
        #wintts(sentence, "en_US-libritts_r-medium.onnx -s " + str(voicenum))
        #return
        # Generate response using CrowBrain
        response = brain.generate(sentence, current_conversation_id)
        if 'error' not in response:
            ai_response = response['content']
            wintts(ai_response, "en_US-libritts_r-medium.onnx -s " + str(voicenum))
        else:
            print(f"Error in AI response: {response['error']}")
            wintts(response['error'], "en_US-libritts_r-medium.onnx -s " + str(voicenum))

def test_voice(vid):
    global voicenum
    voicenum = vid
    wintts("This is my voice for number " + str(vid), "en_US-libritts_r-medium.onnx -s " + str(vid))

def contains_word(text, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


def wintts(text, model):
    global config
    global saidname
    global stopplayback
    stopplayback=False
    saidname = contains_word(text, config.config['name'])

    # Clean up the text
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\*", "", text)
    text = text.strip()
    remove_chars = "&<>[]|^%:\""
    text = "".join(char for char in text if char not in remove_chars)

    # Split the text into sentences or lines
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    
    # Process each non-empty sentence
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        print(sentence)

        # Generate a random filename
        random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + ".wav"
        random_filename = os.path.join("wav", random_filename)

        # Use shell escaping for the sentence to handle special characters
        safe_sentence = shlex.quote(sentence)
        
        command = f"echo {safe_sentence} | piper -m {model} -f {random_filename}"
        os.system(command)

        # Add the file to the queue
        tts_queue.put(random_filename)

def play_and_delete_wav():
    global stopplayback
    global Running
    global is_playing
    global is_talking
    global recorder
    global saidname
    while Running:
        if not tts_queue.empty():
            filename = tts_queue.get()
            print(filename)
            if(not stopplayback):
                is_talking=True
                play_wav(filename)  # Assuming play_wav is a function to play WAV files
            os.remove(filename)  # Delete the WAV file after playing
            tts_queue.task_done()
        else:
            if is_talking:
                
                #print("Playing Done")
                recorder.interrupt_stop_event.set()
                recorder.stop()
                time.sleep(0.1)
                is_talking=False
                saidname=False
            time.sleep(0.1)  # Sleep for a short duration to avoid busy waiting
    print("WAVE THREAD END")

wf = None
vol = 0

stopplayback = False


def callback(in_data, frame_count, time_info, status):
    global vol
    global wf
    global stopplayback
    if stopplayback:
        return (None, pyaudio.paComplete)
    # Read data from file
    data = wf.readframes(frame_count)
    d = np.frombuffer(data, dtype=np.int16)
    v = np.average(np.abs(d))  
    if (not np.isnan(v)):
        vol = v * .0001
    else:
        vol = 0

    return (data, pyaudio.paContinue)



def play_wav(wavefile):
    #print("wavstart")
    athread = threading.Thread(target=wavethread, args=(wavefile,))
    athread.start()
    while is_playing:
        time.sleep(0.01)
    #print("end of fun")

def wavethread(wavefile):
    global wf  # Make wf global so it can be accessed by callback
    global is_playing
    global outputdev
    is_playing = True
    # Open the wav file
    wf = wave.open(wavefile, 'rb')

    p = pyaudio.PyAudio()
    RATE = wf.getframerate()
    CHUNK = int(RATE / 10)

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=RATE,
                    output=True,
                    output_device_index=outputdev,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    # Start the stream
    stream.start_stream()

    # Keep the script running while the audio is playing
    while stream.is_active():
        time.sleep(0.1)

    # Stop stream
    stream.stop_stream()
    stream.close()

    # Close PyAudio and wave file
    wf.close()
    p.terminate()
    is_playing=False


def process_text(text):
    global mute
    global Running
    global Sleeping
    print(text)
    if not mute and not Sleeping and Running:
        handle_completed_sentence(text)

mute = False
recorder = None
listening = False

def my_start_callback():
    global is_talking
    global listening
    global vc
    global lastvolume
    print("Recording started!")
    if(not is_talking):
        listening=True
        lastvolume = vc.get_volume()
        print(lastvolume)
        vc.set_volume(25)

def my_stop_callback():
    global listening
    listening=False
    vc.set_volume(lastvolume)
    print("Recording stopped!")

def tupdate(text):
    global stopplayback
    global brain
    global saidname
    global current_conversation_id
    global Sleeping
    global crow
    global is_talking
    global listening
    print(text)
    if Sleeping:
        if(contains_word(text.lower(),config.config['name'].lower())):
            print("wake")
            Sleeping = False
            crow.SleepTimer = 0
    else:
        crow.SleepTimer = 0
    if is_talking:
        if(contains_word(text.lower(),config.config['name'].lower()) and not saidname):
            #stop current playback
            print("INTERUPT")
            stopplayback=True
            brain.addSystemMessage("[Interupted]",current_conversation_id)
            return
    else:
        listening=True

def transcriptstart():
    global crow
    crow.SleepTimer = 0



whisperprompt = ""



def aibrains():
    global Running
    global mute
    global recorder
    print("BRAINS")
    try:
        print('Listening ... (press Shift + ESC to exit)')
        
        while Running:
            recorder.text(process_text)
       
    except Exception as e:
        print(f'An error occurred in aibrains: {e}')
        
    finally:
        print("end brain")
        Running = False


recorder = None
Sleeping = True
Running = True
crow = None
config = None

vc = Volume.VolumeControl(outputdev)
lastvolume = vc.get_volume()

def main():
    global crow
    global Running
    global mute
    global voicenum
    global vol
    global is_playing
    global listening
    global Sleeping
    global recorder
    global outputdev
    #list_output_devices()

    print("MAIN")

    print(lastvolume)
    try:

        while Running:
            #Sleeping=False

            #update crow visuals
            crow.listen = listening
            crow.Sleeping = Sleeping
            if(is_playing):
                crow.volume = vol
                crow.SleepTimer= 0
            crow.Update()


            if(listening):
                crow.SleepTimer= 0

            if(not Sleeping and crow.SleepTimer>15000):
                print("Sleep Mode")
                Sleeping=True
                crow.SleepTimer = 0

            if keyboard.is_pressed('Esc') and keyboard.is_pressed('Shift'):
                print("Escape key pressed. Exiting loop.")
                break
            if(not crow.running):
                break
        
    except Exception as e:
        print(f'An error occurred in the main loop: {e}')
    finally:
        Running = False
        shutdown()

def shutdown():
    global Running, recorder, brain, crow
    print("Initiating shutdown...")
    
    # Stop the main loop
    Running = False
    
    # Stop the recorder
    if recorder:
        print("Shutting down recorder...")
        recorder.abort()
        recorder.shutdown()
    

    # Stop the brain server
    if brain and brain.server_thread:
        print("Shutting down brain server...")
        brain.app.config['TESTING'] = True  # This should make the server more responsive to shutdown
        requests.get('http://localhost:5000/shutdown')  # Assuming you add a /shutdown route
        brain.server_thread.join(timeout=5)
    
    # Stop Crow
    if crow:
        print("Shutting down Crow...")
        crow.End()
    
    # Close PyAudio
    #if 'audio' in globals():
        print("Closing PyAudio...")
        audio.terminate()

    print("Forcing termination of remaining threads...")
    for thread in [brainthread, playback_thread]:
        if thread and thread.is_alive():
            force_thread_termination(thread)

    print("Shutdown complete. Exiting...")
    os._exit(0)  # Force exit the Python process
    
    print("Shutdown complete.")

def force_thread_termination(thread):
    if thread.is_alive():
        print(f"Force terminating thread: {thread.name}")
        tid = thread.ident
        if tid is not None:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
                print("Exception raise failed")


def name_to_index(device_name, is_input):
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['name'] == device_name:
            if (is_input and device_info['maxInputChannels'] > 0) or \
               (not is_input and device_info['maxOutputChannels'] > 0):
                p.terminate()
                return i
    p.terminate()
    return None  # Device not found

def get_audio_output_devices():
    """
    Prints a list of audio output devices and their device indices on Windows.
    """
    p = pyaudio.PyAudio()
    
    print("Audio Output Devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["maxOutputChannels"] > 0:
            print(f"{i}: {device_info['name']}")
    
    p.terminate()

if __name__ == '__main__':

    get_audio_output_devices()
    delete_wav_files()
    config = CrowConfig.config()
    ainame = config.config['name']
    whisperprompt = "Talking to" + ainame
    voicenum = config.config['voice']
    micnum = name_to_index( config.config['mic'],True)
    outputdev = name_to_index( config.config['speaker'],False)
    print("Output Device: " + str(outputdev))
    if(micnum is None):
        micnum=0
        print("Mic Not Set")

    recorder_config = {
        'input_device_index': micnum,
        'spinner': False,
        'model': 'base.en',
        'language': 'en',
        'silero_sensitivity': 0.4,
        'silero_use_onnx': True,
        'webrtc_sensitivity': 2,
        'device':'cuda',
        'post_speech_silence_duration': 1.0,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.2,
        'realtime_model_type': 'tiny.en',
        'on_recording_start': my_start_callback,
        'on_recording_stop': my_stop_callback,
        'on_transcription_start': transcriptstart,
        'on_realtime_transcription_update': tupdate, 
        #'on_vad_detect_start': vadstart,
        #'on_vad_detect_stop': vadstop,
        #'on_realtime_transcription_stabilized': process_text,
        'initial_prompt':whisperprompt,
    }
    recorder = AudioToTextRecorder(**recorder_config)
    brainthread = threading.Thread(target=aibrains)
    brainthread.start()
    time.sleep(1)
    playback_thread = threading.Thread(target=play_and_delete_wav)
    playback_thread.daemon = True  # Daemonize thread
    playback_thread.start()
    brain = CrowBrain.Init()
    brain.config = config
    brain.set_test_voice_callback(test_voice)
    crow = Crow.Init()
    wintts("Crow is Online", "en_US-libritts_r-medium.onnx -s " + str(voicenum))
    main()
    print("END")