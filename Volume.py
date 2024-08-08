import platform

class VolumeControl:
    def __init__(self, device_index=None):
        self.system = platform.system()
        self.device_index = device_index

        if self.system == "Windows":
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            device = AudioUtilities.GetSpeakers()
            interface = device.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = interface.QueryInterface(IAudioEndpointVolume)
        elif self.system == "Linux":
            import pulsectl
            self.pulse = pulsectl.Pulse('volume-control')
        elif self.system == "Darwin":  # macOS
            import subprocess

    def get_volume(self):
        if self.system == "Windows":
            return round(self.volume.GetMasterVolumeLevelScalar() * 100)
        elif self.system == "Linux":
            sinks = self.pulse.sink_list()
            if self.device_index is not None and 0 <= self.device_index < len(sinks):
                return round(sinks[self.device_index].volume.value_flat * 100)
            elif sinks:
                return round(sinks[0].volume.value_flat * 100)
        elif self.system == "Darwin":
            cmd = f"osascript -e 'output volume of (get volume settings)'"
            if self.device_index is not None:
                cmd = f"osascript -e 'tell application \"System Events\" to get volume settings of audio device \"{self.device_index}\"'"
            return int(subprocess.check_output(cmd, shell=True).strip().split(", ")[0].split(":")[1])

    def set_volume(self, volume):
        volume = max(0, min(100, volume))  # Ensure volume is between 0 and 100
        if self.system == "Windows":
            self.volume.SetMasterVolumeLevelScalar(volume / 100, None)
        elif self.system == "Linux":
            sinks = self.pulse.sink_list()
            if self.device_index is not None and 0 <= self.device_index < len(sinks):
                self.pulse.volume_set_all_chans(sinks[self.device_index], volume / 100)
            elif sinks:
                self.pulse.volume_set_all_chans(sinks[0], volume / 100)
        elif self.system == "Darwin":
            cmd = f"osascript -e 'set volume output volume {volume}'"
            if self.device_index is not None:
                cmd = f"osascript -e 'tell application \"System Events\" to set volume of audio device \"{self.device_index}\" to {volume}'"
            subprocess.call(cmd, shell=True)

    @staticmethod
    def list_devices():
        system = platform.system()
        if system == "Windows":
            from pycaw.pycaw import AudioUtilities
            return [device.FriendlyName for device in AudioUtilities.GetAllDevices()]
        elif system == "Linux":
            import pulsectl
            with pulsectl.Pulse('device-list') as pulse:
                return [sink.name for sink in pulse.sink_list()]
        elif system == "Darwin":
            import subprocess
            cmd = "system_profiler SPAudioDataType | grep -A 1 'Output:' | grep -v 'Output:' | awk -F: '{print $1}' | sed 's/^[ \t]*//'"
            return subprocess.check_output(cmd, shell=True).decode().strip().split('\n')
