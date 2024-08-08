import pygame
import sys
import os
import random
import psutil
import CrowConfig
import webbrowser


# For Windows, we need to set these environment variables
# to make the window clickthrough
if os.name == 'nt':
    import ctypes
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        pass
    import win32gui
    import win32con
    import win32api  # Add this import
    import win32process
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))


# Set the working directory to the script's directory
os.chdir(script_dir)
pygame_icon = pygame.image.load('crow.ico')
pygame.display.set_icon(pygame_icon)

def load_sprite_animation(name):
    sprites = []
    index = 1
    while True:
        filename = f"{name}{index}.png"
        try:
            sprite = pygame.image.load(filename).convert_alpha()
            sprites.append(sprite)
            index += 1
        except FileNotFoundError:
            break
    return sprites

def load_sprite_sheet(filename, sprite_size):
    sprite_sheet = pygame.image.load(filename).convert_alpha()
    sprite_width, sprite_height = sprite_size
    sheet_width, sheet_height = sprite_sheet.get_size()
    sprites = []
    
    for y in range(0, sheet_height, sprite_height):
        for x in range(0, sheet_width, sprite_width):
            sprite = sprite_sheet.subsurface((x, y, sprite_width, sprite_height))
            if not is_blank(sprite):
                sprites.append(sprite)
    
    return sprites

def is_blank(sprite):
    for x in range(sprite.get_width()):
        for y in range(sprite.get_height()):
            if sprite.get_at((x, y)).a != 0:
                return False
    return True



class CrowAnimationController:
    def __init__(self, crow):
        self.crow = crow
        self.head_index = 0
        self.idle_index = 0
        self.fly_index = 0
        self.blink_timer = 0
        self.lookback_timer = 0


        # Animation speeds
        self.head_animation_speed = 10  # frames per second
        self.idle_animation_speed = 1  # frames per second
        self.fly_animation_speed = 10  # frames per second
        self.blink_interval = 10  # frames between blinks
        self.lookback_interval = 120  # frames between lookbacks

        # Timers
        self.idle_timer = 0
        self.fly_timer = 0
        self.blink_cooldown = 0
        self.lookback_cooldown = 0

    def update(self):
        # Update head animation based on volume
        if self.crow.volume > 0:
            target_index = int(len(self.crow.head) * self.crow.volume)
            self.head_index = min(target_index, len(self.crow.head) - 1)
        else:
            self.head_index = 0

        # Update idle animation
        if not self.crow.flying:
            self.idle_timer += 1
            if self.idle_timer >= (60 / self.idle_animation_speed):
                self.idle_index = (self.idle_index + 1) % len(self.crow.idle)
                self.idle_timer = 0

        # Update fly animation
        if self.crow.flying:
            self.fly_timer += 1
            if self.fly_timer >= (60 / self.fly_animation_speed):
                self.fly_index = (self.fly_index + 1) % len(self.crow.fly)
                self.fly_timer = 0

        # Update blink timer
        if self.crow.volume == 0:
            self.blink_cooldown -= 1
            if self.blink_cooldown <= 0:
                self.blink_timer = self.blink_interval
                self.blink_cooldown = random.randint(self.blink_interval, self.blink_interval * 20)
                if random.random() < 0.1:  # 10% chance to look back instead of blink
                    self.lookback_timer = 120
                else:
                    self.lookback_timer = 0

        # Decrement blink timer
        if self.blink_timer > 0:
            self.blink_timer -= 1

        # Update lookback timer
        if self.lookback_timer > 0:
            self.lookback_timer -= 1
            self.lookback_cooldown = self.lookback_interval

        # Update lookback cooldown
        if self.lookback_cooldown > 0:
            self.lookback_cooldown -= 1

    def render(self, screen):
        # Render body
        if self.crow.flying:
            screen.blit(self.crow.fly[self.fly_index], (0, 0))
        else:
            screen.blit(self.crow.idle[self.idle_index], (0, 0))

        # Render head
        if self.crow.listen and not self.crow.Sleeping:
            screen.blit(self.crow.headtilt, (0, 0))           
        elif not self.crow.flying and self.lookback_timer > 0 and self.crow.volume == 0:
            screen.blit(self.crow.headlookback, (0, 0))
        elif self.blink_timer > 0 and self.crow.volume == 0:
            screen.blit(self.crow.headblink, (0, 0))
        else:
            screen.blit(self.crow.head[self.head_index], (0, 0))
        
        return screen
    
def Init():
    return DesktopPet.get_instance()

class DesktopPet:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if DesktopPet._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DesktopPet._instance = self
        #pygame.init()
        self.last_click_time = 0
        self.double_click_threshold = 500  # milliseconds
        self.config = CrowConfig.config()
        self.Sleeping = False
        self.SleepTimer = 0
        self.scale = self.config.config['scale']   
        # Set up the window
    
        self.screen = pygame.display.set_mode((64*self.scale, 64*self.scale), pygame.NOFRAME)
        pygame.display.set_caption("Crow")
        print("title set")
        
        self.right = False


        self.volume = 0 #a float that is a voice volume from 0 to 1 
        self.listen = False 
        self.flying = False

        # Load sprites
        self.head = load_sprite_sheet("images/crowhead.png",(64,64)) #  a set of mouth animations the last is wide open, the first is closed. we should lerp to these from the volume
        self.headblink = pygame.image.load("images/crowhead-blink.png").convert_alpha() #blink randomly every once in a while as long as talking is 0
        self.headlookback = pygame.image.load("images/crowhead-lookback.png").convert_alpha() #rarely instead of blink
        self.headtilt = pygame.image.load("images/crowhead-tilt.png").convert_alpha()  #if listen is true, this should render    
        
        self.idle = load_sprite_animation("images/crow-idle") #when not moving
        self.fly = load_sprite_sheet("images/crowfly.png",(64,64)) #when moving to a new spot
        print("Sprites Loaded")


        
        # Set the window to be transparent
        self.screen.set_colorkey((0,0,0))  # Black will be transparent, any sprites can not be black
        self.screen.fill((0,0,0))
        
        # For Windows, set the window to be clickthrough
        if os.name == 'nt':
            hwnd = pygame.display.get_wm_info()["window"]
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                                   win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
            win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0,0,0), 0, win32con.LWA_COLORKEY)
        
        
        self.clock = pygame.time.Clock()
        self.dragging = False
        self.hwnd = pygame.display.get_wm_info()["window"]
        self.set_always_on_top()
        self.current_window = None
        self.target_x = 0
        self.target_y = 0
        self.current_x = 0
        self.current_y = 0
        self.move_speed = 4  # Adjust this to change animation speed
        self.wincheck = 0

        self.running = True
        self.animation_controller = CrowAnimationController(self)
        self.wincheck = 0



    def set_always_on_top(self):
        win32gui.SetWindowPos(
            self.hwnd,
            win32con.HWND_TOPMOST,
            0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE    
        )


    def get_focused_window(self):
        focused = win32gui.GetForegroundWindow()
        if focused == self.hwnd:
            return None  # Return None if our window is focused
        title = win32gui.GetWindowText(focused)
        if title:
            return focused
        return None
    
    def get_window_info(self, hwnd):
        if hwnd:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                return rect
            except win32gui.error:
                return None
        return None

    def move_to_window(self, hwnd):
        if hwnd and hwnd != self.hwnd:  # Only move if it's not our own window
            rect = self.get_window_info(hwnd)
            if rect:
                window_width = rect[2] - rect[0]
                window_bottom = rect[3]
                
                # Set new target x and y positions
                self.target_x = random.randint(rect[0], rect[2] - self.screen.get_width())
                self.target_y = window_bottom - self.screen.get_height()   
            else:
                # If we can't get window info, move to a default position
                self.target_x = 0
                self.target_y = win32api.GetSystemMetrics(win32con.SM_CYSCREEN) - self.screen.get_height()

    def move_to_taskbar_clock(self):
        # Get the screen size
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

        # Get the taskbar height
        taskbar_hwnd = win32gui.FindWindow("Shell_TrayWnd", None)
        if taskbar_hwnd:
            taskbar_rect = win32gui.GetWindowRect(taskbar_hwnd)
            taskbar_height = taskbar_rect[3] - taskbar_rect[1]
        else:
            # If we can't find the taskbar, assume a default height
            taskbar_height = 40

        # Calculate the position
        # We'll position it slightly to the left of the very corner to avoid overlapping with any system tray icons
        offset_from_right = 100  # Adjust this value as needed
        self.target_x = screen_width - self.screen.get_width() - offset_from_right
        self.target_y = screen_height - self.screen.get_height() - taskbar_height

    def update_position(self):
        # Calculate the distance between the current position and the target position
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        
        # Calculate the length of the distance vector
        distance = (dx ** 2 + dy ** 2) ** 0.5
        
        # If the distance is very small, just snap to the target position
        if distance < self.move_speed:
            self.current_x = self.target_x
            self.current_y = self.target_y
            self.flying = False
        else:
            # Move the pet by a fraction of the distance each frame
            fraction = self.move_speed / distance
            self.current_x += dx * fraction
            self.current_y += dy * fraction
            self.flying = True
            if self.target_x > self.current_x:
                self.right = True
            else:
                self.right = False
        
        # Set new window position
        win32gui.SetWindowPos(self.hwnd, 0, 
                            int(self.current_x),
                            int(self.current_y),
                            0, 0, win32con.SWP_NOSIZE | win32con.SWP_NOZORDER)
        

            

    def launch_webpage(self):
        url = "http://127.0.0.1:" + str(self.config.config['port'])
        webbrowser.open(url)


    def Update(self):
        #while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    current_time = pygame.time.get_ticks()
                    if current_time - self.last_click_time < self.double_click_threshold:
                        self.launch_webpage()
                    self.last_click_time = current_time

 

        self.wincheck+=1
        if self.wincheck > 300:
            self.wincheck = 0
            if(self.Sleeping):
                self.move_to_taskbar_clock()
            else:
                # Check for new focused window
                focused_window = self.get_focused_window()
                if focused_window and focused_window != self.current_window:
                    self.current_window = focused_window
                    self.move_to_window(self.current_window)
                    self.last_window_info = self.get_window_info(self.current_window)
            
            # Check if current window has moved or resized
        if not self.Sleeping and self.current_window:
            current_window_info = self.get_window_info(self.current_window)
            if not self.flying and current_window_info != self.last_window_info:
                self.move_to_window(self.current_window)
                self.last_window_info = current_window_info

        # Update position for animation
        self.update_position()
        
        
        # Clear the screen
        self.screen.fill((0,0,0))  # Fill with the transparent color
        
        # Draw the current sprite
        temp_screen = pygame.Surface((64, 64), pygame.SRCALPHA)
        self.animation_controller.update()
        self.animation_controller.render(temp_screen)

        if self.right:
            temp_screen = pygame.transform.flip(temp_screen, True, False)

        scaled_screen = pygame.transform.scale(temp_screen, (64 * self.scale, 64 * self.scale))

        self.screen.blit(scaled_screen, (0, 0))
        # Update the display
        pygame.display.flip()
        
        self.SleepTimer += self.clock.get_time()
        self.clock.tick(60)
    
    def End(self):
        self.running = False
        print("crow end")
        pygame.quit()
