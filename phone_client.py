#This program is designed to run on a mobile device using Kivy.
#It connects to a laptop server to receive video frames processed by a YOLO model.
#The frames are displayed in real time on the mobile device.
#Ensure the necessary libraries installed: kivy, opencv-python, numpy.

import socket
import cv2
import pickle
import struct
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import threading
import time
import numpy as np

#This is a Kivy application that connects to a laptop server to receive video frames.
class YOLOApp(App):
    # Kivy application class for the YOLO client.
    def build(self):
        Window.clearcolor = (0, 0, 0, 1)
        layout = BoxLayout(orientation='vertical')
        self.status_label = Label(text="Connecting to laptop server...", size_hint=(1, 0.05), font_size='16sp')
        self.img_widget = Image(size_hint=(1, 0.95), allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.status_label)
        layout.add_widget(self.img_widget)
        
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.is_connected = False
        self.network_thread = threading.Thread(target=self.network_loop, daemon=True)
        self.network_thread.start()
        
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return layout

    #This method runs in a separate thread to handle network communication.
    def network_loop(self):
        host = '192.168.241.49' #Replace with the network's IPv4 address.
        port = 9999 #Replace with the port the server is listening on.
        
        #Attempt to connect to the server in a loop.
        while True:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_socket.settimeout(5)
            
            try:
                client_socket.connect((host, port))
                self.is_connected = True
                Clock.schedule_once(lambda dt: self.set_status("Connected. Starting camera..."))
            except (socket.error, ConnectionRefusedError, socket.timeout):
                self.is_connected = False
                #If connection fails, update the status and wait before retrying.
                Clock.schedule_once(lambda dt: self.set_status(f"Connection failed. Retrying in 3s..."))
                time.sleep(3)
                continue

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                Clock.schedule_once(lambda dt: self.set_status("Failed to open camera."))
                client_socket.close()
                time.sleep(3)
                continue
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            Clock.schedule_once(lambda dt: self.set_status("Streaming..."))
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]

            try:
                while self.is_connected:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                    if not ret:
                        continue
                        
                    message = struct.pack("Q", len(buffer)) + buffer.tobytes()
                    client_socket.sendall(message)

                    payload_size = struct.calcsize("Q")
                    data = b""
                    while len(data) < payload_size:
                        packet = client_socket.recv(65536)
                        if not packet:
                            raise ConnectionError("Server disconnected")
                        data += packet
                    
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("Q", packed_msg_size)[0]

                    while len(data) < msg_size:
                        data += client_socket.recv(65536)
                    
                    frame_data = data[:msg_size]
                    data = data[msg_size:]

                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    processed_frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    if processed_frame is not None:
                        with self.frame_lock:
                            self.current_frame = processed_frame

            except (ConnectionResetError, ConnectionError, struct.error, OSError) as e:
                self.is_connected = False
                Clock.schedule_once(lambda dt: self.set_status("Connection lost. Reconnecting..."))
            except Exception as e:
                self.is_connected = False
                Clock.schedule_once(lambda dt: self.set_status(f"Error: {str(e)[:50]}..."))
            finally:
                cap.release()
                client_socket.close()
                time.sleep(2)

    #This method updates the status label in the UI.
    def set_status(self, text):
        self.status_label.text = text

    #This method updates the image widget with the latest frame.
    def update(self, dt):
        with self.frame_lock:
            if self.current_frame is not None:
                frame = self.current_frame
                self.current_frame = None
            else:
                return
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flipped = cv2.flip(rgb_frame, 0)
            buf = flipped.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.img_widget.texture = texture
        except Exception as e:
            print(f"Display error: {e}")

    #This method is called when the application stops.
    def on_stop(self):
        self.is_connected = False

if __name__ == "__main__":
    YOLOApp().run()