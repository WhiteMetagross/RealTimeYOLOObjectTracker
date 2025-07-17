#This program is part of a Real Time YOLO Object Tracker project.
#It connects to a laptop server to receive video frames processed by a YOLO model.
#The frames are processed and displayed in real time on a mobile device using Kivy.

import socket
import cv2
import struct
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes
import threading
import time
import colorsys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from collections import defaultdict, deque
import pickle
import logging
import gc
from threading import Lock
import traceback

#Set up logging.
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

#Set the configuration directory for YOLO.
#This is where the YOLO model configuration files will be stored.
CONFIG_DIR = "./ultralytics_config"
os.environ["YOLO_CONFIG_DIR"] = CONFIG_DIR
os.makedirs(CONFIG_DIR, exist_ok=True)

#Define the COCO classes used by the YOLO model.
#These classes are used to label the detected objects in the video frames.
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

#Function to get the class name from the class ID.
#This function returns the name of the class corresponding to the given class ID.
def get_class_name(class_id):
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"

#Define the convolutional layer with optional Instance Normalization.
#This layer is used in the OSNet architecture for feature extraction.
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    #This method performs the forward pass of the convolutional layer.
    #It applies the convolution, normalization, and activation functions to the input tensor.
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#Define the OSBlock used in the OSNet architecture.
#This block consists of multiple convolutional layers and is used for feature extraction.
class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, IN=False):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = ConvLayer(in_channels, mid_channels, 1, IN=IN)
        self.conv2a = ConvLayer(mid_channels, mid_channels, 3, padding=1, IN=IN)
        self.conv2b = ConvLayer(mid_channels, mid_channels, (1, 3), padding=(0, 1), IN=IN)
        self.conv2c = ConvLayer(mid_channels, mid_channels, (3, 1), padding=(1, 0), IN=IN)
        self.conv3 = ConvLayer(mid_channels * 3, out_channels, 1, IN=IN)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ConvLayer(in_channels, out_channels, 1, IN=IN)
    
    #This method performs the forward pass of the OSBlock.
    #It applies the convolutional layers and combines the outputs with a residual connection.
    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2 = torch.cat([x2a, x2b, x2c], 1)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = x3 + residual
        return out

#Define the OSNetAIN architecture.
#This architecture is used for feature extraction in the ReID (Re-Identification) task.
class OSNetAIN(nn.Module):
    def __init__(self, num_classes=1000, blocks=[2, 2, 2], layers=[64, 256, 384, 512], channels=[64, 256, 384, 512], feature_dim=512, **kwargs):
        super(OSNetAIN, self).__init__()
        self.feature_dim = feature_dim
        
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.conv2 = self._make_layer(OSBlock, channels[0], channels[1], blocks[0], IN=True)
        self.conv3 = self._make_layer(OSBlock, channels[1], channels[2], blocks[1], IN=True)
        self.conv4 = self._make_layer(OSBlock, channels[2], channels[3], blocks[2], IN=True)
        
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], feature_dim)
        
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    #This method creates a sequential layer of OSBlocks.
    #It allows for stacking multiple OSBlocks to form a deeper network.
    def _make_layer(self, block, in_channels, out_channels, blocks, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, IN=IN))
        return nn.Sequential(*layers)
    
    #This method performs the forward pass of the OSNetAIN model.
    #It applies the convolutional layers, pooling, and fully connected layers to the input tensor.
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

#Define the ReID feature extractor using the OSNetAIN model.
class ReIDFeatureExtractor:
    def __init__(self, model_path="osnet_ain_x1_0_imagenet.pth", device='cuda'):
        self.device = device
        self.model = OSNetAIN(num_classes=1000, feature_dim=512)
        self.lock = Lock()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ReID model not found at {model_path}. Please ensure the model file exists.")
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('classifier.'):
                    filtered_state_dict[key] = value
            
            self.model.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"Successfully loaded ReID model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load ReID model: {str(e)}")
            raise RuntimeError(f"Failed to load ReID model: {str(e)}")
        
        self.model.to(device)
        self.model.eval()
        self.model.half()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.batch_size = 32
    
    #This method extracts features from a batch of image crops.
    #It processes the crops, converts them to tensors, and passes them through the model to get the features.
    def extract_features(self, image_crops):
        if len(image_crops) == 0:
            return np.array([])
        
        with self.lock:
            try:
                batch_tensors = []
                valid_crops = []
                
                for crop in image_crops:
                    if crop is not None and crop.shape[0] > 4 and crop.shape[1] > 4:
                        try:
                            if len(crop.shape) == 3 and crop.shape[2] == 3:
                                tensor = self.transform(crop)
                                batch_tensors.append(tensor)
                                valid_crops.append(crop)
                        except:
                            continue
                
                if len(batch_tensors) == 0:
                    return np.array([])
                
                features_list = []
                for i in range(0, len(batch_tensors), self.batch_size):
                    batch_slice = batch_tensors[i:i+self.batch_size]
                    batch = torch.stack(batch_slice).to(self.device).half()
                    
                    with torch.no_grad():
                        features = self.model(batch)
                        features_list.append(features.cpu().numpy())
                    
                    del batch
                
                if features_list:
                    return np.vstack(features_list)
                else:
                    return np.array([])
                    
            except Exception as e:
                logger.error(f"Error in feature extraction: {e}")
                return np.array([])

#Define the TrackState class to hold the state of each tracked object.
class TrackState:
    def __init__(self, track_id, box, feature, class_id, confidence, frame_count):
        self.track_id = track_id
        self.box = box
        self.feature = feature
        self.class_id = class_id
        self.confidence = confidence
        self.last_seen = frame_count
        self.feature_history = deque([feature], maxlen=2)
        self.position_history = deque([box], maxlen=3)
        self.hit_streak = 1
        self.time_since_update = 0
        self.age = 0
        self.stable = False
    
    #This method updates the track state with new information.
    def update(self, box, feature, confidence, frame_count):
        self.box = box
        self.feature = feature
        self.confidence = confidence
        self.last_seen = frame_count
        self.feature_history.append(feature)
        self.position_history.append(box)
        self.hit_streak += 1
        self.time_since_update = 0
        self.age += 1
        
        if self.hit_streak >= 1:
            self.stable = True
    
    #This method predicts the next position of the tracked object based on its history.
    def predict(self):
        self.time_since_update += 1
        self.age += 1
        
        if len(self.position_history) >= 2:
            last_box = self.position_history[-1]
            prev_box = self.position_history[-2]
            
            dx = last_box[0] - prev_box[0]
            dy = last_box[1] - prev_box[1]
            
            predicted_box = [
                last_box[0] + dx * 0.8,
                last_box[1] + dy * 0.8,
                last_box[2] + dx * 0.2,
                last_box[3] + dy * 0.2
            ]
            
            return predicted_box
        
        return self.box
    
    #This method returns the average feature from the feature history.
    def get_average_feature(self):
        if len(self.feature_history) > 0:
            return np.mean(self.feature_history, axis=0)
        return self.feature

#Define the ReIDTracker class to manage tracking of objects across frames.
class ReIDTracker:
    def __init__(self, reid_extractor, max_cosine_distance=0.35, max_time_lost=10, min_confidence=0.3):
        self.reid_extractor = reid_extractor
        self.max_cosine_distance = max_cosine_distance
        self.max_time_lost = max_time_lost
        self.min_confidence = min_confidence
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.lock = Lock()
        self.min_box_area = 200
        self.max_tracks = 30
        
    #This method calculates the cosine distance between two feature vectors.
    def cosine_distance(self, a, b):
        try:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1 - np.dot(a, b) / (norm_a * norm_b)
        except:
            return 1.0
    
    #This method calculates the area of a bounding box.
    def box_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    #This method calculates the Intersection over Union (IoU) between two bounding boxes.
    def box_iou(self, box1, box2):
        try:
            x1_max = max(box1[0], box2[0])
            y1_max = max(box1[1], box2[1])
            x2_min = min(box1[2], box2[2])
            y2_min = min(box1[3], box2[3])
            
            if x2_min <= x1_max or y2_min <= y1_max:
                return 0.0
            
            intersection = (x2_min - x1_max) * (y2_min - y1_max)
            area1 = self.box_area(box1)
            area2 = self.box_area(box2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    #This method updates the tracker with new detections and returns the stable tracks.
    def update(self, detections, frame):
        with self.lock:
            try:
                self.frame_count += 1
                
                valid_detections = []
                crops = []
                
                for det in detections:
                    if len(det) >= 6:
                        x1, y1, x2, y2 = map(int, det[:4])
                        conf = float(det[4])
                        cls = int(det[5])
                        
                        if conf < self.min_confidence:
                            continue
                        
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        box = [x1, y1, x2, y2]
                        
                        if self.box_area(box) >= self.min_box_area and x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                valid_detections.append((box, conf, cls))
                                crops.append(crop)
                
                if len(crops) == 0:
                    self._age_tracks()
                    return self._get_stable_tracks()
                
                features = self.reid_extractor.extract_features(crops)
                
                if features.size == 0:
                    self._age_tracks()
                    return self._get_stable_tracks()
                
                matched_tracks = []
                unmatched_detections = list(range(len(valid_detections)))
                matched_track_ids = set()
                
                for i, (box, conf, cls) in enumerate(valid_detections):
                    if i >= len(features):
                        continue
                        
                    feature = features[i]
                    best_match_id = None
                    best_score = float('inf')
                    
                    for track_id, track in self.tracks.items():
                        if track_id in matched_track_ids:
                            continue
                            
                        if track.class_id != cls:
                            continue
                        
                        predicted_box = track.predict()
                        iou = self.box_iou(box, predicted_box)
                        
                        if iou < 0.03:
                            continue
                        
                        avg_feature = track.get_average_feature()
                        cos_distance = self.cosine_distance(feature, avg_feature)
                        
                        combined_score = cos_distance + (1 - iou) * 0.1
                        
                        if combined_score < self.max_cosine_distance and combined_score < best_score:
                            best_score = combined_score
                            best_match_id = track_id
                    
                    if best_match_id is not None:
                        self.tracks[best_match_id].update(box, feature, conf, self.frame_count)
                        matched_tracks.append((best_match_id, box, cls, conf))
                        matched_track_ids.add(best_match_id)
                        if i in unmatched_detections:
                            unmatched_detections.remove(i)
                
                for i in unmatched_detections:
                    if len(self.tracks) >= self.max_tracks:
                        self._remove_oldest_track()
                    
                    if i < len(valid_detections) and i < len(features):
                        box, conf, cls = valid_detections[i]
                        feature = features[i]
                        
                        track_id = self.next_id
                        self.next_id += 1
                        
                        self.tracks[track_id] = TrackState(
                            track_id, box, feature, cls, conf, self.frame_count
                        )
                        matched_tracks.append((track_id, box, cls, conf))
                
                self._age_tracks()
                return self._get_stable_tracks()
                
            except Exception as e:
                logger.error(f"Error in tracker update: {e}")
                return []
    
    #This method ages the tracks and removes those that have not been seen for too long.
    def _age_tracks(self):
        to_remove = []
        for track_id, track in self.tracks.items():
            if (self.frame_count - track.last_seen > self.max_time_lost or 
                track.time_since_update > self.max_time_lost):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    #This method removes the oldest track if the maximum number of tracks is exceeded.
    def _remove_oldest_track(self):
        if self.tracks:
            oldest_id = min(self.tracks.keys(), key=lambda x: self.tracks[x].last_seen)
            del self.tracks[oldest_id]
    
    #This method returns the stable tracks that have been seen recently and have a stable state.
    def _get_stable_tracks(self):
        stable_tracks = []
        for track_id, track in self.tracks.items():
            if track.stable:
                stable_tracks.append((track_id, track.box, track.class_id, track.confidence))
        return stable_tracks

#Define the ConnectionHandler class to manage the connection and processing of frames.
class ConnectionHandler:
    def __init__(self, model, reid_tracker):
        self.model = model
        self.reid_tracker = reid_tracker
        self.frame_counter = 0
        self.last_process_time = time.time()
    
    #This method processes the incoming video frames from the client.
    def process_frame(self, frame):
        try:
            if frame is None or frame.size == 0:
                return None
                
            self.frame_counter += 1
            
            results = self.model(frame, verbose=False, device='cuda', conf=0.3, iou=0.3, half=True)
            result = results[0]

            detections = []
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        if conf > 0.25:
                            detections.append([x1, y1, x2, y2, conf, cls])
                    except:
                        continue

            tracked_objects = self.reid_tracker.update(detections, frame)
            return tracked_objects
            
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            return []

track_colors = {}
color_lock = Lock()

#This method generates a unique color for each track ID.
def get_track_color(track_id):
    with color_lock:
        if track_id not in track_colors:
            hue = (track_id * 0.618033988749895) % 1.0 #Golden ratio for unique colors generation.
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
            track_colors[track_id] = tuple(int(c * 255) for c in reversed(rgb))
        return track_colors[track_id]

#This method annotates the frame with bounding boxes and labels for tracked objects.s
def annotate_frame(frame, tracked_objects):
    if frame is None:
        return frame
        
    try:
        h, w = frame.shape[:2]
        
        for track_id, box, class_id, conf in tracked_objects:
            try:
                x1, y1, x2, y2 = map(int, box)
                
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                color = get_track_color(track_id)
                class_name = get_class_name(class_id)
                label = f"ID:{track_id} {class_name} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (center_x, center_y), 3, color, -1)
                
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                label_bg_y1 = max(0, y1 - text_height - baseline - 2)
                label_bg_y2 = y1
                
                cv2.rectangle(frame, (x1, label_bg_y1), (x1 + text_width, label_bg_y2), color, -1)
                cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
            except:
                continue
        
        status_text = f"Tracked: {len(tracked_objects)}"
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
        
    except Exception as e:
        logger.error(f"Error in annotate_frame: {e}")
        return frame

#This function handles the client connection and processes incoming frames.
def handle_client(conn, addr, handler):
    payload_size = struct.calcsize("Q")
    data = b""
    frame_counter = 0
    
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
        
        logger.info(f"Client connected: {addr}")

        while True:
            try:
                while len(data) < payload_size:
                    packet = conn.recv(1024*1024)
                    if not packet:
                        break
                    data += packet
                
                if len(data) < payload_size:
                    break

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]
                
                if msg_size > 10 * 1024 * 1024:
                    break

                while len(data) < msg_size:
                    remaining = msg_size - len(data)
                    packet = conn.recv(min(1024*1024, remaining))
                    if not packet:
                        break
                    data += packet
                
                if len(data) < msg_size:
                    break

                frame_data = data[:msg_size]
                data = data[msg_size:]

                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                if frame is None:
                    continue
                
                tracked_objects = handler.process_frame(frame)
                
                if tracked_objects is None:
                    tracked_objects = []

                annotated_frame = annotate_frame(frame, tracked_objects)

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                success, result_payload = cv2.imencode('.jpg', annotated_frame, encode_param)
                
                if success:
                    message = struct.pack("Q", len(result_payload)) + result_payload.tobytes()
                    conn.sendall(message)

                frame_counter += 1
                
                if frame_counter % 100 == 0:
                    torch.cuda.empty_cache()

            except (ConnectionResetError, OSError, BrokenPipeError):
                break
            except Exception as e:
                logger.error(f"Error processing frame from {addr}: {e}")
                break

    except Exception as e:
        logger.error(f"Error handling client {addr}: {e}")
    finally:
        try:
            conn.close()
        except:
            pass
        logger.info(f"Connection from {addr} closed")

#This function starts the server and listens for incoming connections.
def start_server():
    host = '192.168.241.49' #Replace with the server's network's IPv4 address (same as the phone client).
    port = 9999 #This is the port the server will listen on.
    
    logger.info("Loading YOLO model...")
    model = YOLO("yolo11m.pt")
    model.fuse()
    model.to('cuda')
    model.half()
    
    logger.info("Loading ReID model...")
    try:
        reid_extractor = ReIDFeatureExtractor()
        reid_tracker = ReIDTracker(reid_extractor)
        logger.info("ReID model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ReID model: {e}")
        logger.error("Program requires ReID model to function. Please ensure osnet_ain_x1_0_imagenet.pth is in the current directory.")
        return
    
    handler = ConnectionHandler(model, reid_tracker)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        
        logger.info(f"Server listening on {host}:{port}")
        logger.info(f"Using device: {model.device.type.upper()}")

        while True:
            try:
                conn, addr = server_socket.accept()
                thread = threading.Thread(target=handle_client, args=(conn, addr, handler))
                thread.daemon = True
                thread.start()
            except KeyboardInterrupt:
                logger.info("Server shutting down...")
                break
            except Exception as e:
                logger.error(f"Server error: {e}")
                
    finally:
        server_socket.close()

if __name__ == '__main__':
    start_server()