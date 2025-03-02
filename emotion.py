import cv2
import numpy as np
from deepface import DeepFace
from textblob import TextBlob
import speech_recognition as sr

class EmotionAI:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.camera = cv2.VideoCapture(0)  # Access webcam

    def detect_facial_emotion(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Error accessing camera")
            return None
        
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            return emotion
        except Exception as e:
            print("Emotion detection error:", e)
            return None
    
    def analyze_speech_emotion(self):
        with self.microphone as source:
            print("Listening for speech-based emotion analysis...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        
        try:
            text = self.recognizer.recognize_google(audio)
            sentiment = TextBlob(text).sentiment.polarity
            if sentiment > 0:
                return "happy"
            elif sentiment < 0:
                return "sad"
            else:
                return "neutral"
        except sr.UnknownValueError:
            return "Speech not understood"
        except sr.RequestError:
            return "API unavailable"

    def detect_and_respond(self):
        face_emotion = self.detect_facial_emotion()
        speech_emotion = self.analyze_speech_emotion()
        print(f"Facial Emotion: {face_emotion}, Speech Emotion: {speech_emotion}")
        return face_emotion, speech_emotion

if __name__ == "__main__":
    emotion_ai = EmotionAI()
    emotion_ai.detect_and_respond()
