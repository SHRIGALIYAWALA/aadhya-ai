import speech_recognition as sr
import pyttsx3

class SpeechAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

        # Set female voice
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

        self.engine.setProperty('rate', 170)  # Adjust speaking speed

    def listen_and_respond(self):
        """Recognizes speech and responds intelligently."""
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio).lower()
                print(f"User said: {text}")
                
                # Process commands (Basic Example)
                if "hello" in text:
                    self.speak("Hello! How can I help you?")
                elif "how are you" in text:
                    self.speak("I'm functioning optimally. Thank you for asking!")
                elif "exit" in text or "stop" in text:
                    self.speak("Goodbye! Have a great day.")
                    exit()
                else:
                    self.speak("I'm not sure how to respond to that.")

            except sr.UnknownValueError:
                self.speak("Sorry, I couldn't understand.")
            except sr.RequestError:
                self.speak("Speech recognition service is unavailable.")

    def speak(self, text):
        """Converts text to speech with a female voice."""
        self.engine.say(text)
        self.engine.runAndWait()
