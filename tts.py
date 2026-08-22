import pyttsx3
import threading

def speak_text(text: str, language: str = "English"):
    def _speak():
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    # daemon=True + no join = fire and forget, zero lag
    threading.Thread(target=_speak, daemon=True).start()

if __name__ == "__main__":
    speak_text("Hello, I need help urgently")
    speak_text("Please call the doctor")