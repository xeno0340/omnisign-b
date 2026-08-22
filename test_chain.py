from gloss_dict import get_gloss
from sentence_builder import build_sentence
from tts import speak_text

# Test 1 — signs to sentence to speech
signs = ["HELP", "ME", "WATER"]
emotion = "distressed"

sentence = build_sentence(signs, emotion)
print(f"Signs: {signs}")
print(f"Sentence: {sentence}")
speak_text(sentence)

# Test 2 — gloss dict lookup
gloss = get_gloss("i need help")
print(f"\nGloss lookup: {gloss}")
sentence2 = build_sentence(gloss, "neutral")
print(f"Sentence: {sentence2}")
speak_text(sentence2)