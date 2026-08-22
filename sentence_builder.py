from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "./models",
    model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    model_type="llama",
    max_new_tokens=30,
    temperature=0.1
)

KNOWN_PATTERNS = {
    ("HELLO",):               "Hello! How can I help you?",
    ("HI",):                  "Hi there!",
    ("HELP",):                "Please help me!",
    ("HELP", "ME"):           "Please help me urgently!",
    ("ME", "HELP", "NEED"):   "I need help urgently.",
    ("ME", "WATER", "NEED"):  "I need water please.",
    ("WATER",):               "I need water.",
    ("ME", "PAIN"):           "I am in pain.",
    ("PAIN",):                "I am in pain.",
    ("DOCTOR",):              "Please call a doctor.",
    ("DOCTOR", "CALL"):       "Please call a doctor.",
    ("ME", "FOOD", "NEED"):   "I am hungry, I need food.",
    ("FOOD",):                "I need food.",
    ("TOILET", "WHERE"):      "Where is the toilet?",
    ("TOILET",):              "Where is the toilet?",
    ("AMBULANCE", "CALL", "NOW"): "Call an ambulance now!",
    ("ME", "BREATHE", "CANNOT"): "I cannot breathe, help me!",
    ("ME", "FEVER", "HAVE"):  "I have a fever.",
    ("FEVER",):               "I have a fever.",
    ("THANK", "YOU"):         "Thank you.",
    ("THANKS",):              "Thank you.",
    ("PLEASE", "REPEAT"):     "Please repeat that.",
    ("SORRY",):               "I am sorry.",
    ("YES",):                 "Yes.",
    ("NO",):                  "No.",
    ("STOP",):                "Please stop.",
    ("EMERGENCY",):           "This is an emergency, help me now!",
    ("SOS",):                 "Emergency! Please help me immediately!",
    ("NAME",):                "My name is...",
}

def build_sentence(signs: list, emotion: str = "neutral") -> str:
    key = tuple(signs)
    if key in KNOWN_PATTERNS:
        return KNOWN_PATTERNS[key]

    # Check single sign
    if len(signs) == 1:
        single = (signs[0],)
        if single in KNOWN_PATTERNS:
            return KNOWN_PATTERNS[single]

    # Smart fallback — no model
    signs_clean = " ".join(s.lower() for s in signs)
    if emotion == "distressed":
        return f"Please help me urgently with {signs_clean}."
    return f"I need {signs_clean}."

if __name__ == "__main__":
    print(build_sentence(["HELLO"], "happy"))
    print(build_sentence(["HELP", "ME"], "distressed"))
    print(build_sentence(["YES"], "neutral"))