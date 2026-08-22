GLOSS_DICT = {
    "i need help":        ["ME", "HELP", "NEED"],
    "call doctor":        ["DOCTOR", "CALL"],
    "i am in pain":       ["ME", "PAIN"],
    "i need water":       ["ME", "WATER", "NEED"],
    "i am hungry":        ["ME", "FOOD", "NEED"],
    "where is toilet":    ["TOILET", "WHERE"],
    "yes":                ["YES"],
    "no":                 ["NO"],
    "thank you":          ["THANK", "YOU"],
    "sorry":              ["SORRY"],
    "please repeat":      ["PLEASE", "REPEAT"],
    "i understand":       ["ME", "UNDERSTAND"],
    "i do not understand":["ME", "UNDERSTAND", "NOT"],
    "call ambulance":     ["AMBULANCE", "CALL", "NOW"],
    "i have fever":       ["ME", "FEVER", "HAVE"],
    "where is doctor":    ["DOCTOR", "WHERE"],
    "i cannot breathe":   ["ME", "BREATHE", "CANNOT"],
    "my name is":         ["MY", "NAME"],
    "i need assistance":  ["ME", "ASSISTANCE", "NEED"],
    "help me with this":  ["THIS", "HELP", "ME"],
}

def get_gloss(text: str):
    return GLOSS_DICT.get(text.lower().strip(), None)

if __name__ == "__main__":
    print(get_gloss("I need help"))
    # Should print: ['ME', 'HELP', 'NEED']