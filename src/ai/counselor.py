import re, random

OPENERS = [
    "I’m here with you. What’s on your mind right now?",
    "Tell me what feels heaviest; we’ll unpack it step by step.",
    "What happened just before you started feeling this way?"
]

def reply(user_text: str):
    t = user_text.lower().strip()
    if not t:
        return random.choice(OPENERS)

    if any(k in t for k in ["angry","irritat","mad","rage"]):
        return ("Anger is often a protector. What do you wish others understood about this situation?\n"
                "Try: write one boundary you’d set if you could. Then a calm action you’ll take in 10 minutes.")
    if any(k in t for k in ["sad","low","empty","down","cry"]):
        return ("That sounds heavy. Could we name one feeling under the sadness (lonely, tired, ashamed)?\n"
                "Tiny step: message one person with a simple ‘hey’. Then 2 minutes of sunlight or music.")
    if any(k in t for k in ["stress","anx","panic","overwhelm"]):
        return ("Let’s ground for 30 seconds: name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.\n"
                "What’s one controllable next step for today only?")
    if "sleep" in t:
        return ("Sleep tweaks: same wake time daily, no caffeine after 2pm, dim lights 1h before bed, notebook dump.\n"
                "What’s one change you’ll try tonight?")
    # default reflection
    return ("I hear you. If you could rewrite this story, what’s a kinder sentence you’d tell yourself?\n"
            "What’s one 5-minute action you can take after we finish this message?")
