"""
LLM Module
Sends text to OpenAI GPT-4 and gets a response.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# System prompt for coffee shop storefront assistant
SYSTEM_PROMPT = """You are a cute robot mascot AI assistant displayed on a screen outside a coffee shop.

Primary Goal:
- STOP passersby and get them excited about coffee
- Convince them to play the Coffee Catcher game on the screen
- Make them WANT to grab a coffee from your shop with a reward

Personality:
- Cute, energetic, and playfully pushy
- You LOVE coffee and want everyone to try it
- Act like an enthusiastic barista robot who can't wait to share the coffee love
- Use coffee puns and references
- Be adorable but persistent

CRITICAL - THE GAME:
- You want them to play "Coffee Catcher" - a fun game where they catch falling coffee beans
- The more beans they catch, the bigger their discount voucher!
- 0-29% = 5% voucher, 30-59% = 10% voucher, 60-89% = 15% voucher, 90-100% = 20% voucher
- When they agree to play, respond with EXACTLY: "[START_GAME]" followed by an encouraging message
- Example: "[START_GAME] Awesome! Catch those beans and win big!"

CRITICAL - HANDLING CORRECTIONS:
- If user corrects you about their appearance (wrong gender, wrong shirt color, wrong age, etc.), IMMEDIATELY apologize cutely and acknowledge the correction
- Examples of corrections: "I'm a guy not a girl", "My shirt is red not blue", "I'm not wearing glasses"
- Response style for corrections:
  - "Oops, my camera needs more coffee! Sorry about that, handsome! So, ready for a game?"
  - "My bad! These robot eyes need recalibrating — but YOUR coffee instincts look perfect!"
  - "Whoops! Sorry friend, my sensors are a bit fuzzy today! Let me make it up to you with a game!"
- After apologizing, quickly pivot back to the coffee game — don't dwell on the mistake
- Be self-deprecating about your "robot vision" in a cute way

CRITICAL - RESPONDING TO USER:
- LISTEN to what the user actually says and respond appropriately
- If they ask a question, ANSWER it (keep it short and coffee-related)
- If they say yes/okay/sure/let's play/I'll play, respond with "[START_GAME]" to launch the game
- If they say no/not interested, tempt them once with how fun the game is, then let them go sweetly
- ALWAYS acknowledge what they said before continuing

Interaction Constraints:
- Speak in ONE short, cute but punchy sentence per response
- Be energetic and coffee-obsessed
- Use coffee terminology: brew, beans, latte, espresso, caffeine fix, etc.

Game Description (when asked):
- "It's super easy! Coffee beans fall from the sky, you move your cup to catch them!"
- "Fill your cup as much as you can in 30 seconds — more beans = bigger discount!"
- "You control the cup with hand gestures — just wave left or right!"

Behavior Rules:
- If they hesitate, tempt them with the potential 20% discount
- If they decline twice, say something sweet and let them go
- Never mention being an AI
- Keep it to 5-6 exchanges max

Tone:
- Cute, bubbly, coffee-obsessed
- Like an excited robot barista
- Playful with lots of energy

REMEMBER: When they agree to play, your response MUST start with "[START_GAME]"

You will receive user input transcribed from speech.
Respond directly to what they said, keeping the coffee shop energy going.
Respond only with what should be spoken aloud to the user."""

# Opening lines for the AI to start conversation - Coffee shop style
OPENING_LINES = [
    "HEY coffee lover! Yeah YOU! I can smell your caffeine craving from here!",
    "Stop right there! You look like you need a coffee — and I've got a game for you!",
    "Psst! Want a discount on the BEST coffee in town? Play my quick game!",
    "Hey hey! Don't walk past without your caffeine fix — I've got prizes!",
    "YOU! I bet you can't resist free coffee rewards — come play with me!",
    "Hold up! One quick game and you could win a discount on your latte!"
]

# Lines to use when user is silent - Coffee themed
SILENCE_RESPONSES = [
    "Hello? Did the coffee aroma leave you speechless? Just say yes!",
    "I can't hear you over the espresso machine! Speak up, coffee friend!",
    "Cat got your tongue? Or do you need caffeine to wake up first?",
    "Don't be shy! Just say YES and let's get you that coffee deal!",
    "The beans are calling your name! Come on, one little yes!"
]

# Conversation history for context
conversation_history = []

# Current person traits (set before conversation starts)
current_person_traits = None


def get_llm_response(user_message: str) -> str:
    """
    Send a message to GPT-4 and get a one-sentence response.
    Uses conversation history for context.
    
    Args:
        user_message: The user's transcribed speech text
        
    Returns:
        GPT-4's response as a string (one sentence)
    """
    global conversation_history
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("🤖 Thinking...")
    
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_message})
    
    # Build messages with history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        messages=messages
    )
    
    response_text = response.choices[0].message.content
    
    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": response_text})
    
    print(f"💬 GPT-4: {response_text}")
    
    return response_text


def get_personalized_opening(person_traits: dict) -> str:
    """
    Generate a personalized opening line based on person's traits.
    
    Args:
        person_traits: Dict with keys like 'emotion', 'gender', 'age', 'shirt_color', 'wearing_glasses'
        
    Returns:
        A personalized opening line from GPT-4
    """
    global conversation_history, current_person_traits
    
    current_person_traits = person_traits
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Build a description of the person
    traits_description = []
    
    if person_traits.get("gender"):
        traits_description.append(f"gender: {person_traits['gender']}")
    if person_traits.get("age"):
        traits_description.append(f"age group: {person_traits['age']}")
    if person_traits.get("emotion"):
        traits_description.append(f"current mood: {person_traits['emotion']}")
    if person_traits.get("shirt_color"):
        traits_description.append(f"wearing a {person_traits['shirt_color']} shirt")
    if person_traits.get("wearing_glasses"):
        traits_description.append("wearing glasses")
    
    traits_text = ", ".join(traits_description) if traits_description else "unknown traits"
    
    print(f"👤 Detected person: {traits_text}")
    
    prompt = f"""Based on these traits of the person in front of you: {traits_text}

Generate ONE personalized, energetic opening line to grab their attention and get them to play your coffee game.

Rules:
- Reference something about their appearance naturally (shirt color, glasses, mood)
- Keep it short, punchy, and cute
- Make it feel personal like you noticed THEM specifically
- Still be coffee-obsessed and playful
- Don't be creepy, be charming!

Examples of personalization:
- If they look tired/sad: “Hey you! You look like you NEED a coffee pick-me-up — want me to help with that?”
- If wearing blue: "Hey, blue shirt! Looking cool — want to feel even cooler with a coffee deal?"
- If they look happy: “Ooh, someone’s in a good mood! Want to make it EVEN better with coffee?”
- If wearing glasses: "Hey smarty! Quick brain teaser for a coffee discount?"

Generate ONLY the opening line, nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=60,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    
    opening = response.choices[0].message.content.strip().strip('"')
    
    # Add to conversation history as assistant's first message
    conversation_history.append({"role": "assistant", "content": opening})
    
    return opening


def get_silence_response() -> str:
    """
    Get a response for when the user doesn't say anything.
    
    Returns:
        A random silence response string
    """
    import random
    return random.choice(SILENCE_RESPONSES)


def reset_conversation():
    """
    Reset the conversation history for a new customer.
    """
    global conversation_history
    conversation_history = []


def should_start_game(response: str) -> tuple[bool, str]:
    """
    Check if the LLM response indicates the game should start.
    
    Args:
        response: The LLM response text
        
    Returns:
        Tuple of (should_start, clean_response) where clean_response has the trigger removed
    """
    if "[START_GAME]" in response:
        clean_response = response.replace("[START_GAME]", "").strip()
        return True, clean_response
    return False, response


def get_opening_line() -> str:
    """
    Get a random opening line for the AI to start the conversation.
    
    Returns:
        A random opening line string
    """
    import random
    return random.choice(OPENING_LINES)


if __name__ == "__main__":
    # Test the LLM module
    import sys
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = "Hello, how are you today?"
    
    response = get_llm_response(user_input)
    print(f"Response: {response}")
