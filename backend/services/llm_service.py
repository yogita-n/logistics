import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None
except ImportError:
    client = None

def get_rerouting_decision(agent_state, congested_nodes,
                            old_route, new_route) -> dict:
    if not client:
        return {
            "should_reroute": True,
            "reason": "Fallback: Congestion exceeds threshold, rerouting is beneficial.",
            "customer_message": "Heads up! Your delivery agent is taking a slightly different route to avoid some unexpected traffic."
        }

    prompt = f"""
    You are a routing AI for a food delivery platform in Bengaluru.

    Agent has {len(agent_state['orders'])} orders in hand.
    Time elapsed: {agent_state['time_elapsed']} mins.

    Congested road segments detected: {list(congested_nodes.keys())}
    Current route ETA: {old_route[-1]['arrival_time_mins']} mins
    Rerouted route ETA: {new_route[-1]['arrival_time_mins']} mins

    Decide:
    1. should_reroute: true/false
    2. reason: one sentence explanation
    3. customer_message: SMS to send if rerouting (under 20 words)

    Respond ONLY in JSON.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"LLM Service fallback used due to: {e}")
        return {
            "should_reroute": True,
            "reason": "Fallback: Congestion exceeds threshold, rerouting is beneficial.",
            "customer_message": "Heads up! Your delivery agent is taking a slightly different route to avoid some unexpected traffic."
        }
