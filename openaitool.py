import openai
import os
import re
import string

class ChatGPTResponder:
    def __init__(self, api_key: str = None):
        api_key = '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'

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("plz provide OpenAI API key。")

        openai.api_key = self.api_key

    def get_response(self, input_text: str) -> str:
        constant_prompt = (
            "You are a cinematic soundtrack composer and analyst. You will be given an emotion_result, which is a time series of emotional probabilities (e.g. excitement, fear, tension, sadness, relaxing, neutral over time). Analyze the emotional trend in this data and generate a professional, cinematic music description that reflects the emotional progression.\n\n"
            "Format requirements:\n"
            "- Paragraph: Begin with a natural, flowing English paragraph that vividly describes how the music progresses as the emotions evolve.\n\n"
            "- Structured Fields: After the paragraph, provide the following fields with their values (each on a new line, with the field name followed by a colon and the inferred content):\n"
            "  Key: A suitable musical key (or key progression) that matches the emotional arc.\n"
            "  Genre: An appropriate genre or style of music that fits the overall emotional tone.\n"
            "  Mood: A short description of the mood progression, reflecting the sequence of emotions over time.\n"
            "  Tempo: An approximate tempo or tempo range in BPM that aligns with the energy and pacing of the emotional changes.\n"
            "  Instruments: A list of instruments or sections that would dominate the soundtrack, chosen to convey the emotions effectively.\n\n"
            "Additional instructions:\n"
            "- Dynamic Derivation: Do not hardcode any specific key, genre, mood, tempo, or instruments. Derive all choices from the provided emotional data and its evolution.\n"
            "- Language and Tone: Use fluent, vivid, and professional language. The description should be cinematic and evocative, but remain concise and clear. Avoid unnecessary punctuation or symbols; keep the tone polished and focused on music.\n"
            "- Output Only the Prompt: Your final answer should only contain the music description and the structured fields in the format above, with no extra commentary or explanation."
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": constant_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            answer = response.choices[0].message['content'].strip()

            return answer


        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


