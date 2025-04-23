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
            "This is the analysis value of my image"
            "Could you help me analyze what kind of character this image is?"
            "Based on all his values, we can make an estimate"
            "Give me a suitable soundtrack description that fits this image"
            "Give me a hint"
            "I just want the music tips"
            "Tips for placing models that generate music"
            "Preferably in English"
            "Please attach the required major key classification, music type, instruments required"
            "only output the English prompt"
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": constant_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            answer = response.choices[0].message['content'].strip()

            return answer


        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


