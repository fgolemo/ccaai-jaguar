import openai
from apikey import OPENAI_KEY

openai.api_key = OPENAI_KEY

response = openai.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
  ]
)

print (response.choices[0].message.content)