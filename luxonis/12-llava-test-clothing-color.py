import os
import ollama

path = os.path.join(os.path.dirname(__file__), "testimg1.png")

stream = ollama.generate(
    model="llava",
    prompt="What is the person wearing? Respond with 2 words separated by comma. The first word is the color of the shirt, and the second word is the type of clothing. For example, 'blue, shirt' or 'yellow, dress'.",
    images=[path],
    stream=False,
)
# for chunk in stream:
#     print(chunk["response"], end="")
print (stream["response"])

