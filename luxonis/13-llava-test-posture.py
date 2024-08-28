import os
import ollama

path = os.path.join(os.path.dirname(__file__), "testimg3.png")

stream = ollama.generate(
    model="llava",
    prompt="What is arm posture of the person? Respond very briefly and in a neutral way. For example: 'the person is standing with their hands behind their back' or 'the person has their hands above their head'.",
    images=[path],
    stream=False,
)
# for chunk in stream:
#     print(chunk["response"], end="")
print (stream["response"])

