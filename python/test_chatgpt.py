import openai
openai.api_key = "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"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Tell the world about the ChatGPT API in the style of a pirate."}
  ]
)

print(completion.choices[0].message.content)
