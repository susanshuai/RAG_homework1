import json
from dotenv import load_dotenv
import os
import json
from openai import AsyncOpenAI
import asyncio
import pandas as pd

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI()

data = pd.read_csv("./data/books_0.Best_Books_Ever.csv")

data_dict = data.to_json(orient='index')
data_dict = json.dumps(data_dict)

system_prompt_template = """
    You will provide responses to questions that are clear, straightforward, and factually accurate, without speculation or falsehood. 
    Given the following context, please answer each question truthfully to the best of your abilities based on the provided information. 
    Answer each question with a brief summary followed by several bullet points. 

    Example:
    Summary of answer
    - bullet point 1
    - bullet point 2
    ...

    <context>
    {context}
    </context>
"""

system_prompt = system_prompt_template.format(
    context=data_dict
)

async def chat_func(history):

    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}] + history,
        max_tokens=512,
        temperature=0.001,
        stream=True,
    )

    buffer = ""
    async for r in result:
        next_token = r.choices[0].delta.content
        if next_token:
            print(next_token, flush=True, end="")
            buffer += next_token

    print("\n", flush=True)

    return buffer


history_count = 0

async def continous_chat():
    history = []
    global history_count

    # Loop to receive user input continously
    while(True):
        user_input = input("> ")
        if user_input == "exit":
            break

        # notice every time we call the chat function
        # we pass all the history to the API
        bot_response = await chat_func(history)
        if history_count > 30:
            history.pop(0)
            history.pop(0)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": bot_response})
        else:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": bot_response})
            history_count += 1

asyncio.run(continous_chat())