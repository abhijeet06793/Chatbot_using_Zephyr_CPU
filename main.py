import os
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

## load the model
local_llm = r"C:\Users\vibha\PycharmProjects\ZephyrChatbot\Model\mistral-7b-instruct-v0.2.Q2_K.gguf"

config = {
    "max_new_tokens": 1024,
    "repetition_penalty": 1.2,
    "temperature":0.7,
    "top_k": 50,
    "top_p":0.9,
    "stream": True,
    "threads": int(os.cpu_count()/2)
}
llm_init = CTransformers(model=local_llm, model_type="mistral", lib="avx2", **config)

print(llm_init)

#query = r"What is the meaning of life ?"
#result = llm_init.invoke(query)
#print(result)

template = r"""Question: {question}

Answer: Lets think step by step and answer faithfully.
"""

@cl.on_chat_start
async def on_chat_start():
    model = llm_init
    prompt = PromptTemplate(template=template, input_variables=["question"])
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

## link - https://www.youtube.com/watch?v=LbwY760mdqM&ab_channel=AIAnytime