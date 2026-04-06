from fastapi import FastAPI
import uvicorn
#from openai import OpenAI
#from groq import Groq
from mistralai.client import Mistral
from exa_py import Exa
import os
from dotenv import load_dotenv


app = FastAPI()

@app.get("/")
def hello():
    return {"headline": "Hello, dear developer!", "text_info": "I'm Tim Seufert, a 16 y.o. high school developer and hobbyist with a strong focus on developing well designed, user friendly AI solutions. I'm a student working on personal projects to enhance my software development skills. I've built the FakifyAPI a few weeks after finalizing my initial Fakify (fake news detector) project. I wanted to create a way for users of the internet to analyze and validate information from sources like articles or blog posts in a secure and reliable way (through retrieval augmented generation with the Mistral and Exa API's). Today I'am proudly presenting the FakifyAPI. This interface should allow users to embed my Fakify service into their own applications and services. All in all my project aims for an easy use of the Fakify project and making the internet a little bit safer. I hope you will enjoy using this API. See you on the internet :) - Tim Seufert", "api_status": "API is operational", "documentation": "/docs", "about_the_project": "https:github.com/Timhongphuc/FakifyAPI"}

@app.get("/app")
def analysis():

    load_dotenv()

    input_text = "XYZ"
#--------------------------------------------------------------------------
# EXA API (Content Endpoint)

    if input_text:
        exa = Exa(api_key = os.environ.get("EXA_API_KEY"))

        response = exa.get_contents(
            urls=[input_text],
            text= True
        )

        article_content = response.results[0].text
        print(article_content)

#--------------------------------------------------------------------------
# #MISTRAL API (Endpoint No1) Search query for Exa API

    if input_text:
        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

        inputs = [
            {"role": "user", "content": f"Generate ONE search query: {article_content}",}
        ]

        response = client.beta.conversations.start(
            agent_id= os.environ.get("AGENT_ID"), #System prompt is included in the Mistral API Dashboard
            inputs=inputs,
        )

        # Extract the clean markdown content from the response
        search = response.outputs[0].content

        print(search)
        #st.markdown(search)
        print("Received search query!")

#--------------------------------------------------------------------------
# EXA API (Search Endpoint)

    exa = Exa(api_key = os.environ.get("EXA_API_KEY"))

    st.write("Fetching results...")

    result = exa.search_and_contents(
        search,
        category = "news",
        livecrawl = "fallback",
        num_results=4,
        summary = {
        "query": "Your task is to create a brief AI summary of the Webpage. Max. 20 Words"
        },

        text = True,
        type = "auto"
    )

    search_results = result.results[3].text #Take the first 4 Search results (in index 3)
    print(search_results)
    print("Results fetched!")

    #--------------------------------------------------------------------------
    # Mistral API (API Endpoint No2) AI Summary of RAG analysis

    with Mistral(
                    api_key=os.environ.get("MISTRAL_API_KEY"),
                ) as mistral:

                    res = mistral.chat.complete(model="mistral-large-latest", messages=[
                        {
                        "content": f"Please provide me with an comprehensive analysis. These are the sources you can use to fulfill your task: The content of the News article you HAVE to check: {article_content}, similar search results to the topic (to verify credibility). Take a deep look into the sources: {search_results}. These sources are really important. If there are other Websites that provide the same information as given in the article, rate the article as real or likely real. If the topic in the article appears in other sources and the source is trustworthy change the rating accordingly.",
                        "role": "user"
                        },
                        {
                            "content": os.environ.get("SYSTEM_PROMPT"),
                            "role": "system"
                        },
                    ], stream=False)

                    # Handle response
                    results_final = res.choices[0].message.content
                    print(results_final)

                    #with st.chat_message("user"):
                    print("Finished analysis!")

    #----------------------------------------------------------------

    return {"results": f"{results_final}"}


uvicorn.run(app)