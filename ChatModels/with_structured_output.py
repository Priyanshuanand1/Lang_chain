from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv() 

model = ChatGoogleGenerativeAI(model='gemini-2.5-pro') 


class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]    
    sentiment: Annotated[str,"Return sentiment of the review either negatrive, positive  or neutral" ]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""This game delivers fast-paced action and charming visuals, 
                      offering a surprisingly deep experience beneath its simple exterior. Its controls feel 
                      responsive, and progression remains satisfying. Occasional difficulty 
                      spikes appear, but it’s an enjoyable, well-crafted adventure
                       that keeps players engaged throughout.""")  
print(result)    