from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() 
model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')  
result = model.invoke('Who is the prime minister of india')  
print(result.content)    