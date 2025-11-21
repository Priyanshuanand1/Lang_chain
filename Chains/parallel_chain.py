from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

prompt1 = PromptTemplate(
    template= 'Generate short and simple note from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template= 'Generate a 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template= 'Merge the provided notes and quiz into a document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model | parser,
    'quiz' : prompt2 | model | parser
})

merge_chain = prompt3 | model | parser
 
chain = parallel_chain | merge_chain

text = '''
Space: The Final Frontier

Space has long captured the human imagination, representing a vast and mysterious expanse that stretches far beyond the limits of our everyday experience. It is a realm defined by extremes—immense distances, powerful forces, and conditions so hostile that life as we know it can only exist within the thin protective shell of Earth’s atmosphere. Yet, despite these challenges, space remains one of humanity’s greatest inspirations and frontiers for exploration.

At its most basic level, space is the enormous void that exists beyond Earth and its atmosphere. It is filled not with true emptiness, but with stars, planets, moons, asteroids, comets, and an abundance of energy and radiation. The stars that dot our night sky are distant suns, each with unique properties and often surrounded by planetary systems of their own. Our own solar system is a small neighborhood within the Milky Way galaxy, which itself is just one of billions of galaxies in the known universe. This vastness makes space almost incomprehensible; distances are so large that they are measured not in miles or kilometers but in light-years—the distance light travels in a year, about 9.46 trillion kilometers.

Space exploration began in earnest in the mid-20th century, when technological advances allowed humanity to venture beyond Earth. The launch of Sputnik in 1957 marked the beginning of the space age, quickly followed by human missions such as Yuri Gagarin’s first orbit of Earth and the Apollo moon landings. These milestones expanded our understanding of space and demonstrated the potential for human presence beyond our planet. In more recent decades, probes like Voyager, rovers like Perseverance, and telescopes like the James Webb Space Telescope have provided breathtaking images and invaluable scientific data, revealing planets, stars, and galaxies never before seen in such detail.

The importance of space extends far beyond exploration alone. Satellites orbiting Earth enable global communication, weather forecasting, navigation, and scientific monitoring of the environment. Space technology supports disaster response, climate research, and agriculture. Meanwhile, the International Space Station serves as a unique laboratory where astronauts conduct experiments that benefit life on Earth and deepen our understanding of how living organisms adapt to the harsh conditions of microgravity.

Furthermore, space represents a long-term opportunity for humanity’s survival and growth. As Earth faces challenges such as climate change, resource depletion, and population growth, many scientists and visionaries see space as a potential avenue for future habitats or the extraction of resources from asteroids and other celestial bodies. Although these ideas remain speculative and technologically challenging, they highlight the role space could play in shaping our future.

In addition to scientific and practical benefits, space exploration enriches culture and inspires creativity. It encourages curiosity, cooperation among nations, and a desire to understand our place in the universe. Ultimately, space reminds us of both our smallness and our potential. The cosmos is vast and ancient, yet humanity has begun to uncover its secrets. As technology advances, our reach into space will only grow, continuing the timeless human pursuit of discovery.

'''

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii() 