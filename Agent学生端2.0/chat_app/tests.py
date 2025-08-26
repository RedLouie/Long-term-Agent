from translate_agent import translate_Agent
trs_agent = translate_Agent()
input_text = "I would like to learn about the story of Han Xin."
res = trs_agent.translate_chat(input_text)
print(f"{input_text}\n {res}")