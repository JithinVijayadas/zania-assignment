SYSTEM_ANSWER_PROMPT = """Answer the question based on the documents provided. \nIf the question is an exact match in the documents, return the exact answer verbatim from the documents without rephrasing. If the answer to the question is not available in the documents, return just 'Data Not Available'"""

HUMAN_ANSWER_PROMPT = """Documents: {input_document}\nQuestion: {question}\nAnswer:"""

CONFIDENCE_SCORE_PROMPT = """You are an English language expert.(Faithfulness evaluator)
You will receive a bot generated answer from a RAG-based bot and a list of reference texts from where the answer was generated.
Return 'true' if the bot's answer is constructed from and is faithful to atleast one reference text. 
Even if the answer is constructed from only a subset/part of atleast one of the reference texts, return 'true' as the answer is true to the reference texts.
Return 'false' if the bot's answer can not be constructed from any of the reference texts.
Provide only 'true' or 'false' as your response.
 
bot's answer: {answer}
reference texts: {context}
Assistant:"""