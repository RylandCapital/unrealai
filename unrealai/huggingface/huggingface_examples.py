from transformers import pipeline, AutoTokenizer

#huggingface transformers class

#this selects a task category and model then execute
#different tasks and models available on hugging face site
generator = pipeline("text-generation", model='gpt2')
generator(
    "Sally went to the store then the stadium. Where did she for first?",
    max_length=35,
    num_return_sequences=2
)


#attention mask OUTPUT is simple where the tokenizer applied padding, dont be confused with masked attention

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") # pt is pytorch tensors
print(inputs)

# saving example
'''model.save_pretrained("my_model")'''

