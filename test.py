from transformers import AutoTokenizer
import openai


api_key = "esecret_ic2tffl3krl7smiemfu1174lig"
model = "meta-llama/Llama-2-13b-chat-hf"
system_prompt="It is a test for max_tokens"
item="what is the max token in this case"
messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item}
            ]
response = openai.ChatCompletion.create(
        api_base = "https://api.endpoints.anyscale.com/v1",
        api_key=api_key,
        model=model,
        messages= messages,
        temperature=0.2,
        max_tokens=30
    )
print(response)
response_content = response['choices'][0]["message"]["content"]