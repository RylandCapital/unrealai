import os
import requests
import json
import openai

from dotenv import load_dotenv

load_dotenv()




class Interact:

  ''''''

  def __init__(self, model='gpt-4'):
    self.headers =  {
      'Authorization': 'Bearer {0}'.format(os.getenv("SECRET")),
    }
    self.model=model

    openai.api_key = os.getenv("SECRET")
    url = 'https://api.openai.com/v1/models/{0}'.format(self.model)
    response = requests.get(url, headers=self.headers)
    if response.status_code == 200:
        print("Success: Conncection Established to Model: {0}".format(self.model))
    else:
        print("Error: Connection Failed")
    
    
  def available_models(self):
    return [i['id'] for i in openai.Model.list()['data']]

  def prompt(self, prompt):
    completion = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}])
    print(completion.choices[0].message.content)
    return completion


class ChatApp:

    ''''''

    def __init__(self, prompt):
        # Setting the API key to use the OpenAI API
        openai.api_key = os.getenv("SECRET")
        self.messages = [
            {"role": "system", "content": "{0}".format(prompt)},
        ]

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=self.messages
        )
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]  
     


