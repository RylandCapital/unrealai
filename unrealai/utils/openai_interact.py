import os
import requests
import json
import openai

from dotenv import load_dotenv

load_dotenv()




class Interact:

  ''''''

  def __init__(self, model='gpt-3.5-turbo'):
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
    [i['id'] for i in openai.Model.list()['data']]

  def prompt(self, prompt):
    completion = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}])
    print(completion.choices[0].message.content)
     
     
     


