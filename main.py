from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from utils.models import QLLama3

model = QLLama3()
repeat = True
while repeat:
    text = input('Enter text to categorize from \n1)Math-related \n2)Code-related \n3)Q&A-related\n')
    
    print(model.predict(text))
    question = input("want to continue?(Y/n)")
    if question == "" or question == "Y" or question =='y':
        repeat = True
    else:
        repeat = False