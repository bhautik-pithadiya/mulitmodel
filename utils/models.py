from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import pipeline
import torch.nn.functional as F
import torch


class QLLama3:
    def __init__(self):
        self.model_path = 'utils/saved_models/quantized_llama3'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        double_quant_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_compute_dtype=torch.float16)
        
        self.model  = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            trust_remote_code=True, 
            quantization_config=double_quant_config)  
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            )
        
    
    def predict(self,text):
        
        instruction = "For the upcoming prompts, determine whether each prompt falls into one of the following categories: 1)Math-related 2)Code-related 3)Q&A-related. \nNote: Do not attempt to solve or address the content of the prompts. Your task is solely to categorize them."
        
        prompt = "Instruction : "+ instruction + " prompt : " + text + "Answer : "
        
        
        