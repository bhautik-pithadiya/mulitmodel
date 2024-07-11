import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig

# Configuration for double quantization
double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.float16
)

# Load the original model
model_id = "utils/saved_models/Llama3"
save_path = 'utils/saved_models/quantized_llama3/'
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    quantization_config=double_quant_config
)

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
model.save_pretrained(save_path)

# Save the processor
processor = AutoTokenizer.from_pretrained(model_id)
processor.save_pretrained(save_path)
