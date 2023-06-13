from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from fastapi import FastAPI
from pydantic import BaseModel

quantized_model_dir = "TheBloke/guanaco-65B-GPTQ"
model_basename = "Guanaco-65B-GPTQ-4bit.act-order"
use_safetensors = True
use_slow = True
use_triton = True
bits = 4
group_size = 128
desc_act = True

tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=not use_slow)

try:
    quantize_config = BaseQuantizeConfig.from_pretrained(quantized_model_dir)
except:
    quantize_config = BaseQuantizeConfig(
                        bits=bits,
                                group_size=group_size,
                                        desc_act=desc_act
                                            )

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
                    use_safetensors=True,
                        model_basename=model_basename,
                            device="cuda:0",
                                use_triton=use_triton,
                                    quantize_config=quantize_config)

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

app = FastAPI()

class Item(BaseModel):
    input: str

@app.post("/")
async def root(item: Item):
    prompt_template = f'''### Human: {item.input}
    ### Assistant:'''
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output_tokens = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    output = tokenizer.decode(output_tokens[0])
    m = "### Assistant:"
    index = output.index(m)
    new_output = output[index+len(m):].strip()
    try:
        hm_index = new_output.index("### Human:")
        new_output = new_output[:hm_index]
    except:
        pass
    return {"output": new_output}