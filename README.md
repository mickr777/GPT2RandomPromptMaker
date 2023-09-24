# GPT2RandomPromptMaker
A node for InvokeAI utilizes the GPT-2 language model to generate random prompts based on a provided seed and context.

#### Fields:

| Fields | Description |
| -------- | ------------ |
| seed | The initial seed for the prompt generation.|
| context | The context to provide additional information for the prompt.|
| max_length | The maximum length of the generated text.|
| temperature | Controls the randomness of predictions.|
| repetition_penalty | Penalty for repeated content in the generated text.|
| model_hf_name | Hugging Face gpt2 model name to use.|
| local_model | List of the local models to use from the model_cache folder.|

Tested Model Examples
```
Meli/GPT2-Prompt
AUTOMATIC/promptgen-lexart
Gustavosta/MagicPrompt-Stable-Diffusion
succinctly/text2image-prompt-generator
MBZUAI/LaMini-Neo-1.3B (needs more vram then other models)
```

#### Info:
- on first use it should download the needed files.

## Example:
Generated Prompt: An enchanted weapon will be usable by any character regardless of their alignment.

![9acf5aef-7254-40dd-95b3-8eac431dfab0](https://github.com/mickr777/GPT2RandomPromptMaker/assets/115216705/219ced60-2ebc-4a26-88ae-21e989ee72f9)

