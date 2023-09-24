import torch
import os
from typing import Literal, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InvocationContext,
    invocation,
    UIComponent,
    InputField,
)
from invokeai.app.invocations.primitives import StringOutput

CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def list_local_models() -> list:
    if not os.path.exists(CACHE_DIR):
        return []
    # Assuming each model is stored in its own folder within CACHE_DIR
    models = [
        d.replace("models", "").replace("-", " ")
        for d in os.listdir(CACHE_DIR)
        if os.path.isdir(os.path.join(CACHE_DIR, d))
    ]
    return sorted(models, key=lambda x: x.lower())


available_models = list_local_models()

if available_models:
    models_str = ", ".join([repr(m) for m in available_models])
    ModelLiteral = eval(f'Literal["None", {models_str}]')
else:
    ModelLiteral = Literal["None"]


def is_model_cached(model_name: str) -> bool:
    """Check if the model and its tokenizer are cached locally in a custom directory."""
    try:
        AutoTokenizer.from_pretrained(
            model_name, local_files_only=True, cache_dir=CACHE_DIR)
        AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True, cache_dir=CACHE_DIR)
        return True
    except:
        return False


@invocation(
    "Random_Prompt_Maker_GPT2",
    title="Random Prompt Maker Using GPT2",
    tags=["prompt", "gpt2"],
    category="prompt",
    version="1.0.0",
)
class GPT2PromptInvocation(BaseInvocation):
    """Generates a random prompt using GPT-2"""

    # Inputs
    seed: Optional[str] = InputField(
        default="An enchanted", description="Seed for the prompt generation", ui_component=UIComponent.Textarea
    )
    context: Optional[str] = InputField(
        default="Describe a scene where",
        description="Context for the prompt generation",
        ui_component=UIComponent.Textarea,
    )
    max_length: Optional[int] = InputField(
        default=100, description="Max length of the generated text")
    temperature: Optional[float] = InputField(
        default=0.7, description="Controls the randomness of predictions")
    repetition_penalty: Optional[float] = InputField(
        default=1.0,
        description="Penalty for repeated content in the generated text",
        ge=1,
        le=2,
    )
    model_hf_name: Optional[str] = InputField(
        default="gpt2", description="Hugging Face model name to use")
    local_model: Optional[ModelLiteral] = InputField(
        default="None", description="List of the local models to use from the model_cache folder"
    )

    def is_sfw(self, text):
        banned_words = []  # add your banned words here eg. "nude", "murder"
        return not any(banned_word in text for banned_word in banned_words)

    def generate_prompt(self, seed, context=None, trials=0):
        if trials > 5:
            return "\033[1;31mUnable to generate SFW prompt after 5 attempts.\033[0m"

        # Determine which model to use
        if self.local_model and self.local_model != "None":
            model_to_use = os.path.join(CACHE_DIR, self.local_model)
        else:
            model_to_use = self.model_hf_name

        if not is_model_cached(model_to_use):
            print(
                f"\033[1;32;40mDownloading model:   \033[0m \033[1;37;40m{model_to_use}\033[0m")
        else:
            print(
                f"\033[1;32;40mUsing cached model:   \033[0m \033[1;37;40m{model_to_use}\033[0m")

        tokenizer = AutoTokenizer.from_pretrained(
            model_to_use, cache_dir=(None if self.local_model else CACHE_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            model_to_use, cache_dir=(None if self.local_model else CACHE_DIR))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        if context is not None:
            seed_with_context = f"{context} {seed}"
        else:
            seed_with_context = seed

        input_ids = tokenizer.encode(
            seed_with_context, return_tensors="pt").to(device)
        attention_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=device)
        pad_token_id = tokenizer.eos_token_id

        output = model.generate(
            input_ids,
            do_sample=True,
            max_length=self.max_length,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
        )

        prompt = tokenizer.decode(output[0], skip_special_tokens=True)

        if not self.is_sfw(prompt):
            print("\033[1;31mGenerated prompt is NSFW, retrying...\033[0m")
            return self.generate_prompt(seed, context, trials=trials + 1)

        context_end_position = len(context) + 1 if context else 0
        prompt_without_context = prompt[context_end_position:].strip()
        prompt_without_context = prompt_without_context.split("\n\n")[0]

        return prompt_without_context

    def invoke(self, context: InvocationContext) -> StringOutput:
        prompt = self.generate_prompt(self.seed, self.context)
        print(
            f"\033[1;32;40mGenerated Prompt:  \033[0m \033[1;37;40m{prompt}\033[0m")
        return StringOutput(value=prompt)
