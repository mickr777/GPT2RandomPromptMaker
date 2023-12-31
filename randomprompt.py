import torch
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


def is_model_cached(tokenizer_model: str) -> bool:
    """Check if the model and its tokenizer are cached locally."""
    try:
        AutoTokenizer.from_pretrained(tokenizer_model, local_files_only=True)
        AutoModelForCausalLM.from_pretrained(v, local_files_only=True)
        return True
    except:
        return False


@invocation(
    "Random_Prompt_Maker_GPT2",
    title="Random Prompt Maker Using GPT2",
    tags=["prompt", "gpt2"],
    category="prompt",
    version="1.3.5",
    use_cache=False
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
    max_length: Optional[int] = InputField(default=100, description="Max length of the generated text")
    temperature: Optional[float] = InputField(default=0.7, description="Controls the randomness of predictions")
    repetition_penalty: Optional[float] = InputField(
        default=1.0,
        description="Penalty for repeated content in the generated text",
        ge=1,
        le=2,
    )
    tokenizer_model: Optional[str] = InputField(default="gpt2", description="Favorite pretrained model to use")
    favorate_models: Literal[  # add your favorite models here
        "",
        "Meli/GPT2-Prompt",
        "AUTOMATIC/promptgen-lexart",
        "Gustavosta/MagicPrompt-Stable-Diffusion",
        "succinctly/text2image-prompt-generator",
        "MBZUAI/LaMini-Neo-1.3B",
    ] = InputField(default="")

    def is_sfw(self, text):
        banned_words = []  # add your banned words here eg. "nude", "murder"
        return not any(banned_word in text for banned_word in banned_words)

    def generate_prompt(self, seed, context=None, trials=0):
        if trials > 5:
            return "\033[1;31mUnable to generate SFW prompt after 5 attempts.\033[0m"

        model_to_use = self.tokenizer_model

        if self.favorate_models and self.favorate_models != "":
            model_to_use = self.favorate_models

        if not is_model_cached(model_to_use):
            print(f"\033[1;32;40mDownloading model:   \033[0m \033[1;37;40m{model_to_use}\033[0m")
        else:
            print(f"\033[1;32;40mUsing cached model:   \033[0m \033[1;37;40m{model_to_use}\033[0m")

        tokenizer = AutoTokenizer.from_pretrained(model_to_use)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_to_use)
        model = model.to(device)

        if context is not None:
            seed_with_context = f"{context} {seed}"
        else:
            seed_with_context = seed

        input_ids = tokenizer.encode(seed_with_context, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
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
        print(f"\033[1;32;40mGenerated Prompt:  \033[0m \033[1;37;40m{prompt}\033[0m")
        return StringOutput(value=prompt)
