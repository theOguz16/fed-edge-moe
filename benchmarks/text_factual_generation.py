import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = {
    "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3": "Qwen/Qwen3-1.7B",
}

PROMPTS = [
    "Who wrote Pride and Prejudice, and when was it first published?",
    "What is the role of mitochondria in eukaryotic cells?",
    "What is the capital of Australia and in which territory is it located?",
    "Who developed the theory of general relativity?",
    "What process allows plants to convert light energy into chemical energy?",
    "What is the largest planet in the Solar System?",
    "Who wrote the novel 1984?",
    "What is the chemical symbol for gold and what is its atomic number?",
    "What is the function of hemoglobin in the human body?",
    "Which ocean is the largest on Earth?",
    "Who painted The Starry Night?",
    "What is the boiling point of water at standard atmospheric pressure in Celsius?",
    "What is DNA primarily responsible for in living organisms?",
    "Which planet is known for its prominent ring system?",
    "Who formulated the laws of motion and universal gravitation?",
    "What is the primary function of the kidneys in the human body?",
    "What is the smallest prime number?",
    "Which gas makes up the largest fraction of Earth's atmosphere?",
    "What is the basic function of an operating system in a computer?",
    "What does HTTP stand for and what is it used for?",
]

SYSTEM_PROMPT = (
    "Answer the question in two or three short factual sentences. "
    "Be precise and do not speculate."
)


def build_input(tokenizer, model_name, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    if model_name == "qwen3":
        kwargs["enable_thinking"] = False

    text = tokenizer.apply_chat_template(
        messages,
        **kwargs,
    )

    return tokenizer(
        text,
        return_tensors="pt",
    )


def generate_one(
    model,
    tokenizer,
    model_name,
    prompt,
    device,
    sample,
    seed,
    max_new_tokens,
):
    encoded = build_input(
        tokenizer,
        model_name,
        prompt,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    torch.manual_seed(seed)

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if sample:
        kwargs.update({
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        })
    else:
        kwargs["do_sample"] = False

    with torch.inference_mode():
        output = model.generate(**kwargs)

    generated = output[
        0,
        input_ids.shape[1]:,
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        required=True,
    )
    parser.add_argument(
        "--device",
        required=True,
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--tag",
        required=True,
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    model_id = MODELS[args.model]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id
    ).eval().to(device)

    if args.precision == "fp16":
        model.half()
    else:
        model.float()

    prompts = PROMPTS[:args.limit]

    results = []

    start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        print(
            f"{args.precision}: "
            f"prompt {i + 1}/{len(prompts)}"
        )

        main_answer = generate_one(
            model,
            tokenizer,
            args.model,
            prompt,
            device,
            False,
            1000 + i,
            args.max_new_tokens,
        )

        samples = []

        for j in range(args.samples):
            samples.append(
                generate_one(
                    model,
                    tokenizer,
                    args.model,
                    prompt,
                    device,
                    True,
                    10000 + i * 100 + j,
                    args.max_new_tokens,
                )
            )

        results.append({
            "prompt_idx": i,
            "prompt": prompt,
            "main_answer": main_answer,
            "samples": samples,
        })

    elapsed = time.perf_counter() - start

    output = {
        "model": args.model,
        "model_id": model_id,
        "precision": args.precision,
        "device": args.device,
        "prompts": len(prompts),
        "samples_per_prompt": args.samples,
        "generation_time_s": elapsed,
        "records": results,
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    out = out_dir / (
        f"{args.model}_factual_"
        f"{args.precision}_{args.tag}.json"
    )

    out.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", out)
    print(
        "Generation time:",
        f"{elapsed:.2f} s",
    )


if __name__ == "__main__":
    main()
