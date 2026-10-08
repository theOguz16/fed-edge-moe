import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-1.7B"

PROMPT = (
    "What is the capital of Australia and "
    "in which territory is it located?"
)

SYSTEM_PROMPT = (
    "Answer the question in two or three short factual sentences. "
    "Be precise and do not speculate."
)


def build_input(tokenizer):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": PROMPT,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return tokenizer(
        text,
        return_tensors="pt",
    )


def generate_one(
    model,
    tokenizer,
    device,
    sample,
    seed,
    max_new_tokens,
):
    encoded = build_input(tokenizer)

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
        "--precision",
        choices=["fp32", "fp16"],
        required=True,
    )

    parser.add_argument(
        "--device",
        default="mps",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--tag",
        default="mac",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID
    ).eval().to(device)

    if args.precision == "fp16":
        model.half()
    else:
        model.float()

    records = []

    start = time.perf_counter()

    for trial in range(args.trials):
        print(
            f"{args.precision}: "
            f"trial {trial + 1}/{args.trials}"
        )

        main_answer = generate_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            sample=False,
            seed=1000 + trial,
            max_new_tokens=args.max_new_tokens,
        )

        samples = []

        for sample_idx in range(args.samples):
            samples.append(
                generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    sample=True,
                    seed=(
                        50000
                        + trial * 1000
                        + sample_idx
                    ),
                    max_new_tokens=args.max_new_tokens,
                )
            )

        records.append({
            "prompt_idx": trial,
            "prompt": PROMPT,
            "main_answer": main_answer,
            "samples": samples,
        })

    elapsed = time.perf_counter() - start

    output = {
        "model": "qwen3",
        "model_id": MODEL_ID,
        "precision": args.precision,
        "device": args.device,
        "prompts": args.trials,
        "samples_per_prompt": args.samples,
        "generation_time_s": elapsed,
        "records": records,
    }

    out = Path("results") / (
        f"qwen3_australia_robustness_"
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
    print(f"Generation time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
