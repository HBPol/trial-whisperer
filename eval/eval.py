"""Simple grounding/evidence eval over test set."""
import json, sys


def main(fp: str):
    gold = [json.loads(l) for l in open(fp)]
    # TODO: call API locally; compute exact match / citation presence
    print(f"Loaded {len(gold)} gold examples.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/testset.sample.jsonl")