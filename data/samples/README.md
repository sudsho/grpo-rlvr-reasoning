# Sample rollouts

Each JSONL file has 10 sampled rollouts from the trained model with:
- `prompt` (the input)
- `response` (the model's output)
- `gold` (ground truth)
- `verifier_ok` (bool)
- `verifier_reason` (str)

Handy for auditing what the verifier is accepting/rejecting without
having to spin up a full eval.
