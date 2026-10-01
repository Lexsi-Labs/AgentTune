# Quick Start

This builds one agent, runs it once, and evaluates it, all with a scripted policy and an
in-process tool, so it runs anywhere, no model, no network, no GPU.

```python
from agenttune.agentic import Project, DictToolHarness, ReActStrategy

# 1. A harness: the environment the agent acts in. DictToolHarness runs plain Python
#    callables as "tools" in-process — no sandbox, no network.
harness = DictToolHarness({"add": lambda a, b: a + b}, max_steps=4)

# 2. A strategy: the agent's policy. Here it's a scripted function standing in for a
#    model — step 0 calls the "add" tool, then it finishes with the answer.
strategy = ReActStrategy(lambda s: {"name": "add", "arguments": {"a": 2, "b": 3}}
                         if s.step == 0 else {"name": "finish", "arguments": {"answer": "5"}})

# 3. A Project ties strategy + harness together and gives you the lifecycle API.
proj = Project(strategy=strategy, harness=harness)

proj.infer("what is 2+3?")                                    # one episode → EventLog
proj.evaluate([{"task": "what is 2+3?", "expected": "5"}])    # → mean_score 1.0
proj.evaluate_agentic(["what is 2+3?"])                       # real programmatic metrics
```

Every call above produces or consumes the same `EventLog`, the normalized trajectory
format that carries this run through evaluation, training, distillation, and healing
without re-instrumenting anything. See [Basic Concepts](basic-concepts.md) for what that
actually means, and [User Guide: Agentic Spine](../user-guide/agentic-spine.md) for the
full picture (5 strategies, 4 memory backends, the tool library).

## Train a tool-using agent with RL

Pass `tools`, a `train_dataset`, and `reward_funcs`. `create_agentic_trainer` builds the
tool-calling rollout loop, runs generation, scores each completion with the reward
functions, and steps the optimizer. The example below trains a SQL agent on BioGRID.

```python
from datasets import load_dataset
from agenttune.core.backend_factory import create_agentic_trainer
from agenttune.agentic.tools.builtin.sql import SQLDatabaseTool
import textwrap

# 1. Setup a tool
sql_tool = SQLDatabaseTool("sqlite:////content/biogrid.db")
sql_tool.create_from_dataset(
    dataset_name="qgallouedec/biogrid",
    table_name="interactions",
    split="train",
)

def query_biogrid(sql_command: str) -> list:
    """
    Execute a read-only SQL query on the BioGRID database.

    Args:
        sql_command: The SQL query to execute.

    Returns:
        A list of tuples containing the query results.
    """
    result = sql_tool.execute(action="sql_db_query", input=sql_command)
    if result.success:
        return result.output
    return {"error": result.error}

# 2. Format your dataset
def format_example(example):
    preamble = textwrap.dedent("""\
    You have access to the BioGRID SQLite database.
    Use SQL queries to answer the question.
    Final answer must be enclosed in stars, e.g. *Yes* or *No*.
    """)
    return {
        "prompt": [{"role": "user", "content": f"{preamble}\nQuestion: {example['question']}"}],
        "answer": example["answer"],
    }

train_dataset = (
    load_dataset("qgallouedec/biogrid_qa", split="train")
    .filter(lambda ex: ex["question"].startswith("Does the gene "))
    .map(format_example, remove_columns=["question", "answer"])
)

# 3. Create trainer and run
trainer = create_agentic_trainer(
    algorithm="grpo",
    model="Qwen/Qwen3-1.7B",
    train_dataset=train_dataset,
    tools=[query_biogrid],
    reward_funcs=["correctness_reward", "structure_reward", "query_reward"],

    output_dir="./output/grpo_biogrid",
    max_steps=100,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    num_generations=2,
    max_completion_length=1024,
    use_vllm=True,               # needs `pip install -e '.[vllm]'` (Linux + CUDA); False on CPU/Mac
    vllm_mode="colocate",

    log_completions=True,
    report_to="none",
)

results = trainer.train()
print(f"✅ Done — steps: {results.get('total_steps')}  loss: {results.get('final_loss')}")
```

See [User Guide: RL Training](../user-guide/rl-training.md) for the full walkthrough of
this pattern, including the tool and reward internals.

## Next steps

- **Swap in a real model.** Notebook 1 in the [Local Notebooks](../notebooks/local-notebook.md)
  index replaces the scripted policy above with a real model and runs the full
  build → collect → evaluate → train → distill → heal lifecycle.
- **Browse every capability.** [Features](../features.md): one row per feature, each
  linked to a notebook that runs it for real.
