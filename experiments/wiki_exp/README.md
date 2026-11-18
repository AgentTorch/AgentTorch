### Wiki experiment: data and pipeline

This experiment consumes a local folder of JSONL files named `enwiki_namespace_0_*.jsonl`, each line representing a Wikipedia article in the main namespace. For each record we extract four canonical fields: `id` from `identifier` (or `id`), `title` from `name` (or common fallbacks), `url` from `url` (or `main_entity.url`/`is_part_of.url`), and `text` from `description` or `abstract`, falling back to a joined view of section text when needed. These fields are placed into a DataFrame and also serialized into a single input field called `content` so the full context a prompt sees is aligned with what the optimizer sees.

An example of this:



For instance, a typical JSONL line in the source dump might look like:

```json
{
  "identifier": "76716259",
  "name": "Not Again SU",
  "url": "https://en.wikipedia.org/wiki/Not_Again_SU",
  "description": "Student organization in Syracuse, New York",
  "sections": [
    {
      "has_parts": [
        { "heading": "Introduction", "text": "The hashtag and student led organization #NotAgainSU began circulating after several racist incidents occurred on campus..." },
        { "heading": "Demands", "text": "The protestors made a list of 19 demands to the University which was later expanded to 34..." }
      ]
    }
  ]
}
```

From this, the experiment maps `identifier → id`, `name → title`, `url → url`, and prefers `description` (or `abstract`) for `text`, falling back to the joined `sections[].has_parts[].text` when needed. The canonical `{id, title, url, text}` are stored in the DataFrame and also serialized into a single `content` field so DSPy and P3O see the same inputs.
For DSPy, we build a minimal classifier module with few-shot demonstrations created from sampled articles. When available, DSPy’s compile step (MIPROv2 “light”) uses heuristic labels to tune the scaffold; these labels are a proxy only and are not required for the downstream optimization. The tuned module is then converted into an AgentTorch Template via `from_predict`, which exposes a system prompt with a dedicated Attributes block (this is how P3O includes slots).

Attributes/slots are experiment-defined switches that shape the model’s behavior, for example asking it to extract dates, highlight entities, define key terms, summarize events, or surface related concepts. They are not taken from the dataset; rather, they are explicit instruction lines placed in the prompt’s “Attributes:” block by the Template. The LLM reads those instruction lines and responds accordingly, but it does not decide which lines appear; the optimizer does. P3O treats each attribute as a binary decision (on/off) in the prompt template, samples a configuration, runs the LLM to get a structured response, turns that response into a scalar reward, and updates its policy to prefer configurations that yield higher reward. Over iterations this has the effect of “pruning” attributes that do not help under the chosen reward and retaining those that do, without requiring any ground-truth labels.

The reward is computed from the model’s structured output (a numeric score per category) using the MSE-shaped metric from prior experiments. Concretely, the scores are collapsed to a scalar and compared against a scalar target with mean squared error; this error is then mapped to a smooth reward that P3O can optimize with policy-gradient updates. This provides a consistent label-free signal to compare attribute configurations. If you later add gold labels, you can substitute an accuracy-, cross-entropy-, or Brier-based reward while keeping the same slot-optimization loop.

Each run produces a summary JSON in `experiments/wiki_exp/` that captures the extracted prompt scaffold, the final Attribute selections chosen by P3O, the reward and token estimates, and compact previews to inspect how the prompt evolved.