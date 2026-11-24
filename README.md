# Nullwalk

Removing unwanted behaviors from language models by editing only the activation directions the model actually reads

## Motivation

Language models carry many behaviors, such as refusing a request, as directions in their internal activations. This makes it possible to switch a behavior off without any retraining: edit the activations at one layer so that the behavior's signal disappears.

Existing closed-form edits remove that signal everywhere:

- **Directional ablation** deletes the single direction along which behavior prompts and neutral prompts differ on average.
- **LEACE** finds the smallest edit after which no linear classifier can tell the two groups apart.

The model itself does not look everywhere. The later layers that produce the behavior respond to only some directions of the activation space, and changes in the other directions barely affect them. Removing the signal from directions the model does not respond to is unnecessary change, and every unnecessary change risks side effects on unrelated behavior.

## Idea

Nullwalk asks for less: the behavior only has to become invisible in the directions the model's later layers are sensitive to. These directions are estimated from gradients of a score that measures the behavior. Differences the model cannot see are allowed to stay, so the edit walks in the null space of what the model reads.

Like directional ablation and LEACE, the edit is closed-form and training-free: it is computed directly from activations and gradients, without fine-tuning the model. Because the requirement is weaker, the Nullwalk edit is never larger than the LEACE edit. If the model were equally sensitive to every direction, Nullwalk would be exactly LEACE.

## Hypotheses

These are open questions, not results:

1. At the same level of behavior removal, the smaller edit causes less collateral damage.
2. The advantage grows with model size, because the directions the model is sensitive to make up a shrinking fraction of the activation space.

## Status

Work in progress. The first behavior studied is refusal. Results will be added here once they exist.

Final project for the seminar *Interpreting and Analyzing Neural Language Models* (LaCoCo Lab, Saarland University, winter semester 2026/27).

## Builds on

- *Refusal in Language Models Is Mediated by a Single Direction*
- *LEACE: Perfect linear concept erasure in closed form*
- *On the Non-Identifiability of Steering Vectors in Large Language Models*
