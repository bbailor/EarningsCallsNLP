# Technology Mention Labeling Guide

## Purpose and unit of annotation

Label the business significance of the technology discussion in each displayed
passage. The **anchor sentence** contains a term from the configured technology
dictionary; the surrounding text is context from the same speaker turn. Assign
one label to the passage, not a separate label to every term.

Use the strongest level that the passage itself supports. Do not infer use,
impact, or intent from outside knowledge about the company. A technology name
alone is not evidence of deployment or business impact.

## Labels

| Level | Name | Apply when | Example |
|---|---|---|---|
| **0** | No substantive technology claim | The match is a false positive, incidental wording, or the passage does not actually make a claim about the target technology. | “The ML measurement was 200 milliliters.” |
| **1** | Generic or aspirational mention | The passage names a technology or expresses broad interest, plans, or possibility, but gives no concrete application, capability, or investment. | “We believe AI will be important to our industry.” |
| **2** | Concrete use, capability, or investment | The company describes a specific application, product capability, pilot, deployment, or related investment. A measured outcome is not required. | “We are piloting machine learning to improve inventory forecasts.” |
| **3** | Material or strategic technology impact | The passage supports a substantial role in company strategy, scaled operations, differentiated products, revenue, or quantified business outcomes. | “Our deployed AI forecasting system reduced inventory costs by 8% across all warehouses.” |

### Boundary rules

- Use **0** for dictionary matches that are not technology claims. If the
  passage does make a technology claim, do not use 0 merely because it is vague.
- Use **1** for general statements, goals, or speculative benefits. A future
  tense statement is not automatically a concrete plan.
- Use **2** when a specific use, pilot, product feature, capability, or
  investment is described, even if it is small or early-stage.
- Use **3** only when scale, strategic importance, product differentiation,
  revenue significance, or a business outcome is explicit in the passage.
- If a passage contains more than one claim, assign the highest level supported
  by its text.
- Attribute claims to the speaker as stated. Do not treat an analyst’s question
  as a company claim unless an executive affirms it.
- If context is insufficient or ambiguous, choose the lower supported level and
  briefly explain the uncertainty in the notes.

## Annotation procedure

1. Read the anchor sentence and its same-speaker context.
2. Select exactly one level (0–3) and optionally record a short rationale.
3. Use your own name or initials consistently. Do not overwrite another
   annotator’s work.
4. The randomly selected double-label subset must be labeled independently by
   two people. Do not discuss those passages before both labels are submitted.
5. After both annotators finish, calculate unweighted Cohen’s kappa on passages
   labeled by both people in the double-label subset. Report the number of
   jointly labeled passages alongside kappa.

## Agreement calculation

For the four nominal categories, Cohen’s kappa is
`κ = (p_observed - p_expected) / (1 - p_expected)`, where observed agreement is
the fraction of matching labels and expected agreement is the sum of the
products of each annotator’s marginal proportion for each level. Kappa is
undefined if expected agreement is 1.0. Interpret it with the sample size and
the confusion pattern; do not treat one cutoff as a substitute for reviewing
disagreements.

## Calibration and revisions

Before full annotation, have both annotators label the same small practice set,
compare disagreements, and clarify interpretations using the boundary rules.
If the codebook changes, record the revision and re-label affected items
consistently before reporting agreement.
