# Public method interfaces

## Episode and time boundary

The fictional example joins successive alerts from the same subject when adjacent
gaps are no more than ten minutes. The first alert anchors the episode and the
maximum explicit fictional outcome code labels it. No free-text labeling is used.

Recent features admit only buckets whose end is strictly earlier than the alarm.
A bucket ending exactly at the alarm is excluded. The temporal representation has
287 admitted five-minute slots and an explicit leading padding slot. The personal
baseline occupies the seven days preceding the recent 24-hour interval.

## Observation and preprocessing

Episode summaries separate values, observation rates, recency and gaps. Baseline
availability is explicit. PartitionPreprocessor fits fallback scales, imputations
and robust scales using training subjects only. Temporal transforms preserve masks,
elapsed hours and padding. Missing values cannot update the observed-value cache.

## Model scope

GRU-D uses learned input and hidden-state decay. The mTAN adaptation embeds time,
interpolates each observed channel using attention, and aggregates with a GRU.
Neither architecture is claimed as novel. See the established
[GRU-D paper](https://doi.org/10.1038/s41598-018-24271-9) and
[mTAN paper](https://arxiv.org/abs/2101.10318).

Study A's retained encoders consume a distinct primitive/process representation.
The new public tensor adapter does not import the private source-schema pipeline.
AIME and legacy definitions are documented separately in the [code map](code-map.md).

## Evaluation

The public metrics adapter validates binary labels, score alignment and finite
probabilities. AUROC is undefined for one-class inputs; average precision is undefined
when no positive label is present. Undefined values are represented as JSON null.
Scores at or above a selected threshold are prioritized; equal scores cannot be split.
