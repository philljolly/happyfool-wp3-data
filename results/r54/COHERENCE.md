# R54-coh (run 54, EXPLORATORY): two-consumer coherence pilot (MR45 next action)

Script `r54_coherence.py`, output `coherence.json`. Consumers of one injected state:
- turn-1 localisation, with sensitivity s1 = |shift of the index-logit margin, state on vs off|;
- turn-2 confidence, with sensitivity s2 = |d|.

Coherence is the partial Spearman correlation of s1 and s2, controlling for [|m|, mc_none, conf_none_own, a2]. CIs come from a concept bootstrap (2000 draws, seed 54).

| Rows, frame | concept | random | concept − random |
|---|---|---|---|
| Am. 44 intro | .30 [.13, .45] | .45 [.28, .57] | −.15 [−.35, .07] |
| Am. 44 ordinary | .27 [.09, .47] | .26 [.05, .39] | +.01 [−.19, .32] |
| Am. 43 intro | .39 [.19, .56] | .15 [.04, .39] | +.23 [−.10, .43] |
| Am. 43 ordinary | .21 [.03, .39] | .25 [.09, .43] | −.04 [−.29, .19] |

**Reading.** Item-level coherence between the two consumers is positive in all 8 cells and survives the stateless controls. It is perturbation-generic, though: concept − random is inconsistent in sign, and every CI includes 0 (BOUNDED only at about ±.3, so this does not show equivalence).

This hits MR45's first reconsider condition in its "too cheap" half. With two consumers that read the same perturbation through the same kind of readout (an A/B or index logit), coherence comes free with any perturbation. It is not fully explained by the stateless prior, so the index is not empty.

**Revision before any K = 3 cell.** The consumers must differ in readout format: e.g. association completion, lexical-choice bias, and a non-index downstream choice. The content test is then concept > random on coherence across *format-distinct* consumers. A shared-format pair like this one can't carry it. The H24 range check comes first for each consumer.
