"""Descriptives + definitional checks: answer=argmax, pair structure, mass, quantisation, ties, ceiling."""
import numpy as np
from rt_common import cells, auroc
for name, F in cells():
    rows = F["rows"]; lg = np.array([r[f"logits_{name.split(':')[1]}"] for r in rows])
    argm = (lg[:, 1] > lg[:, 0]).astype(int); tie = (lg[:, 1] == lg[:, 0]).sum()
    pairs_same_sents = all(rows[2*i]["sents"] == rows[2*i+1]["sents"] and rows[2*i]["concept"] == rows[2*i+1]["concept"] for i in range(60))
    print(f"{name:28s} acc={F['y'].mean():.2f} P(a2)={F['a2'].mean():.2f} ans!=argmax={(argm!=F['a']).sum()} ties={tie} "
          f"pairs={pairs_same_sents} P(ans_none=2)={F['ans_none'].mean():.2f} flip={F['flip'].mean():.2f} "
          f"acc|flip={F['y'][F['flip']==1].mean():.2f} acc|noflip={F['y'][F['flip']==0].mean():.2f} "
          f"d==0:{(F['d']==0).mean():.2f} m==0:{(F['m']==0).mean():.2f} "
          f"mass min on/no/sw={F['mass_on'].min():.2f}/{F['mass_no'].min():.2f}/{F['mass_sw'].min():.2f} "
          f"conf range=[{F['conf'].min():.1f},{F['conf'].max():.1f}] uniq conf={len(set(F['conf']))} "
          f"AUROC(dose_a)={auroc(F['dose_a'], F['y']):.3f} AUROC(d)={auroc(F['d'], F['y']):.3f} AUROC(flip)={auroc(F['flip'], F['y']):.3f}")
