# Évaluation — preset `precis`

Modèles : Qwen/Qwen2.5-3B / Qwen/Qwen2.5-3B-Instruct · seuil prudent (1 % de faux positifs en calibration) : 0.88

## T-P1 — Performance selon la longueur (données tenues à l'écart)

| Fenêtre (tokens) | Humain / IA | AUROC [IC 95 %] | Faux positifs @0,5 | Rappel @0,5 | Faux positifs @prudent | Rappel @prudent |
|---|---|---|---|---|---|---|
| 50 | 220 / 24 | 0.762 [0.652–0.867] | 34.1 % | 70.8 % | 0.5 % | 4.2 % |
| 100 | 220 / 24 | 0.794 [0.680–0.910] | 21.8 % | 79.2 % | 0.9 % | 8.3 % |
| 250 | 142 / 7 | 0.991 [0.970–1.000] | 7.7 % | 100.0 % | 0.0 % | 28.6 % |
| complet | 220 / 24 | 0.976 [0.956–0.991] | 3.6 % | 75.0 % | 0.0 % | 37.5 % |

## Taux de détection par source (texte complet)

| Source | Réalité | n | Tokens moy. | Signalé IA @0,5 | Signalé IA @prudent | Proba. médiane |
|---|---|---|---|---|---|---|
| allocine | humain | 80 | 257 | 7.5 % | 0.0 % | 12.2 % |
| attaque_paraphrase_llm | IA | 24 | 241 | 91.7 % | 29.2 % | 79.3 % |
| attaque_retrotraduction | IA | 24 | 236 | 66.7 % | 29.2 % | 68.9 % |
| attaque_typos | IA | 24 | 241 | 29.2 % | 0.0 % | 33.6 % |
| attaque_unicode | IA | 24 | 237 | 75.0 % | 29.2 % | 77.6 % |
| code_civil | humain | 40 | 452 | 0.0 % | 0.0 % | 2.9 % |
| critiques_livres | humain | 40 | 214 | 2.5 % | 0.0 % | 11.7 % |
| humain_corrige_llm | humain | 40 | 253 | 12.5 % | 0.0 % | 17.0 % |
| ia_academique | IA | 6 | 262 | 100.0 % | 83.3 % | 92.8 % |
| ia_evasion | IA | 6 | 225 | 50.0 % | 16.7 % | 49.3 % |
| ia_familier | IA | 6 | 216 | 66.7 % | 33.3 % | 73.0 % |
| ia_formel | IA | 6 | 244 | 83.3 % | 16.7 % | 63.6 % |
| wikipedia | humain | 60 | 642 | 1.7 % | 0.0 % | 6.7 % |

## Calibration, documents longs, stabilité

- **T-P4 calibration** : ECE 0.103, Brier 0.050
- **E2E-3 phrase isolée** : abstention sur 80.7 % des 244 premières phrases
- **E2E-1 document 100 % humain** (19119 tokens, 84 séquences) : 4 séquence(s) signalée(s) à tort — verdict « Passages IA probables (4 séquences) »
- **E2E-2 / T-R7 sandwich** (22642 tokens, 10 textes IA insérés) : précision 50.0 %, rappel 60.0 % par séquence (VP 6, FP 6, FN 4) — verdict « Passages IA probables (12 séquences) »
- **Test 1.1 stabilité (taille des morceaux)** : 256 tokens → Binoculars 0.9977, 14 séq. IA, écart moyen 2.7 pts / max 16.2 pts, 384 tokens → Binoculars 0.9986, 12 séq. IA, écart moyen 0.0 pts / max 0.0 pts, 512 tokens → Binoculars 0.9997, 12 séq. IA, écart moyen 2.9 pts / max 24.6 pts
- **T-P5 déterminisme** : identique sur deux exécutions
- **T-P6 vitesse** : 532 tokens/s → ~3.1 min pour 120 pages (~100 000 tokens)
