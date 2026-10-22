# Évaluation — preset `standard`

Modèles : Qwen/Qwen2.5-1.5B / Qwen/Qwen2.5-1.5B-Instruct · seuil prudent (1 % de faux positifs en calibration) : 0.86

## T-P1 — Performance selon la longueur (données tenues à l'écart)

| Fenêtre (tokens) | Humain / IA | AUROC [IC 95 %] | Faux positifs @0,5 | Rappel @0,5 | Faux positifs @prudent | Rappel @prudent |
|---|---|---|---|---|---|---|
| 50 | 220 / 24 | 0.840 [0.728–0.928] | 33.2 % | 79.2 % | 0.0 % | 12.5 % |
| 100 | 220 / 24 | 0.797 [0.678–0.903] | 19.5 % | 62.5 % | 1.4 % | 12.5 % |
| 250 | 142 / 7 | 0.981 [0.949–1.000] | 7.7 % | 100.0 % | 0.7 % | 28.6 % |
| complet | 220 / 24 | 0.964 [0.935–0.989] | 4.1 % | 79.2 % | 0.0 % | 29.2 % |

## Taux de détection par source (texte complet)

| Source | Réalité | n | Tokens moy. | Signalé IA @0,5 | Signalé IA @prudent | Proba. médiane |
|---|---|---|---|---|---|---|
| allocine | humain | 80 | 257 | 6.2 % | 0.0 % | 12.6 % |
| attaque_paraphrase_llm | IA | 24 | 241 | 91.7 % | 62.5 % | 88.4 % |
| attaque_retrotraduction | IA | 24 | 236 | 79.2 % | 41.7 % | 82.6 % |
| attaque_typos | IA | 24 | 241 | 25.0 % | 0.0 % | 27.9 % |
| attaque_unicode | IA | 24 | 237 | 79.2 % | 33.3 % | 74.8 % |
| code_civil | humain | 40 | 452 | 0.0 % | 0.0 % | 6.4 % |
| critiques_livres | humain | 40 | 214 | 2.5 % | 0.0 % | 14.0 % |
| humain_corrige_llm | humain | 40 | 253 | 7.5 % | 0.0 % | 20.3 % |
| ia_academique | IA | 6 | 262 | 100.0 % | 66.7 % | 88.1 % |
| ia_evasion | IA | 6 | 225 | 33.3 % | 16.7 % | 44.6 % |
| ia_familier | IA | 6 | 216 | 83.3 % | 16.7 % | 76.4 % |
| ia_formel | IA | 6 | 244 | 100.0 % | 16.7 % | 73.9 % |
| wikipedia | humain | 60 | 642 | 5.0 % | 0.0 % | 6.0 % |

## Calibration, documents longs, stabilité

- **T-P4 calibration** : ECE 0.108, Brier 0.057
- **E2E-3 phrase isolée** : abstention sur 80.7 % des 244 premières phrases
- **E2E-1 document 100 % humain** (19119 tokens, 84 séquences) : 3 séquence(s) signalée(s) à tort — verdict « Passages IA probables (3 séquences) »
- **E2E-2 / T-R7 sandwich** (22642 tokens, 10 textes IA insérés) : précision 44.4 %, rappel 40.0 % par séquence (VP 4, FP 5, FN 6) — verdict « Passages IA probables (9 séquences) »
- **Test 1.1 stabilité (taille des morceaux)** : 256 tokens → Binoculars 1.0143, 10 séq. IA, écart moyen 3.2 pts / max 40.0 pts, 384 tokens → Binoculars 1.0151, 9 séq. IA, écart moyen 0.0 pts / max 0.0 pts, 512 tokens → Binoculars 1.0137, 9 séq. IA, écart moyen 3.3 pts / max 47.8 pts
- **T-P5 déterminisme** : identique sur deux exécutions
- **T-P6 vitesse** : 1021 tokens/s → ~1.6 min pour 120 pages (~100 000 tokens)
