# Évaluation du détecteur

Mesures du 22 octobre 2026 sur une RTX 4060 Laptop (8 Go). Détail complet par preset : [evaluation-precis.md](evaluation-precis.md) · [evaluation-standard.md](evaluation-standard.md). Reproduction : `uv run --group calibration python scripts/evaluate.py --preset precis`.

## Données

**Calibration** (818 textes) : HC3 (500 réponses humaines et ChatGPT, anglais), Wikipédia FR (150 extraits, dump de novembre 2023), 150 textes IA français de 8 genres produits par SmolLM3-3B, et 18 textes de Claude.

**Test, jamais vu à la calibration** :

| Humain (220 textes) | IA (24 textes + 96 attaques) |
|---|---|
| 80 critiques AlloCiné (familier) | 24 textes rédigés par Claude en 4 catégories : familier, formel, académique, « évasion » (écrit comme un humain, phrases irrégulières, petites fautes) |
| 40 critiques de livres | Attaques sur ces 24 textes : fautes de frappe (5 % des mots), caractères invisibles et homoglyphes, paraphrase par LLM, rétro-traduction FR → EN → FR |
| 40 extraits du Code civil (juridique) | |
| 60 extraits Wikipédia FR | + 40 critiques humaines **corrigées par un LLM** (piège à faux positifs) |

## Comparaison des presets

| Mesure (texte complet sauf mention) | `standard` 1,5B | `precis` 3B |
|---|---|---|
| AUROC [IC 95 %] | 0,964 [0,935–0,989] | **0,976** [0,956–0,991] |
| AUROC fenêtre 250 tokens | 0,981 | **0,991** |
| AUROC fenêtre 100 tokens | **0,797** | 0,794 |
| Faux positifs @0,5 / @seuil prudent | 4,1 % / 0 % | **3,6 % / 0 %** |
| Rappel @0,5 / @seuil prudent | **79 %** / 29 % | 75 % / **38 %** |
| Textes IA « évasion » détectés | 33 % | **50 %** |
| Paraphrase par LLM | **92 %** | **92 %** |
| Rétro-traduction | **79 %** | 67 % |
| Fautes de frappe | 25 % | **29 %** |
| Caractères invisibles / homoglyphes | **79 %** | 75 % |
| Humain corrigé par LLM signalé à tort | **7,5 %** | 12,5 % |
| Document 100 % humain (84 séquences) | **3** signalées à tort | 4 signalées à tort |
| Sandwich : précision / rappel par séquence | 44 % / 40 % | **50 % / 60 %** |
| Stabilité (taille des morceaux), écart moyen / max | 3,2 / 48 pts | **2,7 / 25 pts** |
| Vitesse, 120 pages (~100 000 tokens) | **1,6 min** | 3,1 min |

`precis` est le preset par défaut : meilleure séparation humain / IA, meilleure localisation dans les documents longs, plus stable. `standard` reste deux fois plus rapide.

## Critères go / no-go du plan de tests

| Test | Seuil | `precis` | Statut |
|---|---|---|---|
| T-P1 AUROC à 250 tokens | ≥ 0,90 | 0,991 (seulement 7 textes IA assez longs) | Atteint, échantillon faible |
| T-P2 AUROC document | ≥ 0,90 | 0,976 | Atteint |
| T-P4 calibration (ECE) | < 0,05 | 0,103 | **Non atteint** |
| T-R1 paraphrase | rappel ≥ 50 % | 92 % (paraphrase par LLM) | Atteint, paraphrase humaine non testée |
| T-R7 localisation par section | précision ≥ 80 % | 50 % | **Non atteint** |
| T-B1 rédacteurs non natifs | FPR ≤ 2× base | — | **Non testé** (pas de corpus francophone ouvert) |
| T-P5 déterminisme | identique | identique | Atteint |
| T-P6 vitesse 120 pages | < 10 min | 3,1 min | Atteint |
| E2E-3 phrase isolée | abstention | 81 % des premières phrases | Atteint |

## Ce que l'évaluation a changé

1. **Biais d'auto-détection.** Les textes IA de calibration étaient d'abord générés par Qwen2.5-1.5B-Instruct, c'est-à-dire le modèle de détection lui-même. Le preset `standard` les reconnaissait trop bien, ce qui gonflait ses scores et faisait paraître le 3B moins bon (AUROC de calibration 0,78). Les textes de calibration et les attaques sont désormais produits par SmolLM3, une autre famille de modèles.
2. **Variables de calibration du 3B.** La combinaison Binoculars + Fast-DetectGPT, apprise surtout sur HC3 (anglais, ChatGPT 2022), se transpose mal au français moderne avec le 3B. Binoculars seul fait passer l'AUROC de 0,946 à 0,976. Ce choix a été fait sur ce jeu de test : le chiffre est donc légèrement optimiste.
3. **Plafond de surprise par token (12 nats).** Avec 5 % de mots comportant une coquille, la détection tombait à 17 %. Le plafond l'a fait remonter à 42 % lors de l'expérience (preset `standard`, calibration de l'époque), pour une perte de 0,004 d'AUROC sur les textes propres. Avec la calibration finale, elle est de 25 à 29 %. Un plafond plus bas (8 nats) dégradait nettement les textes propres (rappel de 79 % à 58 %).
4. **Contexte de 256 tokens** au lieu de 128 pour chaque morceau, ce qui réduit l'effet des frontières.
5. **Quantification.** Le 3B en 4 bits (NF4) tient en 5 Go et reste plus précis que le 1,5B en bf16. Le 8 bits (LLM.int8) sature les 8 Go et est trop lent pour être utilisable. À taille égale, le 4 bits coûte environ 0,03 d'AUROC (mesuré sur le 1,5B).

## Limites mesurées

- **Textes courts** : sous 100 tokens, l'AUROC est d'environ 0,79 et les faux positifs atteignent 22 % au seuil de 0,5. L'outil s'abstient sous 60 tokens.
- **Écriture « comme un humain » et fautes de frappe** : c'est la faiblesse principale (50 % et 29 % de détection).
- **Localisation fine** : dans un document mixte, une séquence signalée sur deux est réellement de l'IA. Le signal sert à orienter une relecture, pas à prouver.
- **Calibration** : les probabilités sont trop confiantes (ECE 0,10). Pour un usage réel, recalibrez sur vos propres textes avec `aidetect calibrer`.
- **Non testé** : rédacteurs non natifs (T-B1), paraphrase humaine, PDF scannés (OCR), corpus SemEval-2024 Task 8 et M4 (multi-générateurs).
- **Petits échantillons IA** : 24 textes de test, d'où des intervalles de confiance larges (voir le tableau T-P1 de chaque rapport).
