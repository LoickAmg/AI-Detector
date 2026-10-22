# AI-Detector

Détecteur **local et open-source** de texte généré par IA, du court paragraphe au document de 120 pages.
Aucune API externe : tout tourne sur votre machine (GPU recommandé).

> Le résultat est une **probabilité**, jamais une preuve. Un score élevé doit déclencher une vérification humaine, pas une accusation.

## Principe

Deux méthodes zero-shot publiées, calculées sur les mêmes passes de modèle :

- **Binoculars** (Hans et al., ICML 2024) — rapport entre la perplexité d'un texte et la « perplexité croisée » de deux modèles (base / instruct).
- **Fast-DetectGPT** (Bao et al., ICLR 2024) — écart entre la vraisemblance observée et celle attendue sous le modèle.

Les statistiques sont calculées **token par token**, ce qui permet de noter n'importe quelle portion : phrase (avec une fenêtre de phrases voisines), séquence (~300 tokens, coupée aux fins de paragraphe) ou document entier. Une régression logistique calibrée transforme ces scores en probabilité qui tient compte de la longueur.

Les longs documents sont traités **par séquences** de 384 tokens, chacune avec 128 tokens de contexte précédent, découpées aux fins de phrase.

## Installation

```bash
uv sync
```

PyTorch CUDA 12.8 est installé par défaut. Les modèles Qwen2.5 (multilingues) sont téléchargés au premier lancement.

| Preset | Modèles | Mémoire GPU |
|---|---|---|
| `standard` | Qwen2.5-1.5B / 1.5B-Instruct (bf16) | ~7 Go |
| `rapide` | Qwen2.5-0.5B / 0.5B-Instruct | ~3 Go, possible sur CPU |

## Utilisation

```bash
uv run aidetect analyser memoire.pdf --rapport rapport.html   # pdf, docx, txt, md
uv run aidetect analyser --texte "Un texte à vérifier…"
uv run aidetect serveur                                        # interface web sur http://127.0.0.1:8765
```

Le rapport indique un verdict global, la part estimée de texte IA, une bande par séquence (localisation dans le document, avec numéro de page pour les PDF) et le texte surligné phrase par phrase.

## Calibrer sur vos propres textes

Les probabilités dépendent de la paire de modèles et du type de textes. Pour un usage sérieux, recalibrez avec des textes dont l'origine est certaine :

```bash
uv run aidetect calibrer --humain dossier_humain --ia dossier_ia
```

Le jeu de calibration par défaut se reconstruit avec `uv run --group calibration python scripts/build_calibration_set.py` (HC3, Wikipédia FR, textes Qwen et Claude).

## Limites

- Peu fiable sous ~150 tokens (une ou deux phrases).
- Les textes au ton personnel (e-mails, lettres) écrits par un LLM moderne passent parfois.
- Le texte humain très formel ou corrigé par une IA peut être signalé : utilisez le seuil prudent et la revue humaine.
- Code, tableaux et listes ne sont pas des cibles fiables.
