# AI-Detector

Détecteur **local et open-source** de texte généré par IA, du court paragraphe au document de 120 pages.
Aucune API externe : tout tourne sur votre machine (GPU recommandé).

> Le résultat est une **probabilité**, jamais une preuve. Un score élevé doit déclencher une vérification humaine, pas une accusation.

## Principe

Deux méthodes zero-shot publiées, calculées sur les mêmes passes de modèle :

- **Binoculars** (Hans et al., ICML 2024) — rapport entre la perplexité d'un texte et la « perplexité croisée » de deux modèles (base / instruct).
- **Fast-DetectGPT** (Bao et al., ICLR 2024) — écart entre la vraisemblance observée et celle attendue sous le modèle.

Les statistiques sont calculées **token par token**, ce qui permet de noter n'importe quelle portion : phrase (avec une fenêtre de phrases voisines), séquence (~300 tokens, coupée aux fins de paragraphe) ou document entier. Une régression logistique calibrée transforme ces scores en probabilité qui tient compte de la longueur, avec un **intervalle de confiance à 90 %** (bootstrap par blocs de tokens).

Les longs documents sont traités **par séquences** de 384 tokens, chacune avec 256 tokens de contexte précédent, découpées aux fins de phrase.

### Garde-fous

- **Abstention** sous 60 tokens, et « Indéterminé » quand l'intervalle de confiance couvre les deux issues.
- **Normalisation anti-contournement** : caractères invisibles, homoglyphes cyrilliques/grecs glissés dans des mots latins et lettres stylisées (𝐠𝐫𝐚𝐬) sont neutralisés et signalés.
- **Plafond de surprise par token** (12 nats) : quelques fautes de frappe ne suffisent plus à faire passer un texte IA pour humain.
- Un document majoritairement humain contenant des passages IA est présenté comme tel (« Passages IA probables »), jamais masqué par un score global « humain ».

## Installation

```bash
uv sync
```

PyTorch CUDA 12.8 est installé par défaut. Les modèles Qwen2.5 (multilingues) sont téléchargés au premier lancement.

| Preset | Modèles | Mémoire GPU | 120 pages (RTX 4060) | AUROC test FR |
|---|---|---|---|---|
| `precis` (défaut) | Qwen2.5-3B / 3B-Instruct (4 bits NF4) | ~5 Go | ~3 min | 0,976 |
| `standard` | Qwen2.5-1.5B / 1.5B-Instruct (bf16) | ~7 Go | ~1,6 min | 0,964 |
| `rapide` | Qwen2.5-0.5B / 0.5B-Instruct | ~3 Go, possible sur CPU | — | non calibré |

Sans GPU CUDA, `precis` se replie sur `standard`.

## Utilisation

```bash
uv run aidetect analyser memoire.pdf --rapport rapport.html   # pdf, docx, txt, md
uv run aidetect analyser --texte "Un texte à vérifier…"
uv run aidetect analyser memoire.pdf --preset standard        # deux fois plus rapide
uv run aidetect serveur                                        # interface web sur http://127.0.0.1:8765
```

Le rapport indique un verdict global avec son intervalle de confiance, la part estimée de texte IA, une bande par séquence (localisation dans le document, avec numéro de page pour les PDF) et le texte surligné phrase par phrase.

## Calibrer sur vos propres textes

Les probabilités dépendent de la paire de modèles et du type de textes. Pour un usage sérieux, recalibrez avec des textes dont l'origine est certaine :

```bash
uv run aidetect calibrer --humain dossier_humain --ia dossier_ia
```

Le jeu de calibration par défaut se reconstruit avec `uv run --group calibration python scripts/build_calibration_set.py` : HC3 (humain vs ChatGPT, anglais), Wikipédia FR (novembre 2023) et des textes IA français de genres variés produits par **SmolLM3-3B**, un générateur volontairement d'une autre famille que les modèles de détection. Un détecteur note trop bien les textes de son propre modèle, ce qui fausse la calibration.

## Évaluation

```bash
uv run --group calibration python scripts/evaluate.py --preset precis
```

La suite suit un plan de tests en 5 axes (performance par longueur, biais par domaine, robustesse aux attaques, documents longs, calibration) sur des données **jamais vues à la calibration** : critiques AlloCiné, Code civil, critiques de livres, Wikipédia FR et 24 textes IA rédigés par Claude, dont 6 écrits « comme un humain ».

Résultats détaillés et limites mesurées : [docs/evaluation.md](docs/evaluation.md).

## Limites

- Peu fiable sous ~150 tokens (une ou deux phrases).
- Les textes IA écrits « comme un humain », paraphrasés ou traduits passent souvent.
- Le texte humain corrigé par une IA est parfois signalé : utilisez le seuil prudent et la revue humaine.
- Code, tableaux et listes ne sont pas des cibles fiables.
