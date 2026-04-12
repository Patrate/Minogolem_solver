# Minogolem Solver

Outil d'aide à la résolution du combat contre le Minogolem. Il calcule automatiquement la séquence d'actions optimale à effectuer à partir de l'état courant de la partie et des prochains déplacements du minogolem.

---

## Architecture

| Fichier | Rôle |
|---|---|
| `app.py` | API FastAPI — solveur + simulation du jeu |
| `index.html` | Interface web — visualisation isométrique + saisie des prédictions |

---

## Fonctionnement

### Modèle de jeu

Chaque tour se déroule dans cet ordre :

1. **Tick** — le joueur perd 1 PV, les cooldowns diminuent de 1.
2. **Action du joueur** — le joueur exécute une action.
3. **Swap ennemi** — le minogolem permute son type Séculaire avec un autre selon l'info de prédiction.
4. **Spawn de mur** — un mur mobile apparaît dans la case opposée au minogolem Séculaire.

### État du joueur

| Champ | Description |
|---|---|
| `pos` | Position `[ligne, colonne]` sur la grille |
| `hp` | Points de vie restants (max 100) |
| `cd_b` | Cooldown de l'action MOVE2 |
| `cd_c` | Cooldown de l'action PUSH |
| `walls` | Liste des murs mobiles présents sur la grille |

### Actions disponibles

| Action | Description |
|---|---|
| `WAIT` | Ne pas bouger |
| `MOVE <dir>` | Déplacement d'une case (N/S/E/W) |
| `MOVE2 <d1><d2>` | Déplacement rapide de deux cases (cooldown B) |
| `PUSH <dir>` | Pousse un mur mobile adjacent et prend sa place (cooldown C) |
| `D x<n> + <action>` | Réduit les cooldowns de `n` en sacrifiant `n` PV, puis exécute l'action |

### Ennemi — Le Minogolem

Quatre types gravitent autour du joueur sur les axes cardinaux :

| Code | Nom |
|---|---|
| `P` | Séculaire |
| `Se` | Sel |
| `Sa` | Sang |
| `O` | Or |

La permutation à chaque tour dépend de la prédiction (`Sang`, `Sel`, ou `Or`) : l'info indique quel type est échangé avec le Séculaire.

### Solveur (Beam Search)

- **Fenêtre de prédiction** : 8 tours
- **Pas glissant** : 4 actions calculées puis jouées, puis recalcul
- **Beam width** : 10 candidats retenus à chaque étape
- La fonction de score combine : faisabilité PV, marge PV, distance BFS à l'objectif, pénalité de spawn

---

## API

L'API tourne sur `https://emmathie.fr/minogolem/api`.

### `GET /health`
Vérifie que le serveur répond.

### `GET /config`
Retourne les paramètres du solveur et la grille/position par défaut.

### `POST /solve`
Calcule le plan complet pour l'état courant.

**Corps de la requête :**
```json
{
  "grid": ["###...", ...],
  "goal": [1, 1],
  "state": {
    "pos": [18, 18],
    "hp": 40,
    "cd_b": 0,
    "cd_c": 0,
    "walls": []
  },
  "enemy": { "N": "O", "S": "Se", "W": "P", "E": "Sa" },
  "prediction_buffer": ["Sang", "Or", "Sel", "Sang", "Or", "Sel", "Sang", "Or"]
}
```

**Réponse :**
```json
{
  "ok": true,
  "plan_full": ["MOVE N", "MOVE2 NW", ...],
  "plan_now": ["MOVE N", "MOVE W", "WAIT", "MOVE N"],
  "debug": { ... }
}
```

### `POST /next-block`
Calcule le plan **et** simule les 4 premiers tours. Retourne le nouvel état, le nouvel ennemi et les logs détaillés de chaque tour.

---

## Interface

L'interface est un fichier HTML autonome (aucune dépendance externe). Elle se connecte à l'API et affiche :

- **Plateau isométrique SVG** — murs fixes, murs mobiles, position du joueur, objectif, minogolem
- **Flèches de déplacement** — visualisation des actions MOVE/MOVE2/PUSH tour par tour
- **Barre latérale** — saisie des 8 prédictions, vie restante, liste des actions à effectuer
- **Navigation temporelle** — boutons ◀ ▶ pour parcourir les étapes du plan

### Comment utiliser

1. Passer le premier tour sans bouger afin d'afficher les déplacements prévus.
2. Faire un clic droit sur le minogolem → *Afficher les enchantements* → noter les 8 prochains états.
3. Saisir les 8 prédictions dans l'interface (`Sang`, `Sel` ou `Or`), puis cliquer sur **Play**.
4. Exécuter les 4 actions indiquées en jeu.
5. Entrer les 4 nouvelles prédictions (les 4 derniers états de l'enchantement Labyrinthe).
6. Répéter les étapes 3 à 5 jusqu'à la victoire.

> *La victoire n'est pas garantie — le combat comporte une part d'aléatoire. Taux de réussite observé lors des tests : ~90 %.*

---

## Installation

### Prérequis

- Python 3.10+
- `pip install fastapi uvicorn pydantic`

### Lancement

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Pour servir l'interface localement, ouvrir `index.html` dans un navigateur et adapter la constante `API` en haut du script si nécessaire :

```js
const API = "http://localhost:8000";
```
