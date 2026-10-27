# Game of Thrones character data

The Kaggle dataset ["Game of Thrones"](https://www.kaggle.com/datasets/mylesoneill/game-of-thrones)
by Myles O'Neill. This project uses one of its three files:

| File | Rows | dead / alive | Columns |
|---|---|---|---|
| `character-predictions.csv` | 1,946 characters | 495 / 1,451 | 33 |

## Where it comes from

Scraped from the fan wiki [A Wiki of Ice and Fire](https://awoiaf.westeros.org) by the
"A Song of Ice and Data" team at the Technical University of Munich (2016), who used it to
predict which characters die next. **It describes the books up to book 5, *A Dance with
Dragons***, not the TV show: some characters who die on TV are still alive here.

## Columns

| Column | What it is | Used? |
|---|---|---|
| `isAlive` | 1 = alive at the end of book 5, 0 = dead | **target** (flipped: `dies` = 1 - isAlive) |
| `name` | the character's name | to look characters up, not as a feature |
| `male` | 1 = male, 0 = female | yes → `male` |
| `house` | house or group (347 different values) | yes → `house`, 12 big houses + "other" + "none" |
| `culture` | culture (64 spellings) | yes → `culture`, 14 groups + "other" + "unknown" |
| `title` | title or seat ("Ser", "Winterfell", "Hand of the King") | yes → `title`, 7 groups |
| `isNoble`, `isMarried` | 1 / 0 | yes → `noble`, `married` |
| `age` | age (known for 22%; two broken values of about −298,000) | yes, cleaned to 0–120 or missing |
| `numDeadRelations` | how many relatives are dead | yes → `dead_relatives` |
| `popularity` | 0–1, how much the wiki talks about them | yes |
| `book1` … `book5` | 1 if the character appears in that book | yes |
| `actual`, `pred`, `alive`, `plod` | the Munich team's answer and predictions | **no: leakage** |
| `DateoFdeath` | year of death (only the dead have one) | **no: leakage** |
| `dateOfBirth` | year of birth | no (`age` says the same) |
| `mother`, `father`, `heir`, `spouse`, `isAlive...` | family (86–99% missing) | no |
| `isPopular`, `boolDeadRelations` | `popularity > 0.5`, `numDeadRelations > 0` | no: repeats |
| `S.No` | row number | no |

Books: 1 *A Game of Thrones*, 2 *A Clash of Kings*, 3 *A Storm of Swords*, 4 *A Feast for
Crows*, 5 *A Dance with Dragons*.
