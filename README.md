# Sneaker Price-Comparison RAG Agent

Ask in plain English which site sells a given Nike, adidas or New Balance shoe cheapest,
across the brand stores and the big resale marketplaces. Answers are grounded in real
listings, matched at SKU level, priced **per size**, and cited.

```console
$ sneakerrag ask "cheapest nike dunk low in size 10.5"

Nike Dunk Low Retro — White / Black (style DD1391-100) — MSRP $120.00
  Sold out at the brand store — the listings below are resale asks.
  Cheapest in a US 10.5: GOAT: $190.54 for a US 10.5 (+$12.00 shipping; +$5.00 fees;
  delivered $207.54) — 58.8% above the $120.00 MSRP [3]
  https://www.goat.com/sneakers/nike-dunk-low-retro-white-black-dd1391-100
  Saves $14.89 vs the priciest of the 2 listings compared.
    - Flight Club: $207.93 for a US 10.5 (+$14.50 shipping; delivered $222.43) [4]
    - excluded KicksCrew (no size 10.5) [1]
    - excluded Stadium Goods (no size 10.5) [2]

Sources
  [1] KicksCrew — $144.08 (no US 10.5; price shown is their lowest ask) — https://…
  [2] Stadium Goods — $169.79 (no US 10.5; price shown is their lowest ask) — https://…
  [3] GOAT — $207.54 — https://www.goat.com/sneakers/nike-dunk-low-retro-white-black-dd1391-100
  [4] Flight Club — $222.43 — https://www.flightclub.com/nike-dunk-low-retro-white-black-dd1391-100
```

## Quick start

No dependencies required — the whole pipeline runs on the Python standard library.

```bash
git clone https://github.com/luharry2821/RAG-Agent---Price-Comp.git
cd RAG-Agent---Price-Comp

python3 -m sneakerrag ingest                            # load the bundled sample catalogue
python3 -m sneakerrag chat                              # interactive: type a shoe, get the cheapest site
python3 -m sneakerrag compare "samba og" --size 9 -v    # price table for one size
python3 -m sneakerrag ask "cheapest 990v6 in a 10 under \$200"
python3 -m sneakerrag product DD1391-100                # every seller + the ask curve
python3 -m sneakerrag deals --min-pct 20                # biggest discounts
python3 -m sneakerrag serve                             # JSON API on :8000
```

Install it as a command (`sneakerrag …`) with `pip install -e .`, and add
`pip install -e ".[claude]"` for Claude-written answers.

## Talking to it

`chat` is the quickest way in. Type a shoe; filters you set stick until you change them,
and the index is built once so follow-up questions are instant.

```console
$ sneakerrag chat
sneakerrag — 47 listings across 7 sites, answers by offline template

shoe> panda dunks
Nike Dunk Low Retro — White / Black (style DD1391-100) — MSRP $120.00
  Sold out at the brand store — the listings below are resale asks.
  Cheapest: KicksCrew: $144.08 lowest ask (free shipping) — 20.1% above the $120.00 MSRP [1]
  …

shoe> :size 13
size: 13

shoe> dunk low
  Cheapest in a US 13: KicksCrew: $144.08 for a US 13 …
    - excluded GOAT (no size 13) [3]
```

`:help` lists the commands (`:size`, `:condition`, `:brand`, `:shipping`, `:limit`,
`:filters`, `:sites`, `:stats`). One-shot equivalents are `ask` (prose answer) and `compare`
(price table); `serve` exposes the same thing over HTTP.

**Prices come from the bundled sample data.** Until you run `ingest --live` against a real
source, every number you see is synthetic — the pipeline is real, the prices are invented.

## The sites

| key | site | kind | brands |
|---|---|---|---|
| `nike` | Nike | brand store | Nike |
| `adidas` | adidas | brand store | adidas |
| `newbalance` | New Balance | brand store | New Balance |
| `stadiumgoods` | Stadium Goods | resale | all three |
| `flightclub` | Flight Club | resale | all three |
| `goat` | GOAT | resale | all three |
| `kickscrew` | KicksCrew | resale | all three |

Add or swap a site by editing `sneakerrag/sources/sites.py`.

## Retail and resale are not the same problem

This is the part that shapes the whole design. A brand store sells one price for every
size, new, in stock or not. A resale marketplace is an order book: **each size has its own
ask**, condition varies, and shipping plus authentication/processing fees land at checkout.
So the agent models:

- **per-size price curves** (`size_prices`) — "cheapest Dunk Low" is meaningless without a
  size. In the bundled sample the same GOAT listing asks $199 for a 10 and $144 for a 13;
  the winner changes with the size, and every price the agent quotes is the price for the
  size you asked about;
- **delivered price** — item + shipping + fees. A $150 ask with $17 of extras loses to a
  $160 ask that ships free, and the agent ranks on the receipt, not the sticker;
- **condition** — used listings undercut new ones and are labelled as such; ask for
  "brand new" or "deadstock" and they're filtered out;
- **MSRP premium** — resale asks are quoted against retail, so you can see you're paying
  59% over MSRP before you click;
- **sold out at retail** — when no brand store still carries the shoe, the answer says so
  up front. That's the case these marketplaces exist for.

One caveat the agent surfaces but can't fix: Flight Club and GOAT are both GOAT Group
properties and often show the same inventory, so treat a match between them as one supply
pool rather than two independent quotes.

## How it works

```
ingest ──▶ normalize ──▶ embed ──▶ SQLite catalogue + price history
                                          │
question ──▶ understand ──▶ retrieve (BM25 + vectors) ──▶ cluster by SKU
                                          │
                            price the requested size ──▶ grounded answer + citations
```

**1. Ingest** (`sneakerrag/sources/`). One adapter per site. Live adapters read schema.org
`Product`/`Offer` JSON-LD — the structured data these sites already publish for search
engines — which is far more stable than CSS scraping. Marketplaces publish one `Offer` per
size under an `AggregateOffer`; those become the price curve. The bundled fixtures let
everything run offline.

**2. Normalize** (`normalize.py`). Sellers write the same shoe every possible way
(`Nike Dunk Low Retro Men's Shoes`, `Nike Dunk Low Retro 'White/Black'`,
`Nike Dunk Low Retro White/Black DD1391-100`). This step pulls out brand, model, colorway,
gender and — most importantly — the manufacturer style code, with per-brand patterns
(`DD1391-100`, `B75806`, `M990GL6`).

**3. Match** (`matching.py`). The heart of "the same shoe/SKU":

- a shared style code is authoritative — same code, same shoe;
- listings without a code are joined fuzzily on brand + model + colorway + audience;
- hard vetoes prevent the expensive mistakes: conflicting style codes, different model
  numbers (`990v6` ≠ `990v5`, `Air Max 90` ≠ `Air Max 95`), men's vs women's;
- clustering is agglomerative and re-validated after every merge, so a vague listing can't
  bridge two different colorways. When a colourless listing has two equally good suitors it
  stays on its own rather than quoting you a price for the wrong shoe.

**4. Retrieve** (`retrieve.py`, `aliases.py`, `embeddings.py`). Hybrid search with hard
structured filters (brand, size, condition, stock, budget) applied first — a shopping
constraint is a predicate, not a nudge — and the budget checked against the price of the
*requested* size rather than the seller's "from" price. On top of that:

- **Inverted index, BM25F.** Only documents containing a query term are scored, and every
  per-document statistic is computed once at build time. Term frequencies are weighted by
  field, so a match on the style code or model name counts for far more than the same word
  buried in a marketing title.
- **Shopper vocabulary** (`aliases.py`). People type "panda dunks", "af1s", "sambas",
  "adiddas", "dad shoes". A small domain dictionary — nicknames, plurals, colorway slang,
  common misspellings — plus trigram typo repair against the actual corpus vocabulary.
  Expansions are damped, so an alias can add recall but never outvote what was typed.
  Retail noise words ("cheapest", "shoes", "size") are dropped from the query, which stops
  them crowding out the one token that identifies the shoe.
- **Hybrid fusion.** BM25 and cosine live on incomparable scales. Normalising each by its
  max is sharp at rank 1 (BM25's *margin* is real signal); reciprocal-rank fusion is
  steadier deeper down (it can't be skewed by an outlier). Measured, neither wins outright,
  so the default averages both — see the table below.
- **A relevance floor.** Ask for an Asics and a nearest-neighbour search will hand back a
  Nike; the agent then fluently prices a shoe you never asked about, which is worse than no
  answer. To be returned at all, a listing must satisfy enough of the *typed* words —
  credited back through any expansion, so a nickname cannot vouch for itself — and at least
  one match must identify a model rather than describe one. Colours, cuts ("low", "retro",
  "og") and brand names are descriptive: "Air Jordan 1 Retro High OG Chicago" must not
  return a Dunk Low Retro on the strength of "retro" plus the colours "Chicago" expands to,
  and "New Balance 2002R" must not return a 990v6 on the words "new balance". A query that
  matches nearly everything typed passes regardless, so a pure-colorway search ("silver sea
  salt new balance") still works. Out-of-catalogue queries return nothing, and the agent
  says so.
- **Two-stage, rare terms first.** BM25 is the cheap first pass and only its best candidates
  get the cosine and fusion work; within it, rare terms select candidates while common ones
  ("white", "low") merely refine the ranking rather than walking their whole posting list.

Embeddings are a dependency-free hashed bag of words, bigrams and character 4-grams; short
product titles are dominated by lexical signal, and a neural model would separate `990v6`
from `990v5` *worse*, not better. Set `SNEAKERRAG_EMBEDDER=st` to use
`sentence-transformers` if it's installed — the index detects a real dense model and widens
the vector pass to the whole catalogue automatically. Install `numpy` and the cosine pass
becomes a single matrix multiply; it is optional and everything works without it.

**5. Compare** (`models.py`). Within a cluster, offers are ranked by delivered price for the
requested size, honouring availability, stock and condition.

**6. Answer** (`agent.py`, `llm.py`). Candidate retrieval deliberately ignores the budget and
size filters so the price table stays complete — you can't tell someone what they're saving
if the pricier listings were filtered away. Every listing in the cluster is numbered and
cited, including the ones that failed a constraint, with the reason attached (a citation
whose seller has no stock in your size says so, and flags that its price is a different
size's ask).

## Claude

Claude (`claude-opus-5`) does the two jobs where language matters: turning a messy question
into structured filters, and writing the final answer over the retrieved evidence. It is
never the source of a price — retrieval, matching and ranking are deterministic Python, and
the system prompt forbids inventing a seller, price or URL.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...   # or: ant auth login
python3 -m sneakerrag ask "which site has the cheapest deadstock samba og in a 10, and how much over retail is it"
```

Without credentials the agent falls back to a heuristic query parser and a deterministic
answer writer, so **every command works offline** — `--no-llm` forces that path. Configure
with `SNEAKERRAG_MODEL` and `SNEAKERRAG_EFFORT`.

## Live scraping

The bundled catalogue is synthetic sample data (see `data/fixtures/*.json` and
`tools/make_fixtures.py`) — realistic in shape, invented in price. To pull real listings:

```bash
python3 -m sneakerrag ingest --live --sources goat,kickscrew --query "dunk low"
```

Before you do, read this honestly: **these are third-party sites, and crawling them is a
decision you own, not a default.** The client already refuses paths disallowed by
`robots.txt`, rate-limits per host, caches responses on disk and identifies itself in the
User-Agent — but check each site's Terms of Service first, and prefer an official
affiliate, partner or developer API where one exists. The four resale marketplaces in
particular front their pages with bot protection that will simply refuse an automated
client, and their per-size ask data is the product they sell — expect to need a sanctioned
feed rather than a crawler. `StaticSource` accepts records from one directly, so a licensed
feed drops straight in:

```python
from sneakerrag.sources import SITES_BY_KEY, StaticSource
source = StaticSource(SITES_BY_KEY["goat"], records_from_your_licensed_feed)
```

The `product_link` patterns in `sites.py` are a starting point, not a guarantee — site URL
shapes change, so verify them against the live markup before trusting a live run.

## Performance and quality

`tools/benchmark.py` measures both, so improvements are demonstrated rather than asserted.
Quality is scored on 34 real-shopper phrasings (nicknames, plurals, misspellings, colorway
slang, style codes) plus 9 queries for shoes the catalogue does not carry — including
detailed ones like "Air Jordan 1 Retro High OG Chicago Lost & Found" — where the right
answer is *no* answer.

```bash
python3 tools/benchmark.py             # both halves
python3 tools/benchmark.py --quality
```

Against the previous implementation (max-normalised blend over a full per-query scan, no
vocabulary, no floor), on identical data:

| | recall@1 | MRR | wrong-shoe answers |
|---|---|---|---|
| before | 94% | 0.956 | 9 / 9 |
| after | 100% | 1.000 | 0 / 9 |

| corpus | before, p50 | after, p50 | |
|---|---|---|---|
| 1,000 listings | 63 ms | 2.0 ms | 32× |
| 10,000 listings | 608 ms | 15 ms | 41× |
| 50,000 listings | 3,270 ms | 63 ms | 52× |

Two caveats worth stating plainly: 100% on 34 queries means the eval set is small, not that
retrieval is solved — the "Lost & Found" case above was found by a user typing one shoe name,
not by the suite — extend `GOLDEN` and `NEGATIVES` in the benchmark as you add shoes. And
the speed figures are on a synthetic catalogue with realistic vocabulary spread; a corpus
where every listing shares the same few words behaves worse.

Which fusion to use is a measurement, not a preference:

| configuration | recall@1 | recall@3 | MRR |
|---|---|---|---|
| linear blend, no vocabulary | 94% | 97% | 0.956 |
| linear blend + vocabulary | 100% | 100% | 1.000 |
| RRF + vocabulary | 100% | 100% | 1.000 |
| **hybrid + vocabulary** (shipped) | **100%** | **100%** | **1.000** |

The vocabulary is doing most of the work; fusion mode is a tiebreak at this corpus size.
`VectorIndex(fusion="linear"|"rrf"|"hybrid", expand=False)` flips each knob.

## Price history

Every ingest appends a point per listing, so re-running it on a schedule turns the catalogue
into a price tracker:

```bash
python3 -m sneakerrag product B75806     # per-seller prices, ask curves, and the trail over time
python3 -m sneakerrag deals --min-pct 15 # current discounts, plus drops since first seen
```

## JSON API

```bash
python3 -m sneakerrag serve --port 8000
curl "localhost:8000/ask?q=cheapest+dunk+low+size+10.5"
curl "localhost:8000/compare?q=samba+og&size=9&condition=new&limit=1"
curl "localhost:8000/product?style_code=M990GL6"
```

Endpoints: `/health`, `/sources`, `/stats`, `/compare`, `/ask`, `/product`.

## Python API

```python
from sneakerrag import SneakerAgent

agent = SneakerAgent()
print(agent.answer("cheapest dunk low in a 10.5").text)

spec, products = agent.compare("adidas samba og")
best = products[0].best_offer(size="9")
print(best.source_name, best.price_for("9"), "+", best.extras, "=", best.total_for("9"))
print("saving:", products[0].savings(size="9"), "| still at retail:", products[0].at_retail)
```

## Tests

```bash
python3 -m unittest discover -s tests -t . -v   # 141 tests, no network, no API key
```

Coverage is weighted toward the parts that are easy to get quietly wrong: title parsing,
SKU matching vetoes, per-size and delivered-price ranking, condition filters, retrieval
filters and the relevance floor, nickname and typo handling, price history, and the
live-scrape path (exercised against fixture HTML through a fake HTTP client). Results are
asserted to be identical with and without `numpy`.

## Layout

```
sneakerrag/
  models.py      Listing / Product / QuerySpec / Answer, per-size pricing
  normalize.py   brands, models, colorways, style codes, prices, sizes
  matching.py    SKU clustering — style-code join + guarded fuzzy join
  embeddings.py  dependency-free hashed embeddings (pluggable)
  aliases.py     sneaker nicknames, slang, misspellings, query stopwords
  retrieve.py    inverted-index BM25F + vector index, hybrid fusion, relevance floor
  store.py       SQLite catalogue, cached vectors, price history
  llm.py         Claude wrapper with an offline fallback
  agent.py       the RAG pipeline
  cli.py/api.py  command line and JSON API
  sources/       http (robots-aware) · jsonld · site specs · adapters
data/fixtures/   synthetic sample catalogue for the seven sites
tools/           fixture generator, speed + quality benchmark
tests/           141 unit + integration tests
```
