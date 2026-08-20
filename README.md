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
python3 -m sneakerrag compare "samba og" --size 9 -v    # price table for one size
python3 -m sneakerrag ask "cheapest 990v6 in a 10 under \$200"
python3 -m sneakerrag product DD1391-100                # every seller + the ask curve
python3 -m sneakerrag deals --min-pct 20                # biggest discounts
python3 -m sneakerrag serve                             # JSON API on :8000
```

Install it as a command (`sneakerrag …`) with `pip install -e .`, and add
`pip install -e ".[claude]"` for Claude-written answers.

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

**4. Retrieve** (`retrieve.py`, `embeddings.py`). Hybrid BM25 + vector search over listing
text, with hard structured filters (brand, size, condition, stock, budget) and a large boost
for an exact style-code hit. The budget filter uses the price of the *requested* size, not
the seller's "from" price. Embeddings are a dependency-free hashed bag of words, bigrams and
character 4-grams; short product titles are dominated by lexical signal, and a neural model
would separate `990v6` from `990v5` *worse*, not better. Set `SNEAKERRAG_EMBEDDER=st` to use
`sentence-transformers` if it's installed.

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
python3 -m unittest discover -s tests -t . -v   # 113 tests, no network, no API key
```

Coverage is weighted toward the parts that are easy to get quietly wrong: title parsing,
SKU matching vetoes, per-size and delivered-price ranking, condition filters, retrieval
filters, price history, and the live-scrape path (exercised against fixture HTML through a
fake HTTP client).

## Layout

```
sneakerrag/
  models.py      Listing / Product / QuerySpec / Answer, per-size pricing
  normalize.py   brands, models, colorways, style codes, prices, sizes
  matching.py    SKU clustering — style-code join + guarded fuzzy join
  embeddings.py  dependency-free hashed embeddings (pluggable)
  retrieve.py    hybrid BM25 + vector index with structured filters
  store.py       SQLite catalogue, cached vectors, price history
  llm.py         Claude wrapper with an offline fallback
  agent.py       the RAG pipeline
  cli.py/api.py  command line and JSON API
  sources/       http (robots-aware) · jsonld · site specs · adapters
data/fixtures/   synthetic sample catalogue for the seven sites
tools/           fixture generator
tests/           113 unit + integration tests
```
