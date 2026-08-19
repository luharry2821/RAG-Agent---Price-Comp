# Sneaker Price-Comparison RAG Agent

Ask in plain English which site sells a given Nike, adidas or New Balance shoe cheapest,
and get an answer grounded in real listings — with the SKU-level matching that makes the
comparison trustworthy, and a citation for every price.

```console
$ sneakerrag ask "cheapest nike air max 90 in size 10.5"

Nike Air Max 90 — White / Black (style HM0089-100)
  Cheapest: Foot Locker — $112.71 — 13.3% off $130.00, free shipping [1]
  https://www.footlocker.com/product/~nike-air-max-90/HM0089-100.html
  Saves $17.29 vs the most expensive of 2 listings checked.
    - Nike: $130.00 / $130.00 delivered [4]
    - excluded JD Sports: $114.40 (no size 10.5) [2]
    - excluded Dick's Sporting Goods: $129.48 (out of stock) [3]

Size filter: US 10.5.

Sources
  [1] Foot Locker — $112.71 — https://www.footlocker.com/product/~nike-air-max-90/HM0089-100.html
  [2] JD Sports — $121.39 — https://www.jdsports.com/product/nike-air-max-90/16000000/
  [3] Dick's Sporting Goods — $129.48 — https://www.dickssportinggoods.com/p/nike-air-max-90/22000000
  [4] Nike — $130.00 — https://www.nike.com/t/nike-air-max-90-hm0089/HM0089-100
```

## Quick start

No dependencies required — the whole pipeline runs on the Python standard library.

```bash
git clone https://github.com/luharry2821/RAG-Agent---Price-Comp.git
cd RAG-Agent---Price-Comp

python3 -m sneakerrag ingest                       # load the bundled sample catalogue
python3 -m sneakerrag compare "adidas samba og" -v # price table across sites
python3 -m sneakerrag ask "new balance 990v6 under \$180 including shipping"
python3 -m sneakerrag deals --min-pct 20           # biggest discounts
python3 -m sneakerrag serve                        # JSON API on :8000
```

Install it as a command (`sneakerrag …`) with `pip install -e .`, and add
`pip install -e ".[claude]"` for Claude-written answers.

## The six sites

| key | site | brands |
|---|---|---|
| `nike` | Nike | Nike |
| `adidas` | adidas | adidas |
| `newbalance` | New Balance | New Balance |
| `footlocker` | Foot Locker | all three |
| `jdsports` | JD Sports | all three |
| `dickssportinggoods` | Dick's Sporting Goods | all three |

Brand stores anchor MSRP; the three multi-brand retailers are where the discounts turn up.
Add a site by appending a `SiteSpec` to `sneakerrag/sources/sites.py`.

## How it works

```
ingest ──▶ normalize ──▶ embed ──▶ SQLite catalogue + price history
                                          │
question ──▶ understand ──▶ retrieve (BM25 + vectors) ──▶ cluster by SKU
                                          │
                                   compare prices ──▶ grounded answer + citations
```

**1. Ingest** (`sneakerrag/sources/`). One adapter per retailer. Live adapters read
schema.org `Product`/`Offer` JSON-LD — the same structured data retailers publish for
search engines — which is far more stable than CSS scraping. The bundled fixtures let
everything run offline.

**2. Normalize** (`normalize.py`). Retailers write the same shoe six different ways
(`Nike Air Max 90 Men's Shoes`, `Air Max 90 - Men's - White/Black`,
`Nike Air Max 90 (HM0089-100)`). This step pulls out brand, model, colorway, gender and —
most importantly — the manufacturer style code, with per-brand patterns
(`DD1391-100`, `B75806`, `M990GL6`).

**3. Match** (`matching.py`). The heart of "the same shoe/SKU":

- a shared style code is authoritative — same code, same shoe;
- listings without a code are joined fuzzily on brand + model + colorway + audience;
- hard vetoes prevent the expensive mistakes: conflicting style codes, different model
  numbers (`990v6` ≠ `990v5`, `Air Max 90` ≠ `Air Max 95`), men's vs women's;
- clustering is agglomerative and re-validated after every merge, so a vague listing
  can't bridge two different colorways. When a colourless listing has two equally good
  suitors, it stays on its own rather than quoting you a price for the wrong shoe.

**4. Retrieve** (`retrieve.py`, `embeddings.py`). Hybrid BM25 + vector search over listing
text, with hard structured filters (brand, size, stock, budget) and a large boost for an
exact style-code hit. Embeddings are a dependency-free hashed bag of words, bigrams and
character 4-grams; short product titles are dominated by lexical signal, and a neural
model would separate `990v6` from `990v5` *worse*, not better. Set
`SNEAKERRAG_EMBEDDER=st` to use `sentence-transformers` if it's installed.

**5. Compare** (`models.py`). Within a cluster, offers are ranked by *delivered* price
(item + shipping), honouring size availability and stock. Shipping regularly flips the
winner — a $85 shoe with $20 delivery loses to a $95 shoe with free shipping.

**6. Answer** (`agent.py`, `llm.py`). Candidate retrieval deliberately ignores the
budget and size filters so the price table stays complete — you can't tell someone what
they're saving if the pricier listings were filtered away. Every listing in the cluster is
numbered and cited, including the ones that failed a constraint, with the reason attached.

## Claude

Claude (`claude-opus-5`) does the two jobs where language matters: turning a messy question
into structured filters, and writing the final answer over the retrieved evidence. It is
never the source of a price — retrieval, matching and ranking are deterministic Python, and
the system prompt forbids inventing a retailer, price or URL.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...   # or: ant auth login
python3 -m sneakerrag ask "which site has the cheapest samba og in a 10, and how much do I save"
```

Without credentials the agent falls back to a heuristic query parser and a deterministic
answer writer, so **every command works offline** — `--no-llm` forces that path. Configure
with `SNEAKERRAG_MODEL` and `SNEAKERRAG_EFFORT`.

## Live scraping

The bundled catalogue is synthetic sample data (see `data/fixtures/*.json` and
`tools/make_fixtures.py`) — realistic in shape, invented in price. To pull real listings:

```bash
python3 -m sneakerrag ingest --live --sources footlocker,jdsports --query "air max 90"
```

Before you do, please read this honestly: **these are third-party sites, and crawling them
is a decision you own, not a default.** The client already refuses paths disallowed by
`robots.txt`, rate-limits per host, caches responses on disk and identifies itself in the
User-Agent — but check each retailer's Terms of Service first, and prefer an official
affiliate or partner API where one exists (most of these retailers have one). Several sites
also front their pages with bot protection that will simply refuse an automated client; a
sanctioned data feed is the durable answer, and `StaticSource` accepts records from one
directly.

## Price history

Every ingest appends a point per listing, so re-running it on a schedule turns the catalogue
into a price tracker:

```bash
python3 -m sneakerrag product B75806     # per-site prices + the trail over time
python3 -m sneakerrag deals --min-pct 15 # current discounts, plus drops since first seen
```

## JSON API

```bash
python3 -m sneakerrag serve --port 8000
curl "localhost:8000/ask?q=cheapest+air+max+90+size+10.5"
curl "localhost:8000/compare?q=samba+og&limit=1"
curl "localhost:8000/product?style_code=M990GL6"
```

Endpoints: `/health`, `/sources`, `/stats`, `/compare`, `/ask`, `/product`.

## Python API

```python
from sneakerrag import SneakerAgent

agent = SneakerAgent()
answer = agent.answer("cheapest 990v6 in size 10")
print(answer.text)

spec, products = agent.compare("adidas samba og")
best = products[0].best_offer(size="10")
print(best.source_name, best.total_price, best.url)
print("saving:", products[0].savings(size="10"))
```

## Tests

```bash
python3 -m unittest discover -s tests -t . -v   # 89 tests, no network, no API key
```

Coverage is weighted toward the parts that are easy to get quietly wrong: title parsing,
SKU matching vetoes, delivered-price ranking, retrieval filters, price history, and the
live-scrape path (exercised against fixture HTML through a fake HTTP client).

## Layout

```
sneakerrag/
  models.py      Listing / Product / QuerySpec / Answer
  normalize.py   brands, models, colorways, style codes, prices, sizes
  matching.py    SKU clustering — style-code join + guarded fuzzy join
  embeddings.py  dependency-free hashed embeddings (pluggable)
  retrieve.py    hybrid BM25 + vector index with structured filters
  store.py       SQLite catalogue, cached vectors, price history
  llm.py         Claude wrapper with an offline fallback
  agent.py       the RAG pipeline
  cli.py/api.py  command line and JSON API
  sources/       http (robots-aware) · jsonld · site specs · adapters
data/fixtures/   synthetic sample catalogue for the six sites
tools/           fixture generator
tests/           89 unit + integration tests
```
