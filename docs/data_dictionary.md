# Synthetic transaction data

**synthetic data, patterns modelled on common fraud signals; not real transactions**

All distributions, coordinates, risk weights, costs and thresholds in this project are **assumptions** for teaching. No published fraud statistics are used. The 1.5% target is an **assumption**, deliberately raised to make charts and metrics readable; the idea that operational rates would be lower is also an assumption, not a claim about any particular bank or payment network.

The full preset creates 50,000 original transactions for 5,000 possible customers over 90 calendar days, 1 January–31 March 2025. It then appends 150 exact duplicate rows. The tiny preset creates 5,000 originals for 500 possible customers plus 15 duplicates. Both use the single seed in `config.yaml`. Tiny is a separate smoke-test fixture, not a subset used in the full experiment.

| Column | Type / units | Meaning and generation assumption |
|---|---|---|
| txn_id | string | Unique event ID before deliberate duplication. |
| timestamp | datetime, local naive time | Event time in the 90-day period; no timezone conversion is modelled. |
| customer_id | integer | Customer identifier; excluded from model inputs. |
| amount_inr | float, INR | Typical customer spend times a lognormal factor; 4% receive a larger multiplier, then 12% are replaced with 100, 500 or 1,000 INR. |
| merchant_category | string | grocery, food, fuel, travel, shopping, utilities, electronics or gaming; 65% of category draws use the customer's preferred category. |
| city | string | Mumbai, Delhi, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Jaipur, Ahmedabad or Lucknow; 90% of city draws use the customer's home city. |
| distance_from_home_km | float, km | Approximate straight-line city distance plus a small positive local distance; about 2.5% missing. Coordinates and conversion factors are rough teaching assumptions. |
| device_type | string | mobile, desktop or pos_terminal, with assumed probabilities 60%, 15%, 25%; about 2% missing. |
| is_new_merchant | 0/1 | First observed payment to an internal merchant ID for this customer in the generated period. These IDs are used during generation and discarded; there is only one output table. |
| txns_last_1h | integer, count | Number of preceding customer transactions within the previous 3,600 seconds, computed during generation before duplicates are added. |
| customer_avg_amount_inr | float, INR | Customer's fixed nominal typical spend, drawn once from a lognormal distribution; not a statistic computed from this table or future rows. Tail events mean the realised average can differ. |
| is_fraud | 0/1 | Noisy synthetic label sampled from the documented rule below. |

## Time and customer patterns

Evening hours 18–22 have 2.2 times the base sampling weight; hours 00–05 have 0.35 times the weight. Weekends add 0.25 to the day weight and days 1–3 add 0.20. About 18% of generated events follow the previous generated event for the same customer within 15 minutes, creating short bursts. These are assumptions, not measured Indian spending patterns.

Each customer has a fixed home city, preferred category and typical spend. First-observed merchant flags are more frequent early in the period because no pre-period history exists. This produces a changing fraud rate across time; the notebooks report the actual split rates instead of assuming they match.

## Explainable noisy fraud rule

The assumed risk score adds:

- 2.4 for hour before 06:00;
- 2.8 for amount exceeding three times the customer's typical spend;
- 2.7 for distance exceeding 200 km;
- 1.6 for a first-observed merchant;
- 2.2 for at least three previous-hour transactions;
- 1.0 for electronics or gaming;
- 0.5 for desktop.

Probability is `0.0025 + 0.90 * sigmoid(1.8 * (score - intercept))`. Sigmoid maps a score to a number between zero and one. Bisection finds an intercept giving an **expected** 1.5% fraud rate; independent Bernoulli draws create the actual labels. Background fraud and random draws make perfect prediction impossible. This is a transparent simulation rule, not a fitted model or a published fraud formula.

## Deliberate data defects

Rates apply before appended duplicates, so final CSV rates differ slightly.

| Defect | Assumption / amount | Treatment |
|---|---|---|
| Missing distance | 2.5% | Training-only numeric median inside each pipeline. |
| Missing device | 2.0% | Literal `unknown`; pipeline also has categorical imputation. |
| Exact duplicates | 0.3% extra rows | Drop exact duplicate rows. |
| Zero/negative amounts | 10 full / 4 tiny | Remove invalid rows. |
| Category case and spaces | 1.0% | Strip spaces and convert to lowercase. |

Timestamps produce `hour` and `day_of_week`; four additional features are `is_night`, `amount_to_avg_ratio`, `is_far_from_home`, and `log_amount`. Missing distance stays missing in its derived flag until training-only imputation. IDs and raw timestamps are excluded from model inputs. No transaction history is computed in a notebook.

## Reproducibility and limits

The same seed, configuration, Python and package versions reproduce the data and model metrics. Recorded runtime, run IDs, Git SHAs, registry version numbers and timestamps naturally change between runs. Binary equality across operating systems or CPU architectures is not promised. Labels and costs are assumptions; this project cannot validate a real fraud policy, and its precision is not a production estimate.
