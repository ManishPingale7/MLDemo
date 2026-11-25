# Synthetic transaction data — teaching-v2

**synthetic data, patterns modelled on common fraud signals; not real transactions**

This version is intentionally easy: rare risky transactions combine several clear warning signs, and label noise is light. It was designed for high precision and recall in an introductory ML/MLOps presentation. Every probability, distribution, coordinate, cost and threshold below is an **assumption**, not a published statistic or a measured banking pattern. High scores on this dataset do not establish performance on real transactions.

The full preset creates 50,000 original transactions for 5,000 possible customers over 90 calendar days, 1 January–31 March 2025, then appends 150 exact duplicates. Tiny creates 5,000 originals for 500 possible customers plus 15 duplicates. Both use the single seed in `config.yaml`; tiny is a separate CI fixture with the same schema.

| Column | Type / units | Meaning |
|---|---|---|
| txn_id | string | Unique event ID before deliberate duplication; excluded from model inputs. |
| timestamp | datetime, local naive time | Event time in the 90-day period, without timezone conversion. |
| customer_id | integer | Customer identifier; excluded from model inputs. |
| amount_inr | float, INR | Payment amount; routine lognormal spending with a heavy tail and round amounts, or a large relative amount for the risky profile. |
| merchant_category | string | grocery, food, fuel, travel, shopping, utilities, electronics or gaming. |
| city | string | Mumbai, Delhi, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Jaipur, Ahmedabad or Lucknow. |
| distance_from_home_km | float, km | Approximate straight-line city distance plus a small positive local distance; about 2.5% missing. |
| device_type | string | mobile, desktop or pos_terminal; about 2% missing. |
| is_new_merchant | 0/1 | First observed payment to an internal merchant ID for this customer within the generated period. Merchant IDs are discarded; there is only one output table. |
| txns_last_1h | integer, count | Number of preceding customer transactions within the previous 3,600 seconds, computed during generation before duplicates are added. |
| customer_avg_amount_inr | float, INR | Customer's fixed nominal typical spend, drawn once; not calculated from this table or future rows. Realised average spending can differ. |
| is_fraud | 0/1 | Lightly noisy label sampled after the transaction profile is generated. |

## How the two profiles are generated

Each customer has a fixed home city, preferred category and typical spend. The generator samples an internal routine/risky profile **before** producing the transaction attributes. Profile membership is never exported or used as a model input; labels are drawn separately afterwards. No ID, missingness flag or hidden profile column is supplied to a classifier.

| Attribute | Routine profile assumption | Risky profile assumption |
|---|---|---|
| Relative amount | Typical spend × lognormal factor; 4% receive a 4–15× tail multiplier; 12% are replaced by INR 100/500/1,000. | Typical spend × a uniform multiplier from 7 to 15. |
| Location | 90% of city draws use home city; the rest choose a city at random. | A different city from the customer's home is selected. |
| Merchant category | 65% of draws use the preferred category; the rest choose at random. | 15% shopping, 50% electronics, 35% gaming. |
| Device | 60% mobile, 15% desktop, 25% POS terminal. | 85% desktop, 15% mobile. |
| Merchant familiarity | A mixture of recurring merchants and occasional other merchants. | A fresh internal merchant ID, so this is a first-observed payment. |
| Initial hour | Evening activity is favoured and overnight activity is less common. | 75% are initially assigned an hour before 06:00; the remainder use ordinary hours. |
| Shopping burst | 18% chance of following the previous generated transaction for its customer within 15 minutes. | 55% chance of that same burst behaviour. |

A burst inherits the preceding event's customer and time, so final night-time proportions differ from the initial hour assignments. Previous-hour counts are calculated from the resulting sorted events, not invented from the label. Different cities can be less than 200 km apart; the model learns from several signals together.

Routine payments can also be large, distant, new-merchant or desktop transactions. The **combination** of signals makes risky profiles easy to identify. This is an intentional simplification for teaching, not a realistic estimate of how separable actual fraud would be.

## Labels and prevalence

Assumed fraud probability is **97% for the risky profile** and **0.02% for the routine profile**, configured in `config.yaml`. The risky-profile sampling rate is:

`(target_fraud_rate - routine_fraud_probability) / (risky_fraud_probability - routine_fraud_probability)`

With the supplied settings this is approximately 1.526%, giving an **expected** fraud rate of 1.5%. Bernoulli draws determine the actual labels. Some risky-profile transactions are legitimate and a few routine-profile transactions are fraudulent, so the example is not perfectly separable. Small tiny-CI splits can nevertheless score perfectly by chance.

The 1.5% prevalence is an **assumption** chosen for readable charts. The idea that real operational fraud rates would be lower is also an assumption here, not a claim about a particular institution. No published fraud statistics are used.

## Time patterns and deliberate defects

Evening hours 18–22 have 2.2 times the ordinary hour weight; hours 00–05 have 0.35 times the weight. Weekends add 0.25 to the day weight and days 1–3 add 0.20. First-observed merchant flags are more common early because no pre-period history is modelled.

Defect rates apply before appended duplicates, so final CSV rates differ slightly. Defects are sampled independently of the fraud label.

| Defect | Assumption / amount | Treatment |
|---|---|---|
| Missing distance | 2.5% | Training-only numeric median inside each pipeline. |
| Missing device | 2.0% | Literal `unknown`; pipeline also supports categorical imputation. |
| Exact duplicates | 0.3% extra rows | Drop exact duplicates. |
| Zero/negative amounts | 10 full / 4 tiny | Remove invalid rows. |
| Category case and spaces | 1.0% | Strip spaces and convert to lowercase. |

Timestamps produce `hour` and `day_of_week`; four engineered features are `is_night`, `amount_to_avg_ratio`, `is_far_from_home`, and `log_amount`. Missing distance remains missing in its derived flag until training-only imputation. No transaction history is computed in a notebook.

## Reproducibility and versioning

The same seed, configuration, Python and package versions reproduce the data and predictive metrics. Runtime, run IDs, Git SHAs and registry version numbers naturally vary. Bit-for-bit equality across operating systems or CPU architectures is not promised.

`teaching-v2` replaces the earlier overlapping-risk dataset with these clearer profiles. Old and new scores describe different datasets and must not be interpreted as a model improvement on the same benchmark. MLflow logs the dataset version, and the quality gate rejects a model tagged for a different version.
