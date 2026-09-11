# CDR/IPDR Friend-or-Interaction Neural Network

This is the supervised relationship-learning component of SynthNet. It uses the
same full-file identity validator as the CGAN generator, upload API, and IPDR
correlator, so normalization, explicit mappings, and source lineage stay
consistent across the workflow.

The program learns **your definition** of a positive relationship from a base-truth CSV. It treats `friend`, `interacting`, `yes`, `true`, or `1` as positive and `stranger`, `not_interacting`, `no`, `false`, or `0` as negative. “Friend or interacting” is one combined binary class.

It supports both CSV families produced by the existing generator:

| Input | Normalized export | Compact export |
|---|---|---|
| CDR identities | `Mobile_No`, `Other_Party_No` | `caller_id`, `receiver_msisdn` |
| CDR time | `Call_Date` + `Call_Initiation_Time(CIT)` | `timestamp` |
| IPDR identity | `Landline/MSISDN/MDN/Leased Circuit ID for Internet Access` | `subscriber_id` |
| IPDR time | `TIME1 (dd/MM/yyyy HH:mm:ss)` | `timestamp` |

The companion `subscribers.csv` is auto-detected when it sits beside the CDR/IPDR files. It connects `SUB_...`, MSISDN, source identifier, IMEI, and IMSI aliases. You can also choose it explicitly.

## Quick start (Windows GUI)

1. Open PowerShell in this folder.
2. Run `./start_app.ps1`.
3. In **Train model**, choose the CDR, IPDR, and base-truth files.
4. After training, open **Predict pairs**, enter two identifiers, and score them.

Before either operation, choose the explicit CDR/IPDR identity answer. The
application scans every row in chunks, reports subject/subscriber coverage and
observation periods, and stores that decision with the model metadata. Matching
identifiers support a mapping; they do not prove personal identity.

The GUI defaults to the generated `*_schema_clean.csv` files in the parent folder when they exist.

## Base-truth format

Create a CSV like this:

```csv
person_a,person_b,label
+919876500001,+919876500002,1
+919876500001,+919876500003,0
SUB_0000004,SUB_0000007,friend
SUB_0000004,SUB_0000011,stranger
```

Use identifiers present in the supplied data or its `subscribers.csv`. Pair order does not matter. Duplicate pairs are collapsed; contradictory duplicate labels stop training with an error.

The minimum is 10 unique labeled pairs with at least 3 positives and 3 negatives. For a useful model, supply hundreds or thousands of reviewed examples, including realistic hard negatives—not only random strangers. Keep labels from the same observation window as the CDR/IPDR whenever possible.

An empty starter file is at `templates/base_truth_template.csv`.

## Command line

Train:

```powershell
python train_model.py `
  --cdr "..\call_logs_cdr_schema_clean.csv" `
  --ipdr "..\internet_sessions_ipdr_schema_clean.csv" `
  --subscribers "..\subscribers.csv" `
  --base-truth "my_base_truth.csv" `
  --identity-answer same_person `
  --identity-confirmation `
  --output-dir "trained_model"
```

Predict one pair:

```powershell
python predict_pairs.py `
  --model "trained_model\model.pt" `
  --cdr "..\call_logs_cdr_schema_clean.csv" `
  --ipdr "..\internet_sessions_ipdr_schema_clean.csv" `
  --person-a "+919876500001" `
  --person-b "+919876500002" `
  --identity-answer same_person `
  --identity-confirmation
```

Predict a CSV of pairs:

```powershell
python predict_pairs.py `
  --model "trained_model\model.pt" `
  --cdr "..\call_logs_cdr_schema_clean.csv" `
  --ipdr "..\internet_sessions_ipdr_schema_clean.csv" `
  --pairs "pairs.csv" `
  --output "predictions.csv" `
  --identity-answer same_person `
  --identity-confirmation
```

For evidence-only validation, including IPDR-only input, use the shared CLI:

```powershell
python ..\synthnet_validate.py --cdr "..\cdr.csv" --ipdr "..\ipdr.csv" `
  --identity-answer multiple_person_dataset --generation-mode multi_person_conditioned
```

`different_people` and `not_sure` preserve separate-source operations but are
not accepted by the relationship-training command, which requires a
per-person combined mapping. Use the validator/correlator for those cases.

Run `python train_model.py --help` or `python predict_pairs.py --help` for column overrides and training options.

## What the network uses

The CSVs are streamed in chunks. For each labeled or requested unordered pair, the program derives:

- direct CDR event count, direction balance, reciprocity, duration, active days, night/weekend ratios, and contact share;
- each person’s CDR degree plus shared-contact count and Jaccard similarity;
- IPDR session/data-volume balance, hourly-activity cosine similarity, shared destinations/ports/cells, and same-hour activity overlap.

Those engineered values feed a PyTorch multilayer perceptron with two hidden layers, class weighting, validation-based early stopping, and a validation-selected decision threshold. The model output folder contains:

- `model.pt` — model weights, scaler, feature list, and threshold;
- `metrics.json` — train/validation/held-out-test metrics and coverage;
- `base_truth_predictions.csv` — each labeled pair, split, probability, and prediction;
- `engineered_pair_features.csv` — auditable pair features;
- `feature_importance.csv` — relative first-layer input weights (diagnostic, not causal importance).

## Important limitations

A positive prediction is a statistical pattern match, **not proof** that two people are friends. CDR/IPDR behavior can reflect family, work, shared infrastructure, automated traffic, or data-quality errors. Validate performance on representative held-out labels, review false positives and false negatives, comply with applicable privacy and telecom rules, restrict access to the source/output files, and do not use the score as the sole basis for consequential decisions.

