from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from friendship_nn.features import FEATURE_NAMES, extract_pair_features
from friendship_nn.identities import IdentityResolver
from friendship_nn.model import load_artifact, predict_with_artifact, save_artifact, train_neural_network
from friendship_nn.pairs import prepare_pairs
from friendship_nn.schemas import resolve_data_schema


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.people = [f"+9199000000{index:02d}" for index in range(8)]
        self.cdr_path = self.root / "cdr.csv"
        self.ipdr_path = self.root / "ipdr.csv"
        self.subscribers_path = self.root / "subscribers.csv"
        self._write_fixture_data()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _is_positive(self, first: int, second: int) -> bool:
        return first // 2 == second // 2 or (first < 4 and second < 4)

    def _write_fixture_data(self) -> None:
        cdr_rows: list[dict[str, object]] = []
        serial = 1
        for first in range(len(self.people)):
            for second in range(first + 1, len(self.people)):
                if not self._is_positive(first, second):
                    continue
                for repeat in range(4):
                    for caller, receiver in ((first, second), (second, first)):
                        cdr_rows.append(
                            {
                                "SL_NO": serial,
                                "Mobile_No": self.people[caller],
                                "Other_Party_No": self.people[receiver],
                                "Call_Date": f"{repeat + 1:02d}/06/2026",
                                "Call_Initiation_Time(CIT)": f"{10 + repeat:02d}:15:00",
                                "Call_Duration": 120 + 10 * repeat,
                                "Call_Type": "VOICE",
                                "First_Cell_id": f"CELL_{caller % 2}",
                            }
                        )
                        serial += 1
        pd.DataFrame(cdr_rows).to_csv(self.cdr_path, index=False)

        ipdr_rows: list[dict[str, object]] = []
        for person_index, person in enumerate(self.people):
            group = person_index // 2
            for repeat in range(6):
                ipdr_rows.append(
                    {
                        "Landline/MSISDN/MDN/Leased Circuit ID for Internet Access": person,
                        "TIME1 (dd/MM/yyyy HH:mm:ss)": f"2026/06/{repeat + 1:02d} {9 + group:02d}:00:00",
                        "Destination IP Address": f"10.10.{group}.{repeat % 2}",
                        "Destination Port": 443 if repeat % 2 else 5222,
                        "First CELL ID": f"CELL_{group}",
                        "Session Duration (Seconds)": 300 + repeat,
                        "Data Volume Up Link": 100 + repeat,
                        "Data Volume Down Link": 300 + repeat,
                    }
                )
        pd.DataFrame(ipdr_rows).to_csv(self.ipdr_path, index=False)
        pd.DataFrame(
            {
                "subscriber_id": [f"SUB_{index:04d}" for index in range(len(self.people))],
                "msisdn": self.people,
                "source_identifier": [f"SRC_{index:04d}" for index in range(len(self.people))],
            }
        ).to_csv(self.subscribers_path, index=False)

    def _pairs(self, resolver: IdentityResolver) -> pd.DataFrame:
        rows = []
        for first in range(len(self.people)):
            for second in range(first + 1, len(self.people)):
                rows.append(
                    {
                        "person_a": self.people[first],
                        "person_b": self.people[second],
                        "source_row": len(rows) + 2,
                        "label": int(self._is_positive(first, second)),
                    }
                )
        return prepare_pairs(pd.DataFrame(rows), resolver)

    def test_schema_and_feature_extraction_match_generated_exports(self) -> None:
        cdr_schema = resolve_data_schema(self.cdr_path, "cdr")
        ipdr_schema = resolve_data_schema(self.ipdr_path, "ipdr")
        resolver = IdentityResolver.from_subscribers_csv(self.subscribers_path)
        pairs = self._pairs(resolver)
        features, coverage = extract_pair_features(
            pairs, cdr_schema, ipdr_schema, resolver, chunk_size=17
        )
        self.assertEqual(list(features.columns), FEATURE_NAMES)
        self.assertEqual(len(features), 28)
        self.assertGreater(coverage["pairs_with_direct_cdr"], 0)
        positive_index = int(pairs.index[pairs["label"] == 1][0])
        negative_index = int(pairs.index[pairs["label"] == 0][0])
        self.assertGreater(features.loc[positive_index, "cdr_events_total"], 0)
        self.assertEqual(features.loc[negative_index, "cdr_events_total"], 0)
        self.assertEqual(features.loc[positive_index, "cdr_reciprocity_ratio"], 1.0)

    def test_neural_network_train_save_load_predict(self) -> None:
        cdr_schema = resolve_data_schema(self.cdr_path, "cdr")
        ipdr_schema = resolve_data_schema(self.ipdr_path, "ipdr")
        resolver = IdentityResolver.from_subscribers_csv(self.subscribers_path)
        pairs = self._pairs(resolver)
        features, _ = extract_pair_features(pairs, cdr_schema, ipdr_schema, resolver, chunk_size=19)
        artifact, metrics, probabilities, splits, importance = train_neural_network(
            features,
            pairs["label"].to_numpy(dtype=np.int64),
            FEATURE_NAMES,
            epochs=30,
            batch_size=8,
            seed=7,
            device_name="cpu",
        )
        model_path = self.root / "model.pt"
        save_artifact(artifact, model_path)
        loaded = load_artifact(model_path)
        loaded_probabilities, predictions = predict_with_artifact(loaded, features, "cpu")
        self.assertEqual(len(probabilities), len(pairs))
        self.assertEqual(len(loaded_probabilities), len(pairs))
        self.assertEqual(len(predictions), len(pairs))
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertEqual(len(importance), len(FEATURE_NAMES))
        self.assertIn("test", metrics)
        np.testing.assert_allclose(probabilities, loaded_probabilities, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

