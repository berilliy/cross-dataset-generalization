"""Prospective external test. Only source-fitted pipelines enter score_external."""
from hashlib import sha256
import json
from zipfile import ZipFile
import numpy as np
import pandas as pd

SOURCE_ID = "https://doi.org/10.24432/C5P338"
BENIGN = {"MQTT_Publish", "Thing_Speak", "Wipro_bulb", "Amazon-Alexa"}
ATTACK = {"DOS_SYN_Hping", "ARP_poisioning", "NMAP_UDP_SCAN", "NMAP_XMAS_TREE_SCAN",
          "NMAP_OS_DETECTION", "NMAP_TCP_scan", "DDOS_Slowloris",
          "Metasploit_Brute_Force_SSH", "NMAP_FIN_SCAN"}


def read_release(root):
    archive = root / "external_data" / "rt-iot2022.zip"
    provenance = json.loads((archive.parent / "release.json").read_text(encoding="utf-8"))
    assert sha256(archive.read_bytes()).hexdigest() == provenance["sha256"]
    with ZipFile(archive) as z, z.open("RT_IOT2022") as stream:
        frame = pd.read_csv(stream, usecols=["flow_duration", "proto", "Attack_type"])
    assert len(frame) == 123117 and frame.notna().all().all()
    assert set(frame.Attack_type) <= (BENIGN | ATTACK)
    assert set(frame.proto) <= {"tcp", "udp", "icmp"}
    assert frame.flow_duration.ge(0).all()
    features = pd.DataFrame({"log_duration": np.log1p(frame.flow_duration.astype("float32"))})
    for name in ("tcp", "udp", "icmp"):
        features[name] = frame.proto.eq(name).astype("int8")
    features["other"] = features[["tcp", "udp", "icmp"]].sum(axis=1).eq(0).astype("int8")
    truth = pd.DataFrame({"row_number": np.arange(len(frame)), "attack_label": frame.Attack_type,
                          "target": frame.Attack_type.isin(ATTACK).astype("int8")})
    return features, truth


def source_overlap(train, external, features):
    # Exact within-representation equality, including identical float dtypes.
    seen = pd.util.hash_pandas_object(train[features], index=False)
    test = pd.util.hash_pandas_object(external[features], index=False)
    return float(test.isin(seen).mean())


def score_external(model, features, target_x, sources, seed, variant, name, threshold, overlap, results):
    score_dir = results / "external_scores"
    score_dir.mkdir(exist_ok=True)
    score = model.predict_proba(target_x[features])[:, 1]
    assert np.isfinite(score).all()
    key = f"{sources}__{seed}__{variant}__{name}"
    pd.DataFrame({"row_number": target_x.index, "score": score}).to_parquet(score_dir / (key+".parquet"), index=False)
    return dict(key=key, sources=sources, seed=seed, variant=variant, model=name,
                source_threshold=threshold, feature_overlap=overlap)


def evaluate_external(truth, runs, results, evaluate):
    truth.to_parquet(results / "external_target.parquet", index=False)
    truth.groupby(["attack_label", "target"]).size().rename("n").to_csv(results / "external_label_counts.csv")
    records = []
    errors = []
    assert len(runs) == len({r["key"] for r in runs}) == 90
    for run in runs:
        scores = pd.read_parquet(results / "external_scores" / (run["key"]+".parquet"))
        assert scores.row_number.equals(truth.row_number)
        for policy, threshold in [("default_0.5", .5), ("source_cal_fpr_1pct", run["source_threshold"])]:
            records.append(dict(**run, policy=policy, threshold=threshold, **evaluate(truth.target, scores.score, threshold)))
            frame = truth.assign(pred=scores.score >= threshold)
            for (label, target), group in frame.groupby(["attack_label", "target"]):
                errors.append(dict(key=run["key"], policy=policy, attack_label=label, target=target,
                                   n=len(group), errors=int(group.pred.ne(target).sum())))
    pd.DataFrame(records).to_csv(results / "external_metrics.csv", index=False)
    pd.DataFrame(errors).to_csv(results / "external_error_by_label.csv", index=False)


def calibration_analysis(dataset, model, cal, cal_score, test, test_score, results, evaluate, choose):
    frame = cal[["_row_key", "_event_time", "target"]].copy().assign(score=cal_score)
    frame.to_parquet(results / f"calibration_scores_{dataset}_{model}.parquet", index=False)
    midpoint = frame._event_time.min() + (frame._event_time.max() - frame._event_time.min()) / 2
    windows = {"full": frame, "early": frame[frame._event_time < midpoint], "late": frame[frame._event_time >= midpoint]}
    assert len(windows["early"]) + len(windows["late"]) == len(frame)
    rows = []
    for window, group in windows.items():
        benign = group.loc[group.target.eq(0)]
        for alpha in (.01, .005, .001):
            row = dict(dataset=dataset, model=model, window=window, alpha=alpha, benign_n=len(benign),
                       calibration_n=len(group), start=group._event_time.min(), end=group._event_time.max(), midpoint=midpoint)
            if len(benign) < 30:
                rows.append(dict(**row, status="insufficient benign rows"))
                continue
            threshold = choose(group.target, group.score, alpha)
            achieved = float((benign.score >= threshold).mean())
            assert achieved <= alpha + 1e-12
            rows.append(dict(**row, status="evaluated", threshold=threshold, calibration_fpr=achieved,
                             **evaluate(test.target, test_score, threshold)))
    return rows

