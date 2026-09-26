"""Phase 8.3: Investigate and fix the Cap-64 ranker truncation loss (85 pairs).
1. Confirm the mechanism: raw inverted index retrieves the candidate, but coarse ranker scores it below top 64.
2. Test targeted fix: modest boost or slot reservation for phonetic/indic matches.
3. Re-measure recovery on 85 pairs and full 784 diagnostic set.
"""
import sqlite3, json, time
from pathlib import Path
from collections import Counter
import joblib, numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from src.config import Config
from src.disk_blocking import DiskBlocker
from src.disk_store import connect, fetch_records
from src.coarse_ranker import coarse_features

def main():
    print("="*80)
    print("Phase 8.3: Cap-64 Ranker Truncation Analysis & Targeted Fix")
    print("="*80)

    # 1. Load diagnostic 784 missed pairs from validation
    val_data = joblib.load('work/real_v1/features/validation.joblib')
    v1_pairs = set(zip(val_data['pairs'].source1_entity_id, val_data['pairs'].candidate_entity_id))

    con = connect('work/cap64_rebuilt/train/records.sqlite', readonly=True)
    con.row_factory = sqlite3.Row
    val_anchors = list(val_data['truth_counts'].keys())

    missed_pairs = []
    for start in range(0, len(val_anchors), 900):
        chunk = val_anchors[start:start+900]
        q = f'SELECT entity_id, matches FROM labels WHERE entity_id IN ({",".join(["?"]*len(chunk))})'
        for row in con.execute(q, chunk):
            s1 = row['entity_id']
            matches = [m.strip() for m in row['matches'].split(',') if m.strip()]
            for m in matches:
                if (s1, m) not in v1_pairs:
                    missed_pairs.append((s1, m))

    print(f"Total diagnostic missed pairs: {len(missed_pairs)}")
    missed_set = set(missed_pairs)

    missed_s1_set = {p[0] for p in missed_pairs}
    rids = [r[0] for r in con.execute(f'SELECT rid FROM anchors WHERE entity_id IN ({",".join(["?"]*len(missed_s1_set))})', list(missed_s1_set))]
    anchors = fetch_records(con, 'anchors', rids)

    cfg = Config.load('config/real.json')
    from dataclasses import replace
    cfg = replace(cfg, working_dir='work/cap64_rebuilt', retrieval_posting_limit=500, max_candidates_per_anchor=64)
    blocker = DiskBlocker(cfg, 'train')

    # Step 1: Raw retrieval
    rids_raw, masks, offsets, text, indexer, repeated = blocker.raw_retrieve(anchors)
    print(f"Raw retrieve: {len(rids_raw)} pairs for {len(anchors)} anchors")

    # Coarse ranking scores
    if len(rids_raw) and blocker.ranker is not None:
        rank = blocker.ranker.predict_proba(coarse_features(anchors, text, indexer, repeated, masks,
            similarity_threads=min(2, blocker.config.threads)))[:, 1]
    else:
        rank = np.zeros(len(rids_raw), dtype=np.float32)

    # Standard cap 64 truncation
    standard_selected = []
    for i in range(len(anchors)):
        lo, hi = int(offsets[i]), int(offsets[i+1])
        keep = np.arange(lo, hi)
        if hi - lo > blocker.config.max_candidates_per_anchor:
            keep = keep[np.argsort(-rank[lo:hi], kind='stable')[:blocker.config.max_candidates_per_anchor]]
        standard_selected.extend(keep.tolist())

    s1_arr = anchors.entity_id.to_numpy()[repeated]
    cand_arr = text['entity_id'][indexer]

    raw_pair_set = set(zip(s1_arr, cand_arr))
    standard_kept_set = set(zip(s1_arr[standard_selected], cand_arr[standard_selected]))

    recovered_standard = missed_set & standard_kept_set
    in_raw_dropped_standard = (missed_set & raw_pair_set) - standard_kept_set
    not_in_raw = missed_set - raw_pair_set

    print(f"\nStandard Cap-64 Baseline:")
    print(f"  Recovered in Cap-64:             {len(recovered_standard)} / {len(missed_pairs)} ({len(recovered_standard)/len(missed_pairs)*100:.2f}%)")
    print(f"  In Raw Index but Dropped by Cap: {len(in_raw_dropped_standard)} / {len(missed_pairs)} ({len(in_raw_dropped_standard)/len(missed_pairs)*100:.2f}%)")
    print(f"  Not in Raw Index:                {len(not_in_raw)} / {len(missed_pairs)} ({len(not_in_raw)/len(missed_pairs)*100:.2f}%)")

    # Step 2: Confirm the mechanism on the 85 dropped pairs
    print(f"\nAnalyzing the {len(in_raw_dropped_standard)} dropped pairs:")
    dropped_ranks = []
    dropped_masks = []
    dropped_pool_sizes = []
    for i in range(len(anchors)):
        lo, hi = int(offsets[i]), int(offsets[i+1])
        s1 = anchors.entity_id.iloc[i]
        for idx in range(lo, hi):
            pair = (s1, cand_arr[idx])
            if pair in in_raw_dropped_standard:
                # find rank position within anchor
                local_ranks = rank[lo:hi]
                order = np.argsort(-local_ranks, kind='stable')
                pos = int(np.where(order == (idx - lo))[0][0])
                dropped_ranks.append(pos)
                dropped_masks.append(int(masks[idx]))
                dropped_pool_sizes.append(hi - lo)

    print(f"  Average candidate pool size for dropped anchors: {np.mean(dropped_pool_sizes):.1f} (min={min(dropped_pool_sizes)}, max={max(dropped_pool_sizes)})")
    print(f"  Average rank position of true pair: {np.mean(dropped_ranks):.1f} (all > 64: min={min(dropped_ranks)}, max={max(dropped_ranks)})")
    mask_counts = Counter(dropped_masks)
    print(f"  Masks of dropped pairs: {dict(mask_counts)}")
    for bit, rname in enumerate(["country_name", "city_name", "postal", "name_ngram", "address", "strong_name", "name_number", "address_number", "name_pair"]):
        has_bit = sum(1 for m in dropped_masks if (m & (1 << bit)) != 0)
        print(f"    Bit {bit} ({rname}): {has_bit}/{len(dropped_masks)} ({has_bit/len(dropped_masks)*100:.1f}%)")

    # Step 3: Test Targeted Fixes
    # The SXN rule corresponds to bit 6 (name_number) and SXW/compound to bit 7 (address_number).
    # Phonetic matches have bit 6 or 7 set from supplement index.
    # Let's test two approaches:
    # A) Phonetic rank boost: rank_boosted = rank + alpha * is_phonetic
    # B) Reserved slots: reserve k slots for candidates with phonetic channel match (bit 6 or 7)

    print(f"\nTesting Targeted Fixes:")
    results = []

    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
        boosted_rank = rank.copy()
        # bit 6 (name_number/SXN) and bit 7 (address_number/SXW)
        phonetic_mask = ((masks & (1 << 6)) != 0) | ((masks & (1 << 7)) != 0)
        boosted_rank += alpha * phonetic_mask.astype(np.float32)

        boosted_selected = []
        for i in range(len(anchors)):
            lo, hi = int(offsets[i]), int(offsets[i+1])
            keep = np.arange(lo, hi)
            if hi - lo > blocker.config.max_candidates_per_anchor:
                keep = keep[np.argsort(-boosted_rank[lo:hi], kind='stable')[:blocker.config.max_candidates_per_anchor]]
            boosted_selected.extend(keep.tolist())

        kept_set = set(zip(s1_arr[boosted_selected], cand_arr[boosted_selected]))
        rec_85 = len(in_raw_dropped_standard & kept_set)
        rec_784 = len(missed_set & kept_set)
        regressed = len(recovered_standard - kept_set)
        results.append({
            'method': f'phonetic_boost_{alpha:.2f}',
            'rescued_from_85': rec_85,
            'total_diagnostic_recovered': rec_784,
            'regressed_prior_recovered': regressed,
            'net_gain': rec_784 - len(recovered_standard)
        })
        print(f"  Boost alpha={alpha:.2f}: Rescued {rec_85}/85, Net 784 recovery: {rec_784}/784 (+{rec_784 - len(recovered_standard)}), Regressed: {regressed}")

    # Approach B: Reserved slots (e.g. reserve up to k slots for phonetic candidates)
    for k_res in [2, 4, 8, 12, 16]:
        res_selected = []
        phonetic_mask = ((masks & (1 << 6)) != 0) | ((masks & (1 << 7)) != 0)
        for i in range(len(anchors)):
            lo, hi = int(offsets[i]), int(offsets[i+1])
            if hi - lo <= blocker.config.max_candidates_per_anchor:
                res_selected.extend(range(lo, hi))
                continue
            
            # Separate phonetic and non-phonetic
            order = np.argsort(-rank[lo:hi], kind='stable') + lo
            phon_idx = [idx for idx in order if phonetic_mask[idx]]
            norm_idx = [idx for idx in order if not phonetic_mask[idx]]

            # Take top k_res from phonetic, fill remainder from normal
            take_phon = phon_idx[:k_res]
            remainder_needed = blocker.config.max_candidates_per_anchor - len(take_phon)
            take_norm = norm_idx[:remainder_needed]

            kept = sorted(take_phon + take_norm)
            res_selected.extend(kept)

        kept_set = set(zip(s1_arr[res_selected], cand_arr[res_selected]))
        rec_85 = len(in_raw_dropped_standard & kept_set)
        rec_784 = len(missed_set & kept_set)
        regressed = len(recovered_standard - kept_set)
        results.append({
            'method': f'reserved_slots_{k_res}',
            'rescued_from_85': rec_85,
            'total_diagnostic_recovered': rec_784,
            'regressed_prior_recovered': regressed,
            'net_gain': rec_784 - len(recovered_standard)
        })
        print(f"  Reserved slots k={k_res}: Rescued {rec_85}/85, Net 784 recovery: {rec_784}/784 (+{rec_784 - len(recovered_standard)}), Regressed: {regressed}")

    # Approach C: SXN bit only boost (since SXN is pure Indic transliteration Soundex)
    for alpha in [0.10, 0.15, 0.20, 0.25]:
        sxn_boosted = rank.copy()
        sxn_mask = ((masks & (1 << 6)) != 0)
        sxn_boosted += alpha * sxn_mask.astype(np.float32)

        boosted_selected = []
        for i in range(len(anchors)):
            lo, hi = int(offsets[i]), int(offsets[i+1])
            keep = np.arange(lo, hi)
            if hi - lo > blocker.config.max_candidates_per_anchor:
                keep = keep[np.argsort(-sxn_boosted[lo:hi], kind='stable')[:blocker.config.max_candidates_per_anchor]]
            boosted_selected.extend(keep.tolist())

        kept_set = set(zip(s1_arr[boosted_selected], cand_arr[boosted_selected]))
        rec_85 = len(in_raw_dropped_standard & kept_set)
        rec_784 = len(missed_set & kept_set)
        regressed = len(recovered_standard - kept_set)
        results.append({
            'method': f'sxn_only_boost_{alpha:.2f}',
            'rescued_from_85': rec_85,
            'total_diagnostic_recovered': rec_784,
            'regressed_prior_recovered': regressed,
            'net_gain': rec_784 - len(recovered_standard)
        })
        print(f"  SXN-only Boost alpha={alpha:.2f}: Rescued {rec_85}/85, Net 784 recovery: {rec_784}/784 (+{rec_784 - len(recovered_standard)}), Regressed: {regressed}")

    # Save results
    report_file = Path('reports/improvements/phase8_ranker_truncation_fix.json')
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        'total_diagnostic_misses': len(missed_pairs),
        'standard_baseline': {
            'recovered_in_cap64': len(recovered_standard),
            'dropped_by_cap': len(in_raw_dropped_standard),
            'not_in_raw': len(not_in_raw)
        },
        'dropped_pairs_analysis': {
            'avg_pool_size': float(np.mean(dropped_pool_sizes)),
            'avg_rank_pos': float(np.mean(dropped_ranks)),
            'mask_counts': {str(k): v for k, v in mask_counts.items()}
        },
        'experiments': results
    }
    report_file.write_text(json.dumps(report_data, indent=2))
    print(f"\nReport saved to {report_file}")
    con.close()

if __name__ == '__main__':
    main()
