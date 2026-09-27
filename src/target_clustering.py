"""Target-to-Target Intra-Source Profile Fusion and Disambiguation.

Clusters mutually corroborating S2 and S3 target records to resolve missing fields,
merge alternate legal forms, and construct enriched composite target profiles.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from anyascii import anyascii
from rapidfuzz.fuzz import ratio, token_sort_ratio


class TargetClusterer:
    """Discovers and merges duplicate or fragmented target records across S2 and S3."""

    def __init__(self, name_match_threshold: float = 0.85, addr_match_threshold: float = 0.80):
        self.name_thresh = name_match_threshold
        self.addr_thresh = addr_match_threshold

    @staticmethod
    def _clean_text(s: str | None) -> str:
        if not s or pd.isna(s):
            return ""
        t = anyascii(str(s)).lower().strip()
        for k, v in [('pvt', 'private'), ('ltd', 'limited'), ('inc', 'incorporated'), ('corp', 'corporation'), ('co', 'company')]:
            t = t.replace(k, v)
        for char in '()[]{},.-_':
            t = t.replace(char, ' ')
        return ' '.join(t.split())

    def cluster_targets(self, targets_df: pd.DataFrame) -> dict[str, list[str]]:
        """Cluster target records by shared exact or near-exact business identity.

        Args:
            targets_df: DataFrame containing ['entity_id', 'business_name', 'business_address', 'country']

        Returns:
            Dictionary mapping representative target_id to list of clustered target_ids.
        """
        records = targets_df.to_dict('records')
        
        # 1. Inverted index on (country, first 2 name tokens) for candidate pairing
        buckets = defaultdict(list)
        for idx, r in enumerate(records):
            c = self._clean_text(r.get('country', ''))
            n = self._clean_text(r.get('business_name', ''))
            tokens = n.split()[:2]
            if c and tokens:
                key = (c, tuple(tokens))
                buckets[key].append(idx)

        # 2. Build adjacency list for connected components
        adj = defaultdict(set)
        for key, indices in buckets.items():
            if len(indices) > 50:  # Skip generic large buckets
                continue
            for i in range(len(indices)):
                idx1 = indices[i]
                r1 = records[idx1]
                n1 = self._clean_text(r1['business_name'])
                a1 = self._clean_text(r1['business_address'])
                
                for j in range(i + 1, len(indices)):
                    idx2 = indices[j]
                    r2 = records[idx2]
                    n2 = self._clean_text(r2['business_name'])
                    a2 = self._clean_text(r2['business_address'])

                    # Compare name and address
                    nr = max(ratio(n1, n2), token_sort_ratio(n1, n2)) / 100.0
                    if nr >= self.name_thresh:
                        if not a1 or not a2:
                            # If one address is missing, connect based on high name similarity
                            adj[idx1].add(idx2)
                            adj[idx2].add(idx1)
                        else:
                            ar = max(ratio(a1, a2), token_sort_ratio(a1, a2)) / 100.0
                            if ar >= self.addr_thresh:
                                adj[idx1].add(idx2)
                                adj[idx2].add(idx1)

        # 3. Extract connected components
        visited = set()
        clusters = {}
        for idx in range(len(records)):
            if idx in visited:
                continue
            component = []
            queue = [idx]
            visited.add(idx)
            while queue:
                curr = queue.pop(0)
                component.append(records[curr]['entity_id'])
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            leader_id = component[0]
            clusters[leader_id] = component

        return clusters

    def create_enriched_profiles(
        self, targets_df: pd.DataFrame, clusters: dict[str, list[str]]
    ) -> pd.DataFrame:
        """Construct composite records where missing fields are populated from cluster peers."""
        id_to_record = targets_df.set_index('entity_id').to_dict('index')
        enriched_rows = []

        for leader_id, member_ids in clusters.items():
            # Aggregate available fields
            best_name = ""
            best_address = ""
            best_country = ""

            for mid in member_ids:
                rec = id_to_record[mid]
                n = str(rec.get('business_name', '') or '')
                a = str(rec.get('business_address', '') or '')
                c = str(rec.get('country', '') or '')

                if len(n) > len(best_name):
                    best_name = n
                if len(a) > len(best_address):
                    best_address = a
                if not best_country and c:
                    best_country = c

            # Produce enriched entries for all cluster members
            for mid in member_ids:
                orig = id_to_record[mid]
                enriched_rows.append({
                    'entity_id': mid,
                    'business_name': orig.get('business_name') or best_name,
                    'business_address': orig.get('business_address') or best_address,
                    'country': orig.get('country') or best_country,
                    'cluster_leader': leader_id,
                    'cluster_size': len(member_ids),
                })

        return pd.DataFrame(enriched_rows)
