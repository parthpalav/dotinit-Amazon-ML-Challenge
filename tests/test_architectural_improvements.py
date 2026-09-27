"""Unit and integration tests for proposed architectural improvements."""
import numpy as np
import pandas as pd
import torch

from src.soft_macro_loss import SoftMacroF05Loss
from src.field_cross_encoder import FieldDisentangledCrossEncoder, MaxSimInteraction
from src.target_clustering import TargetClusterer
from src.dense_blocking import HybridDenseBlocker
from src.domain_adaptation import MaskedEntityDataset, PseudoLabelExtractor


def test_soft_macro_f05_loss_gradient_flow():
    """Verify differentiable Soft-Macro F0.5 loss computes valid gradients."""
    loss_fn = SoftMacroF05Loss(beta=0.5, singleton_penalty_weight=1.0)
    
    # 4 candidates across 2 anchors
    logits = torch.tensor([2.0, -1.0, 3.0, -2.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Anchor 0 has 1 positive; Anchor 1 is a singleton (0 positives)
    anchor_indices = torch.tensor([0, 0, 1, 1])

    loss = loss_fn(logits, labels, anchor_indices, is_logits=True)
    assert loss.item() > 0.0
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Gradients for Anchor 0 positive should encourage higher logits (negative loss gradient)
    assert logits.grad[0] < 0.0
    # Gradients for Anchor 1 false positive on singleton should encourage lower logits (positive loss gradient)
    assert logits.grad[2] > 0.0


def test_field_disentangled_cross_encoder():
    """Verify forward pass of field-disentangled neural cross-encoder."""
    model = FieldDisentangledCrossEncoder(vocab_size=1000, embed_dim=32, hidden_dim=64)
    model.eval()

    batch_size = 4
    a_name = torch.randint(1, 1000, (batch_size, 8))
    t_name = torch.randint(1, 1000, (batch_size, 10))
    a_addr = torch.randint(1, 1000, (batch_size, 16))
    t_addr = torch.randint(1, 1000, (batch_size, 20))
    a_legal = torch.randint(0, 10, (batch_size,))
    t_legal = torch.randint(0, 10, (batch_size,))
    country_match = torch.tensor([[1.0], [1.0], [0.0], [1.0]])

    with torch.no_grad():
        logits = model(a_name, t_name, a_addr, t_addr, a_legal, t_legal, country_match)

    assert logits.shape == (batch_size, 1)
    assert torch.isfinite(logits).all()


def test_target_clusterer_and_enrichment():
    """Verify connected-component target clustering and missing attribute enrichment."""
    data = [
        {
            'entity_id': 'S2-001',
            'business_name': 'Acme Global Logistics Pvt Ltd',
            'business_address': '123 Market St, San Francisco, CA',
            'country': 'US',
        },
        {
            'entity_id': 'S3-002',
            'business_name': 'Acme Global Logistics (Limited)',
            'business_address': '',  # Missing address
            'country': 'US',
        },
        {
            'entity_id': 'S2-003',
            'business_name': 'Beta Tech Corp',
            'business_address': '456 Tech Blvd, Austin, TX',
            'country': 'US',
        },
    ]
    df = pd.DataFrame(data)
    clusterer = TargetClusterer(name_match_threshold=0.85, addr_match_threshold=0.80)
    clusters = clusterer.cluster_targets(df)

    # S2-001 and S3-002 should cluster together
    assert len(clusters) == 2
    leader = [k for k, v in clusters.items() if 'S2-001' in v][0]
    assert 'S3-002' in clusters[leader]

    enriched = clusterer.create_enriched_profiles(df, clusters)
    assert len(enriched) == 3
    
    # Verify S3-002 received the address from S2-001
    s3_row = enriched[enriched['entity_id'] == 'S3-002'].iloc[0]
    assert s3_row['business_address'] == '123 Market St, San Francisco, CA'


def test_hybrid_dense_blocker_rrf():
    """Verify dense vector nearest neighbor search and Reciprocal Rank Fusion."""
    rng = np.random.default_rng(42)
    anchor_embeds = rng.standard_normal((3, 32)).astype(np.float32)
    target_embeds = rng.standard_normal((5, 32)).astype(np.float32)

    anchor_ids = ['S1-1', 'S1-2', 'S1-3']
    target_ids = ['S2-A', 'S2-B', 'S2-C', 'S2-D', 'S2-E']

    blocker = HybridDenseBlocker(top_k_dense=2, rrf_k=60)
    dense_results = blocker.compute_cosine_candidates(
        anchor_embeds, target_embeds, anchor_ids, target_ids, top_k=2
    )
    assert len(dense_results) == 6  # 3 anchors * 2 top-k

    sparse_df = pd.DataFrame([
        {'source1_entity_id': 'S1-1', 'candidate_entity_id': 'S2-A'},
        {'source1_entity_id': 'S1-2', 'candidate_entity_id': 'S2-C'},
    ])

    fused = blocker.fuse_sparse_and_dense(sparse_df, dense_results)
    assert len(fused) >= 6
    assert 'rrf_score' in fused.columns
    assert 'in_sparse' in fused.columns
    assert 'in_dense' in fused.columns
    # Scores must be descending per anchor
    for anc_id, group in fused.groupby('source1_entity_id'):
        scores = group['rrf_score'].to_numpy()
        assert np.all(np.diff(scores) <= 0)


def test_pseudo_label_extractor():
    """Verify extraction of ultra-high-confidence pseudo-labels with target uniqueness."""
    extractor = PseudoLabelExtractor(confidence_threshold=0.95, max_per_anchor=1)
    
    test_pairs = [
        ('S1-100', 'S2-900', 0.99, 'France'),
        ('S1-101', 'S2-900', 0.96, 'France'),  # S2-900 conflict (should be rejected)
        ('S1-102', 'S2-901', 0.98, 'France'),
        ('S1-103', 'S2-902', 0.85, 'France'),  # Below 0.95 threshold (should be rejected)
    ]

    pseudo = extractor.extract_pseudo_labels(test_pairs)
    assert len(pseudo) == 2
    assert pseudo[0] == ('S1-100', 'S2-900', 1, 'France')
    assert pseudo[1] == ('S1-102', 'S2-901', 1, 'France')
