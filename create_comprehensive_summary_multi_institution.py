#!/usr/bin/env python3
"""
Multi-Institution Comprehensive Category Summary - Text Tables Only

Adapted from create_comprehensive_summary.py to:
- Read from rag_actions_extraction_MATCHED.csv
- Generate separate tables for each institution
- Output to single text file with all institutions
- Remove visual/PNG outputs (text tables only)

Features per institution:
1. Forward-looking transitions (where actions go)
2. Backward-looking origins (where actions came from)
3. All 7 derived metrics
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
import csv

def load_data(filepath):
    """Load and clean the processed CSV data."""
    df = pd.read_csv(filepath)
    df = df.dropna(how='all')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    print(f"✓ Loaded {len(df)} rows")
    print(f"  Institutions: {df['Institution'].unique()}")
    print(f"  Periods: {sorted(df['Period'].dropna().unique())}")

    return df

def parse_fuzzy_match(match_text):
    """
    Parse Gemini fuzzy match column to extract relationships.
    Format: "LineID (Relationship, Score); LineID (Relationship, Score); ..."
    Returns: List of tuples (lineid, relationship, score)
    """
    if pd.isna(match_text) or match_text == "":
        return []

    matches = []
    match_text = str(match_text).strip()
    parts = match_text.split(";")

    for part in parts:
        part = part.strip()
        pattern = r"(\d+)\s*\(([^,]+),\s*([\d.]+)\)"
        match = re.search(pattern, part)

        if match:
            lineid = int(match.group(1))
            relationship = match.group(2).strip()
            score = float(match.group(3))
            matches.append((lineid, relationship, score))

    return matches

def get_primary_category(category_str):
    """Get primary category from category string."""
    if pd.isna(category_str):
        return 'Uncategorised'

    category_str = str(category_str).strip()

    if ';' in category_str:
        return category_str.split(';')[0].strip()
    else:
        return category_str

def classify_trajectory(relationship_type):
    """
    Classify Gemini relationship types into C, R, S, N, U.
    """
    if relationship_type in ["Identical", "Extended", "Continued"]:
        return "C"
    elif relationship_type in ["Related", "Narrowed", "Partial", "Replaced"]:
        return "R"
    elif relationship_type in ["Stopped", "Unrelated"]:
        return "S"
    elif relationship_type == "Unknown":
        return "U"
    else:
        if relationship_type:
            return "R"
        return "U"

def get_composite_code(rag_value, relationship_type):
    """
    Create composite code from RAG status and trajectory type.

    Args:
        rag_value: 'R', 'A', 'G', or None/NaN
        relationship_type: 'Identical', 'Extended', 'Narrowed', 'Related', 'Unrelated', 'Stopped'

    Returns:
        Composite code string (e.g., 'GC', 'AR', 'RS', 'U-')
    """
    # Classify trajectory into C/R/S
    if relationship_type in ["Identical", "Extended", "Continued"]:
        traj = "C"
    elif relationship_type in ["Related", "Narrowed", "Partial", "Replaced"]:
        traj = "R"
    elif relationship_type in ["Stopped", "Unrelated"]:
        traj = "S"
    else:
        traj = "-"  # No match

    # Get RAG prefix
    if pd.isna(rag_value) or rag_value == "":
        rag = "U"  # Unknown/Unassessed
    else:
        rag = str(rag_value).strip().upper()
        if rag not in ['R', 'A', 'G']:
            rag = "U"

    return f"{rag}{traj}"

def is_new_action(action_row, df_inst, period):
    """
    Determine if an action is 'New' (no predecessors from previous period).
    """
    if period == 1.0:
        return False

    line_id = action_row['LineID']

    # Check if any RAG_AP action from previous period links to this one
    prev_period = period - 1.0
    fuzzy_col = f"Fuzzy Match RAG_AP{int(prev_period)} → AP{int(period)} (LineID, Relationship)"

    if fuzzy_col not in df_inst.columns:
        return True  # No data = assume new

    # Check all RAG_AP rows from previous period
    rag_prev = df_inst[df_inst['Document Type'] == f'RAG_AP{int(prev_period)}']

    for _, rag_row in rag_prev.iterrows():
        matches = parse_fuzzy_match(rag_row.get(fuzzy_col, ""))
        for match_lineid, _, _ in matches:
            if match_lineid == line_id:
                return False  # Has predecessor

    return True  # No predecessor found = new

def build_comprehensive_summary(df, institution):
    """
    Build comprehensive category summary for one institution.
    Adapted to work with rag_actions_extraction_MATCHED.csv format.
    """
    # Filter to single institution
    df_inst = df[df["Institution"] == institution].copy()

    summary = defaultdict(lambda: {
        'periods': defaultdict(lambda: {'T': 0, 'actions': [], 'action_ids': []}),
        'forward_transitions': {},
        'backward_origins': {},
        'new_actions': {2.0: 0, 3.0: 0},
        'action_trajectories': [],
        'composite_transitions': {}  # NEW: Track composite RAG-trajectory codes
    })

    # First pass: assign actions to primary categories (AP actions only to avoid double-counting)
    for _, row in df_inst.iterrows():
        doc_type = row['Document Type']

        # Only count AP actions
        if doc_type in ['AP1', 'AP2', 'AP3']:
            period = row['Period']
            line_id = row['LineID']
            primary_category = get_primary_category(row.get('Category', ''))

            if pd.notna(period):
                summary[primary_category]['periods'][float(period)]['T'] += 1
                summary[primary_category]['periods'][float(period)]['actions'].append(row)
                summary[primary_category]['periods'][float(period)]['action_ids'].append(line_id)

    # Second pass: analyze forward transitions from RAG_AP rows
    # Period 1->2: RAG_AP1 → AP2
    rag_ap1 = df_inst[df_inst['Document Type'] == 'RAG_AP1']
    for _, row in rag_ap1.iterrows():
        category = get_primary_category(row.get('Category', ''))
        fuzzy_col = "Fuzzy Match RAG_AP1 → AP2 (LineID, Relationship)"

        # GET RAG VALUE from this row
        rag_value = row.get('RAG value', None)

        if fuzzy_col in df_inst.columns:
            matches = parse_fuzzy_match(row.get(fuzzy_col, ""))

            if matches:
                best_match = matches[0]
                relationship = best_match[1]
                traj_code = classify_trajectory(relationship)
                # CREATE COMPOSITE CODE
                composite = get_composite_code(rag_value, relationship)
            else:
                traj_code = "S"  # No match = stopped
                composite = get_composite_code(rag_value, "Stopped")

            trans_key = "1->2"
            if trans_key not in summary[category]['forward_transitions']:
                summary[category]['forward_transitions'][trans_key] = {
                    'C': 0, 'R': 0, 'S': 0, 'U': 0
                }
            summary[category]['forward_transitions'][trans_key][traj_code] += 1
            summary[category]['action_trajectories'].append(traj_code)

            # STORE COMPOSITE CODE
            if trans_key not in summary[category]['composite_transitions']:
                summary[category]['composite_transitions'][trans_key] = defaultdict(int)
            summary[category]['composite_transitions'][trans_key][composite] += 1

    # Period 2->3: RAG_AP2 → AP3
    rag_ap2 = df_inst[df_inst['Document Type'] == 'RAG_AP2']
    for _, row in rag_ap2.iterrows():
        category = get_primary_category(row.get('Category', ''))
        fuzzy_col = "Fuzzy Match RAG_AP2 → AP3 (LineID, Relationship)"

        # GET RAG VALUE from this row
        rag_value = row.get('RAG value', None)

        if fuzzy_col in df_inst.columns:
            matches = parse_fuzzy_match(row.get(fuzzy_col, ""))

            if matches:
                best_match = matches[0]
                relationship = best_match[1]
                traj_code = classify_trajectory(relationship)
                # CREATE COMPOSITE CODE
                composite = get_composite_code(rag_value, relationship)
            else:
                traj_code = "S"
                composite = get_composite_code(rag_value, "Stopped")

            trans_key = "2->3"
            if trans_key not in summary[category]['forward_transitions']:
                summary[category]['forward_transitions'][trans_key] = {
                    'C': 0, 'R': 0, 'S': 0, 'U': 0
                }
            summary[category]['forward_transitions'][trans_key][traj_code] += 1
            summary[category]['action_trajectories'].append(traj_code)

            # STORE COMPOSITE CODE
            if trans_key not in summary[category]['composite_transitions']:
                summary[category]['composite_transitions'][trans_key] = defaultdict(int)
            summary[category]['composite_transitions'][trans_key][composite] += 1

    # Third pass: analyze backward origins (where AP actions came from)
    # Period 2 origins
    if 2.0 in [float(p) for p in df_inst['Period'].dropna().unique()]:
        ap2_actions = df_inst[df_inst['Document Type'] == 'AP2']

        for _, action in ap2_actions.iterrows():
            category = get_primary_category(action.get('Category', ''))

            if is_new_action(action, df_inst, 2.0):
                if '2_from_1' not in summary[category]['backward_origins']:
                    summary[category]['backward_origins']['2_from_1'] = {'C': 0, 'R': 0, 'N': 0}
                summary[category]['backward_origins']['2_from_1']['N'] += 1
                summary[category]['new_actions'][2.0] += 1
            else:
                # Find predecessor relationship
                line_id = action['LineID']
                rag_ap1 = df_inst[df_inst['Document Type'] == 'RAG_AP1']

                for _, rag_row in rag_ap1.iterrows():
                    fuzzy_col = "Fuzzy Match RAG_AP1 → AP2 (LineID, Relationship)"
                    if fuzzy_col in df_inst.columns:
                        matches = parse_fuzzy_match(rag_row.get(fuzzy_col, ""))

                        for match_lineid, relationship, _ in matches:
                            if match_lineid == line_id:
                                traj_code = classify_trajectory(relationship)

                                if '2_from_1' not in summary[category]['backward_origins']:
                                    summary[category]['backward_origins']['2_from_1'] = {'C': 0, 'R': 0, 'N': 0}

                                if traj_code == 'C':
                                    summary[category]['backward_origins']['2_from_1']['C'] += 1
                                elif traj_code == 'R':
                                    summary[category]['backward_origins']['2_from_1']['R'] += 1
                                break

    # Period 3 origins
    if 3.0 in [float(p) for p in df_inst['Period'].dropna().unique()]:
        ap3_actions = df_inst[df_inst['Document Type'] == 'AP3']

        for _, action in ap3_actions.iterrows():
            category = get_primary_category(action.get('Category', ''))

            if is_new_action(action, df_inst, 3.0):
                if '3_from_2' not in summary[category]['backward_origins']:
                    summary[category]['backward_origins']['3_from_2'] = {'C': 0, 'R': 0, 'N': 0}
                summary[category]['backward_origins']['3_from_2']['N'] += 1
                summary[category]['new_actions'][3.0] += 1
            else:
                line_id = action['LineID']
                rag_ap2 = df_inst[df_inst['Document Type'] == 'RAG_AP2']

                for _, rag_row in rag_ap2.iterrows():
                    fuzzy_col = "Fuzzy Match RAG_AP2 → AP3 (LineID, Relationship)"
                    if fuzzy_col in df_inst.columns:
                        matches = parse_fuzzy_match(rag_row.get(fuzzy_col, ""))

                        for match_lineid, relationship, _ in matches:
                            if match_lineid == line_id:
                                traj_code = classify_trajectory(relationship)

                                if '3_from_2' not in summary[category]['backward_origins']:
                                    summary[category]['backward_origins']['3_from_2'] = {'C': 0, 'R': 0, 'N': 0}

                                if traj_code == 'C':
                                    summary[category]['backward_origins']['3_from_2']['C'] += 1
                                elif traj_code == 'R':
                                    summary[category]['backward_origins']['3_from_2']['R'] += 1
                                break

    return summary

def calculate_all_metrics(summary):
    """Calculate all 7 comprehensive metrics for each category."""
    metrics = {}

    for category, data in summary.items():
        cat_metrics = {}

        periods = data['periods']
        forward_trans = data['forward_transitions']
        new_actions = data['new_actions']
        trajectories = data['action_trajectories']

        # 1. Growth Rate
        if 1.0 in periods and 3.0 in periods:
            t1 = periods[1.0]['T']
            t3 = periods[3.0]['T']
            cat_metrics['growth_rate'] = ((t3 - t1) / t1 * 100) if t1 > 0 else None
        else:
            cat_metrics['growth_rate'] = None

        # 2. Continuation Rate
        cont_rates = []
        for trans_key in ['1->2', '2->3']:
            if trans_key in forward_trans:
                trans = forward_trans[trans_key]
                total = trans['C'] + trans['R'] + trans['S']
                if total > 0:
                    cont_rates.append((trans['C'] + trans['R']) / total * 100)

        cat_metrics['continuation_rate'] = np.mean(cont_rates) if cont_rates else None

        # 3. Trajectory Diversity
        if trajectories:
            traj_counts = Counter(trajectories)
            total = len(trajectories)
            entropy = 0
            for count in traj_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            max_entropy = np.log2(len(traj_counts)) if len(traj_counts) > 1 else 1
            cat_metrics['trajectory_diversity'] = (entropy / max_entropy * 100) if max_entropy > 0 else 0
        else:
            cat_metrics['trajectory_diversity'] = None

        # 4-7: Simplified metrics (splitting/merging/longevity/coherence require complex tracking)
        # For now, set to None - can be enhanced if needed
        cat_metrics['splitting_ratio'] = None
        cat_metrics['merging_ratio'] = None
        cat_metrics['longevity_score'] = None

        # Category Coherence
        if trajectories:
            traj_numeric = [{'C': 1, 'R': 0.5, 'S': 0, 'U': 0.25}[t] for t in trajectories]
            if len(traj_numeric) > 1:
                coherence = 1 - np.std(traj_numeric)
                cat_metrics['category_coherence'] = max(0, coherence * 100)
            else:
                cat_metrics['category_coherence'] = 100.0
        else:
            cat_metrics['category_coherence'] = None

        # New Action Injection Rate
        if 2.0 in periods:
            t2 = periods[2.0]['T']
            n2 = new_actions.get(2.0, 0)
            cat_metrics['injection_rate_2'] = (n2 / t2 * 100) if t2 > 0 else None
        else:
            cat_metrics['injection_rate_2'] = None

        if 3.0 in periods:
            t3 = periods[3.0]['T']
            n3 = new_actions.get(3.0, 0)
            cat_metrics['injection_rate_3'] = (n3 / t3 * 100) if t3 > 0 else None
        else:
            cat_metrics['injection_rate_3'] = None

        metrics[category] = cat_metrics

    return metrics

def calculate_composite_metrics(summary):
    """
    Calculate composite RAG-trajectory metrics.

    Returns dict: {category: {metric_name: value}}
    """
    composite_metrics = {}

    for category, data in summary.items():
        cat_metrics = {}

        if 'composite_transitions' not in data:
            composite_metrics[category] = {
                'success_rate': None,
                'successful_continuity_rate': None,
                'adaptive_capacity': None,
                'resilience_score': None,
                'completion_continuation_ratio': None,
                'failure_response_profile': None,
                'strategic_selectivity': None,
                'composite_1_2': {},
                'composite_2_3': {}
            }
            continue

        comp_trans = data['composite_transitions']

        # Store raw composite counts for display
        cat_metrics['composite_1_2'] = dict(comp_trans.get('1->2', {}))
        cat_metrics['composite_2_3'] = dict(comp_trans.get('2->3', {}))

        # Combine both transitions for overall metrics
        all_composites = defaultdict(int)
        for trans_key in ['1->2', '2->3']:
            if trans_key in comp_trans:
                for code, count in comp_trans[trans_key].items():
                    all_composites[code] += count

        # Count by RAG status
        G_total = sum(count for code, count in all_composites.items() if code.startswith('G'))
        A_total = sum(count for code, count in all_composites.items() if code.startswith('A'))
        R_total = sum(count for code, count in all_composites.items() if code.startswith('R'))
        total_with_rag = G_total + A_total + R_total

        # 1. Success Rate: (GC + GR + GS) / Total with RAG × 100%
        if total_with_rag > 0:
            cat_metrics['success_rate'] = (G_total / total_with_rag) * 100
        else:
            cat_metrics['success_rate'] = None

        # 2. Successful Continuity Rate: (GC + GR) / (GC + GR + GS) × 100%
        GC = all_composites.get('GC', 0)
        GR = all_composites.get('GR', 0)
        GS = all_composites.get('GS', 0)
        if (GC + GR + GS) > 0:
            cat_metrics['successful_continuity_rate'] = ((GC + GR) / (GC + GR + GS)) * 100
        else:
            cat_metrics['successful_continuity_rate'] = None

        # 3. Adaptive Capacity: (AR + RR) / (A_total + R_total) × 100%
        AR = all_composites.get('AR', 0)
        RR = all_composites.get('RR', 0)
        challenged_total = A_total + R_total
        if challenged_total > 0:
            cat_metrics['adaptive_capacity'] = ((AR + RR) / challenged_total) * 100
        else:
            cat_metrics['adaptive_capacity'] = None

        # 4. Resilience Score: (RC + AC) / (R_total + A_total) × 100%
        RC = all_composites.get('RC', 0)
        AC = all_composites.get('AC', 0)
        if challenged_total > 0:
            cat_metrics['resilience_score'] = ((RC + AC) / challenged_total) * 100
        else:
            cat_metrics['resilience_score'] = None

        # 5. Completion-to-Continuation Ratio: (GC + GR) / G_total
        if G_total > 0:
            cat_metrics['completion_continuation_ratio'] = (GC + GR) / G_total
        else:
            cat_metrics['completion_continuation_ratio'] = None

        # 6. Failure Response Profile: RC / (RC + RR + RS + R-)
        RS = all_composites.get('RS', 0)
        R_dash = all_composites.get('R-', 0)
        red_responses = RC + RR + RS + R_dash
        if red_responses > 0:
            cat_metrics['failure_response_profile'] = (RC / red_responses) * 100
        else:
            cat_metrics['failure_response_profile'] = None

        # 7. Strategic Selectivity: (GS + RS) / Total stopped
        AS = all_composites.get('AS', 0)
        US = all_composites.get('US', 0)
        total_stopped = GS + AS + RS + US
        if total_stopped > 0:
            cat_metrics['strategic_selectivity'] = ((GS + RS) / total_stopped) * 100
        else:
            cat_metrics['strategic_selectivity'] = None

        composite_metrics[category] = cat_metrics

    return composite_metrics

def format_flow(flow_dict, show_unknown=True):
    """Format flow dictionary as compact string."""
    parts = []
    for key in ['C', 'R', 'S', 'N', 'U']:
        if key == 'U' and not show_unknown:
            continue
        if flow_dict.get(key, 0) > 0:
            parts.append(f"{key}{flow_dict[key]}")
    return ' '.join(parts) if parts else '—'

def format_composite_flow(comp_dict):
    """Format composite flow dictionary as compact string."""
    if not comp_dict:
        return "—"

    # Prioritize showing the most meaningful codes
    priority = ['GC', 'GR', 'GS', 'AC', 'AR', 'AS', 'RC', 'RR', 'RS']
    parts = []

    for code in priority:
        if comp_dict.get(code, 0) > 0:
            parts.append(f"{code}{comp_dict[code]}")

    # Add any other codes not in priority list
    for code, count in comp_dict.items():
        if code not in priority and count > 0:
            parts.append(f"{code}{count}")

    return ' '.join(parts[:6])  # Limit to 6 codes for readability

def create_institution_summary(summary, metrics, composite_metrics, institution):
    """Create text summary with composite metrics for one institution."""

    lines = []

    # Institution header
    lines.append("\n" + "=" * 160)
    lines.append(f"INSTITUTION: {institution}")
    lines.append("=" * 160)

    # Sort categories by total actions
    category_order = sorted(
        summary.keys(),
        key=lambda c: sum(summary[c]['periods'][p]['T'] for p in [1.0, 2.0, 3.0]
                         if p in summary[c]['periods']),
        reverse=True
    )

    # Main header with composite metrics
    header = (
        f"{'Category':<38s} | "
        f"{'P1':>3s} | {'P1→P2 (Composite)':>18s} | "
        f"{'P2':>3s} | {'P2→P3 (Composite)':>18s} | "
        f"{'P3':>3s} | "
        f"{'Succ%':>5s} | {'SCont%':>6s} | {'Adapt%':>6s} | {'Resil%':>6s}"
    )
    lines.append(header)
    lines.append("-" * 160)

    for category in category_order:
        data = summary[category]
        cat_metrics = metrics[category]
        cat_comp = composite_metrics[category]

        # Period totals
        t1 = data['periods'].get(1.0, {}).get('T', 0)
        t2 = data['periods'].get(2.0, {}).get('T', 0)
        t3 = data['periods'].get(3.0, {}).get('T', 0)

        # COMPOSITE FLOWS (instead of just C/R/S)
        comp_1_2 = format_composite_flow(cat_comp.get('composite_1_2', {}))
        comp_2_3 = format_composite_flow(cat_comp.get('composite_2_3', {}))

        # Composite metrics
        succ = f"{cat_comp['success_rate']:4.0f}%" if cat_comp['success_rate'] is not None else " N/A"
        scont = f"{cat_comp['successful_continuity_rate']:5.0f}%" if cat_comp['successful_continuity_rate'] is not None else "  N/A"
        adapt = f"{cat_comp['adaptive_capacity']:5.0f}%" if cat_comp['adaptive_capacity'] is not None else "  N/A"
        resil = f"{cat_comp['resilience_score']:5.0f}%" if cat_comp['resilience_score'] is not None else "  N/A"

        # Format row
        cat_display = category[:37] if len(category) <= 37 else category[:34] + "..."

        row = (
            f"{cat_display:<38s} | "
            f"{t1:>3d} | {comp_1_2:>18s} | "
            f"{t2:>3d} | {comp_2_3:>18s} | "
            f"{t3:>3d} | "
            f"{succ:>5s} | {scont:>6s} | {adapt:>6s} | {resil:>6s}"
        )
        lines.append(row)

    lines.append("=" * 160)

    return lines

def write_csv_output(all_summaries, all_metrics, all_composite_metrics, csv_file):
    """
    Write comprehensive summary data to CSV file.

    Args:
        all_summaries: Dict {institution: summary_dict}
        all_metrics: Dict {institution: metrics_dict}
        all_composite_metrics: Dict {institution: composite_metrics_dict}
        csv_file: Output file path
    """

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'Institution',
            'Category',
            'P1',
            'P1→P2 (Composite)',
            'P2',
            'P2→P3 (Composite)',
            'P3',
            'Success Rate (%)',
            'Successful Continuity Rate (%)',
            'Adaptive Capacity (%)',
            'Resilience Score (%)'
        ])

        # Write data for each institution
        for institution in sorted(all_summaries.keys()):
            summary = all_summaries[institution]
            metrics = all_metrics[institution]
            composite_metrics = all_composite_metrics[institution]

            # Sort categories by total actions (same as text output)
            category_order = sorted(
                summary.keys(),
                key=lambda c: sum(summary[c]['periods'][p]['T'] for p in [1.0, 2.0, 3.0]
                                 if p in summary[c]['periods']),
                reverse=True
            )

            for category in category_order:
                data = summary[category]
                cat_comp = composite_metrics[category]

                # Period totals
                t1 = data['periods'].get(1.0, {}).get('T', 0)
                t2 = data['periods'].get(2.0, {}).get('T', 0)
                t3 = data['periods'].get(3.0, {}).get('T', 0)

                # Composite flows
                comp_1_2 = format_composite_flow(cat_comp.get('composite_1_2', {}))
                comp_2_3 = format_composite_flow(cat_comp.get('composite_2_3', {}))

                # Composite metrics (numeric values, not formatted strings)
                succ = round(cat_comp['success_rate'], 2) if cat_comp['success_rate'] is not None else ''
                scont = round(cat_comp['successful_continuity_rate'], 2) if cat_comp['successful_continuity_rate'] is not None else ''
                adapt = round(cat_comp['adaptive_capacity'], 2) if cat_comp['adaptive_capacity'] is not None else ''
                resil = round(cat_comp['resilience_score'], 2) if cat_comp['resilience_score'] is not None else ''

                # Write row
                writer.writerow([
                    institution,
                    category,
                    t1,
                    comp_1_2,
                    t2,
                    comp_2_3,
                    t3,
                    succ,
                    scont,
                    adapt,
                    resil
                ])

def main():
    """Main execution."""
    print("=" * 80)
    print("MULTI-INSTITUTION COMPREHENSIVE SUMMARY GENERATOR")
    print("Text Tables and CSV Output - All Institutions in One File")
    print("=" * 80)

    # File paths
    input_file = "rag_actions_extraction_MATCHED.csv"
    output_file = "comprehensive_summary_all_institutions.txt"
    csv_output_file = "comprehensive_summary_all_institutions.csv"

    # Load
    df = load_data(input_file)

    # Get institutions
    institutions = df['Institution'].unique()
    print(f"\nProcessing {len(institutions)} institutions...")

    all_lines = []
    all_summaries = {}
    all_metrics = {}
    all_composite_metrics = {}

    # Header for file
    all_lines.append("=" * 160)
    all_lines.append("COMPREHENSIVE CATEGORY SUMMARY - ALL INSTITUTIONS")
    all_lines.append("Period-to-Period Flow Analysis with Composite RAG-Trajectory Metrics")
    all_lines.append("=" * 160)

    # Process each institution
    for institution in institutions:
        print(f"\n  Processing {institution}...")

        # Build summary
        summary = build_comprehensive_summary(df, institution)

        # Calculate basic metrics
        metrics = calculate_all_metrics(summary)

        # Calculate composite RAG-trajectory metrics
        composite_metrics = calculate_composite_metrics(summary)

        # Store for CSV output
        all_summaries[institution] = summary
        all_metrics[institution] = metrics
        all_composite_metrics[institution] = composite_metrics

        # Generate text summary
        inst_lines = create_institution_summary(summary, metrics, composite_metrics, institution)
        all_lines.extend(inst_lines)

    # Add legend at end
    all_lines.append("\n" + "=" * 160)
    all_lines.append("LEGEND")
    all_lines.append("=" * 160)
    all_lines.append("\nComposite Flow Codes (RAG Status + Trajectory):")
    all_lines.append("  GC = Green→Continued (successful continuation)")
    all_lines.append("  GR = Green→Related (successful refinement)")
    all_lines.append("  GS = Green→Stopped (successful completion & pivot)")
    all_lines.append("  AC = Amber→Continued (persistent challenge)")
    all_lines.append("  AR = Amber→Related (adaptive refinement)")
    all_lines.append("  AS = Amber→Stopped (stalled effort)")
    all_lines.append("  RC = Red→Continued (resilient persistence)")
    all_lines.append("  RR = Red→Related (strategic pivot from failure)")
    all_lines.append("  RS = Red→Stopped (acknowledged failure)")
    all_lines.append("  UC/UR/US = Unassessed (no RAG data)")
    all_lines.append("\nRAG Status:")
    all_lines.append("  G (Green) = Completed/Successful")
    all_lines.append("  A (Amber) = In progress/Ongoing")
    all_lines.append("  R (Red) = Delayed/Failed")
    all_lines.append("  U (Unknown) = No RAG assessment available")
    all_lines.append("\nColumns:")
    all_lines.append("  P1, P2, P3 = Total actions in each period (AP counts only)")
    all_lines.append("  P1→P2, P2→P3 = Composite forward transitions (RAG status from RAG_AP + trajectory to next AP)")
    all_lines.append("\nComposite Metrics:")
    all_lines.append("  Succ%  = Success rate: (All Green codes) / (Total with RAG) × 100%")
    all_lines.append("  SCont% = Successful continuity: (GC+GR) / (GC+GR+GS) × 100%")
    all_lines.append("  Adapt% = Adaptive capacity: (AR+RR) / (All Amber+Red) × 100%")
    all_lines.append("  Resil% = Resilience: (RC+AC) / (All Red+Amber) × 100%")
    all_lines.append("\nNotes:")
    all_lines.append("  • Actions assigned to PRIMARY category (first listed if multiple)")
    all_lines.append("  • RAG values from RAG_AP documents, trajectories from Gemini AI matching")
    all_lines.append("  • Composite codes combine RAG status with relationship types (Extended/Related/etc.)")

    # Save text output to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(all_lines))

    # Write CSV output
    print(f"\n  Writing CSV output...")
    write_csv_output(all_summaries, all_metrics, all_composite_metrics, csv_output_file)

    # Print to console
    for line in all_lines:
        print(line)

    print(f"\n✓ Saved comprehensive summary to: {output_file}")
    print(f"✓ Saved CSV output to: {csv_output_file}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
