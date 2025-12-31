#!/usr/bin/env python3
"""
Multi-Institution Graphical Category Summary Visualization
Adapted from create_graphical_summary_Orig.py to work with rag_actions_extraction_MATCHED.csv

Key changes:
- Reads from rag_actions_extraction_MATCHED.csv format
- Processes multiple institutions
- Generates one plot per institution
- Extracts trajectory data from "Fuzzy Match" columns with Gemini relationship types
- Uses "Period" column instead of "AP/RAG_AP Number"

Author: Adapted for multi-institution analysis
"""

import os
import re
from collections import defaultdict

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Standardized colour scheme
COLOURS = {
    # Trajectory types
    "Continued": "#27ae60",  # Green - positive continuity
    "Extended": "#27ae60",  # Green - Extended is like Continued
    "Related": "#f39c12",  # Orange - modified continuity
    "Narrowed": "#f39c12",  # Orange - Narrowed is modification
    "Stopped": "#e74c3c",  # Red - discontinuity
    "New": "#3498db",  # Blue - innovation
    "Unknown": "#95a5a6",  # Grey - data gap
    "Identical": "#27ae60",  # Green - same as Continued
    # Metric gradients (for heatmaps)
    "high": "#27ae60",  # Green - good performance
    "medium": "#f39c12",  # Orange - moderate
    "low": "#e74c3c",  # Red - poor performance
    # Period backgrounds
    "period1": "#ecf0f1",  # Light grey
    "period2": "#d5dbdb",  # Medium grey
    "period3": "#bdc3c7",  # Darker grey
    # Category importance
    "high_priority": "#2c3e50",  # Dark blue-grey
    "medium_priority": "#7f8c8d",  # Medium grey
    "low_priority": "#bdc3c7",  # Light grey
}


def load_data(filepath):
    """Load processed CSV data."""
    df = pd.read_csv(filepath)
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def parse_fuzzy_match(match_text):
    """
    Parse Gemini fuzzy match column to extract relationships.

    Format: "LineID (Relationship, Score); LineID (Relationship, Score); ..."
    Example: "45 (Extended, 0.95); 48 (Related, 0.70)"

    Returns: List of tuples (lineid, relationship, score)
    """
    if pd.isna(match_text) or match_text == "":
        return []

    matches = []
    match_text = str(match_text).strip()

    # Split by semicolon for multiple matches
    parts = match_text.split(";")

    for part in parts:
        part = part.strip()
        # Pattern: "LineID (Relationship, Score)"
        pattern = r"(\d+)\s*\(([^,]+),\s*([\d.]+)\)"
        match = re.search(pattern, part)

        if match:
            lineid = int(match.group(1))
            relationship = match.group(2).strip()
            score = float(match.group(3))
            matches.append((lineid, relationship, score))

    return matches


def get_primary_category(category_str):
    """Get primary category from potentially multi-category string."""
    if pd.isna(category_str):
        return "Uncategorised"

    category_str = str(category_str).strip()

    if ";" in category_str:
        return category_str.split(";")[0].strip()
    else:
        return category_str


def classify_trajectory(relationship_type):
    """
    Classify Gemini relationship types into C, R, S, N, U.

    Gemini types: Identical, Extended, Narrowed, Related, Unrelated
    """
    if relationship_type in ["Identical", "Extended", "Continued"]:
        return "C"  # Continued
    elif relationship_type in ["Related", "Narrowed", "Partial", "Replaced"]:
        return "R"  # Related
    elif relationship_type in ["Stopped", "Unrelated"]:
        return "S"  # Stopped
    elif relationship_type == "Unknown":
        return "U"  # Unknown
    else:
        # Default: if we have any match, treat as Related
        if relationship_type:
            return "R"
        return "U"


def get_composite_code(rag_value, relationship_type):
    """
    Create composite code from RAG status and trajectory type.

    RAG: R (Red/delayed), A (Amber/ongoing), G (Green/complete), U (Unknown)
    Trajectory: C (Continued), R (Related), S (Stopped), - (no match)

    Returns composite codes like "GC", "AR", "RS", etc.
    """
    # Classify trajectory
    traj = classify_trajectory(relationship_type)

    # Get RAG prefix
    if pd.isna(rag_value) or rag_value == "":
        rag = "U"
    else:
        rag = str(rag_value).strip().upper()
        if rag not in ['R', 'A', 'G']:
            rag = "U"

    return f"{rag}{traj}"


def calculate_composite_metrics(composite_transitions):
    """
    Calculate composite RAG-trajectory metrics from composite codes.

    Returns dict with:
    - success_rate: % of actions that are Green
    - successful_continuity_rate: % of Green actions that continue (GC+GR) vs stop (GS)
    - adaptive_capacity: % of challenged actions (Amber/Red) that adapt (AR+RR)
    - resilience_score: % of challenged actions that persist unchanged (AC+RC)
    """
    metrics = {}

    for category, trans_data in composite_transitions.items():
        cat_metrics = {
            'success_rate': None,
            'successful_continuity_rate': None,
            'adaptive_capacity': None,
            'resilience_score': None
        }

        # Aggregate all composite codes across both transitions
        all_codes = []
        for trans_key in ["1->2", "2->3"]:
            if trans_key in trans_data:
                all_codes.extend(trans_data[trans_key])

        if not all_codes:
            metrics[category] = cat_metrics
            continue

        # Count composite code types
        code_counts = defaultdict(int)
        for code in all_codes:
            code_counts[code] += 1

        # Extract RAG status counts
        G_total = sum(count for code, count in code_counts.items() if code.startswith('G'))
        A_total = sum(count for code, count in code_counts.items() if code.startswith('A'))
        R_total = sum(count for code, count in code_counts.items() if code.startswith('R'))
        total_with_rag = G_total + A_total + R_total

        # 1. Success Rate
        if total_with_rag > 0:
            cat_metrics['success_rate'] = (G_total / total_with_rag) * 100

        # 2. Successful Continuity Rate
        GC = code_counts.get('GC', 0)
        GR = code_counts.get('GR', 0)
        GS = code_counts.get('GS', 0)
        green_sustained = GC + GR
        green_total = GC + GR + GS

        if green_total > 0:
            cat_metrics['successful_continuity_rate'] = (green_sustained / green_total) * 100

        # 3. Adaptive Capacity
        AR = code_counts.get('AR', 0)
        RR = code_counts.get('RR', 0)
        adaptive_total = AR + RR
        challenged_total = A_total + R_total

        if challenged_total > 0:
            cat_metrics['adaptive_capacity'] = (adaptive_total / challenged_total) * 100

        # 4. Resilience Score
        AC = code_counts.get('AC', 0)
        RC = code_counts.get('RC', 0)
        resilient_total = AC + RC

        if challenged_total > 0:
            cat_metrics['resilience_score'] = (resilient_total / challenged_total) * 100

        metrics[category] = cat_metrics

    return metrics


def build_summary_and_metrics(df, institution):
    """
    Build category summary with all necessary data for visualization.
    Adapted to work with rag_actions_extraction_MATCHED.csv format.
    Now extracts composite RAG-trajectory codes.
    """
    # Filter to single institution
    df_inst = df[df["Institution"] == institution].copy()

    summary = defaultdict(
        lambda: {
            "periods": defaultdict(lambda: {"T": 0}),
            "forward_transitions": {},
            "composite_transitions": {},  # NEW: store composite codes
            "metrics": {},
        }
    )

    # Count actions per category per period
    for _, row in df_inst.iterrows():
        period = row["Period"]
        primary_category = get_primary_category(row.get("Category", ""))

        if pd.notna(period):
            summary[primary_category]["periods"][float(period)]["T"] += 1

    # Calculate forward transitions from RAG_AP → AP cross-period matches
    # We need to look at RAG_AP rows and their fuzzy matches
    # NOW ALSO extract RAG values to create composite codes

    # Period 1->2: RAG_AP1 → AP2
    rag_ap1 = df_inst[df_inst["Document Type"] == "RAG_AP1"]
    for _, row in rag_ap1.iterrows():
        category = get_primary_category(row.get("Category", ""))
        fuzzy_col = "Fuzzy Match RAG_AP1 → AP2 (LineID, Relationship)"

        # Extract RAG value for this action
        rag_value = row.get("RAG value", "")

        if fuzzy_col in df_inst.columns:
            matches = parse_fuzzy_match(row.get(fuzzy_col, ""))

            if matches:
                # Take the best match (first one, highest score)
                best_match = matches[0]
                relationship = best_match[1]
                traj_code = classify_trajectory(relationship)

                # Create composite code
                composite_code = get_composite_code(rag_value, relationship)

                trans_key = "1->2"
                if trans_key not in summary[category]["forward_transitions"]:
                    summary[category]["forward_transitions"][trans_key] = {
                        "C": 0, "R": 0, "S": 0, "U": 0
                    }
                summary[category]["forward_transitions"][trans_key][traj_code] += 1

                # Store composite code
                if trans_key not in summary[category]["composite_transitions"]:
                    summary[category]["composite_transitions"][trans_key] = []
                summary[category]["composite_transitions"][trans_key].append(composite_code)
            else:
                # No match = Stopped
                composite_code = get_composite_code(rag_value, "Stopped")

                trans_key = "1->2"
                if trans_key not in summary[category]["forward_transitions"]:
                    summary[category]["forward_transitions"][trans_key] = {
                        "C": 0, "R": 0, "S": 0, "U": 0
                    }
                summary[category]["forward_transitions"][trans_key]["S"] += 1

                if trans_key not in summary[category]["composite_transitions"]:
                    summary[category]["composite_transitions"][trans_key] = []
                summary[category]["composite_transitions"][trans_key].append(composite_code)

    # Period 2->3: RAG_AP2 → AP3
    rag_ap2 = df_inst[df_inst["Document Type"] == "RAG_AP2"]
    for _, row in rag_ap2.iterrows():
        category = get_primary_category(row.get("Category", ""))
        fuzzy_col = "Fuzzy Match RAG_AP2 → AP3 (LineID, Relationship)"

        # Extract RAG value for this action
        rag_value = row.get("RAG value", "")

        if fuzzy_col in df_inst.columns:
            matches = parse_fuzzy_match(row.get(fuzzy_col, ""))

            if matches:
                best_match = matches[0]
                relationship = best_match[1]
                traj_code = classify_trajectory(relationship)

                # Create composite code
                composite_code = get_composite_code(rag_value, relationship)

                trans_key = "2->3"
                if trans_key not in summary[category]["forward_transitions"]:
                    summary[category]["forward_transitions"][trans_key] = {
                        "C": 0, "R": 0, "S": 0, "U": 0
                    }
                summary[category]["forward_transitions"][trans_key][traj_code] += 1

                # Store composite code
                if trans_key not in summary[category]["composite_transitions"]:
                    summary[category]["composite_transitions"][trans_key] = []
                summary[category]["composite_transitions"][trans_key].append(composite_code)
            else:
                composite_code = get_composite_code(rag_value, "Stopped")

                trans_key = "2->3"
                if trans_key not in summary[category]["forward_transitions"]:
                    summary[category]["forward_transitions"][trans_key] = {
                        "C": 0, "R": 0, "S": 0, "U": 0
                    }
                summary[category]["forward_transitions"][trans_key]["S"] += 1

                if trans_key not in summary[category]["composite_transitions"]:
                    summary[category]["composite_transitions"][trans_key] = []
                summary[category]["composite_transitions"][trans_key].append(composite_code)

    # Calculate composite metrics
    composite_transitions_dict = {
        cat: data["composite_transitions"]
        for cat, data in summary.items()
    }
    composite_metrics = calculate_composite_metrics(composite_transitions_dict)

    # Calculate key metrics (keeping old metrics for compatibility)
    for category, data in summary.items():
        periods = data["periods"]

        # Growth rate
        if 1.0 in periods and 3.0 in periods:
            t1 = periods[1.0]["T"]
            t3 = periods[3.0]["T"]
            growth = ((t3 - t1) / t1 * 100) if t1 > 0 else 0
        else:
            growth = None

        # Continuation rate
        cont_rates = []
        for trans_key in ["1->2", "2->3"]:
            if trans_key in data["forward_transitions"]:
                trans = data["forward_transitions"][trans_key]
                total = trans["C"] + trans["R"] + trans["S"]
                if total > 0:
                    cont_rates.append((trans["C"] + trans["R"]) / total * 100)

        cont_rate = np.mean(cont_rates) if cont_rates else None

        # Total actions (use AP counts only to avoid double-counting)
        total_actions = 0
        for period in [1.0, 2.0, 3.0]:
            # Count AP actions for this period
            period_df = df_inst[(df_inst["Period"] == period) &
                               (df_inst["Document Type"] == f"AP{int(period)}")]
            category_count = len(period_df[period_df["Category"].apply(
                lambda x: get_primary_category(x) == category)])
            total_actions += category_count

        # Merge old metrics with composite metrics
        summary[category]["metrics"] = {
            "growth_rate": growth,
            "continuation_rate": cont_rate,
            "total_actions": total_actions,
        }

        # Add composite metrics
        if category in composite_metrics:
            summary[category]["metrics"].update(composite_metrics[category])

    return summary


def get_metric_colour(value, metric_type="percentage"):
    """
    Get colour based on metric value.

    For percentages: high (>75%) = green, medium (40-75%) = orange, low (<40%) = red
    For growth: positive = green, zero = orange, negative = red
    """
    if value is None or np.isnan(value):
        return COLOURS["Unknown"]

    if metric_type == "growth":
        if value > 50:
            return COLOURS["high"]
        elif value >= -10:
            return COLOURS["medium"]
        else:
            return COLOURS["low"]
    else:  # percentage
        if value >= 75:
            return COLOURS["high"]
        elif value >= 40:
            return COLOURS["medium"]
        else:
            return COLOURS["low"]


def draw_period_column(ax, x_center, y_positions, category_data, period, width=0.12):
    """
    Draw a column for one period with count boxes and colour coding.

    Returns dict of {category: (x, y)} for arrow connections.
    """
    positions = {}

    for idx, (category, data) in enumerate(category_data):
        y = y_positions[idx]

        # Get count for this period
        count = data["periods"].get(period, {}).get("T", 0)

        if count > 0:
            # Determine box colour based on continuation rate
            cont_rate = data["metrics"]["continuation_rate"]
            box_colour = get_metric_colour(cont_rate, "percentage")

            # Draw box
            box = FancyBboxPatch(
                (x_center - width / 2, y - 0.25),
                width,
                0.5,
                boxstyle="round,pad=0.02",
                facecolor=box_colour,
                edgecolor="black",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.add_patch(box)

            # Add count text
            ax.text(
                x_center,
                y,
                str(count),
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
                color="white",
            )

            positions[category] = (x_center, y)
        else:
            # Draw empty placeholder
            ax.text(
                x_center, y, "—", fontsize=12, ha="center", va="center", color="#cccccc"
            )

    return positions


def draw_flow_arrows(ax, start_positions, end_positions, category_data, transition_key):
    """
    Draw flow arrows between periods with colour coding by trajectory type.
    """
    for category, data in category_data:
        if category in start_positions and category in end_positions:
            x_start, y_start = start_positions[category]
            x_end, y_end = end_positions[category]

            # Get transition data
            trans = data["forward_transitions"].get(transition_key, {})
            total = trans.get("C", 0) + trans.get("R", 0) + trans.get("S", 0)

            if total > 0:
                # Calculate dominant trajectory type
                if trans.get("C", 0) >= trans.get("R", 0) and trans.get("C", 0) >= trans.get("S", 0):
                    arrow_colour = COLOURS["Continued"]
                    linestyle = "-"
                elif trans.get("R", 0) > trans.get("S", 0):
                    arrow_colour = COLOURS["Related"]
                    linestyle = "--"
                else:
                    arrow_colour = COLOURS["Stopped"]
                    linestyle = ":"

                # Calculate continuation rate for arrow thickness
                cont_rate = ((trans.get("C", 0) + trans.get("R", 0)) / total) if total > 0 else 0
                linewidth = 1 + (cont_rate / 100) * 2  # 1-3 pt range

                # Draw arrow
                arrow = FancyArrowPatch(
                    (x_start + 0.06, y_start),
                    (x_end - 0.06, y_end),
                    arrowstyle="->,head_width=0.3,head_length=0.2",
                    color=arrow_colour,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=0.6,
                    zorder=2,
                )
                ax.add_patch(arrow)


def draw_composite_metric_bars(
    ax, x_position, y_positions, category_data, metric_name, width=0.08
):
    """
    Draw horizontal bars for composite RAG-trajectory metrics.

    Metrics:
    - success_rate: % actions that are Green (completed)
    - successful_continuity_rate: % Green actions that continue (GC+GR)
    - adaptive_capacity: % challenged actions that adapt (AR+RR)
    - resilience_score: % challenged actions that persist (AC+RC)
    """
    for idx, (category, data) in enumerate(category_data):
        y = y_positions[idx]

        value = data["metrics"].get(metric_name)

        if value is not None and not np.isnan(value):
            # Normalize to 0-1 range
            normalized = value / 100
            bar_width = width * normalized
            bar_colour = get_metric_colour(value, "percentage")

            # Draw bar
            rect = Rectangle(
                (x_position, y - 0.1),
                bar_width,
                0.2,
                facecolor=bar_colour,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
            )
            ax.add_patch(rect)

            # Add value text
            ax.text(
                x_position + width + 0.01,
                y,
                f"{value:.0f}%",
                fontsize=7,
                ha="left",
                va="center",
            )
        else:
            # Show N/A for missing values
            ax.text(
                x_position + width / 2,
                y,
                "N/A",
                fontsize=6,
                ha="center",
                va="center",
                color="#cccccc",
            )


def create_graphical_summary(df, institution, output_file, top_n=20):
    """
    Create main graphical summary visualization for one institution.
    """
    print("\n" + "=" * 80)
    print(f"CREATING GRAPHICAL SUMMARY FOR: {institution}")
    print("=" * 80)

    # Build data
    summary = build_summary_and_metrics(df, institution)

    # Sort categories by total actions and filter out "Uncategorised"
    category_order = sorted(
        summary.items(), key=lambda x: x[1]["metrics"]["total_actions"], reverse=True
    )

    # Remove "Uncategorised" category
    category_order = [(cat, data) for cat, data in category_order if cat != "Uncategorised"]

    # Take top N after filtering
    category_order = category_order[:top_n]

    if len(category_order) == 0:
        print(f"  Warning: No data for institution {institution}")
        return None

    print(f"\nShowing top {len(category_order)} categories by total action count (excluding Uncategorised)")

    # Create figure
    fig_height = max(12, len(category_order) * 0.6)
    fig = plt.figure(figsize=(18, fig_height))
    ax = plt.subplot(111)

    # Set up layout
    y_spacing = 1.0
    y_positions = [i * y_spacing for i in range(len(category_order))]
    y_positions.reverse()  # Top to bottom

    # Column x-positions
    x_p1 = 0.20
    x_p2 = 0.40
    x_p3 = 0.60
    # Four composite metrics
    x_metric_success = 0.75
    x_metric_scont = 0.88
    x_metric_adapt = 1.01
    x_metric_resil = 1.14

    # Draw category labels
    for idx, (category, data) in enumerate(category_order):
        y = y_positions[idx]

        # Truncate long names
        cat_display = category[:35] + "..." if len(category) > 38 else category

        # Colour based on total importance
        total = data["metrics"]["total_actions"]
        if total >= 10:
            label_colour = COLOURS["high_priority"]
            fontweight = "bold"
        elif total >= 5:
            label_colour = COLOURS["medium_priority"]
            fontweight = "normal"
        else:
            label_colour = COLOURS["low_priority"]
            fontweight = "normal"

        ax.text(
            0.02,
            y,
            cat_display,
            fontsize=9,
            ha="left",
            va="center",
            color=label_colour,
            fontweight=fontweight,
        )

    # Draw period headers
    for x, label in [(x_p1, "Period 1"), (x_p2, "Period 2"), (x_p3, "Period 3")]:
        ax.text(
            x,
            max(y_positions) + 1,
            label,
            fontsize=11,
            fontweight="bold",
            ha="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#3498db",
                edgecolor="black",
                alpha=0.8,
            ),
            color="white",
        )

    # Draw metric headers (4 composite metrics)
    ax.text(
        x_metric_success + 0.04,
        max(y_positions) + 1,
        "Success%",
        fontsize=9,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#27ae60", alpha=0.6),
        color="white",
    )

    ax.text(
        x_metric_scont + 0.04,
        max(y_positions) + 1,
        "SCont%",
        fontsize=9,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#3498db", alpha=0.6),
        color="white",
    )

    ax.text(
        x_metric_adapt + 0.04,
        max(y_positions) + 1,
        "Adapt%",
        fontsize=9,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f39c12", alpha=0.6),
        color="white",
    )

    ax.text(
        x_metric_resil + 0.04,
        max(y_positions) + 1,
        "Resil%",
        fontsize=9,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e74c3c", alpha=0.6),
        color="white",
    )

    # Draw period columns
    print("\nDrawing period columns...")
    pos_p1 = draw_period_column(ax, x_p1, y_positions, category_order, 1.0)
    pos_p2 = draw_period_column(ax, x_p2, y_positions, category_order, 2.0)
    pos_p3 = draw_period_column(ax, x_p3, y_positions, category_order, 3.0)

    # Draw flow arrows
    print("Drawing flow arrows...")
    draw_flow_arrows(ax, pos_p1, pos_p2, category_order, "1->2")
    draw_flow_arrows(ax, pos_p2, pos_p3, category_order, "2->3")

    # Draw composite metric bars
    print("Drawing composite metric visualizations...")
    draw_composite_metric_bars(
        ax, x_metric_success, y_positions, category_order, "success_rate", width=0.10
    )
    draw_composite_metric_bars(
        ax, x_metric_scont, y_positions, category_order, "successful_continuity_rate", width=0.10
    )
    draw_composite_metric_bars(
        ax, x_metric_adapt, y_positions, category_order, "adaptive_capacity", width=0.10
    )
    draw_composite_metric_bars(
        ax, x_metric_resil, y_positions, category_order, "resilience_score", width=0.10
    )

    # Create legend
    legend_elements = [
        mlines.Line2D(
            [],
            [],
            color=COLOURS["Continued"],
            linewidth=2,
            linestyle="-",
            label="Continued/Extended (C)",
        ),
        mlines.Line2D(
            [],
            [],
            color=COLOURS["Related"],
            linewidth=2,
            linestyle="--",
            label="Related/Narrowed (R)",
        ),
        mlines.Line2D(
            [],
            [],
            color=COLOURS["Stopped"],
            linewidth=2,
            linestyle=":",
            label="Stopped/Unrelated (S)",
        ),
        mpatches.Patch(facecolor=COLOURS["high"], alpha=0.7, label="High (≥75%)"),
        mpatches.Patch(facecolor=COLOURS["medium"], alpha=0.7, label="Medium (40-75%)"),
        mpatches.Patch(facecolor=COLOURS["low"], alpha=0.7, label="Low (<40%)"),
    ]

    # Position legend to avoid overlap with plot elements
    # Place it below the plot area instead of upper right
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=8,
        framealpha=0.9,
        ncol=3
    )

    # Title
    ax.set_title(
        f"Category Flow Analysis: {institution} Technician Commitment\n"
        + "Action Counts, Trajectories, and Composite RAG-Trajectory Metrics",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Formatting - expand to accommodate 4 metric columns
    ax.set_xlim(0, 1.35)
    ax.set_ylim(min(y_positions) - 0.8, max(y_positions) + 1.5)
    ax.axis("off")

    # Add subtitle/notes - moved higher to avoid legend
    notes = (
        "Numbers in boxes = action count | Arrow thickness = continuation rate | "
        "Arrow style: solid=Continued/Extended, dashed=Related/Narrowed, dotted=Stopped\n"
        "Box colour = continuation rate | Composite Metrics: Success% (Green), SCont% (Green sustained), "
        "Adapt% (Amber/Red adapted), Resil% (Amber/Red persisted)\n"
        "Metric bars: green=high (≥75%), orange=medium (40-75%), red=low (<40%)"
    )

    fig.text(
        0.5,
        0.08,  # Raised from 0.02 to 0.08 to avoid legend overlap
        notes,
        ha="center",
        fontsize=7,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Save - add extra space at bottom for legend
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave 10% space at bottom
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved graphical summary to: {output_file}")

    plt.close()

    return output_file


def main():
    """Main execution."""
    print("=" * 80)
    print("MULTI-INSTITUTION GRAPHICAL CATEGORY SUMMARY GENERATOR")
    print("=" * 80)

    input_file = "rag_actions_extraction_MATCHED.csv"
    output_dir = "visualizations"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created output directory: {output_dir}")

    # Load data
    print(f"\nLoading data from: {input_file}")
    df = load_data(input_file)

    # Get unique institutions
    institutions = df["Institution"].unique()
    print(f"\nFound {len(institutions)} institutions: {', '.join(institutions)}")

    # Create visualizations for each institution
    generated_files = []

    for institution in institutions:
        output_file = os.path.join(
            output_dir,
            f"category_summary_{institution.lower()}.png"
        )

        result = create_graphical_summary(df, institution, output_file, top_n=20)

        if result:
            generated_files.append(result)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(generated_files)} visualizations:")
    for i, filepath in enumerate(generated_files, 1):
        print(f"  {i}. {filepath}")


if __name__ == "__main__":
    main()
