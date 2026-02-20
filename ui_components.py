"""Reusable UI components for Streamlit HTML snippets."""

from __future__ import annotations

from html import escape
from typing import Iterable, Sequence

import streamlit as st


def card(content_html: str, class_name: str = "ui-card", style: str = "") -> str:
    style_attr = f' style="{style}"' if style else ""
    return f'<div class="{class_name}"{style_attr}>{content_html}</div>'


def metric_card(
    label: str,
    value: str,
    value_color: str = "",
    label_class: str = "summary-metric-label",
    value_class: str = "summary-metric-value",
    card_class: str = "summary-metric-card",
    subtext: str | None = None,
    sub_class: str = "summary-metric-sub",
) -> str:
    value_style = f' style="color:{value_color};"' if value_color else ""
    html = (
        f'<div class="{card_class}">'
        f'<div class="{label_class}">{escape(str(label))}</div>'
        f'<div class="{value_class}"{value_style}>{escape(str(value))}</div>'
    )
    if subtext:
        html += f'<div class="{sub_class}">{escape(str(subtext))}</div>'
    html += "</div>"
    return html


def pill(text: str, class_name: str) -> str:
    return f'<span class="{class_name}">{escape(str(text))}</span>'


def section_title(text: str, margin_top_px: int = 0) -> str:
    style = f"margin-top:{margin_top_px}px;" if margin_top_px else ""
    return f'<p class="section-title" style="{style}">{escape(str(text))}</p>'


def legend(
    items: Sequence[tuple[str, str]] | None = None,
    wrapper_style: str = "display:flex;flex-wrap:wrap;gap:8px 10px;margin:6px 0 10px 0;",
) -> None:
    pairs = items or [
        ("Green = Favorable", "#69f0ae"),
        ("Red = Adverse Risk", "#ff8a80"),
        ("Blue = Quality/Info", "#58a6ff"),
        ("Amber = Caution/Mixed", "#ffa657"),
        ("Purple = Cost", "#d2a8ff"),
    ]
    spans = []
    for label, color in pairs:
        spans.append(
            '<span style="background:var(--surface-1);border:1px solid var(--border-1);'
            'border-radius:999px;padding:4px 10px;font-size:0.78rem;'
            f'color:{color};">{escape(label)}</span>'
        )
    st.markdown(f'<div style="{wrapper_style}">{"".join(spans)}</div>', unsafe_allow_html=True)
