"""LUMIA-format XML filter — compressed patient timeline format.

Transformations applied to a patient XML at a given cutoff:

- Drop entries with ``timestamp > cutoff``.
- Replace each surviving entry's ``timestamp`` attribute with a relative
  ``time_delta`` of the form ``"<y>y <d>d <h>h <m>m"`` measured backward from
  the cutoff (negative for past events).
- In ``<person>``: replace ``<birthdate>`` with ``<age_at_prediction>``
  (integer years at cutoff); remove ``<payerplan>``; remove the original
  ``<age>`` element.
- Strip identifier attributes from every ``<event>`` (``note_id``,
  ``procedure_occurrence_id``, ``image_occurrence_id``, ``image_series_uid``,
  ``image_study_uid``, ``visit_source_concept_id``).
- Remove the ``person_id`` attribute on the root.
- Hoist ``<entry>`` children out of ``<events>`` containers so they sit
  directly under ``<encounter>``; delete now-empty ``<encounter>`` blocks.

This is the canonical implementation; ``experiments/fewshot_with_labels/lumia_filter.py``
re-exports these names so existing scripts keep working.
"""

from __future__ import annotations

import contextlib
import io
import logging
from datetime import datetime
from typing import Optional
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

__all__ = ["filter_xml_by_date", "get_filtered_lumia_xml"]


def filter_xml_by_date(input_filename, cutoff_date_str, output_filename=None):
    """Apply LUMIA filtering to a patient XML file at ``cutoff_date_str``.

    Returns the modified ``ElementTree``. Prints progress to stdout (legacy
    behavior — ``get_filtered_lumia_xml`` wraps this with stdout suppression).
    """
    cutoff_date = datetime.strptime(cutoff_date_str, "%Y-%m-%d")

    print(f"Reading from {input_filename}...")
    tree = ET.parse(input_filename)
    root = tree.getroot()

    parent_map = {c: p for p in tree.iter() for c in p}

    kept_count = 0
    removed_count = 0
    removed_encounter_count = 0

    first_person_block = root.find('.//person')
    birthdate = first_person_block.find('.//birthdate')
    payerplan = first_person_block.find('.//payerplan')
    old_age_element = first_person_block.find('.//age')

    birthdate_str = birthdate.text.strip()
    birthdate_datetime = datetime.strptime(birthdate_str, "%Y-%m-%d")
    age_at_pred_time = str(int((cutoff_date - birthdate_datetime).days // 365.25))
    first_person_block.remove(birthdate)

    age_element = ET.Element("age_at_prediction")
    age_element.text = age_at_pred_time
    first_person_block.insert(0, age_element)

    if payerplan is not None:
        first_person_block.remove(payerplan)
    if old_age_element is not None:
        first_person_block.remove(old_age_element)

    root.insert(0, first_person_block)

    if 'person_id' in root.attrib:
        del root.attrib['person_id']

    for events in root.findall('.//events'):
        for entry in events.findall('entry'):
            timestamp_str = entry.get('timestamp')
            if timestamp_str:
                try:
                    entry_date = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
                    if entry_date > cutoff_date:
                        events.remove(entry)
                        removed_count += 1
                    else:
                        kept_count += 1
                        delta = entry_date - cutoff_date
                        years = delta.days // 365
                        days = delta.days % 365
                        hours = delta.seconds // 3600
                        minutes = (delta.seconds % 3600) // 60
                        time_delta_str = f"{years}y {days}d {hours}h {minutes}m"
                        entry.set('time_delta', time_delta_str)
                        del entry.attrib['timestamp']
                except ValueError:
                    print(f"Warning: Could not parse date format for '{timestamp_str}'")
            else:
                print("Warning: Found an <entry> without a timestamp.")

            attrs_to_remove = [
                "note_id", "procedure_occurrence_id",
                "image_occurrence_id", "image_series_uid",
                "image_study_uid", "visit_source_concept_id",
            ]
            for event in entry.findall('event'):
                for attr in attrs_to_remove:
                    event.attrib.pop(attr, None)

    for encounter in root.findall('.//encounter'):
        if len(encounter.findall('.//entry')) == 0:
            encounter_parent = parent_map.get(encounter)
            if encounter_parent is not None and encounter in encounter_parent:
                encounter_parent.remove(encounter)
                removed_encounter_count += 1

        for person in encounter.findall('.//person'):
            encounter.remove(person)

        for events in encounter.findall('.//events'):
            encounter.extend(events)
            encounter.remove(events)

    print(f"Done! Kept {kept_count} entries. Removed {removed_count} entries.")
    print(f"Cleaned up {removed_encounter_count} empty <encounter> blocks.")

    ET.indent(tree)

    if output_filename:
        tree.write(output_filename, encoding='utf-8', xml_declaration=True)
        print(f"Saved filtered data to {output_filename}")

    return tree


def get_filtered_lumia_xml(
    xml_path: str,
    cutoff_date_str: str,
    max_chars: Optional[int] = None,
    quiet: bool = True,
) -> str:
    """LUMIA-filter ``xml_path`` at ``cutoff_date_str`` and return the
    serialized XML as a unicode string.

    When ``max_chars`` is positive, drop oldest encounters until the serialized
    XML fits the cap (matches the eviction policy in the Vertex batch script).

    ``quiet=True`` (default) suppresses the per-call ``print`` statements in
    ``filter_xml_by_date`` so callers that loop over thousands of patients
    don't pollute stdout.
    """
    if quiet:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            tree = filter_xml_by_date(xml_path, cutoff_date_str)
    else:
        tree = filter_xml_by_date(xml_path, cutoff_date_str)

    root = tree.getroot()
    if max_chars is not None and max_chars > 0:
        encounters = root.findall("encounter")
        while encounters:
            buf = io.BytesIO()
            tree.write(buf, encoding="utf-8", xml_declaration=True)
            if len(buf.getvalue()) <= max_chars:
                break
            root.remove(encounters.pop(0))

    ET.indent(tree)
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")
