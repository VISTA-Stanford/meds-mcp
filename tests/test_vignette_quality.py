"""
Heuristic tests for vignette generation quality.

The primary failure mode we guard against: vignettes that enumerate procedures
without their findings ("CT head performed", "CBC obtained") instead of
leading with clinically meaningful results.

Tests are deterministic (no LLM calls) and use fixed fixture vignettes drawn
directly from the documented quality guidelines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Quality checker
# ---------------------------------------------------------------------------

# Words that signal a sentence is describing a test/action/procedure.
_PROCEDURE_WORDS = re.compile(
    r"\b(performed|obtained|underwent|received|completed|done|collected|ordered|"
    r"assessed|evaluated|imaged|scanned|tested|measured|recorded|documented)\b",
    re.IGNORECASE,
)

# Concrete clinical result language: specific enough that it unambiguously
# represents a finding/impression, not merely part of a procedure name.
# Deliberately excludes broad words like "without" that appear in procedure
# names ("MRI with and without contrast") rather than findings.
_CONCRETE_RESULT = re.compile(
    # Specific negative findings: "no acute...", "no interval change", etc.
    r"no\s+(acute|enhancing|focal|interval|hypermetabolic|identified|"
    r"lesion|mass|metasta|consolidation|fracture|effusion|lymphadenopathy|"
    r"malignancy|mets|disease|abnormality|process|biopsy|treatment|"
    r"confirmed|evidence\s+of)|"
    # Status / impression words tightly coupled to a result
    r"\b(stable|unchanged|within\s+normal\s+(limits|range)|normal\s+(testes|exam|examination)|"
    r"unremarkable|patent|intact|negative|positive|elevated|reduced|decreased|increased|"
    r"bilateral|unilateral)\b|"
    # Verbs that introduce findings
    r"\b(show(s|ed)|reveal(s|ed)|demonstrat(es|ed))\b|"
    r"\bconsistent\s+with\b|\bevidence\s+of\b|"
    # Numeric clinical values (lab results, measurements, scores)
    r"\d+\.?\d*\s*(ng/mL|g/dL|mg/dL|mmol|mEq|mm|cm|%|IU|U/L|K/uL)|"
    # Specific lab/score keywords
    r"\b(ejection\s+fraction|WBC|Hgb|Plt|PSA|Ki67|ECOG|TPS)\b",
    re.IGNORECASE,
)

# Numeric clinical values standalone (used for the has_quantitative_values flag).
_QUANTITATIVE_VALUE = re.compile(
    r"\d+\.?\d*\s*(ng/mL|g/dL|mg/dL|mmol|mEq|mm|cm|%|IU|U/L|K/uL)|"
    r"(<\s*\d|\d+\s*-\s*\d+\s*mm)",
    re.IGNORECASE,
)

# Trajectory / compression language: indicates the author summarised serial findings.
_TRAJECTORY_WORDS = re.compile(
    r"\b(stable|no interval change|no interval growth|serial|surveillance|"
    r"unchanged|over\s+(several|multiple|the\s+past)|"
    r"remain(ed|s)|follow-?up)\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@dataclass
class VignetteQualityReport:
    n_sentences: int
    n_procedure_sentences: int
    n_concrete_result_sentences: int   # sentences with specific clinical findings
    n_procedure_only_sentences: int    # procedure words present, no concrete result
    has_quantitative_values: bool
    has_trajectory_language: bool
    procedure_only_fraction: float     # n_procedure_only / n_sentences
    concrete_result_fraction: float    # n_concrete_result / n_sentences

    def passes(self) -> bool:
        """True when the vignette meets the minimum quality bar."""
        return (
            self.n_concrete_result_sentences >= 2
            and self.procedure_only_fraction <= 0.5
            and (self.has_quantitative_values or self.has_trajectory_language)
        )

    def failure_reasons(self) -> list[str]:
        reasons = []
        if self.n_concrete_result_sentences < 2:
            reasons.append(
                f"too few concrete result sentences: {self.n_concrete_result_sentences} < 2"
            )
        if self.procedure_only_fraction > 0.5:
            reasons.append(
                f"procedure-only fraction too high: {self.procedure_only_fraction:.2f} > 0.50"
            )
        if not self.has_quantitative_values and not self.has_trajectory_language:
            reasons.append(
                "no quantitative values and no trajectory language found"
            )
        return reasons


def check_vignette_quality(vignette: str) -> VignetteQualityReport:
    sentences = _sentences(vignette)
    n = len(sentences)
    proc_count = result_count = proc_only_count = 0
    for s in sentences:
        has_proc = bool(_PROCEDURE_WORDS.search(s))
        has_result = bool(_CONCRETE_RESULT.search(s))
        if has_proc:
            proc_count += 1
        if has_result:
            result_count += 1
        if has_proc and not has_result:
            proc_only_count += 1

    has_quant = bool(_QUANTITATIVE_VALUE.search(vignette))
    has_traj = bool(_TRAJECTORY_WORDS.search(vignette))

    return VignetteQualityReport(
        n_sentences=n,
        n_procedure_sentences=proc_count,
        n_concrete_result_sentences=result_count,
        n_procedure_only_sentences=proc_only_count,
        has_quantitative_values=has_quant,
        has_trajectory_language=has_traj,
        procedure_only_fraction=round(proc_only_count / n, 3) if n else 1.0,
        concrete_result_fraction=round(result_count / n, 3) if n else 0.0,
    )


# ---------------------------------------------------------------------------
# Fixtures: procedure-only (BAD) vs findings-first (GOOD) vignettes.
# These are the canonical examples from the vignette quality guidelines.
# ---------------------------------------------------------------------------

BAD_VIGNETTE = """\
A 55-year-old white man underwent multiple imaging studies including CT scans of \
the head, abdomen, pelvis, chest, and whole body, as well as ultrasound of the \
scrotum and MRI of the brain with and without contrast over several years. \
Laboratory evaluations included a complete blood count with differential and \
prostate-specific antigen testing. He was also tested twice for SARS-CoV-2 RNA \
by nucleic acid amplification. Radiologic assessments encompassed chest CTs and \
whole-body PET/CT scans, with additional imaging of the orbits and brain. \
The clinical timeline suggests ongoing surveillance and evaluation, though no \
specific diagnosis, histology, or treatment details were documented in the \
available records.\
"""

GOOD_VIGNETTE = """\
A 55-year-old White man is undergoing surveillance for several indeterminate \
pulmonary nodules. Over several years, he has had serial CT scans of the chest, \
abdomen, pelvis, and head, as well as whole-body CT and PET/CT imaging. \
The pulmonary nodules remain subcentimeter and stable, with no interval growth. \
PET/CT shows no hypermetabolic lesions concerning for malignancy. \
MRI of the brain with and without contrast shows no enhancing lesions and only \
mild chronic microvascular ischemic changes. Orbital imaging shows no mass or \
inflammatory process. Laboratory studies show a normal complete blood count with \
differential. Prostate-specific antigen is mildly elevated at 5.8 ng/mL but \
stable compared with prior testing. Scrotal ultrasound shows normal testes \
bilaterally with a small left hydrocele and no focal mass. Two SARS-CoV-2 RNA \
nucleic acid amplification tests are negative. There is no documented biopsy, \
histologic diagnosis, molecular testing, or oncologic treatment initiation. \
No confirmed malignancy is identified in the available records.\
"""

MINIMAL_FINDINGS_VIGNETTE = """\
A 72-year-old woman with hypertension presented with dyspnea. Chest X-ray \
showed bilateral pleural effusions. Echocardiogram revealed reduced ejection \
fraction of 35%. She was started on diuretics with clinical improvement.\
"""

PROCEDURE_HEAVY_VIGNETTE = """\
A 60-year-old man underwent chest CT. He also had an echocardiogram performed. \
A complete blood count was obtained. Liver function tests were collected. \
Urine culture was obtained. A bone scan was performed. \
Pulmonary function tests were completed.\
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBadVignetteDetection:
    """The procedure-only vignette should be flagged as failing quality."""

    def test_bad_vignette_fails_quality_check(self):
        report = check_vignette_quality(BAD_VIGNETTE)
        assert not report.passes(), (
            f"Expected bad vignette to fail quality check, but it passed. "
            f"Report: {report}"
        )

    def test_bad_vignette_has_high_procedure_only_fraction(self):
        report = check_vignette_quality(BAD_VIGNETTE)
        assert report.procedure_only_fraction > 0.4, (
            f"Expected procedure-only fraction > 0.4, got {report.procedure_only_fraction}"
        )

    def test_bad_vignette_has_few_concrete_result_sentences(self):
        report = check_vignette_quality(BAD_VIGNETTE)
        assert report.n_concrete_result_sentences < 2, (
            f"Expected < 2 concrete result sentences in bad vignette, "
            f"got {report.n_concrete_result_sentences}"
        )

    def test_bad_vignette_has_no_quantitative_values(self):
        report = check_vignette_quality(BAD_VIGNETTE)
        assert not report.has_quantitative_values, (
            "Expected bad vignette to lack quantitative clinical values"
        )

    def test_pure_procedure_list_fails(self):
        report = check_vignette_quality(PROCEDURE_HEAVY_VIGNETTE)
        assert not report.passes(), (
            "A vignette that only lists procedures should fail quality check"
        )
        assert report.procedure_only_fraction >= 0.5


class TestGoodVignetteAccepted:
    """The findings-first vignette should pass all quality checks."""

    def test_good_vignette_passes_quality_check(self):
        report = check_vignette_quality(GOOD_VIGNETTE)
        assert report.passes(), (
            f"Good vignette should pass quality check. "
            f"Failures: {report.failure_reasons()}"
        )

    def test_good_vignette_has_multiple_concrete_result_sentences(self):
        report = check_vignette_quality(GOOD_VIGNETTE)
        assert report.n_concrete_result_sentences >= 2, (
            f"Expected >= 2 concrete result sentences, got {report.n_concrete_result_sentences}"
        )

    def test_good_vignette_has_quantitative_values(self):
        report = check_vignette_quality(GOOD_VIGNETTE)
        assert report.has_quantitative_values, (
            "Good vignette should contain specific quantitative values (e.g. 5.8 ng/mL)"
        )

    def test_good_vignette_has_trajectory_language(self):
        report = check_vignette_quality(GOOD_VIGNETTE)
        assert report.has_trajectory_language, (
            "Good vignette should use trajectory/compression language (e.g. 'stable', 'serial')"
        )

    def test_good_vignette_low_procedure_only_fraction(self):
        report = check_vignette_quality(GOOD_VIGNETTE)
        assert report.procedure_only_fraction <= 0.25, (
            f"Good vignette should have <= 25% procedure-only sentences, "
            f"got {report.procedure_only_fraction:.2f}"
        )

    def test_minimal_findings_vignette_passes(self):
        report = check_vignette_quality(MINIMAL_FINDINGS_VIGNETTE)
        assert report.passes(), (
            f"A short vignette with clear findings should pass. "
            f"Failures: {report.failure_reasons()}"
        )


class TestQualityReportFields:
    """Unit tests for the quality report structure and field values."""

    def test_report_has_all_fields(self):
        report = check_vignette_quality(GOOD_VIGNETTE)
        assert report.n_sentences > 0
        assert isinstance(report.has_quantitative_values, bool)
        assert isinstance(report.has_trajectory_language, bool)
        assert 0.0 <= report.procedure_only_fraction <= 1.0
        assert 0.0 <= report.concrete_result_fraction <= 1.0

    def test_empty_vignette_fails(self):
        report = check_vignette_quality("")
        assert not report.passes()
        assert report.n_sentences == 0

    def test_good_has_higher_concrete_result_fraction_than_bad(self):
        good = check_vignette_quality(GOOD_VIGNETTE)
        bad = check_vignette_quality(BAD_VIGNETTE)
        assert good.concrete_result_fraction > bad.concrete_result_fraction, (
            "Good vignette should have a higher concrete result fraction than bad vignette"
        )

    def test_failure_reasons_non_empty_for_bad(self):
        report = check_vignette_quality(BAD_VIGNETTE)
        reasons = report.failure_reasons()
        assert len(reasons) > 0, "Bad vignette should produce at least one failure reason"

    def test_failure_reasons_empty_for_good(self):
        report = check_vignette_quality(GOOD_VIGNETTE)
        reasons = report.failure_reasons()
        assert len(reasons) == 0, (
            f"Good vignette should have no failure reasons, got: {reasons}"
        )
