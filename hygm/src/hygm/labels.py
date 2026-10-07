"""The labels every model is built around (#353, #436): no imports beyond validation's, so any module can use them."""

from .validation import PERSON_LABEL, USER_LABEL

#: Value types: one node per mention, since every "3" in a session is a different fact.
VALUE_LABELS: tuple[str, ...] = ("Duration", "Quantity", "Money", "Date", "TimeWindow")

#: The fixed core (#353). Derivation never merges or retires these, and they
#: sit outside the active-type cap.
CORE_LABELS: tuple[str, ...] = (USER_LABEL, PERSON_LABEL, *VALUE_LABELS)

#: Types that collect what the model has no better type for. The adoption gate
#: measures the share of mentions landing here, so these stay catch-alls in
#: every version, whatever a learned run merges into them.
CATCH_ALL_LABELS: tuple[str, ...] = ("Topic", "Artifact")
