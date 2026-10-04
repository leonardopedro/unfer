"""Sort the `all()` table in unfer_protocol/src/codes.rs by code number.

Each row is parsed into `(number, name, description)`, the rows are sorted, and
they are re-emitted in one canonical layout. Because that is a reformat as well
as a reorder, the script asserts the parsed content is unchanged -- as a
multiset of tuples -- before writing, so a regex that fails to match a row
aborts instead of silently dropping it.

Run after adding a code, or to restore the ordering invariant that
`codes::tests::the_table_is_sorted_and_free_of_duplicates` enforces.
"""

import io
import re
import sys
from collections import Counter

PATH = "unfer_protocol/src/codes.rs"

ROW = re.compile(
    r'\(\s*\n\s+(?P<num>\d{4}),\s*\n\s+"(?P<name>[^"]*)",\s*\n\s+"(?P<desc>[^"]*)",\s*\n\s*\)',
    re.S,
)

text = io.open(PATH, encoding="utf-8").read()
start = text.index("pub fn all()")
open_bracket = text.index("&[", text.index("]", start)) + len("&[")
close_ = text.index("\n    ]", open_bracket)
head, body, tail = text[:open_bracket], text[open_bracket:close_], text[close_:]

rows = [(m.group("num"), m.group("name"), m.group("desc")) for m in ROW.finditer(body)]

# Coverage guard: every code-number line in the table must belong to a matched
# row. Counting `(` would not work -- 31 of them sit inside descriptions, in
# prose like "(unparseable NDJSON)".
number_lines = len(re.findall(r"^\s+\d{4},\s*$", body, re.M))
if number_lines != len(rows):
    sys.exit(
        "table has %d code-number lines but the regex parsed %d rows -- "
        "refusing to write, because a row would be lost" % (number_lines, len(rows))
    )
if len({r[0] for r in rows}) != len(rows):
    sys.exit("duplicate code numbers in the table; resolve before sorting")

before = Counter(rows)
order = sorted(rows, key=lambda r: int(r[0]))
assert Counter(order) == before, "sorting changed the contents"

emitted = "".join(
    '        (\n            %s,\n            "%s",\n            "%s",\n        ),\n' % r
    for r in order
)
# Trailing newline already provided by the final row; drop it so `tail` (which
# begins with "\n    ]") keeps the table's original closing shape.
emitted = emitted[:-1]

moved = sum(1 for a, b in zip(rows, order) if a[0] != b[0])
io.open(PATH, "w", encoding="utf-8").write(head + emitted + tail)
print("rows: %d   positions changed: %d" % (len(rows), moved))
if moved:
    print("  first moved: %s -> position %d" % (rows[-1][0], order.index(rows[-1])))