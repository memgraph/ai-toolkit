# hygm conformance fixtures

Files every reader of hygm's formats runs, in any language, so that readers
agree on what is valid and why something isn't. The rules are in
[`../docs/contract.md`](../docs/contract.md); the JSON Schemas ship in the
package under `src/hygm/schema/`.

```
model/
  imports/<name>.model.yaml      the models the import fixtures import; not fixtures themselves
  valid/<name>.yaml|.json        must load; read, written and read again, it is the same model
  invalid/<code>/<name>.yaml     must be refused with error code <code> (contract, section 9)
mapping/
  models/<name>.model.yaml       the models the mapping fixtures name; not fixtures themselves
  valid/<name>.yaml|.json        must load with its model; read, written and read again, it is the same mapping
  invalid/<code>/<name>.yaml     must be refused with error code <code>
```

- Each invalid fixture holds exactly one problem, so its directory names the
  one code a reader must report. A reader may report others beside it only
  when the same problem breaks several rules.
- A fixture whose problem is structural also fails its format version's JSON
  Schema; the others need the contract's rules a schema can't express.
- YAML fixtures avoid YAML 1.1 booleans (`on`, `yes`, `y`) as plain
  strings, which readers disagree on.

- A model fixture's `imports` paths are relative to the fixture, as the
  contract says (4.12), and point into `model/imports/`, so the fixture
  directories hold only fixtures. `import_cycle/self.yaml` imports itself.
  A reader resolves them from the fixture's own location; a round trip
  writes the `imports` back and resolves them the same way. A missing
  import is `unknown_model`; an imported file that is itself invalid is
  refused with its own code.
- A mapping fixture's `model` is a path relative to the fixture, into
  `mapping/models/`, so the fixture directories hold only mappings. One
  shared model (`shop.model.yaml`) serves every fixture; `broken.model.yaml`
  exists only to be refused (`unknown_model`).
- A mapping reader reports exactly the fixture's code for every invalid
  mapping fixture, never another beside it.
- Two mapping codes are not yet in the contract's table, proposed with this
  reader: `duplicate_source` (two sources with one name) and
  `endpoint_columns` (a `table` relation's `from`/`to` column count differs
  from the `id` of the endpoint's table binding in the same source, or from
  the endpoint type's `key` when it has none there). `duplicate_property`
  also covers one property bound twice in one binding.

hygm runs them in `tests/test_conformance.py` (models) and
`tests/test_mapping_conformance.py` (mappings).
