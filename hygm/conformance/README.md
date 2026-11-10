# hygm conformance fixtures

Files every reader of hygm's formats runs, in any language, so that readers
agree on what is valid and why something isn't. The rules are in
[`../docs/contract.md`](../docs/contract.md); the JSON Schemas ship in the
package under `src/hygm/schema/`.

```
model/
  valid/<name>.yaml|.json        must load; read, written and read again, it is the same model
  invalid/<code>/<name>.yaml     must be refused with error code <code> (contract, section 9)
```

- Each invalid fixture holds exactly one problem, so its directory names the
  one code a reader must report. A reader may report others beside it only
  when the same problem breaks several rules.
- A fixture whose problem is structural also fails its format version's JSON
  Schema; the others need the contract's rules a schema can't express.
- YAML fixtures avoid YAML 1.1 booleans (`on`, `yes`, `y`) as plain
  strings, which readers disagree on.

hygm runs them in `tests/test_conformance.py`.
