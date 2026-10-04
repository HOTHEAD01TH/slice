# slice

Company inbox. Eight seeded items (email, ticket, lead) stay in memory. Each one is sent to a local Laya model with four triage questions: `queue` (choice: billing, technical, account, sales), `urgency` (score), `churn` (noul), and `refund` (noul). The call sets `min_confidence` to 0.6. If any answer abstains (`abstention` of `abstained` or `unevaluated`, `low_confidence`, or no usable value), the item goes to the unsure column. That column is the only human work queue. Routed items stay visible and are not the work queue. Nothing is auto-replied. There is no mock score path, and the client does not read `action.act_probability`.

## Setup

Python 3.10 or newer.

```bash
python -m pip install laya
```

The first decision downloads weights from https://huggingface.co/convaiinnovations/laya. Boot does not download them.

## Sidecar

```bash
python3 sidecar.py
```

Listens on `127.0.0.1:8101`. It calls `laya.Router.predict` and `predict_batch` on `POST /v1/systemone` and `POST /v1/systemone/batch`. If Laya is not installed or a predict cannot reach a loaded model, the sidecar returns HTTP 503 `{"error":"Laya not running"}` and does not invent scores.

This sidecar is not the archived third-party `laya-serve` package. That package is a fake unless `LAYA_SERVE_BACKEND=laya`. Do not point slice at a fake. If you replace this sidecar with that package anyway, set `LAYA_SERVE_BACKEND=laya` or it will return scores with no model. Official `laya[serve]` (`laya-serve` from `pip install "laya[serve]"`) is a real Router and does not use that variable.

`LAYA_SIDECAR_PORT` changes the sidecar port. `LAYA_URL` (default `http://127.0.0.1:8101`) is what the Bun app calls.

## Web app

```bash
bun install && bun run dev
```

Listens on port 3001. `bun run start` runs without the file watcher. Open http://localhost:3001. If the sidecar cannot reach a loaded model, the page shows Laya not running.
