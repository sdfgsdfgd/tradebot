# q user-service installation

These units are templates for the eventual q checkout. Validate one manual run
before enabling the timer:

```bash
mkdir -p ~/.config/systemd/user
install -m 0644 deploy/systemd/tradebot-news.service ~/.config/systemd/user/
install -m 0644 deploy/systemd/tradebot-news.timer ~/.config/systemd/user/
install -m 0644 deploy/systemd/tradebot-mcl-narrative-prospective.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start tradebot-news.service
journalctl --user -u tradebot-news.service -n 100 --no-pager
systemctl --user enable --now tradebot-news.timer
```

Enabling the timer starts a four-hour clock; it does not immediately repeat the
manual validation run. Subsequent runs remain spaced from the previous run's
completion. To test a temporary two-hour cadence, create a timer drop-in and
change both clock values:

```ini
[Timer]
OnActiveSec=
OnActiveSec=2h
OnUnitInactiveSec=
OnUnitInactiveSec=2h
```

Remove the drop-in to restore four-hour cadence. `AccuracySec=15min` lets
systemd coalesce wakeups; it is not polling or a fifteen-minute loop.

The service pins `gpt-5.6-sol`; the application pins `max` reasoning with
strict Codex config validation. Native reasoning summaries and page-search
progress stay on stderr and therefore appear in this unit's journal, while
stdout remains the final command receipt. The service atomically curates
`~/.codex/trade-research.md` and `~/.codex/trade-events.jsonl`; do not point
multiple concurrent service instances at those files.

## Current runtime ownership

q runs one isolated systemd failure domain per cadence and authority class:

- `tradebot-xsp-shadow` owns selected P-009 source/reconciliation/execution;
- `tradebot-xsp-pressure-tape` owns read-only RTH seconds evidence;
- `tradebot-gold-live` and `tradebot-gold-onset` own Gold live and H4 source;
- `tradebot-mcl-live`, `tradebot-mcl-turn-tape`, and
  `tradebot-mcl-predictive-onset-runtime` own MCL live, seconds evidence, and
  the current generation-pointer predictive observers;
- `tradebot-news` owns four-hour publication and triggers the read-only MCL
  narrative companion;
- `tradebot-ib-gateway-tunnel` is the shared localhost broker transport.

The old `tradebot-xsp-quotes`, `tradebot-mcl-predictive-onset`, and
`tradebot-mcl-predictive-onset-stage114` pairs are retired. Do not install or
enable them as part of the current runtime. Their repository templates remain
historical/recovery evidence only.

## XSP forward evidence

Install these units only from one clean, pushed revision containing the
pressure recorder, shadow evaluator, and their tests. Keep the dedicated runtime
separate from the dashboard environment:

The current q execution checkout tracks only the isolated news branch and has
one deliberate service-file override. After the combined `main` is pushed,
converge it without a merge or stash. Stop only the timer first; never switch
source underneath an active one-shot:

```bash
systemctl --user stop tradebot-news.timer
news_state=$(systemctl --user show tradebot-news.service -p ActiveState --value)
test "$news_state" = inactive || test "$news_state" = failed
git fetch origin refs/heads/main:refs/remotes/origin/main
test "$(git status --porcelain)" = " M deploy/systemd/tradebot-news.service"
cmp -s deploy/systemd/tradebot-news.service ~/.config/systemd/user/tradebot-news.service
git restore --source=HEAD --worktree deploy/systemd/tradebot-news.service
git switch --track -c main origin/main
```

The restore affects only the redundant checkout copy after it has been proven
identical to the still-loaded user unit; it never changes that unit. The
subsequent install deliberately replaces both with the pushed combined
template. A disposable exact-lineage rehearsal proved that switching with the
dirty file fails even when its bytes match, while this sequence ends cleanly on
the intended target.

```bash
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
/usr/bin/python3 -c 'import sys; assert (3, 12) <= sys.version_info[:2] < (3, 14), sys.version'
/usr/bin/python3 -m venv ~/.local/share/tradebot/venv
~/.local/share/tradebot/venv/bin/pip install --requirement requirements.txt
~/.local/share/tradebot/venv/bin/pip check
~/.local/share/tradebot/venv/bin/python -c 'import ib_insync, textual; from zoneinfo import ZoneInfo; ZoneInfo("US/Eastern")'
install -m 0644 deploy/systemd/tradebot-news.{service,timer} ~/.config/systemd/user/
install -m 0644 deploy/systemd/tradebot-mcl-narrative-prospective.service ~/.config/systemd/user/
install -m 0644 deploy/systemd/tradebot-ib-gateway-tunnel.service ~/.config/systemd/user/
install -m 0644 deploy/systemd/tradebot-xsp-shadow.{service,timer} ~/.config/systemd/user/
install -m 0644 deploy/systemd/tradebot-xsp-pressure-tape.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemd-analyze --user verify \
  ~/.config/systemd/user/tradebot-news.{service,timer} \
  ~/.config/systemd/user/tradebot-mcl-narrative-prospective.service \
  ~/.config/systemd/user/tradebot-ib-gateway-tunnel.service \
  ~/.config/systemd/user/tradebot-xsp-shadow.{service,timer} \
  ~/.config/systemd/user/tradebot-xsp-pressure-tape.{service,timer}
cmp -s deploy/systemd/tradebot-news.service ~/.config/systemd/user/tradebot-news.service
cmp -s deploy/systemd/tradebot-mcl-narrative-prospective.service ~/.config/systemd/user/tradebot-mcl-narrative-prospective.service
```

Successful atomic news publication triggers the read-only MCL companion. It
freezes eligible TA-owned forecasts before their four-hour outcome and settles
older forecasts idempotently; companion failure never rewrites the published
news snapshot or grants instrument/order authority.

The tunnel is localhost-only, broker-enforced read-only, and started on demand.
It retries every 30 seconds without a start ceiling. The long-running producer
soft-depends on it and owns broker reconnect/backoff, so a temporarily sleeping
Mac cannot exhaust recovery before the recorder starts; the bounded shadow
one-shot retains its hard dependency. Arm the forward producer before Sunday
`20:15 ET`; waiting for Monday's first eligible RTH bar would irretrievably
lose the GTH prefix. Keep only the shadow timer disabled until its first
live-boundary proof:

```bash
systemctl --user disable --now tradebot-xsp-shadow.timer
systemctl --user enable --now \
  tradebot-news.timer tradebot-xsp-pressure-tape.timer
```

After installation and before the next `20:15 ET` GTH boundary, manually run
the one-shot observer once. It must append a `CLOSED/run_not_started`
preflight that freezes the next untouched GTH boundary without contacting
IBKR. After the first completed `20:15..20:20` SPY bar, run it again and verify
an `EVALUATED` paired XSP/SPY checkpoint with `order_authority=none`; only then
arm the remaining GTH/RTH schedule:

```bash
systemctl --user start tradebot-xsp-shadow.service
journalctl --user -u tradebot-xsp-shadow.service -n 100 --no-pager
systemctl --user enable --now tradebot-xsp-shadow.timer
```

The pressure-tape timer starts one bounded recorder per cash session; the shadow
timer runs the selected P-009 evaluation after each completed GTH/RTH bar and once at
`16:17 ET` to close-align the final RTH bar. A selected cash sleeve begins its
already-validated SELL ladder at `15:57 ET` (`12:57` on early-close days),
forbids every later BUY, and lets later RTH/Curb/GTH recurrences only reconcile
or reduce a position that did not flatten.
The pressure-tape timer is persistent so a q reboot resumes the current capture window;
the recorder's closed-window guard prevents an expired catch-up from reaching
the broker. The high-frequency shadow timer remains non-persistent.
The installed service remains broker-read-only while no validated
`xsp_selected_live_transport.json` exists. A selection transaction must
separately install `tradebot-xsp-shadow-selected-live.conf` as a systemd
drop-in; a selected file without that explicit writable boundary fails closed.
Neither timer starts a profitability clock while `NO_TRADE` remains selected.

After the v3 UPRO/SPXU transport has passed its fresh broker preview and the
operator explicitly accepts the RTH-first cash scope, activate it only through
the one-shot transaction below:

```bash
deploy/systemd/tradebot-xsp-select-live \
  --accept-rth-only-cash-scope \
  /path/to/preview.json
```

It requires clean deployed source and an active read-only cadence, pauses the
timer, writes one fresh RTH checkpoint, freezes the selection from a separate
read-only cash snapshot against the frozen v3 cash receipt, installs the
writable drop-in, and immediately runs the selected owner. The explicit scope
flag acknowledges that this first canary is RTH-only; it does not authorize
the still-unqualified GTH cash lane. Failure before selection restores
read-only cadence.
Failure after selection preserves the immutable selection, leaves cadence
disabled, and requires reconciliation; it never deletes or silently replaces
selected authority. The command is deliberately not a recurring selector.

## MCL predictive-onset accumulation

Install only the generation-pointer runtime after the crowned Stage-112 worker
and continuous turn tape are proven independently. It is a separate read-only
failure domain: empty cycles append nothing, current generation files select
the predictive and shock-wave owners, and no observer can alter direction,
timing, orders, capital, or live/recorder services.

```bash
install -m 0644 deploy/systemd/tradebot-mcl-predictive-onset-runtime.{service,timer} \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemd-analyze --user verify \
  ~/.config/systemd/user/tradebot-mcl-predictive-onset-runtime.{service,timer}
systemctl --user start tradebot-mcl-predictive-onset-runtime.service
journalctl --user -u tradebot-mcl-predictive-onset-runtime.service -n 100 --no-pager
systemctl --user enable --now tradebot-mcl-predictive-onset-runtime.timer
```

Repeating an invocation with no unseen complete treatment must leave every
ledger byte-identical. The five-minute timer runs after the live checkpoint and
recorder flush, is not persistent, and never fabricates missed treatments
across a closure or silently substitutes an older generation.
