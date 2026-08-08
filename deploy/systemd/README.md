# q live-runtime runbook

q owns one native IB Gateway and one centralized strategy control plane. The
Gateway is infrastructure, not a strategy-bundle member: stopping a strategy
must never kill it, and a failed Gateway must never be restarted by a later
strategy timer tick.

## Authority boundaries

- `tradebot-ib-gateway.service` is the sole q GUI owner. It uses
  `Restart=always` with a fifteen-second delay and a four-start/five-minute
  limit, lives outside `tradebot-live.target`, and alerts the Mac only after
  bounded restarts or semantic login fail.
- `tradebot-ib-gateway-login.service` reads credentials from q's unlocked
  desktop keyring, prefers AT-SPI labels, and falls back to freshly OCR-derived
  field boxes. It never uses fixed desktop coordinates. Human fingerprint/2FA
  remains mandatory.
- `tradebot-ib-gateway.timer` runs Sunday `17:20 ET` through a non-destructive
  ensure unit: it starts an absent Gateway and always inspects an existing Java
  login/2FA screen. Broker consumers use `After=` for ordering only; none pulls
  the owner in; none pulls it in.
- `tradebot-ib-preflight.service` performs one read-only broker reconciliation
  before futures and cash entry windows and atomically publishes
  `%t/tradebot-ib-preflight.json` with mode `0600`.
- `tradebot-ib-sentinel.timer` reads only systemd state, the private login and
  preflight receipts, and decisive recent IB errors every minute. It is silent
  for wholly unarmed bundles and sends the Mac a candidate-specific alert for
  an under-armed timer, failed owner/feed, missing next firing, stale entry
  authority in a live window, or lost q-local broker authority. Cache, quote,
  and preflight warmth receive one continuous thirty-minute grace; Gateway,
  2FA, missing schedules, and decisive in-session IB failures alert immediately.
- Writable owners may start only with a current `reduction_ready` receipt. The
  shared broker client classifies every actual submission immediately before
  `placeOrder`: a bounded close uses reduction readiness; every flat-account
  entry, same-side increase, oversize close, or reversal requires
  `entry_ready`.
- A failed/missing/stale/tampered receipt denies the relevant authority. No
  probe restarts Gateway, changes a runtime, submits an order, or cancels one.
- `tradebot-live.target` controls only timers belonging to currently armed
  strategy bundles. It contains no static strategy dependencies.
- The four scheduled one-shot recovery flows are repository-owned units, not
  anonymous `systemd-run` jobs. Their pinned `/var/tmp` scripts remain
  byte-identical; the MCL emergency path additionally takes the shared account
  lock and requires the current reduction receipt before it can submit its one
  exact, held-position reduction. Each failure is candidate-specific on the Mac.
- Core membership records a proven strategy identity only. It grants no
  capital, runtime, or order authority.

The entry probe proves the expected account, fresh positions and open orders,
every selected conId's qualification and live/snapshot price path, absence of
an active `1100` loss or three-loss flap storm, and that every selected bundle
timer is enabled and active. Reduction remains available only when account
state and quotes for every held position are current.

## Current bundles and calendars

The portfolio cockpit changes each row as one transaction and rolls back a
partial systemd change:

| Bundle | Primary owner | Support owners | Calendar |
| --- | --- | --- | --- |
| XSP P-009 | `xsp-shadow` | `xsp-pressure-tape` | Cash/GTH observations, weekdays only |
| Gold Stage 76 | `gold-live` | `gold-onset` | Sunday-Friday, excluding the daily `17:00-18:00 ET` maintenance break |
| MCL Stage 112 | `mcl-live` | `mcl-turn-tape`, `mcl-predictive-onset-runtime` | Sunday `18:00 ET` through Friday `17:00 ET`, excluding daily maintenance |

The MCL tape is a bounded session process started at `17:55 ET` Sunday-Thursday;
it no longer survives through the weekend. News gets one Sunday pre-open run
and its weekday cadence, with no Saturday loop. `START`/`STOP` in the cockpit
is the sole strategy-bundle lifecycle surface; it never selects a strategy,
graduates it, allocates capital, or touches Gateway.

## One-time installation

Install IB Gateway stable at:

```text
/home/x/.local/share/tradebot/ibgateway/ibgateway
```

The repository launcher resolves the current Mutter Xauthority file at each
start, validates single-file ownership, and fails closed if q has no
unambiguous logged-in graphical session. Never hard-code the generated
`.mutter-Xwaylandauth.*` suffix. It also refuses to launch while q port `4001`
is already owned and refuses to start without the root-owned localhost firewall.

On q, from the exact clean deployed revision:

```bash
mkdir -p ~/.config/systemd/user ~/.local/bin
install -m 0755 \
  deploy/systemd/tradebot-ib-gateway-launch \
  deploy/systemd/tradebot-ib-gateway-login \
  ~/.local/bin/
install -m 0755 deploy/systemd/tradebot-cli ~/.local/bin/tradebot
install -m 0644 \
  deploy/systemd/tradebot-ib-gateway-login.service \
  deploy/systemd/tradebot-ib-gateway-ensure.service \
  deploy/systemd/tradebot-{ib-gateway,ib-preflight,ib-sentinel}.{service,timer} \
  deploy/systemd/tradebot-operator-alert@.service \
  deploy/systemd/tradebot-live.target \
  deploy/systemd/tradebot-{gold-live,gold-onset}.{service,timer} \
  deploy/systemd/tradebot-{mcl-live,mcl-turn-tape,mcl-predictive-onset-runtime}.{service,timer} \
  deploy/systemd/tradebot-{mcl-emergency-flatten,mcl-stage131-successor,gold-fail-closed-rollover,xsp-p009-fresh-rollover}.{service,timer} \
  deploy/systemd/tradebot-mcl-narrative-prospective.service \
  deploy/systemd/tradebot-news.{service,timer} \
  deploy/systemd/tradebot-{xsp-shadow,xsp-pressure-tape}.{service,timer} \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemd-analyze --user verify ~/.config/systemd/user/tradebot-*.{service,timer,target}
```

Then validate GUI discovery without launching Gateway:

```bash
~/.local/bin/tradebot-ib-gateway-launch --check
```

On the Mac:

```bash
mkdir -p ~/.local/bin
install -m 0755 deploy/macos/tradebot-operator-alert deploy/macos/tradebot ~/.local/bin/
```

The alert accepts only the fixed conditions in the script, deduplicates
each for fifteen minutes, shows a critical modal, and plays built-in sounds at
the current macOS output volume. It never changes volume or mute state. q uses
BatchMode SSH, so alert delivery can never block on a password prompt.

Install the mandatory q loopback firewall before Gateway is permitted to run:

```bash
sudo install -d -m 0755 /etc/nftables.d
sudo install -m 0600 deploy/systemd/tradebot-ib-loopback-firewall.nft /etc/nftables.d/
sudo install -m 0644 deploy/systemd/tradebot-ib-loopback-firewall.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tradebot-ib-loopback-firewall.service
```

No IBKR username or password belongs in this repository, a unit environment,
the journal, or an automation command line. q's user manager cannot access its
TPM device, so the login worker uses the already-unlocked GNOME Secret Service:

```bash
sudo apt-get install libsecret-tools imagemagick tesseract-ocr
secret-tool store --label='Tradebot IB Gateway username' service tradebot-ib-gateway credential username
secret-tool store --label='Tradebot IB Gateway password' service tradebot-ib-gateway credential password
```

Both commands prompt without placing the value in shell history. Human
fingerprint/2FA remains mandatory.

Install the Python environment before arming anything:

```bash
/usr/bin/python3 -m venv ~/.local/share/tradebot/venv
~/.local/share/tradebot/venv/bin/pip install --requirement requirements.txt
~/.local/share/tradebot/venv/bin/pip check
```

Do not install or enable `tradebot-xsp-quotes.*`, `tradebot-mcl-predictive-onset.*`, or
`tradebot-mcl-predictive-onset-stage114.*` as current runtime units. They are
retained only as rollback and historical evidence.

## Safe commissioning and cutover

Never launch q Gateway while another host owns the live IBKR session. Before
cutover, prove from the current owner that positions and open orders are
understood, stop/disarm order-capable workers, then close that Gateway cleanly.
Do not use an automated kill or restart as a migration primitive.

On q, commission in this order:

```bash
systemctl --user enable --now tradebot-ib-gateway.service
systemctl --user status tradebot-ib-gateway.service --no-pager
```

The semantic worker fills the GUI login; complete human 2FA. Then run the
read-only proof:

```bash
systemctl --user start tradebot-ib-preflight.service
journalctl --user -u tradebot-ib-preflight.service -n 100 --no-pager
```

Do not arm entry-capable bundles until the receipt says both
`reduction_ready=true` and `entry_ready=true` for the expected account. Once
the manual proof succeeds:

```bash
systemctl --user enable tradebot-live.target
systemctl --user enable --now \
  tradebot-ib-gateway.timer \
  tradebot-ib-preflight.timer \
  tradebot-ib-sentinel.timer \
  tradebot-news.timer
```

Arm each selected strategy through the portfolio cockpit, not by enabling its
individual timers by hand. An armed timer joins `tradebot-live.target`; an
unarmed one does not. Stopping the target stops the grouped schedules and
continuous support services but does not stop Gateway.

One-shot recovery timers are enabled through `timers.target`, not
`tradebot-live.target`. They therefore survive a strategy-bundle stop or
restart while preserving their own fail-closed transaction and alert boundary.

## Sunday sequence

1. `17:20 ET`: q starts Gateway and its semantic login worker.
2. q fills credentials from the desktop keyring, raises the Mac alert only
   when IB requests fingerprint/2FA, and waits for an authenticated API port.
   No strategy is armed merely because login succeeds.
3. `17:40 ET`: preflight reconciles account, positions, open orders, selected
   contracts, connectivity, and armed bundle membership.
4. A successful entry receipt permits the already-selected strategy schedules
   to reach the shared last-mile order gate. Failure leaves entries closed and
   raises the Mac alert.
5. `17:55 ET`: the bounded MCL tape starts; futures execution begins no earlier
   than the exchange session boundary.

If 2FA is missed, attend to the GUI and rerun only preflight. Do not restart a
healthy Gateway to refresh the receipt.

## Research-owner commissioning

The control plane does not replace product evidence gates:

- XSP pressure tape must be armed before the intended cash-session boundary.
  The shadow owner remains broker-read-only without a validated selected-live
  transport and its explicit writable systemd drop-in. Commission a new cash
  transport only through `deploy/systemd/tradebot-xsp-select-live`; never make
  recurring runtime code select its own successor.
- MCL turn tape and predictive-onset runtime remain observation-only. Their
  generation pointers choose immutable owners; empty cycles append nothing and
  cannot change direction, timing, orders, capital, or the Stage-112 worker.
- Gold onset remains a non-submitting source owner. The live canary retains its
  existing selection, limit-order, risk, and profitability contracts.
- Graduation advances through `PROVEN_24H`, `PROVEN_48H`, and
  `CORE_ELIGIBLE`. A five-session `PROMOTE` can publish a content-addressed Core
  profile, but Core membership never activates a timer or allocates capital.

Changing Gateway topology therefore requires successor runtime evidence where
an artifact hashes a current unit, while historical strategy ledgers and
committed preregistrations remain byte-identical.

## Failure handling

- Gateway exits: entries and reductions lose fresh broker access; the Mac gets
  `IB GATEWAY EXITED`. No consumer restarts it. Inspect positions/open orders,
  then start it manually if safe.
- Preflight fails: the Mac gets `IBKR NOT READY`; entries remain closed. Read
  the receipt reasons and Gateway journal before intervening.
- Partial bundle: cockpit reports `BROKEN`, denies control shortcuts, and
  rolls back its own partial `START`/`STOP` transaction.
- Connectivity flap: one unpaired `1100` or three losses in ten minutes closes
  entry authority. A later successful probe is required; elapsed time alone
  does not reopen it.
- Existing position: a fresh reduction receipt permits only bounded
  position-reducing orders. It never permits a reversal or increase.

Useful inspection commands:

```bash
systemctl --user status tradebot-ib-gateway.service tradebot-ib-preflight.service --no-pager
systemctl --user list-timers 'tradebot-*' --all
journalctl --user -u tradebot-ib-gateway.service -u tradebot-ib-preflight.service -n 200 --no-pager
python3 -m tradebot.live.ib_preflight require entry --receipt "$XDG_RUNTIME_DIR/tradebot-ib-preflight.json"
python3 -m tradebot.live.ib_preflight require reduction --receipt "$XDG_RUNTIME_DIR/tradebot-ib-preflight.json"
```

There is no Mac transport or SSH tunnel fallback. q is the single Gateway and
runtime owner; a transport incident remains fail-closed until q-local broker
truth is reconciled.
