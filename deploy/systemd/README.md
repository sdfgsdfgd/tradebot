# q user-service installation

These units are templates for the eventual q checkout. Validate one manual run
before enabling the timer:

```bash
mkdir -p ~/.config/systemd/user
install -m 0644 deploy/systemd/tradebot-news.service ~/.config/systemd/user/
install -m 0644 deploy/systemd/tradebot-news.timer ~/.config/systemd/user/
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

## XSP forward evidence

Install these units only from one clean, pushed revision containing the quote
recorder, shadow evaluator, and their tests. Keep the dedicated runtime
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
~/.local/share/tradebot/venv/bin/python -c 'import ib_insync, textual'
install -m 0644 deploy/systemd/tradebot-news.{service,timer} ~/.config/systemd/user/
install -m 0644 deploy/systemd/tradebot-ib-gateway-tunnel.service ~/.config/systemd/user/
install -m 0644 deploy/systemd/tradebot-xsp-{quotes,shadow}.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemd-analyze --user verify \
  ~/.config/systemd/user/tradebot-news.{service,timer} \
  ~/.config/systemd/user/tradebot-ib-gateway-tunnel.service \
  ~/.config/systemd/user/tradebot-xsp-{quotes,shadow}.{service,timer}
cmp -s deploy/systemd/tradebot-news.service ~/.config/systemd/user/tradebot-news.service
```

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
  tradebot-news.timer tradebot-xsp-quotes.timer
```

At Monday `09:37 ET`, manually run the one-shot shadow after the completed
`09:30..09:35` cash bar. Verify that it appended an `EVALUATED` checkpoint with
`order_authority=none`, then arm the remaining `09:42..16:02` schedule:

```bash
systemctl --user start tradebot-xsp-shadow.service
journalctl --user -u tradebot-xsp-shadow.service -n 100 --no-pager
systemctl --user enable --now tradebot-xsp-shadow.timer
```

The quote timer starts one bounded recorder per exchange session; the shadow
timer runs one non-submitting evaluation after each completed cash-RTH bar.
Neither timer starts a profitability clock while `NO_TRADE` remains selected.
