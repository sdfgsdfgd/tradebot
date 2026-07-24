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
