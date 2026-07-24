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

The default is one run per hour. For a two- or four-hour cadence, create a
timer drop-in and change only `OnUnitInactiveSec`:

```ini
[Timer]
OnUnitInactiveSec=
OnUnitInactiveSec=2h
```

Use `4h` instead for four-hour cadence. `AccuracySec=5min` lets systemd coalesce
wakeups; it is not polling or a five-minute loop.

The service pins `gpt-5.6-sol`; the application pins `max` reasoning with
strict Codex config validation. Native reasoning summaries and page-search
progress stay on stderr and therefore appear in this unit's journal, while
stdout remains the final command receipt.
