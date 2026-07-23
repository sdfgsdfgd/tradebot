"""Portfolio search state, paging, and quote-presentation helpers."""

from __future__ import annotations

import asyncio
import time

from ib_insync import Contract, Ticker
from rich.text import Text

from ...live.options import LiveOptionPackageDraft, LiveOptionPackageQuote, quote_live_option_package
from ..common import _safe_num


class PortfolioSearchState:
    def _open_search(self) -> None:
        self._search_active = True
        self._search_query = ""
        self._search_mode_index = 0
        self._search_results = []
        self._search_selected = 0
        self._search_scroll = 0
        self._search_side = 0
        self._search_opt_expiry_index = 0
        self._search_opt_underlyers = []
        self._search_opt_underlyer_descriptions = {}
        self._search_symbol_labels = {}
        self._search_opt_underlyer_index = 0
        self._search_opt_chain_cache = {}
        self._search_opt_chain_page_cache = {}
        self._search_timing = {}
        self._clear_search_option_draft()
        self._reset_search_expiry_paging()
        self._search_loading = False
        self._search_error = None
        self._search_generation += 1
        self._cancel_search_task()
        self._search.display = True
        self._search.focus()
        self._render_search()

    def _close_search(self) -> None:
        self._search_active = False
        self._search_query = ""
        self._search_results = []
        self._search_selected = 0
        self._search_scroll = 0
        self._search_side = 0
        self._search_opt_expiry_index = 0
        self._search_opt_underlyers = []
        self._search_opt_underlyer_descriptions = {}
        self._search_symbol_labels = {}
        self._search_opt_underlyer_index = 0
        self._search_opt_chain_cache = {}
        self._search_opt_chain_page_cache = {}
        self._search_timing = {}
        self._clear_search_option_draft()
        self._reset_search_expiry_paging()
        self._search_loading = False
        self._search_error = None
        self._search_generation += 1
        self._cancel_search_task()
        self._clear_search_tickers()
        self._search.display = False
        self._search.update("")
        self._table.focus()

    def _cancel_search_task(self) -> None:
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()
        self._search_task = None
        self._cancel_search_expiry_prefetch()

    def _cancel_search_expiry_prefetch(self) -> None:
        task = getattr(self, "_search_expiry_prefetch_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._search_expiry_prefetch_task = None
        self._search_expiry_prefetch_generation = -1

    def _reset_search_expiry_paging(self) -> None:
        self._search_expiry_has_more = False
        self._search_expiry_next_offset = 0
        self._search_expiry_total = 0
        self._search_expiry_loading_more = False
        self._search_expiry_auto_advance_from = None

    def _cycle_search_mode(self, step: int) -> None:
        count = len(self._SEARCH_MODES)
        if count <= 0:
            return
        self._search_mode_index = (self._search_mode_index + int(step)) % count
        self._search_selected = 0
        self._search_scroll = 0
        self._search_side = 0
        self._search_opt_expiry_index = 0
        self._search_opt_underlyers = []
        self._search_opt_underlyer_descriptions = {}
        self._search_symbol_labels = {}
        self._search_opt_underlyer_index = 0
        self._search_opt_chain_cache = {}
        self._search_opt_chain_page_cache = {}
        self._search_timing = {}
        self._clear_search_option_draft()
        self._reset_search_expiry_paging()
        self._queue_search()

    def _move_search_selection(self, delta: int) -> None:
        total = self._search_row_count()
        if total <= 0:
            self._render_search()
            return
        max_index = total - 1
        self._search_selected = min(max(self._search_selected + int(delta), 0), max_index)
        self._ensure_search_visible()
        self._render_search()

    def _search_mode(self) -> str:
        return self._SEARCH_MODES[self._search_mode_index]

    @staticmethod
    def _is_option_search_mode(mode: str) -> bool:
        return str(mode or "").strip().upper() in ("OPT", "FOP")

    def _search_option_sec_type(self) -> str | None:
        mode = self._search_mode()
        return mode if self._is_option_search_mode(mode) else None

    def _option_draft(self) -> LiveOptionPackageDraft:
        draft = getattr(self, "_search_option_draft", None)
        return draft if isinstance(draft, LiveOptionPackageDraft) else LiveOptionPackageDraft()

    def _clear_search_option_draft(self) -> None:
        self._search_option_draft = LiveOptionPackageDraft()
        self._search_option_notice = None

    def _cycle_selected_option_leg(self) -> None:
        contract = self._selected_opt_contract()
        if contract is None:
            return
        try:
            self._search_option_draft = self._option_draft().cycle(contract)
            self._search_option_notice = None
        except ValueError as exc:
            self._search_option_notice = str(exc)
        self._render_search()

    def _adjust_selected_option_leg_ratio(self, delta: int) -> None:
        contract = self._selected_opt_contract()
        if contract is None:
            return
        try:
            self._search_option_draft = self._option_draft().adjust_ratio(contract, delta)
            self._search_option_notice = None
        except ValueError as exc:
            self._search_option_notice = str(exc)
        self._render_search()

    def _search_option_package_quote(self, *, mode: str = "MID") -> LiveOptionPackageQuote | None:
        draft = self._option_draft()
        if len(draft.legs) < 2:
            return None
        tickers = tuple(
            self._client.ticker_for_con_id(int(getattr(leg.contract, "conId", 0) or 0))
            for leg in draft.legs
        )
        if any(ticker is None for ticker in tickers):
            return None
        symbol = str(getattr(draft.legs[0].contract, "symbol", "") or "")
        return quote_live_option_package(
            symbol=symbol, legs=draft.legs, tickers=tickers, quantity=1, intent="enter", mode=mode
        )

    def _search_opt_expiries(self) -> list[str]:
        option_sec_type = self._search_option_sec_type()
        if option_sec_type is None:
            return []
        expiries = {
            str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            for contract in self._search_results
            if str(getattr(contract, "secType", "") or "").strip().upper() == option_sec_type
        }
        cleaned = [value for value in expiries if value]
        return sorted(cleaned)

    def _current_opt_expiry(self) -> str | None:
        expiries = self._search_opt_expiries()
        if not expiries:
            self._search_opt_expiry_index = 0
            return None
        self._search_opt_expiry_index = min(
            max(int(self._search_opt_expiry_index), 0),
            len(expiries) - 1,
        )
        return expiries[self._search_opt_expiry_index]

    def _cycle_search_expiry(self, step: int) -> None:
        if self._search_option_sec_type() is None:
            return
        expiries = self._search_opt_expiries()
        if not expiries:
            return
        direction = int(step)
        at_last_expiry = self._search_opt_expiry_index >= (len(expiries) - 1)
        if (
            direction > 0
            and at_last_expiry
            and bool(getattr(self, "_search_expiry_has_more", False))
        ):
            self._search_expiry_auto_advance_from = len(expiries)
            self._queue_search_next_expiry_page()
            return
        count = len(expiries)
        self._search_opt_expiry_index = (self._search_opt_expiry_index + direction) % count
        self._search_selected = self._default_opt_row_index()
        self._search_scroll = 0
        self._ensure_search_visible()
        self._render_search()

    def _current_opt_underlyer(self) -> str | None:
        if not self._search_opt_underlyers:
            self._search_opt_underlyer_index = 0
            return None
        self._search_opt_underlyer_index = min(
            max(int(self._search_opt_underlyer_index), 0),
            len(self._search_opt_underlyers) - 1,
        )
        symbol = str(self._search_opt_underlyers[self._search_opt_underlyer_index] or "").strip().upper()
        return symbol or None

    def _current_opt_underlyer_description(self) -> str:
        symbol = self._current_opt_underlyer()
        if not symbol:
            return ""
        text = str(self._search_opt_underlyer_descriptions.get(symbol, "") or "").strip()
        if not text:
            return ""
        if text.strip().upper() == symbol:
            return ""
        return text

    def _search_label_for_symbol(self, symbol: str, *, sec_type: str = "") -> str:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            return ""
        label = str(self._search_symbol_labels.get(normalized, "") or "").strip()
        if not label and sec_type == "OPT":
            label = str(self._search_opt_underlyer_descriptions.get(normalized, "") or "").strip()
        if label and label.upper() != normalized:
            return label
        if sec_type == "STK":
            return "Equity"
        if sec_type == "FUT":
            return "Futures"
        if sec_type == "OPT":
            return "Equity Option"
        if sec_type == "FOP":
            return "Futures Option"
        return ""

    def _search_label_for_contract(self, contract: Contract | None) -> str:
        if contract is None:
            return ""
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        return self._search_label_for_symbol(symbol, sec_type=sec_type)

    def _search_name_line(self) -> Text | None:
        if not self._search_query.strip():
            return None
        mode = self._search_mode()
        symbol = ""
        label = ""
        style = self._SECTION_HEADER_STYLE_BY_TYPE.get(mode, "bold white")
        if mode == "OPT":
            symbol = self._current_opt_underlyer() or ""
            label = self._search_label_for_symbol(symbol, sec_type="OPT")
        else:
            contract = self._selected_search_contract()
            if contract is None and self._search_results:
                contract = self._search_results[0]
            if contract is not None:
                symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
                sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
                label = self._search_label_for_symbol(symbol, sec_type=sec_type)
                style = self._SECTION_HEADER_STYLE_BY_TYPE.get(sec_type, style)
        if not symbol:
            return None
        line = Text("Name ", style="dim")
        line.append(symbol, style=style)
        if label:
            line.append("  •  ", style="dim")
            line.append(label, style="bold #b7cadc")
        return line

    def _cycle_search_opt_underlyer(self, step: int) -> None:
        if self._search_mode() != "OPT":
            return
        count = len(self._search_opt_underlyers)
        if count <= 1:
            self._render_search()
            return
        self._search_generation += 1
        self._cancel_search_task()
        self._search_opt_underlyer_index = (self._search_opt_underlyer_index + int(step)) % count
        self._clear_search_option_draft()
        self._search_selected = 0
        self._search_scroll = 0
        self._search_side = 0
        self._search_opt_expiry_index = 0
        self._reset_search_expiry_paging()
        self._queue_search_opt_underlyer_load()

    def _queue_search_opt_underlyer_load(self) -> None:
        if self._search_mode() != "OPT":
            return
        query = self._search_query.strip()
        symbol = self._current_opt_underlyer()
        if not query or not symbol:
            self._search_results = []
            self._search_loading = False
            self._search_timing = {}
            self._reset_search_expiry_paging()
            self._render_search()
            return
        cached = self._search_opt_chain_cache.get(symbol)
        if cached is not None:
            self._search_loading = False
            self._search_expiry_loading_more = False
            self._search_error = None
            self._search_results = list(cached)
            self._apply_cached_opt_chain_paging(symbol)
            self._search_selected = self._default_opt_row_index()
            self._ensure_search_visible()
            self._init_search_timing(
                generation=self._search_generation,
                query=query,
                mode="OPT",
                source="underlyer-cycle",
                fetch_limit=self._SEARCH_OPT_FETCH_LIMIT,
            )
            self._set_search_timing(
                status="done",
                phase="done",
                opt_deepen_pending=False,
                chain_cache_hit=True,
                contracts_ms=0.0,
                rows=len(self._search_results),
                candidate_count=0,
                qualified_count=0,
                total_ms=0.0,
            )
            self._render_search()
            return
        self._init_search_timing(
            generation=self._search_generation,
            query=query,
            mode="OPT",
            source="underlyer-cycle",
            fetch_limit=self._SEARCH_OPT_FETCH_LIMIT,
        )
        self._set_search_timing(phase="contracts", opt_deepen_pending=False)
        self._search_loading = True
        self._search_expiry_loading_more = False
        self._search_error = None
        self._search_results = []
        self._render_search()
        generation = self._search_generation
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._search_loading = False
            self._finalize_search_timing(
                generation=generation,
                status="error",
                error="runtime_loop_unavailable",
            )
            self._render_search()
            return
        self._search_task = loop.create_task(
            self._run_search_opt_underlyer(
                generation,
                query,
                symbol,
                fetch_limit=self._SEARCH_OPT_FETCH_LIMIT,
            )
        )

    def _option_pair_rows(self) -> list[tuple[Contract | None, Contract | None]]:
        option_sec_type = self._search_option_sec_type()
        if option_sec_type is None:
            return []
        target_expiry = self._current_opt_expiry()
        by_strike: dict[float, list[Contract | None]] = {}
        for contract in self._search_results:
            if str(getattr(contract, "secType", "") or "").strip().upper() != option_sec_type:
                continue
            expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
            if target_expiry and expiry != target_expiry:
                continue
            try:
                strike = float(getattr(contract, "strike", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if strike not in by_strike:
                by_strike[strike] = [None, None]
            right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
            if right == "P":
                by_strike[strike][1] = contract
            else:
                by_strike[strike][0] = contract
        rows: list[tuple[Contract | None, Contract | None]] = []
        for strike in sorted(by_strike.keys()):
            call, put = by_strike[strike]
            rows.append((call, put))
        return rows

    def _search_option_contracts(self) -> dict[int, Contract]:
        if not self._search_active or self._search_option_sec_type() is None:
            return {}
        contracts: dict[int, Contract] = {}
        for call_contract, put_contract in self._option_pair_rows():
            for contract in (call_contract, put_contract):
                if contract is None:
                    continue
                con_id = int(getattr(contract, "conId", 0) or 0)
                if con_id:
                    contracts.setdefault(con_id, contract)
        for leg in self._option_draft().legs:
            con_id = int(getattr(leg.contract, "conId", 0) or 0)
            if con_id:
                contracts.setdefault(con_id, leg.contract)
        return contracts

    def _sync_search_option_tickers(self) -> None:
        wanted = self._search_option_contracts()
        wanted_ids = set(wanted.keys())
        for con_id in list(self._search_ticker_con_ids):
            if con_id in wanted_ids:
                continue
            self._client.release_ticker(con_id, owner="search")
            self._search_ticker_con_ids.discard(con_id)
            self._search_ticker_loading.discard(con_id)
        for con_id, contract in wanted.items():
            self._start_search_ticker_load(con_id, contract)

    def _clear_search_tickers(self) -> None:
        for con_id in list(self._search_ticker_con_ids):
            self._client.release_ticker(con_id, owner="search")
        self._search_ticker_con_ids.clear()
        self._search_ticker_loading.clear()

    def _start_search_ticker_load(self, con_id: int, contract: Contract) -> None:
        if con_id in self._search_ticker_con_ids or con_id in self._search_ticker_loading:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._search_ticker_loading.add(con_id)
        loop.create_task(self._load_search_ticker(con_id, contract))

    async def _load_search_ticker(self, con_id: int, contract: Contract) -> None:
        try:
            await self._client.ensure_ticker(contract, owner="search")
        except Exception:
            return
        finally:
            self._search_ticker_loading.discard(con_id)
        wanted = self._search_option_contracts()
        if con_id not in wanted:
            self._client.release_ticker(con_id, owner="search")
            return
        self._search_ticker_con_ids.add(con_id)
        self._render_search()

    @staticmethod
    def _option_quote_value(ticker: Ticker | None) -> float | None:
        if ticker is None:
            return None
        bid = _safe_num(getattr(ticker, "bid", None))
        ask = _safe_num(getattr(ticker, "ask", None))
        if bid is not None and ask is not None and bid > 0 and ask > 0 and bid <= ask:
            return (float(bid) + float(ask)) / 2.0
        for value in (
            _safe_num(getattr(ticker, "last", None)),
            _safe_num(getattr(ticker, "close", None)),
        ):
            if value is not None and value > 0:
                return float(value)
        return None

    def _search_option_cell_text(self, right: str, contract: Contract | None) -> str:
        con_id = int(getattr(contract, "conId", 0) or 0) if contract is not None else 0
        if not con_id:
            return f"□ {right} --"
        selected = self._option_draft().leg_for(contract)
        marker = "□" if selected is None else f"{selected.action[0]}{selected.ratio}"
        ticker = self._client.ticker_for_con_id(con_id)
        quote = self._option_quote_value(ticker)
        if quote is not None:
            return f"{marker} {right} {quote:.2f}"
        if con_id in self._search_ticker_loading:
            return f"{marker} {right} ..."
        return f"{marker} {right} --"

    @staticmethod
    def _option_row_strike(
        call_contract: Contract | None,
        put_contract: Contract | None,
    ) -> float | None:
        source = call_contract if call_contract is not None else put_contract
        if source is None:
            return None
        try:
            strike = float(getattr(source, "strike", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        return strike if strike > 0 else None

    def _opt_reference_price(self) -> float | None:
        timing = getattr(self, "_search_timing", None)
        if not isinstance(timing, dict):
            return None
        mode = str(timing.get("mode", "") or "").strip().upper()
        if mode != "OPT":
            return None
        try:
            value = float(timing.get("ref_price"))
        except (TypeError, ValueError):
            return None
        return float(value) if value > 0 else None

    def _default_opt_row_index(self) -> int:
        rows = self._option_pair_rows()
        if not rows:
            return 0
        strikes: list[float] = []
        for call_contract, put_contract in rows:
            strike = self._option_row_strike(call_contract, put_contract)
            if strike is not None:
                strikes.append(float(strike))
        if not strikes:
            return 0
        target = self._opt_reference_price()
        if target is None:
            target = strikes[len(strikes) // 2]
        best_idx = 0
        best_delta = float("inf")
        for idx, (call_contract, put_contract) in enumerate(rows):
            strike = self._option_row_strike(call_contract, put_contract)
            if strike is None:
                continue
            delta = abs(float(strike) - float(target))
            if delta < best_delta:
                best_delta = delta
                best_idx = idx
        return best_idx

    def _opt_underlyer_label(self) -> str:
        symbol = self._current_opt_underlyer()
        if not symbol:
            return ""
        total = len(self._search_opt_underlyers)
        if total <= 0:
            return symbol
        index = min(
            max(int(self._search_opt_underlyer_index), 0),
            total - 1,
        ) + 1
        return f"[{index}/{total}] {symbol}"

    def _search_row_count(self) -> int:
        if self._search_option_sec_type() is not None:
            return len(self._option_pair_rows())
        return len(self._search_results)

    def _ensure_search_visible(self) -> None:
        total = self._search_row_count()
        if total <= 0:
            self._search_scroll = 0
            return
        max_scroll = max(0, total - self._SEARCH_LIMIT)
        if self._search_selected < self._search_scroll:
            self._search_scroll = self._search_selected
        elif self._search_selected >= (self._search_scroll + self._SEARCH_LIMIT):
            self._search_scroll = self._search_selected - self._SEARCH_LIMIT + 1
        self._search_scroll = min(max(self._search_scroll, 0), max_scroll)

    def _init_search_timing(
        self,
        *,
        generation: int,
        query: str,
        mode: str,
        source: str,
        fetch_limit: int,
    ) -> None:
        self._search_timing = {
            "generation": int(generation),
            "query": str(query or ""),
            "mode": str(mode or "").strip().upper(),
            "source": str(source or "").strip().lower(),
            "fetch_limit": int(fetch_limit),
            "status": "loading",
            "phase": "queued",
            "started_mono": float(time.monotonic()),
        }

    def _set_search_timing(self, *, generation: int | None = None, **values: object) -> None:
        timing = getattr(self, "_search_timing", None)
        if not isinstance(timing, dict):
            timing = {}
            self._search_timing = timing
        if generation is not None:
            current_generation = self._timing_int(timing.get("generation"))
            if current_generation != int(generation):
                return
        timing.update(values)

    def _finalize_search_timing(
        self,
        *,
        generation: int | None = None,
        status: str,
        rows: int | None = None,
        error: str | None = None,
    ) -> None:
        timing = getattr(self, "_search_timing", None)
        if not isinstance(timing, dict) or not timing:
            return
        if generation is not None:
            current_generation = self._timing_int(timing.get("generation"))
            if current_generation != int(generation):
                return
        started_raw = timing.get("started_mono")
        try:
            started_mono = float(started_raw)
        except (TypeError, ValueError):
            started_mono = 0.0
        total_ms = None
        if started_mono > 0:
            total_ms = (time.monotonic() - started_mono) * 1000.0
        timing["status"] = str(status or "").strip().lower() or "done"
        timing["phase"] = "done" if timing["status"] in ("done", "ok", "success") else timing.get("phase")
        if rows is not None:
            timing["rows"] = int(rows)
        if error:
            timing["error"] = str(error)
        if total_ms is not None:
            timing["total_ms"] = float(total_ms)

    def _set_search_expiry_paging_from_timing(self, timing: dict[str, object] | None) -> None:
        payload = timing if isinstance(timing, dict) else {}
        self._search_expiry_has_more = bool(payload.get("has_more_expiries"))
        self._search_expiry_next_offset = max(
            0,
            self._timing_int(payload.get("next_expiry_offset")),
        )
        self._search_expiry_total = max(
            0,
            self._timing_int(payload.get("expiry_count")),
        )
        self._search_expiry_loading_more = False

    def _cache_opt_chain_paging(self, symbol: str, timing: dict[str, object] | None) -> None:
        key = str(symbol or "").strip().upper()
        if not key:
            return
        payload = timing if isinstance(timing, dict) else {}
        state = {
            "has_more_expiries": bool(payload.get("has_more_expiries")),
            "next_expiry_offset": max(0, self._timing_int(payload.get("next_expiry_offset"))),
            "expiry_count": max(0, self._timing_int(payload.get("expiry_count"))),
        }
        if not hasattr(self, "_search_opt_chain_page_cache"):
            self._search_opt_chain_page_cache = {}
        self._search_opt_chain_page_cache[key] = state

    def _apply_cached_opt_chain_paging(self, symbol: str) -> None:
        key = str(symbol or "").strip().upper()
        if not key:
            self._reset_search_expiry_paging()
            return
        cache = getattr(self, "_search_opt_chain_page_cache", {})
        if not isinstance(cache, dict):
            self._reset_search_expiry_paging()
            return
        payload = cache.get(key)
        if not isinstance(payload, dict):
            self._reset_search_expiry_paging()
            return
        self._set_search_expiry_paging_from_timing(payload)

    @staticmethod
    def _timing_ms_text(value: object) -> str:
        try:
            ms = float(value)
        except (TypeError, ValueError):
            return "..."
        if ms < 0:
            return "..."
        return f"{ms:.0f}ms"

    @staticmethod
    def _timing_int(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _search_opt_first_paint_limit(self, fetch_limit: int) -> int:
        # Run OPT first page as a single deterministic pass to avoid
        # partial-table regressions during qualify-first/qualify-rest splits.
        return max(1, int(fetch_limit or 1))

    def _set_opt_search_contract_timing(
        self,
        *,
        generation: int,
        contract_timing: dict[str, object],
        contracts_started: float,
        phase: str | None = None,
        deepen_pending: bool | None = None,
    ) -> None:
        values: dict[str, object] = {
            "chain_cache_hit": False,
            "contracts_ms": contract_timing.get(
                "total_ms",
                (time.monotonic() - contracts_started) * 1000.0,
            ),
            "candidate_count": self._timing_int(contract_timing.get("candidate_count")),
            "qualified_count": self._timing_int(contract_timing.get("qualified_count")),
            "contract_stage": str(contract_timing.get("stage", "") or ""),
            "contract_reason": str(contract_timing.get("reason", "") or ""),
            "chain_ms": contract_timing.get("chain_ms"),
            "ref_price_ms": contract_timing.get("ref_price_ms"),
            "ref_price": contract_timing.get("ref_price"),
            "ref_price_source": str(contract_timing.get("ref_price_source", "") or ""),
            "qualify_ms": contract_timing.get("qualify_ms"),
            "selected_expiry_count": self._timing_int(contract_timing.get("selected_expiry_count")),
            "expiry_count": self._timing_int(contract_timing.get("expiry_count")),
            "strike_count": self._timing_int(contract_timing.get("strike_count")),
            "rows_per_expiry": self._timing_int(contract_timing.get("rows_per_expiry")),
            "first_limit": self._timing_int(contract_timing.get("first_limit")),
            "split_active": bool(contract_timing.get("split_active")),
            "qualify_ms_first": contract_timing.get("qualify_ms_first"),
            "qualify_ms_rest": contract_timing.get("qualify_ms_rest"),
        }
        if phase is not None:
            values["phase"] = str(phase)
        if deepen_pending is not None:
            values["opt_deepen_pending"] = bool(deepen_pending)
        self._set_search_timing(generation=generation, **values)

    def _apply_opt_progress_rows(
        self,
        *,
        generation: int,
        symbol: str,
        results: list[Contract],
        contract_timing: dict[str, object],
        contracts_started: float,
    ) -> None:
        if generation != self._search_generation:
            return
        incoming_rows = list(results)
        incoming_count = len(incoming_rows)
        current_count = len(self._search_results)
        cached_rows = self._search_opt_chain_cache.get(symbol)
        cached_count = len(cached_rows) if cached_rows is not None else 0
        # Guard against out-of-order progress callbacks regressing an already-expanded table.
        if current_count > 0 and incoming_count < current_count:
            return
        if cached_count > 0 and incoming_count < cached_count:
            return
        if not self._search_loading and current_count > 0:
            return
        self._search_error = None
        self._search_loading = True
        self._search_results = incoming_rows
        self._search_opt_chain_cache[symbol] = list(incoming_rows)
        self._search_selected = self._default_opt_row_index()
        self._ensure_search_visible()
        self._set_opt_search_contract_timing(
            generation=generation,
            contract_timing=contract_timing,
            contracts_started=contracts_started,
            phase="contracts-deepen",
            deepen_pending=True,
        )
        self._render_search()

    def _search_timing_line(self) -> Text | None:
        if not self._search_query.strip():
            return None
        timing = getattr(self, "_search_timing", None)
        if not isinstance(timing, dict) or not timing:
            return None
        mode = str(timing.get("mode", self._search_mode()) or "").strip().upper()
        parts: list[str] = []
        phase = str(timing.get("phase", "") or "").strip().lower()
        if phase:
            parts.append(f"phase={phase}")
        if mode == "OPT":
            under_ms = self._timing_ms_text(timing.get("underlyer_ms"))
            under_source = str(timing.get("underlyer_source", "") or "").strip().lower()
            under_suffix: list[str] = []
            under_matching_ms = timing.get("underlyer_matching_ms")
            if under_matching_ms is not None:
                under_suffix.append(f"match={self._timing_ms_text(under_matching_ms)}")
            fallback_calls = self._timing_int(timing.get("underlyer_fallback_calls"))
            if fallback_calls > 0:
                under_suffix.append(
                    f"fb={fallback_calls}/{self._timing_ms_text(timing.get('underlyer_fallback_ms'))}"
                )
            rank_ms = timing.get("underlyer_rank_ms")
            if rank_ms is not None:
                under_suffix.append(f"rank={self._timing_ms_text(rank_ms)}")
            if under_source:
                under_ms = f"{under_ms} {under_source}"
            if under_suffix:
                under_ms = f"{under_ms} ({', '.join(under_suffix)})"
            chain_ms = self._timing_ms_text(timing.get("contracts_ms"))
            chain_suffix: list[str] = []
            if bool(timing.get("chain_cache_hit")):
                chain_suffix.append("cache")
            if bool(timing.get("opt_deepen_pending")):
                chain_suffix.append("bg")
            candidates = self._timing_int(timing.get("candidate_count"))
            qualified = self._timing_int(timing.get("qualified_count"))
            if candidates > 0:
                chain_suffix.append(f"q {qualified}/{candidates}")
            chain_detail: list[str] = []
            secdef_ms = timing.get("chain_ms")
            if secdef_ms is not None:
                chain_detail.append(f"secdef={self._timing_ms_text(secdef_ms)}")
            ref_ms = timing.get("ref_price_ms")
            if ref_ms is not None:
                ref_source = str(timing.get("ref_price_source", "") or "").strip().lower()
                ref_part = f"ref={self._timing_ms_text(ref_ms)}"
                if ref_source:
                    ref_part = f"{ref_part}/{ref_source}"
                chain_detail.append(ref_part)
            qualify_ms = timing.get("qualify_ms")
            if qualify_ms is not None:
                chain_detail.append(f"qual={self._timing_ms_text(qualify_ms)}")
            if bool(timing.get("split_active")):
                first_limit = self._timing_int(timing.get("first_limit"))
                if first_limit > 0:
                    chain_detail.append(f"fp={first_limit}")
                qualify_ms_first = timing.get("qualify_ms_first")
                if qualify_ms_first is not None:
                    chain_detail.append(f"q1={self._timing_ms_text(qualify_ms_first)}")
                qualify_ms_rest = timing.get("qualify_ms_rest")
                if qualify_ms_rest is not None:
                    chain_detail.append(f"q2={self._timing_ms_text(qualify_ms_rest)}")
            selected_expiry_count = self._timing_int(timing.get("selected_expiry_count"))
            expiry_count = self._timing_int(timing.get("expiry_count"))
            if expiry_count > 0:
                chain_detail.append(f"exp={selected_expiry_count}/{expiry_count}")
            rows_per_expiry = self._timing_int(timing.get("rows_per_expiry"))
            if rows_per_expiry > 0:
                chain_detail.append(f"rpe={rows_per_expiry}")
            strike_count = self._timing_int(timing.get("strike_count"))
            if strike_count > 0:
                chain_detail.append(f"strk={strike_count}")
            contract_stage = str(timing.get("contract_stage", "") or "").strip().lower()
            contract_reason = str(timing.get("contract_reason", "") or "").strip().lower()
            if contract_stage and contract_stage not in ("done", "cache"):
                chain_detail.append(f"stg={contract_stage}")
            if contract_reason and contract_reason not in ("ok", "cache-hit"):
                chain_detail.append(f"why={contract_reason}")
            chain_detail_text = ""
            if chain_detail:
                chain_detail_text = f" ({' '.join(chain_detail)})"
            parts.extend(
                [
                    f"db={self._timing_ms_text(timing.get('debounce_ms'))}",
                    f"under={under_ms}",
                    f"chain={chain_ms}{(' ' + ','.join(chain_suffix)) if chain_suffix else ''}{chain_detail_text}",
                ]
            )
        else:
            parts.extend(
                [
                    f"db={self._timing_ms_text(timing.get('debounce_ms'))}",
                    f"contracts={self._timing_ms_text(timing.get('contracts_ms'))}",
                    f"labels={self._timing_ms_text(timing.get('labels_ms'))}",
                ]
            )
        total_ms = timing.get("total_ms")
        if total_ms is not None:
            parts.append(f"total={self._timing_ms_text(total_ms)}")
        rows = timing.get("rows")
        if rows is not None:
            parts.append(f"rows={self._timing_int(rows)}")
        line = Text("Timing ", style="dim")
        line.append(" | ".join(parts), style="dim")
        return line

    def _search_fetch_limit_for_mode(self, mode: str) -> int:
        cleaned = str(mode or "").strip().upper()
        if cleaned == "OPT":
            return int(self._SEARCH_OPT_FETCH_LIMIT)
        if self._is_option_search_mode(cleaned):
            return int(self._SEARCH_FETCH_LIMIT)
        return int(self._SEARCH_LIMIT)

    @staticmethod
    def _search_contract_key(contract: Contract | None) -> tuple[object, ...] | None:
        if contract is None:
            return None
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id > 0:
            return ("conid", con_id)
        strike_raw = getattr(contract, "strike", None)
        try:
            strike = round(float(strike_raw or 0.0), 6)
        except (TypeError, ValueError):
            strike = 0.0
        return (
            str(getattr(contract, "secType", "") or "").strip().upper(),
            str(getattr(contract, "symbol", "") or "").strip().upper(),
            str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip(),
            str(getattr(contract, "right", "") or "").strip().upper()[:1],
            float(strike),
            str(getattr(contract, "exchange", "") or "").strip().upper(),
            str(getattr(contract, "tradingClass", "") or "").strip().upper(),
        )

    def _merge_search_contract_rows(
        self,
        existing: list[Contract],
        incoming: list[Contract],
    ) -> list[Contract]:
        out: list[Contract] = []
        seen: set[tuple[object, ...]] = set()
        for contract in list(existing or []) + list(incoming or []):
            key = self._search_contract_key(contract)
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(contract)
        return out

    def _queue_search_next_expiry_page(self) -> None:
        if not self._search_active:
            return
        mode = self._search_mode()
        if not self._is_option_search_mode(mode):
            return
        if self._search_loading or self._search_expiry_loading_more:
            return
        if not bool(self._search_expiry_has_more):
            return
        query = self._search_query.strip()
        if not query:
            return
        offset = max(0, int(self._search_expiry_next_offset))
        generation = int(self._search_generation)
        symbol = self._current_opt_underlyer() if mode == "OPT" else None
        if mode == "OPT" and not symbol:
            return
        self._cancel_search_expiry_prefetch()
        self._search_expiry_loading_more = True
        self._search_loading = True
        self._search_error = None
        self._set_search_timing(
            generation=generation,
            phase="contracts-paging",
            opt_deepen_pending=True,
        )
        self._render_search()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._search_loading = False
            self._search_expiry_loading_more = False
            self._finalize_search_timing(
                generation=generation,
                status="error",
                error="runtime_loop_unavailable",
            )
            self._render_search()
            return
        fetch_limit = self._search_fetch_limit_for_mode(mode)
        self._search_task = loop.create_task(
            self._run_search_next_expiry_page(
                generation=generation,
                query=query,
                mode=mode,
                fetch_limit=fetch_limit,
                expiry_offset=offset,
                opt_underlyer_symbol=symbol,
            )
        )
