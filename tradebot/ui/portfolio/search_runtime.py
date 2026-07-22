"""Portfolio search I/O, result rendering, and selection runtime."""

from __future__ import annotations

import asyncio
import time
from typing import cast

from ib_insync import Contract, PortfolioItem
from rich.text import Text

from ..common import _SyntheticPortfolioItem
from ..positions import PositionDetailScreen


class PortfolioSearchRuntime:
    def _queue_search_idle_expiry_prefetch(self, *, generation: int) -> None:
        if int(generation) != int(self._search_generation):
            return
        if not bool(getattr(self, "_search_active", True)):
            return
        mode = self._search_mode()
        if not self._is_option_search_mode(mode):
            return
        if not bool(getattr(self, "_search_expiry_has_more", False)):
            return
        if bool(getattr(self, "_search_loading", False)) or bool(
            getattr(self, "_search_expiry_loading_more", False)
        ):
            return
        if int(getattr(self, "_search_expiry_prefetch_generation", -1)) == int(generation):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._cancel_search_expiry_prefetch()
        self._search_expiry_prefetch_generation = int(generation)
        self._search_expiry_prefetch_task = loop.create_task(
            self._run_search_idle_expiry_prefetch(int(generation))
        )

    async def _run_search_idle_expiry_prefetch(self, generation: int) -> None:
        try:
            await asyncio.sleep(float(self._SEARCH_EXPIRY_IDLE_PREFETCH_SEC))
        except asyncio.CancelledError:
            return
        try:
            if int(generation) != int(self._search_generation):
                return
            if not bool(getattr(self, "_search_active", True)):
                return
            if not bool(getattr(self, "_search_expiry_has_more", False)):
                return
            if bool(getattr(self, "_search_loading", False)) or bool(
                getattr(self, "_search_expiry_loading_more", False)
            ):
                return
            self._queue_search_next_expiry_page()
        finally:
            if int(generation) == int(getattr(self, "_search_expiry_prefetch_generation", -1)):
                self._search_expiry_prefetch_task = None
                self._search_expiry_prefetch_generation = -1

    async def _run_search_next_expiry_page(
        self,
        *,
        generation: int,
        query: str,
        mode: str,
        fetch_limit: int,
        expiry_offset: int,
        opt_underlyer_symbol: str | None,
    ) -> None:
        contracts_started = time.monotonic()
        contract_timing: dict[str, object] = {}
        try:
            kwargs: dict[str, object] = {
                "mode": str(mode or "").strip().upper(),
                "limit": int(fetch_limit),
                "timing": contract_timing,
                "expiry_offset": int(max(0, expiry_offset)),
            }
            if str(mode or "").strip().upper() == "OPT":
                kwargs["opt_underlyer_symbol"] = str(opt_underlyer_symbol or "").strip().upper()
            rows = await self._client.search_contracts(query, **kwargs)
        except asyncio.CancelledError:
            if generation == self._search_generation:
                self._search_expiry_loading_more = False
                self._search_loading = False
                self._set_search_timing(generation=generation, phase="cancelled")
                self._finalize_search_timing(generation=generation, status="cancelled")
            return
        except Exception as exc:
            if generation != self._search_generation:
                return
            self._search_expiry_loading_more = False
            self._search_loading = False
            self._search_error = str(exc)
            self._set_search_timing(generation=generation, phase="error")
            self._finalize_search_timing(
                generation=generation,
                status="error",
                error=str(exc),
            )
            self._render_search()
            return
        if generation != self._search_generation:
            return
        merged_rows = self._merge_search_contract_rows(self._search_results, list(rows or []))
        self._search_results = merged_rows
        mode_clean = str(mode or "").strip().upper()
        if mode_clean == "OPT":
            symbol = str(opt_underlyer_symbol or "").strip().upper()
            if symbol:
                self._search_opt_chain_cache[symbol] = list(self._search_results)
                self._cache_opt_chain_paging(symbol, contract_timing)
            self._set_opt_search_contract_timing(
                generation=generation,
                contract_timing=contract_timing,
                contracts_started=contracts_started,
                phase="contracts-paging",
                deepen_pending=bool(contract_timing.get("has_more_expiries")),
            )
        else:
            self._set_search_timing(
                generation=generation,
                phase="contracts-paging",
                contracts_ms=contract_timing.get(
                    "total_ms",
                    (time.monotonic() - contracts_started) * 1000.0,
                ),
                candidate_count=self._timing_int(contract_timing.get("candidate_count")),
                qualified_count=self._timing_int(contract_timing.get("qualified_count")),
                contract_stage=str(contract_timing.get("stage", "") or ""),
                contract_reason=str(contract_timing.get("reason", "") or ""),
                selected_expiry_count=self._timing_int(contract_timing.get("selected_expiry_count")),
                expiry_count=self._timing_int(contract_timing.get("expiry_count")),
                rows_per_expiry=self._timing_int(contract_timing.get("rows_per_expiry")),
                opt_deepen_pending=bool(contract_timing.get("has_more_expiries")),
            )
        self._set_search_expiry_paging_from_timing(contract_timing)
        if not self._search_expiry_has_more:
            self._search_expiry_auto_advance_from = None
        auto_from = self._search_expiry_auto_advance_from
        if auto_from is not None:
            expiries_now = self._search_opt_expiries()
            if len(expiries_now) > int(auto_from):
                self._search_opt_expiry_index = int(auto_from)
                self._search_selected = self._default_opt_row_index()
                self._search_scroll = 0
            self._search_expiry_auto_advance_from = None
        self._search_expiry_loading_more = False
        self._search_loading = False
        self._ensure_search_visible()
        self._set_search_timing(
            generation=generation,
            phase="done",
            opt_deepen_pending=False,
            rows=len(self._search_results),
        )
        self._finalize_search_timing(
            generation=generation,
            status="done",
            rows=len(self._search_results),
        )
        if self._is_option_search_mode(mode):
            self._queue_search_idle_expiry_prefetch(generation=generation)
        self._render_search()

    def _queue_search(self) -> None:
        self._search_error = None
        self._search_selected = 0
        self._search_scroll = 0
        self._search_opt_expiry_index = 0
        self._search_opt_underlyers = []
        self._search_opt_underlyer_descriptions = {}
        self._search_symbol_labels = {}
        self._search_opt_underlyer_index = 0
        self._search_opt_chain_cache = {}
        self._search_opt_chain_page_cache = {}
        self._search_timing = {}
        self._reset_search_expiry_paging()
        self._cancel_search_task()
        query = self._search_query.strip()
        if not query:
            self._search_results = []
            self._search_loading = False
            self._render_search()
            return
        self._search_loading = True
        self._render_search()
        self._search_generation += 1
        generation = self._search_generation
        mode = self._search_mode()
        fetch_limit = self._search_fetch_limit_for_mode(mode)
        self._init_search_timing(
            generation=generation,
            query=query,
            mode=mode,
            source="query",
            fetch_limit=fetch_limit,
        )
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
            self._run_search(
                generation,
                query,
                mode,
                fetch_limit=fetch_limit,
            )
        )

    async def _run_search(self, generation: int, query: str, mode: str, *, fetch_limit: int) -> None:
        self._set_search_timing(generation=generation, phase="debounce")
        try:
            debounce_started = time.monotonic()
            await asyncio.sleep(self._SEARCH_DEBOUNCE_SEC)
            self._set_search_timing(
                generation=generation,
                debounce_ms=(time.monotonic() - debounce_started) * 1000.0,
            )
            if mode == "OPT":
                self._set_search_timing(generation=generation, phase="underlyers")
                underlyer_timing: dict[str, object] = {}
                underlyer_started = time.monotonic()
                underlyers = await self._client.search_option_underlyers(
                    query,
                    limit=self._SEARCH_OPT_UNDERLYER_LIMIT,
                    timing=underlyer_timing,
                )
                self._set_search_timing(
                    generation=generation,
                    underlyer_ms=underlyer_timing.get(
                        "total_ms",
                        (time.monotonic() - underlyer_started) * 1000.0,
                    ),
                    underlyer_source=str(underlyer_timing.get("source", "") or "").strip().lower(),
                    underlyer_count=len(underlyers),
                    underlyer_matching_ms=underlyer_timing.get("matching_ms"),
                    underlyer_matching_rows=self._timing_int(underlyer_timing.get("matching_rows")),
                    underlyer_direct_ms=underlyer_timing.get("direct_ms"),
                    underlyer_fallback_ms=underlyer_timing.get("fallback_chain_ms"),
                    underlyer_fallback_calls=self._timing_int(
                        underlyer_timing.get("fallback_chain_calls")
                    ),
                    underlyer_rank_ms=underlyer_timing.get("rank_ms"),
                )
                if generation != self._search_generation:
                    return
                self._search_opt_underlyers = []
                self._search_opt_underlyer_descriptions = {}
                self._search_symbol_labels = {}
                for symbol, description in underlyers:
                    cleaned_symbol = str(symbol or "").strip().upper()
                    if not cleaned_symbol or cleaned_symbol in self._search_opt_underlyers:
                        continue
                    self._search_opt_underlyers.append(cleaned_symbol)
                    label = str(description or "").strip()
                    self._search_opt_underlyer_descriptions[cleaned_symbol] = label
                    if label:
                        self._search_symbol_labels[cleaned_symbol] = label
                if not self._search_opt_underlyers:
                    results: list[Contract] = []
                    self._set_search_timing(
                        generation=generation,
                        phase="contracts",
                        opt_deepen_pending=False,
                        chain_cache_hit=False,
                        contracts_ms=0.0,
                        candidate_count=0,
                        qualified_count=0,
                        contract_stage="underlyers-empty",
                        contract_reason="no-underlyers",
                    )
                else:
                    self._search_opt_underlyer_index = min(
                        max(int(self._search_opt_underlyer_index), 0),
                        len(self._search_opt_underlyers) - 1,
                    )
                    symbol = self._search_opt_underlyers[self._search_opt_underlyer_index]
                    cached = self._search_opt_chain_cache.get(symbol)
                    if cached is None:
                        self._set_search_timing(
                            generation=generation,
                            phase="contracts",
                            opt_deepen_pending=False,
                            chain_cache_hit=False,
                        )
                        contract_timing: dict[str, object] = {}
                        contract_started = time.monotonic()
                        first_limit = self._search_opt_first_paint_limit(fetch_limit)

                        async def _on_opt_progress(rows: list[Contract], progress_timing: dict[str, object]) -> None:
                            self._apply_opt_progress_rows(
                                generation=generation,
                                symbol=symbol,
                                results=list(rows or []),
                                contract_timing=progress_timing,
                                contracts_started=contract_started,
                            )

                        results = await self._client.search_contracts(
                            query,
                            mode=mode,
                            limit=fetch_limit,
                            opt_underlyer_symbol=symbol,
                            timing=contract_timing,
                            opt_first_limit=first_limit,
                            opt_progress=_on_opt_progress,
                        )
                        self._set_opt_search_contract_timing(
                            generation=generation,
                            contract_timing=contract_timing,
                            contracts_started=contract_started,
                            phase="contracts",
                            deepen_pending=False,
                        )
                        if generation != self._search_generation:
                            return
                        self._search_opt_chain_cache[symbol] = list(results)
                        self._cache_opt_chain_paging(symbol, contract_timing)
                        self._set_search_expiry_paging_from_timing(contract_timing)
                    else:
                        results = list(cached)
                        self._apply_cached_opt_chain_paging(symbol)
                        self._set_search_timing(
                            generation=generation,
                            phase="contracts",
                            opt_deepen_pending=False,
                            chain_cache_hit=True,
                            contracts_ms=0.0,
                            candidate_count=0,
                            qualified_count=0,
                            contract_stage="cache",
                            contract_reason="cache-hit",
                        )
            else:
                self._set_search_timing(generation=generation, phase="contracts")
                contracts_started = time.monotonic()
                contract_timing: dict[str, object] = {}
                results = await self._client.search_contracts(
                    query,
                    mode=mode,
                    limit=fetch_limit,
                    timing=contract_timing if self._is_option_search_mode(mode) else None,
                )
                self._set_search_timing(
                    generation=generation,
                    contracts_ms=contract_timing.get(
                        "total_ms",
                        (time.monotonic() - contracts_started) * 1000.0,
                    ),
                )
                if self._is_option_search_mode(mode):
                    self._set_search_expiry_paging_from_timing(contract_timing)
                if generation != self._search_generation:
                    return
                symbols: list[str] = []
                for contract in results:
                    symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
                    if not symbol or symbol in symbols:
                        continue
                    symbols.append(symbol)
                self._search_symbol_labels = {}
                if symbols:
                    self._set_search_timing(generation=generation, phase="labels")
                    labels_started = time.monotonic()
                    self._search_symbol_labels = await self._client.search_contract_labels(
                        query,
                        mode=mode,
                        symbols=symbols,
                    )
                    self._set_search_timing(
                        generation=generation,
                        labels_ms=(time.monotonic() - labels_started) * 1000.0,
                    )
                    if generation != self._search_generation:
                        return
        except asyncio.CancelledError:
            self._search_expiry_loading_more = False
            self._set_search_timing(generation=generation, phase="cancelled")
            self._finalize_search_timing(generation=generation, status="cancelled")
            return
        except Exception as exc:
            if generation != self._search_generation:
                return
            self._search_loading = False
            self._search_expiry_loading_more = False
            self._search_results = []
            self._search_error = str(exc)
            self._set_search_timing(generation=generation, phase="error")
            self._finalize_search_timing(
                generation=generation,
                status="error",
                rows=0,
                error=str(exc),
            )
            self._render_search()
            return
        if generation != self._search_generation:
            return
        had_partial_opt_rows = mode == "OPT" and bool(self._search_results)
        self._search_loading = False
        self._search_expiry_loading_more = False
        self._search_results = list(results)
        total = self._search_row_count()
        if mode == "OPT":
            if had_partial_opt_rows:
                self._search_selected = min(max(self._search_selected, 0), max(total - 1, 0))
            else:
                self._search_selected = self._default_opt_row_index()
        elif mode == "FOP":
            self._search_selected = self._default_opt_row_index()
        else:
            self._search_selected = min(max(self._search_selected, 0), max(total - 1, 0))
        self._ensure_search_visible()
        self._set_search_timing(generation=generation, phase="done", opt_deepen_pending=False)
        self._finalize_search_timing(
            generation=generation,
            status="done",
            rows=len(self._search_results),
        )
        self._queue_search_idle_expiry_prefetch(generation=generation)
        self._render_search()

    async def _run_search_opt_underlyer(
        self,
        generation: int,
        query: str,
        symbol: str,
        *,
        fetch_limit: int,
    ) -> None:
        try:
            self._set_search_timing(generation=generation, phase="contracts", opt_deepen_pending=False)
            contract_timing: dict[str, object] = {}
            contracts_started = time.monotonic()
            first_limit = self._search_opt_first_paint_limit(fetch_limit)

            async def _on_opt_progress(rows: list[Contract], progress_timing: dict[str, object]) -> None:
                self._apply_opt_progress_rows(
                    generation=generation,
                    symbol=symbol,
                    results=list(rows or []),
                    contract_timing=progress_timing,
                    contracts_started=contracts_started,
                )

            results = await self._client.search_contracts(
                query,
                mode="OPT",
                limit=fetch_limit,
                opt_underlyer_symbol=symbol,
                timing=contract_timing,
                opt_first_limit=first_limit,
                opt_progress=_on_opt_progress,
            )
            self._set_opt_search_contract_timing(
                generation=generation,
                contract_timing=contract_timing,
                contracts_started=contracts_started,
                phase="contracts",
                deepen_pending=False,
            )
            self._cache_opt_chain_paging(symbol, contract_timing)
            self._set_search_expiry_paging_from_timing(contract_timing)
        except asyncio.CancelledError:
            self._search_expiry_loading_more = False
            self._set_search_timing(generation=generation, phase="cancelled")
            self._finalize_search_timing(generation=generation, status="cancelled")
            return
        except Exception as exc:
            if generation != self._search_generation:
                return
            self._search_loading = False
            self._search_expiry_loading_more = False
            self._search_results = []
            self._search_error = str(exc)
            self._set_search_timing(generation=generation, phase="error")
            self._finalize_search_timing(
                generation=generation,
                status="error",
                rows=0,
                error=str(exc),
            )
            self._render_search()
            return
        if generation != self._search_generation:
            return
        self._search_opt_chain_cache[symbol] = list(results)
        had_partial_opt_rows = bool(self._search_results)
        self._search_loading = False
        self._search_expiry_loading_more = False
        self._search_results = list(results)
        if had_partial_opt_rows:
            total = self._search_row_count()
            self._search_selected = min(max(self._search_selected, 0), max(total - 1, 0))
        else:
            self._search_selected = self._default_opt_row_index()
        self._ensure_search_visible()
        self._set_search_timing(generation=generation, phase="done", opt_deepen_pending=False)
        self._finalize_search_timing(
            generation=generation,
            status="done",
            rows=len(self._search_results),
        )
        self._render_search()

    def _render_search(self) -> None:
        if not hasattr(self, "_search"):
            return
        if not self._search_active:
            self._search.display = False
            return
        self._search.display = True
        line1 = Text("Search ", style="bold")
        for index, mode in enumerate(self._SEARCH_MODES):
            if index > 0:
                line1.append(" ", style="dim")
            if index == self._search_mode_index:
                line1.append(f"[{mode}]", style="bold #0d1117 on #7ab6ff")
            else:
                line1.append(mode, style="dim")
        line1.append("  > ", style="dim")
        if self._search_query:
            line1.append(self._search_query, style="bold white")
        else:
            line1.append("type symbol...", style="dim")
        lines: list[Text] = [line1]
        name_line = self._search_name_line()
        if name_line is not None:
            lines.append(name_line)
        mode = self._search_mode()
        if self._is_option_search_mode(mode):
            self._sync_search_option_tickers()
        else:
            self._clear_search_tickers()
        if self._search_error:
            lines.append(Text(f"Error: {self._search_error}", style="red"))
        elif self._search_loading:
            lines.append(Text("Searching...", style="yellow"))
        elif not self._search_query.strip():
            lines.append(
                Text(
                    "Tab/Shift+Tab mode | Up/Down scroll | Enter details | Esc close",
                    style="dim",
                )
            )
        timing_line = self._search_timing_line()
        if timing_line is not None:
            lines.append(timing_line)
        if self._is_option_search_mode(mode):
            underlyer_label = self._opt_underlyer_label()
            if mode == "OPT":
                underlyer_line = Text("Underlyer ", style="dim")
                if underlyer_label:
                    underlyer_line.append(underlyer_label, style="bold #86dca9")
                    underlyer_desc = self._current_opt_underlyer_description()
                    if underlyer_desc:
                        underlyer_line.append("  |  ", style="dim")
                        underlyer_line.append(underlyer_desc, style="bold #b7cadc")
                    if len(self._search_opt_underlyers) > 1:
                        underlyer_line.append("  (< / > or Ctrl+Left/Right)", style="dim")
                else:
                    underlyer_line.append("n/a", style="dim")
                lines.append(underlyer_line)

            has_query = bool(self._search_query.strip())
            if mode == "OPT" and not self._search_opt_underlyers:
                if has_query and not self._search_loading:
                    lines.append(Text("No matches", style="dim"))
            elif not self._search_results:
                if has_query and not self._search_loading:
                    lines.append(Text("No option chain rows", style="dim"))
            else:
                expiries = self._search_opt_expiries()
                expiry_line = Text("Expiry ", style="dim")
                if expiries:
                    self._search_opt_expiry_index = min(
                        max(self._search_opt_expiry_index, 0),
                        len(expiries) - 1,
                    )
                    for idx, expiry in enumerate(expiries):
                        if idx > 0:
                            expiry_line.append(" ", style="dim")
                        if idx == self._search_opt_expiry_index:
                            expiry_line.append(f"[{expiry}]", style="bold #0d1117 on #ffcc84")
                        else:
                            expiry_line.append(expiry, style="bold #ffcc84")
                    if bool(getattr(self, "_search_expiry_loading_more", False)):
                        expiry_line.append(" ", style="dim")
                        expiry_line.append("[+...]", style="bold yellow")
                    elif bool(getattr(self, "_search_expiry_has_more", False)):
                        expiry_line.append(" ", style="dim")
                        expiry_line.append("[+more]", style="bold #86dca9")
                    expiry_line.append("  ([ / ])", style="dim")
                else:
                    expiry_line.append("n/a", style="dim")
                lines.append(expiry_line)
                side_line = Text("Side ", style="dim")
                if self._search_side == 0:
                    side_line.append("[CALL]", style="bold #0d1117 on #8fbfff")
                    side_line.append(" PUT", style="dim")
                else:
                    side_line.append("CALL ", style="dim")
                    side_line.append("[PUT]", style="bold #0d1117 on #8fbfff")
                side_line.append("  (left/right)", style="dim")
                lines.append(side_line)
                rows = self._option_pair_rows()
                total = len(rows)
                start = min(self._search_scroll, max(0, total - self._SEARCH_LIMIT))
                end = min(start + self._SEARCH_LIMIT, total)
                for idx in range(start, end):
                    call_contract, put_contract = rows[idx]
                    lines.append(
                        self._search_option_pair_line(
                            idx=idx,
                            call_contract=call_contract,
                            put_contract=put_contract,
                            active=idx == self._search_selected,
                        )
                    )
                selected_contract = self._selected_opt_contract()
                if selected_contract is not None:
                    selected_label = self._search_label_for_contract(selected_contract)
                    selected_line = Text("Selected ", style="dim")
                    selected_line.append(
                        self._search_contract_description(selected_contract, label=selected_label),
                        style="bold #9fd7ff",
                    )
                    lines.append(selected_line)
                if total > self._SEARCH_LIMIT:
                    lines.append(Text(f"Rows {start + 1}-{end}/{total}", style="dim"))
        else:
            if not self._search_results:
                lines.append(Text("No matches", style="dim"))
                self._search.update(Text("\n").join(lines))
                return
            total = len(self._search_results)
            start = min(self._search_scroll, max(0, total - self._SEARCH_LIMIT))
            end = min(start + self._SEARCH_LIMIT, total)
            for idx in range(start, end):
                lines.append(
                    self._search_result_line(
                        self._search_results[idx],
                        row=idx,
                        active=idx == self._search_selected,
                    )
                )
            if total > self._SEARCH_LIMIT:
                lines.append(Text(f"Rows {start + 1}-{end}/{total}", style="dim"))
        self._search.update(Text("\n").join(lines))

    def _search_result_line(self, contract: Contract, *, row: int, active: bool) -> Text:
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        symbol = str(getattr(contract, "symbol", "") or "?").strip().upper() or "?"
        style = self._SECTION_HEADER_STYLE_BY_TYPE.get(sec_type, "bold white")
        line = Text(f"{row + 1}. ", style="dim")
        line.append(sec_type.ljust(3), style=style)
        line.append(" ")
        line.append(symbol, style="bold")
        label = self._search_label_for_symbol(symbol, sec_type=sec_type)
        if label:
            line.append("  •  ", style="dim")
            line.append(self._clip_cell(label, 34), style="bold #b7cadc")
        expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
        if sec_type in ("OPT", "FOP"):
            right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
            strike = getattr(contract, "strike", None)
            strike_text = ""
            if strike not in (None, ""):
                try:
                    strike_text = f"{float(strike):.2f}"
                except (TypeError, ValueError):
                    strike_text = str(strike)
            if expiry:
                line.append(f" {expiry}", style="dim")
            if right:
                line.append(f" {right}", style="bold")
            if strike_text:
                line.append(f" {strike_text}", style="bold")
        elif sec_type == "FUT" and expiry:
            line.append(f" {expiry}", style="dim")
        exchange = str(getattr(contract, "exchange", "") or "").strip().upper()
        if exchange:
            line.append(f"  {exchange}", style="dim")
        if active:
            line.stylize("bold on #1d2a38")
        return line

    @staticmethod
    def _clip_cell(text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) <= width:
            return text.ljust(width)
        if width == 1:
            return "…"
        return text[: width - 1] + "…"

    def _search_option_pair_line(
        self,
        *,
        idx: int,
        call_contract: Contract | None,
        put_contract: Contract | None,
        active: bool,
    ) -> Text:
        strike = self._option_row_strike(call_contract, put_contract)
        strike_text = f"{strike:.2f}" if strike is not None else "--"
        left_style = "bold #8fbfff" if call_contract else "dim"
        right_style = "bold #ff9ac2" if put_contract else "dim"
        if active and self._search_side == 0:
            left_style = f"{left_style} on #1d2a38"
        if active and self._search_side == 1:
            right_style = f"{right_style} on #1d2a38"

        call_text = self._clip_cell(self._search_option_cell_text("C", call_contract), 10)
        put_text = self._clip_cell(self._search_option_cell_text("P", put_contract), 10)
        line = Text(f"{idx + 1}. ", style="dim")
        line.append(call_text, style=left_style)
        line.append("  |  ", style="dim")
        line.append(f"K={strike_text}", style="bold #ffcc84")
        line.append("  |  ", style="dim")
        line.append(put_text, style=right_style)
        return line

    @staticmethod
    def _search_contract_description(contract: Contract, *, label: str = "") -> str:
        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper() or "?"
        expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
        right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
        strike_raw = getattr(contract, "strike", None)
        strike_text = ""
        if strike_raw not in (None, ""):
            try:
                strike_text = f"{float(strike_raw):.2f}"
            except (TypeError, ValueError):
                strike_text = str(strike_raw)
        local_symbol = str(getattr(contract, "localSymbol", "") or "").strip()
        exchange = str(getattr(contract, "exchange", "") or "").strip().upper()
        con_id = int(getattr(contract, "conId", 0) or 0)
        summary_parts: list[str] = [symbol]
        if expiry:
            summary_parts.append(expiry)
        if sec_type in ("OPT", "FOP"):
            if right == "C":
                summary_parts.append("CALL")
            elif right == "P":
                summary_parts.append("PUT")
            elif right:
                summary_parts.append(right)
            if strike_text:
                summary_parts.append(strike_text)
        summary = " ".join(part for part in summary_parts if part) or symbol
        clean_label = str(label or "").strip()
        if clean_label and clean_label.upper() != summary.upper():
            summary = f"{summary}  •  {clean_label}"
        detail_parts: list[str] = []
        if local_symbol and local_symbol.upper() != summary.upper():
            detail_parts.append(local_symbol)
        if exchange:
            detail_parts.append(exchange)
        if con_id > 0:
            detail_parts.append(f"conId {con_id}")
        if detail_parts:
            return f"{summary}  |  {'  |  '.join(detail_parts)}"
        return summary

    def _selected_opt_contract(self) -> Contract | None:
        rows = self._option_pair_rows()
        if not rows:
            return None
        index = min(max(self._search_selected, 0), len(rows) - 1)
        call_contract, put_contract = rows[index]
        contract = call_contract if self._search_side == 0 else put_contract
        if contract is None:
            contract = put_contract if self._search_side == 0 else call_contract
        return contract

    def _selected_search_contract(self) -> Contract | None:
        if not self._search_results:
            return None
        if self._search_option_sec_type() is not None:
            return self._selected_opt_contract()
        index = min(max(self._search_selected, 0), len(self._search_results) - 1)
        return self._search_results[index]

    def _open_search_selection(self) -> None:
        contract = self._selected_search_contract()
        if contract is None:
            return
        item = self._portfolio_item_for_contract(contract)
        self._close_search()
        self.push_screen(
            PositionDetailScreen(
                self._client,
                item,
                self._config.detail_refresh_sec,
            )
        )

    def _portfolio_item_for_contract(self, contract: Contract) -> PortfolioItem:
        con_id = int(getattr(contract, "conId", 0) or 0)
        if con_id:
            for item in self._snapshot.items:
                if int(getattr(getattr(item, "contract", None), "conId", 0) or 0) == con_id:
                    return item

        def _norm_expiry(raw: object) -> str:
            text = str(raw or "").strip()
            if len(text) >= 8 and text[:8].isdigit():
                return text[:8]
            if len(text) >= 6 and text[:6].isdigit():
                return text[:6]
            return text

        def _exchange_set(value: object) -> set[str]:
            out: set[str] = set()
            for attr in ("exchange", "primaryExchange"):
                text = str(getattr(value, attr, "") or "").strip().upper()
                if text:
                    out.add(text)
            return out

        def _strike(value: object) -> float | None:
            raw = getattr(value, "strike", None)
            if raw is None:
                return None
            try:
                number = float(raw)
            except (TypeError, ValueError):
                return None
            return number if number > 0 else None

        sec_type = str(getattr(contract, "secType", "") or "").strip().upper()
        symbol = str(getattr(contract, "symbol", "") or "").strip().upper()
        target_expiry = _norm_expiry(getattr(contract, "lastTradeDateOrContractMonth", ""))
        target_right = str(getattr(contract, "right", "") or "").strip().upper()[:1]
        target_strike = _strike(contract)
        target_exchanges = _exchange_set(contract)
        for item in self._snapshot.items:
            existing = getattr(item, "contract", None)
            if existing is None:
                continue
            if str(getattr(existing, "secType", "") or "").strip().upper() != sec_type:
                continue
            if str(getattr(existing, "symbol", "") or "").strip().upper() != symbol:
                continue
            if sec_type == "FUT":
                existing_expiry = _norm_expiry(getattr(existing, "lastTradeDateOrContractMonth", ""))
                if target_expiry and existing_expiry and target_expiry != existing_expiry:
                    continue
                existing_exchanges = _exchange_set(existing)
                if target_exchanges and existing_exchanges and target_exchanges.isdisjoint(existing_exchanges):
                    continue
            elif sec_type in ("OPT", "FOP"):
                existing_expiry = _norm_expiry(getattr(existing, "lastTradeDateOrContractMonth", ""))
                if target_expiry and existing_expiry and target_expiry != existing_expiry:
                    continue
                existing_right = str(getattr(existing, "right", "") or "").strip().upper()[:1]
                if target_right and existing_right and target_right != existing_right:
                    continue
                existing_strike = _strike(existing)
                if (
                    target_strike is not None
                    and existing_strike is not None
                    and abs(target_strike - existing_strike) > 1e-6
                ):
                    continue
            return item
        return cast(PortfolioItem, _SyntheticPortfolioItem(contract=contract))
