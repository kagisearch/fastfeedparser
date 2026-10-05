"""Threads keep their own parsers only while the trees in flight fit a budget.

A document that would take the estimated memory past the budget is parsed
with the one shared pair of parsers, whose lock inside lxml makes such
documents parse one at a time. Nothing waits in Python, so no state left
behind by an interrupted parse can block a later one.
"""

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from fastfeedparser import main, parse

_WAIT_SECONDS = 20
_FEED = (
    b'<rss version="2.0"><channel><title>t</title>'
    b"<item><title>e</title></item></channel></rss>"
)


def _titles(feed=_FEED):
    return [entry.title for entry in parse(feed).entries]


class _OtherParse:
    """A parse held open in another thread, so the test thread is not alone."""

    def __init__(self, monkeypatch, hold_in="fromstring"):
        self.parsers = []
        self.inside = threading.Event()
        self.release = threading.Event()
        self.errors = []
        real_fromstring = main.etree.fromstring
        real_structure = main._detect_feed_structure

        def hold():
            self.inside.set()
            assert self.release.wait(timeout=_WAIT_SECONDS)

        def fromstring(text, parser=None):
            if threading.current_thread().name == "other":
                if hold_in == "fromstring":
                    hold()
            else:
                self.parsers.append(parser)
            return real_fromstring(text, parser=parser)

        def structure(*args, **kwargs):
            if threading.current_thread().name == "other" and hold_in == "after-parse":
                hold()
            return real_structure(*args, **kwargs)

        monkeypatch.setattr(main.etree, "fromstring", fromstring)
        monkeypatch.setattr(main, "_detect_feed_structure", structure)
        self.thread = threading.Thread(target=self._run, name="other")

    def _run(self):
        try:
            assert _titles() == ["e"]
        except Exception as e:
            self.errors.append(e)

    def __enter__(self):
        self.thread.start()
        assert self.inside.wait(timeout=_WAIT_SECONDS)
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.thread.join(timeout=_WAIT_SECONDS)
        assert not self.thread.is_alive()
        assert not self.errors


def _shared():
    return (main._SHARED_XML_PARSERS.recover, main._SHARED_XML_PARSERS.strict)


def test_thread_uses_its_own_parsers_while_the_budget_holds(monkeypatch):
    with _OtherParse(monkeypatch) as other:
        assert _titles() == ["e"]
        assert len(other.parsers) == 1
        assert other.parsers[0] not in _shared()
        assert other.parsers[0] is main._THREAD_XML_PARSERS.recover


def test_document_over_the_budget_uses_the_shared_parsers(monkeypatch):
    monkeypatch.setattr(main, "_MAX_TREE_BYTES_IN_FLIGHT", 1)
    with _OtherParse(monkeypatch) as other:
        assert _titles() == ["e"]
        assert other.parsers == [main._SHARED_XML_PARSERS.recover]


def test_strict_parse_over_the_budget_uses_the_shared_parsers(monkeypatch):
    monkeypatch.setattr(main, "_MAX_TREE_BYTES_IN_FLIGHT", 1)
    # This header looks malformed, which sends the document to the strict parser.
    feed = b'<?xml version="1.0" encoding="utf-16"?>' + _FEED
    with _OtherParse(monkeypatch) as other:
        assert _titles(feed) == ["e"]
        assert other.parsers == [main._SHARED_XML_PARSERS.strict]


def test_thread_parsing_alone_uses_its_own_parsers_whatever_the_size(monkeypatch):
    monkeypatch.setattr(main, "_MAX_TREE_BYTES_IN_FLIGHT", 1)
    parsers = []
    real_fromstring = main.etree.fromstring

    def fromstring(text, parser=None):
        parsers.append(parser)
        return real_fromstring(text, parser=parser)

    monkeypatch.setattr(main.etree, "fromstring", fromstring)
    assert _titles() == ["e"]
    assert parsers == [main._THREAD_XML_PARSERS.recover]


def test_slow_document_does_not_hold_up_other_parses(monkeypatch):
    # The other thread has parsed its document and is stuck working on it.
    monkeypatch.setattr(main, "_MAX_TREE_BYTES_IN_FLIGHT", 1)
    done = []

    def many():
        done.extend(_titles() for _ in range(50))

    with _OtherParse(monkeypatch, hold_in="after-parse") as other:
        worker = threading.Thread(target=many)
        worker.start()
        worker.join(timeout=_WAIT_SECONDS)
        assert not worker.is_alive()
        assert done == [["e"]] * 50
        assert other.thread.is_alive()


def test_entry_left_by_a_parse_that_never_finished_cannot_block(monkeypatch):
    # What an exception raised from a signal handler can leave behind when it
    # lands before the cleanup: another thread's entry, and this thread's own.
    monkeypatch.setitem(main._IN_FLIGHT, -1, 1 << 60)
    monkeypatch.setitem(main._IN_FLIGHT, threading.get_ident(), 1 << 60)
    worker = threading.Thread(target=_titles)
    worker.start()
    worker.join(timeout=_WAIT_SECONDS)
    assert not worker.is_alive()
    assert _titles() == ["e"]
    # This thread's stale entry is gone after its next parse.
    assert threading.get_ident() not in main._IN_FLIGHT


@pytest.mark.parametrize("bad", [b"<rss", b"not xml at all", b""])
def test_a_parse_that_raises_is_no_longer_in_flight(bad):
    with pytest.raises(ValueError):
        parse(bad)
    assert threading.get_ident() not in main._IN_FLIGHT


def test_parse_called_again_inside_a_parse_completes(monkeypatch):
    real_info = main._parse_feed_info
    inner = []

    def info_that_parses(*args, **kwargs):
        if not inner:
            inner.append(None)
            inner.append(_titles())
        return real_info(*args, **kwargs)

    monkeypatch.setattr(main, "_parse_feed_info", info_that_parses)
    assert _titles() == ["e"]
    assert inner == [None, ["e"]]
    assert threading.get_ident() not in main._IN_FLIGHT


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs signal.setitimer")
def test_exceptions_raised_from_a_signal_handler_leave_parse_working():
    # A timeout handler that raises can land anywhere inside parse(), cleanup
    # included. Whatever it leaves behind, later parses must still finish.
    class Timeout(Exception):
        pass

    armed = []

    def on_alarm(signum, frame):
        if armed:
            raise Timeout

    previous = signal.signal(signal.SIGALRM, on_alarm)
    interrupted = 0
    try:
        deadline = time.monotonic() + _WAIT_SECONDS
        delay = 0.00002
        while interrupted < 300 and time.monotonic() < deadline:
            try:
                armed.append(1)
                signal.setitimer(signal.ITIMER_REAL, delay)
                for _ in range(200):
                    parse(_FEED)
            except Timeout:
                interrupted += 1
            finally:
                del armed[:]
                signal.setitimer(signal.ITIMER_REAL, 0)
            delay = delay * 1.07 if delay < 0.002 else 0.00002
    finally:
        del armed[:]
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert interrupted

    results = []
    worker = threading.Thread(target=lambda: results.append(_titles()))
    worker.start()
    worker.join(timeout=_WAIT_SECONDS)
    assert results == [["e"]]
    assert _titles() == ["e"]


def test_estimate_counts_tags_as_well_as_bytes():
    text = b"<rss><channel><item><description>" + b"word " * 20000 + b"</description></item></channel></rss>"
    cell = b"<a><b>x</b></a>"
    dense = b"<rss><channel>" + cell * ((len(text) - 30) // len(cell)) + b"</channel></rss>"
    assert abs(len(dense) - len(text)) < 100
    assert main._estimated_tree_bytes(text) < 5 * len(text)
    assert main._estimated_tree_bytes(dense) > 10 * main._estimated_tree_bytes(text)
    assert main._estimated_tree_bytes(text + text) > 1.9 * main._estimated_tree_bytes(text)


def test_small_document_is_counted_whole():
    doc = b"<a>" * 100 + b"x" * 1000
    assert len(doc) <= 3 * main._TREE_SAMPLE_BYTES
    assert main._estimated_tree_bytes(doc) == 4 * len(doc) + 200 * 100


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_can_use_the_shared_parsers():
    # Another thread keeps the shared parsers busy while the parent forks. A
    # child that inherited their lock in the held state would never finish.
    script = textwrap.dedent(
        """
        import os, signal, sys, threading, time
        from fastfeedparser import main, parse

        item = b"<item><title>e</title><description>" + b"word " * 2000 + b"</description></item>"
        big = b'<rss version="2.0"><channel><title>t</title>' + item * 300 + b"</channel></rss>"
        small = b'<rss version="2.0"><channel><title>t</title><item><title>e</title></item></channel></rss>'
        main._MAX_TREE_BYTES_IN_FLIGHT = 1
        main._IN_FLIGHT[-1] = 0  # never alone, so every parse is over the budget
        stop = threading.Event()

        def churn():
            while not stop.is_set():
                parse(big)

        worker = threading.Thread(target=churn)
        worker.start()
        time.sleep(0.05)
        failed = 0
        for _ in range(40):
            pid = os.fork()
            if pid == 0:
                main._IN_FLIGHT[-1] = 0
                ok = [entry.title for entry in parse(small).entries] == ["e"]
                os._exit(0 if ok else 1)
            deadline = time.monotonic() + 10
            status = None
            while time.monotonic() < deadline:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    break
                time.sleep(0.005)
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                failed += 1
                break
            failed += os.WEXITSTATUS(status) != 0
        stop.set()
        worker.join(30)
        sys.exit(1 if failed else 0)
        """
    )
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        env=dict(os.environ, PYTHONPATH=src),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr


def test_first_thread_in_flight_is_counted(monkeypatch):
    # The other thread chose its parsers while it was alone. Its tree still
    # counts against a second document that arrives while it is in flight.
    one_document = main._estimated_tree_bytes(_FEED)
    monkeypatch.setattr(main, "_MAX_TREE_BYTES_IN_FLIGHT", int(1.5 * one_document))
    with _OtherParse(monkeypatch) as other:
        assert _titles() == ["e"]
        assert other.parsers == [main._SHARED_XML_PARSERS.recover]


def test_two_documents_within_the_budget_both_keep_their_own_parsers(monkeypatch):
    one_document = main._estimated_tree_bytes(_FEED)
    monkeypatch.setattr(main, "_MAX_TREE_BYTES_IN_FLIGHT", int(2.5 * one_document))
    with _OtherParse(monkeypatch) as other:
        assert _titles() == ["e"]
        assert other.parsers == [main._THREAD_XML_PARSERS.recover]


def test_parser_choice_outside_parse_leaves_no_entry():
    assert main._xml_parsers(_FEED) is main._THREAD_XML_PARSERS
    assert main._parse_xml_root(_FEED).tag == "rss"
    assert threading.get_ident() not in main._IN_FLIGHT
