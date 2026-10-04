"""Threads may parse side by side only within a byte budget.

A parse holds several times its document's size in memory while it runs, so
the total size of the documents being parsed at once is bounded. A document
larger than the whole budget is parsed alone, and documents are admitted in
arrival order so a large one is not starved by a stream of small ones.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from fastfeedparser import main, parse

_WAIT_SECONDS = 10


def _feed(padding: int = 0) -> bytes:
    return (
        '<rss version="2.0"><channel><title>t</title>'
        f"<item><title>e</title><description>{'x' * padding}</description></item>"
        "</channel></rss>"
    ).encode()


_SMALL = _feed()
_LARGE = _feed(4 * len(_SMALL))


class _Tracker:
    """Wrap _parse_content to record what is in flight and hold parses open."""

    def __init__(self, monkeypatch, hold_seconds: float = 0.0):
        self.lock = threading.Lock()
        self.bytes_now = 0
        self.parses_now = 0
        self.peak_bytes = 0
        self.peak_parses = 0
        self.admitted = []
        self.hold_seconds = hold_seconds
        self.gate = None
        real = main._parse_content

        def tracking(content, **options):
            with self.lock:
                self.bytes_now += len(content)
                self.parses_now += 1
                self.peak_bytes = max(self.peak_bytes, self.bytes_now)
                self.peak_parses = max(self.peak_parses, self.parses_now)
                self.admitted.append(threading.current_thread().name)
            try:
                if self.gate is not None and threading.current_thread().name == "held":
                    assert self.gate.wait(timeout=_WAIT_SECONDS)
                if self.hold_seconds:
                    time.sleep(self.hold_seconds)
                return real(content, **options)
            finally:
                with self.lock:
                    self.bytes_now -= len(content)
                    self.parses_now -= 1

        monkeypatch.setattr(main, "_parse_content", tracking)


def _run_threads(targets):
    errors = []

    def guarded(target):
        try:
            target()
        except Exception as e:
            errors.append(e)

    workers = [
        threading.Thread(target=guarded, args=(target,), name=name)
        for name, target in targets
    ]
    for worker in workers:
        worker.start()
    return workers, errors


def _join(workers, errors):
    for worker in workers:
        worker.join(timeout=_WAIT_SECONDS)
    assert not [worker.name for worker in workers if worker.is_alive()]
    assert not errors


def _wait_until(condition):
    deadline = time.monotonic() + _WAIT_SECONDS
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.001)


def test_documents_in_flight_stay_within_the_budget(monkeypatch):
    monkeypatch.setattr(main, "_MAX_BYTES_IN_FLIGHT", 2 * len(_SMALL))
    tracker = _Tracker(monkeypatch, hold_seconds=0.01)
    start = threading.Barrier(8)

    def work():
        start.wait(timeout=_WAIT_SECONDS)
        for _ in range(2):
            assert [entry.title for entry in parse(_SMALL).entries] == ["e"]

    _join(*_run_threads([(f"w{i}", work) for i in range(8)]))
    assert tracker.peak_parses == 2
    assert tracker.peak_bytes == 2 * len(_SMALL)


def test_document_larger_than_the_budget_is_parsed_alone(monkeypatch):
    monkeypatch.setattr(main, "_MAX_BYTES_IN_FLIGHT", 1)
    tracker = _Tracker(monkeypatch, hold_seconds=0.005)
    start = threading.Barrier(4)

    def work():
        start.wait(timeout=_WAIT_SECONDS)
        assert [entry.title for entry in parse(_SMALL).entries] == ["e"]

    _join(*_run_threads([(f"w{i}", work) for i in range(4)]))
    assert tracker.peak_parses == 1
    assert len(tracker.admitted) == 4


def test_large_document_is_admitted_before_later_small_ones(monkeypatch):
    # The budget fits the large document only once nothing else is in flight.
    monkeypatch.setattr(main, "_MAX_BYTES_IN_FLIGHT", len(_LARGE))
    tracker = _Tracker(monkeypatch)
    tracker.gate = threading.Event()

    def parse_small():
        parse(_SMALL)

    def parse_large():
        parse(_LARGE)

    held, errors = _run_threads([("held", parse_small)])
    _wait_until(lambda: tracker.admitted == ["held"])
    large, large_errors = _run_threads([("large", parse_large)])
    _wait_until(lambda: len(main._IN_FLIGHT.waiting) == 1)
    later, later_errors = _run_threads([(f"later{i}", parse_small) for i in range(3)])
    _wait_until(lambda: len(main._IN_FLIGHT.waiting) == 4)
    # The small ones would fit beside the held parse, but they wait their turn.
    assert tracker.admitted == ["held"]

    tracker.gate.set()
    _join(held + large + later, errors + large_errors + later_errors)
    assert tracker.admitted[:2] == ["held", "large"]
    assert sorted(tracker.admitted[2:]) == ["later0", "later1", "later2"]


def test_a_parse_that_raises_gives_its_bytes_back(monkeypatch):
    monkeypatch.setattr(main, "_MAX_BYTES_IN_FLIGHT", len(_SMALL))
    for bad in (b"<rss", b"not xml at all", b""):
        with pytest.raises(ValueError):
            parse(bad)
    assert main._IN_FLIGHT.total == 0
    assert not main._IN_FLIGHT.sizes
    assert not main._IN_FLIGHT.waiting
    # A document that needs the whole budget is admitted straight away.
    assert [entry.title for entry in parse(_SMALL).entries] == ["e"]


@pytest.mark.parametrize("admitted_first", [False, True])
def test_a_waiter_that_is_interrupted_does_not_stay_counted(admitted_first, monkeypatch):
    monkeypatch.setattr(main, "_MAX_BYTES_IN_FLIGHT", len(_SMALL))
    tracker = _Tracker(monkeypatch)
    tracker.gate = threading.Event()
    held, errors = _run_threads([("held", lambda: parse(_SMALL))])
    _wait_until(lambda: tracker.admitted == ["held"])

    real_event = threading.Event

    class InterruptedEvent(real_event):
        def wait(self, timeout=None):
            if admitted_first:
                # Let the held parse finish; as it leaves it admits this
                # waiter, and only then does the interrupt arrive.
                tracker.gate.set()
                assert real_event.wait(self, _WAIT_SECONDS)
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(main.threading, "Event", InterruptedEvent)
        with pytest.raises(KeyboardInterrupt):
            parse(_SMALL)
    assert not main._IN_FLIGHT.waiting

    tracker.gate.set()
    _join(held, errors)
    assert main._IN_FLIGHT.total == 0
    assert not main._IN_FLIGHT.sizes
    assert [entry.title for entry in parse(_SMALL).entries] == ["e"]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_does_not_inherit_parses_in_flight():
    # The parent forks while another of its threads is inside a parse. That
    # thread does not exist in the child, so its bytes must not count there.
    script = textwrap.dedent(
        """
        import os, sys, threading
        from fastfeedparser import main, parse

        feed = (b'<rss version="2.0"><channel><title>t</title>'
                b"<item><title>e</title></item></channel></rss>")
        main._MAX_BYTES_IN_FLIGHT = len(feed)
        inside, release = threading.Event(), threading.Event()
        real = main._parse_content

        def held(content, **options):
            if threading.current_thread().name == "held":
                inside.set()
                release.wait(10)
            return real(content, **options)

        main._parse_content = held
        worker = threading.Thread(target=parse, args=(feed,), name="held")
        worker.start()
        assert inside.wait(10)
        pid = os.fork()
        if pid == 0:
            ok = [entry.title for entry in parse(feed).entries] == ["e"]
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
        release.set()
        worker.join(10)
        sys.exit(os.WEXITSTATUS(status))
        """
    )
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = dict(os.environ, PYTHONPATH=src)
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
