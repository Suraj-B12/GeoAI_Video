"""frames._InOrder: the gate thread of the decode pass. Items are processed one at a time in submission order
(so outputs are those of a serial loop), the queue is bounded, and an exception in the worker reaches the
submitting thread. Expected values are the plain serial results."""
import threading
import time

import pytest

from geoai.frames import _InOrder


def test_items_run_in_order_on_one_other_thread():
    seen, threads = [], set()

    def fn(x):
        threads.add(threading.get_ident())
        time.sleep(0.001 * (x % 3))          # uneven work must not reorder anything
        seen.append(x * x)

    w = _InOrder(fn, 2, "t")
    for i in range(50):
        w.submit(i)
    w.close()
    assert seen == [i * i for i in range(50)]
    assert threads and threading.get_ident() not in threads and len(threads) == 1


def test_queue_is_bounded():
    gate = threading.Event()
    done = []
    w = _InOrder(lambda x: (gate.wait(), done.append(x)), 2, "t")
    t0 = time.perf_counter()
    w.submit(0)                               # taken by the worker, which waits
    time.sleep(0.05)
    w.submit(1)
    w.submit(2)                               # queue full now (2 waiting)
    blocked = threading.Thread(target=w.submit, args=(3,))
    blocked.start()
    time.sleep(0.1)
    assert blocked.is_alive()                 # a third waiting item blocks the submitter
    gate.set()
    blocked.join(5)
    w.close()
    assert done == [0, 1, 2, 3] and w.wait_s > 0 and time.perf_counter() - t0 < 5


def test_worker_exception_reaches_the_submitter_and_stops_processing():
    done = []

    def fn(x):
        if x == 3:
            raise ValueError("bad segment")
        done.append(x)

    w = _InOrder(fn, 1, "t")
    with pytest.raises(ValueError, match="bad segment"):
        for i in range(100):
            w.submit(i)
            time.sleep(0.002)
    w.abort()
    assert done == [0, 1, 2]
    w2 = _InOrder(fn, 8, "t")
    for i in range(5):
        w2.submit(i)
    with pytest.raises(ValueError, match="bad segment"):
        w2.close()
    assert done == [0, 1, 2, 0, 1, 2]         # nothing after the failing item ran


def test_abort_skips_what_is_queued():
    release = threading.Event()
    done = []
    w = _InOrder(lambda x: (release.wait(), done.append(x)), 10, "t")
    for i in range(6):
        w.submit(i)
    time.sleep(0.05)                          # item 0 is running, 1..5 wait
    threading.Timer(0.05, release.set).start()
    w.abort()
    assert done == [0]
