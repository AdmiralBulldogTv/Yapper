import io
from contextlib import contextmanager


@contextmanager
def buffer():
    buf = io.BytesIO()
    try:
        yield buf
    finally:
        buf.seek(0)
        buf.truncate(0)
        buf.close()
