"""Apache Arrow memory-mapped DB — zero-copy worker sharing.

Faz 1.1 (v4 vision plan): Mevcut pickle-based pool 27 MB DB'yi her trial'a
kopyalıyor. Arrow memory-mapped dosya kullanarak tüm worker'lar aynı RAM
bölgesini paylaşır → zero-copy view, trial başlatma overhead'i sıfır.

Kullanım:
    from engine.data.arrow_db import MarketDB, materialize_arrow_db

    # Bir kez oluştur (mining başlamadan önce)
    arrow_path = materialize_arrow_db("data/market_db.parquet",
                                        out_path="data/market_db.arrow")

    # Worker içinde
    db = MarketDB(arrow_path)
    train_df = db.slice(start="2012-01-01", end="2015-12-31")  # zero-copy

Backward compat:
    pyarrow yoksa import hatası verir; çağıran kod try/except ile
    eski pickle yoluna düşer.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def _check_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyarrow.feather as feather
        return pa, pq, feather
    except ImportError as exc:
        raise ImportError(
            "pyarrow gerekli — `pip install pyarrow` ile yükle. "
            "Faz 1.1 hızlandırması için zorunlu."
        ) from exc


def materialize_arrow_db(parquet_path: str,
                          out_path: Optional[str] = None) -> str:
    """Parquet'i Arrow IPC formatına dönüştür (memory-mappable).

    Parquet sıkıştırılmış olduğu için memory-map için uygun değildir.
    Arrow IPC (Feather v2) uncompressed, columnar, mmap-friendly.

    Returns
    -------
    str : Arrow dosyasının yolu (varsa cached, yoksa yeni yazılan)
    """
    pa, pq, feather = _check_pyarrow()

    parquet_path = Path(parquet_path)
    if out_path is None:
        out_path = parquet_path.with_suffix(".arrow")
    out_path = Path(out_path)

    # Cache: Arrow zaten yeniyse yeniden yazma
    if out_path.exists() and out_path.stat().st_mtime >= parquet_path.stat().st_mtime:
        logger.info("Arrow DB cache hit: %s", out_path)
        return str(out_path)

    logger.info("Arrow DB oluşturuluyor: %s → %s", parquet_path, out_path)
    table = pq.read_table(str(parquet_path))
    feather.write_feather(table, str(out_path), compression="uncompressed")
    logger.info("Arrow DB hazır: %s (%.1f MB)", out_path,
                out_path.stat().st_size / 1e6)
    return str(out_path)


class MarketDB:
    """Memory-mapped Arrow Table wrapper.

    Tüm worker'lar aynı OS sayfa cache'ini paylaşır → tek kopya.
    """

    def __init__(self, arrow_path: str):
        pa, pq, feather = _check_pyarrow()
        self._path = arrow_path
        # memory_map=True → mmap, copy yok
        self._table = feather.read_table(arrow_path, memory_map=True)
        self._cached_pdf: Optional["pd.DataFrame"] = None
        logger.debug("MarketDB mmap açıldı: %s (%d satır × %d kolon)",
                     arrow_path, self._table.num_rows, self._table.num_columns)

    @property
    def table(self):
        """Raw Arrow Table (zero-copy)."""
        return self._table

    @property
    def num_rows(self) -> int:
        return self._table.num_rows

    @property
    def schema(self):
        return self._table.schema

    def slice(self, start: Union[str, "pd.Timestamp"],
              end: Union[str, "pd.Timestamp"]):
        """Tarih aralığında alt-tablo döndür (zero-copy filter).

        Returns Arrow Table; .to_pandas() ile DataFrame'e çevirilebilir.
        """
        import pandas as pd
        import pyarrow as pa
        import pyarrow.compute as pc

        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)

        date_col = self._table["Date"]
        # Arrow tablosundaki gerçek timestamp tipini kullan (s, ms, us, ns)
        date_type = date_col.type
        try:
            start_scalar = pa.scalar(start_ts, type=date_type)
            end_scalar   = pa.scalar(end_ts,   type=date_type)
        except Exception:
            # Tip eşleşmiyorsa cast et
            date_col = pc.cast(date_col, pa.timestamp("ns"))
            start_scalar = pa.scalar(start_ts.to_datetime64(), type=pa.timestamp("ns"))
            end_scalar   = pa.scalar(end_ts.to_datetime64(),   type=pa.timestamp("ns"))

        mask = pc.and_(
            pc.greater_equal(date_col, start_scalar),
            pc.less_equal(date_col, end_scalar),
        )
        return self._table.filter(mask)

    def to_pandas(self, cache: bool = True) -> "pd.DataFrame":
        """Tam DataFrame view (cached). Worker içinde bir kez çağrılır."""
        if self._cached_pdf is not None and cache:
            return self._cached_pdf
        df = self._table.to_pandas()
        if cache:
            self._cached_pdf = df
        return df

    def slice_pandas(self, start, end) -> "pd.DataFrame":
        """Tarih aralığını DataFrame olarak döndür (filter + to_pandas)."""
        return self.slice(start, end).to_pandas()


def pa_scalar(value):
    """pyarrow scalar wrapper (timestamp uyumlu)."""
    import pyarrow as pa
    import pandas as pd
    if isinstance(value, pd.Timestamp):
        return pa.scalar(value.to_datetime64(), type=pa.timestamp("ms"))
    return pa.scalar(value)
