"""
engine/db_builder.py - TÜM BIST EVRENİ (RONI'NİN TAM LİSTESİ) VERİTABANI İNŞA EDİCİ

Değişiklikler:
  N4  : shares_outstanding yfinance.info'dan çekilip market_db'ye eklendi.
        factor_neutralize.py log(price × shares) yerine log(price) kullanıyor →
        artık gerçek market-cap proxy mevcut.
  N9  : Exponential backoff (3 deneme) + missing-ticker JSON raporu.
"""
import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Ayarlar
DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)

# SENİN LİSTEN: Tek bir sembol bile eksik kalmadan buraya gömdük
TICKERS = [
    "XU100.IS", "THYAO.IS", "GARAN.IS", "ASELS.IS", "EREGL.IS", "KCHOL.IS", "SAHOL.IS", "SISE.IS", "TOASO.IS",
    "AKBNK.IS", "YKBNK.IS", "BIMAS.IS", "FROTO.IS", "PGSUS.IS", "TAVHL.IS", "ADESE.IS", "ADLVY.IS", "ADGYO.IS",
    "AFYON.IS", "AGHOL.IS", "AGESA.IS", "AGROT.IS", "AAGYO.IS", "AHSGY.IS", "AHGAZ.IS", "AKSFA.IS", "AKFK.IS",
    "AKM.IS", "AKMEN.IS", "AKCVR.IS", "AKCNS.IS", "AKDFA.IS", "AKYHO.IS", "AKENR.IS", "AKFGY.IS", "AKFIS.IS",
    "AKFYE.IS", "AKHAN.IS", "ATEKS.IS", "AKSGY.IS", "AKMGY.IS", "AKSA.IS", "AKSEN.IS", "AKGRT.IS", "AKSUE.IS",
    "AKTVK.IS", "AFB.IS", "AKTIF.IS", "ALCAR.IS", "ALGYO.IS", "ALARK.IS", "ALBRK.IS", "ALK.IS", "ALCTL.IS",
    "ALFAS.IS", "ALJF.IS", "ALKIM.IS", "ALKA.IS", "ALNUS.IS", "ANC.IS", "AYCES.IS", "ALTNY.IS", "ALKLC.IS",
    "ALVES.IS", "ANSGR.IS", "AEFES.IS", "ANHYT.IS", "ASUZU.IS", "ANGEN.IS", "ANELE.IS", "ARCLK.IS", "ARDYZ.IS",
    "ARENA.IS", "ARFYE.IS", "ARMGD.IS", "ARSAN.IS", "ARSVY.IS", "ARTMS.IS", "ARZUM.IS", "ASGYO.IS", "ASTOR.IS",
    "ATAGY.IS", "ATATR.IS", "ATAVK.IS", "ATA.IS", "ATAYM.IS", "ATAKP.IS", "AGYO.IS", "ATLFA.IS", "ATSYH.IS",
    "ATLAS.IS", "ATATP.IS", "AVOD.IS", "AVGYO.IS", "AVTUR.IS", "AVHOL.IS", "AVPGY.IS", "AYDEM.IS", "AYEN.IS",
    "AYES.IS", "AYGAZ.IS", "AZTEK.IS", "A1CAP.IS", "ACP.IS", "A1YEN.IS", "BAGFS.IS", "BAHKM.IS", "BAKAB.IS",
    "BALAT.IS", "BALSU.IS", "BNTAS.IS", "BANVT.IS", "BARMA.IS", "BSRFK.IS", "BASGZ.IS", "BASCM.IS", "BEGYO.IS",
    "BTCIM.IS", "BSOKE.IS", "BYDNR.IS", "BAYRK.IS", "BERA.IS", "BRKT.IS", "BRKSN.IS", "BESLR.IS", "BESTE.IS",
    "BJKAS.IS", "BEYAZ.IS", "BIENY.IS", "BIGTK.IS", "BLCYT.IS", "BLKOM.IS", "BINBN.IS", "BIOEN.IS", "BRKVY.IS",
    "BRKO.IS", "BIGEN.IS", "BRLSM.IS", "BRMEN.IS", "BIZIM.IS", "BLUME.IS", "BMSTL.IS", "BMSCH.IS", "BNPPI.IS",
    "BOBET.IS", "BORSK.IS", "BORLS.IS", "BRSAN.IS", "BRYAT.IS", "BFREN.IS", "BOSSA.IS", "BRISA.IS", "BULGS.IS",
    "BLS.IS", "BLSMD.IS", "BURCE.IS", "BURVA.IS", "BRGAN.IS", "BUR.IS", "BRGFK.IS", "BUCIM.IS", "BVSAN.IS",
    "BIGCH.IS", "CRFSA.IS", "CASA.IS", "CEMZY.IS", "CEOEM.IS", "CCOLA.IS", "CONSE.IS", "COSMO.IS", "CRDFA.IS",
    "CVKMD.IS", "CWENE.IS", "CGCAM.IS", "CAGFA.IS", "CMSAN.IS", "CANTE.IS", "CATES.IS", "CLEBI.IS", "CELHA.IS",
    "CLKMT.IS", "CEMAS.IS", "CEMTS.IS", "CMBTN.IS", "CMENT.IS", "CIMSA.IS", "CUSAN.IS", "DVRLK.IS", "DYBNK.IS",
    "DAGI.IS", "DAPGM.IS", "DARDL.IS", "DGATE.IS", "DCTTR.IS", "DGRVK.IS", "DMSAS.IS", "DENVA.IS", "DENGE.IS",
    "DNZEN.IS", "DENFA.IS", "DZGYO.IS", "DERIM.IS", "DERHL.IS", "DESA.IS", "DESPC.IS", "DSTKF.IS", "DMD.IS",
    "DSYAT.IS", "DEVA.IS", "DNISI.IS", "DIRIT.IS", "DITAS.IS", "DKVRL.IS", "DMRGD.IS", "DOCO.IS", "DOFRB.IS",
    "DOFER.IS", "DOHOL.IS", "DGNMO.IS", "DOGVY.IS", "ARASE.IS", "DOGUB.IS", "DGGYO.IS", "DOAS.IS", "DFKTR.IS",
    "DOKTA.IS", "DURDO.IS", "DURKN.IS", "DUNYH.IS", "DNYVA.IS", "DYOBY.IS", "EBEBK.IS", "ECOGR.IS", "ECZYT.IS",
    "EDATA.IS", "EDIP.IS", "EFOR.IS", "EGEEN.IS", "EGGUB.IS", "EGPRO.IS", "EGSER.IS", "EPLAS.IS", "EGEGY.IS",
    "ECZIP.IS", "ECILC.IS", "EKER.IS", "EKIZ.IS", "EKOFA.IS", "EKOS.IS", "EKSUN.IS", "ELITE.IS", "EMKEL.IS",
    "EMNIS.IS", "EMIRV.IS", "EKTVK.IS", "DMLKT.IS", "EKGYO.IS", "EMVAR.IS", "EMPAE.IS", "ENDAE.IS", "ENJSA.IS",
    "ENERY.IS", "ENKAI.IS", "ENPRA.IS", "ENSRI.IS", "ERBOS.IS", "ERCB.IS", "ERGLI.IS", "KIMMR.IS", "ERSU.IS",
    "ESCAR.IS", "ESCOM.IS", "ESEN.IS", "ETILR.IS", "EUKYO.IS", "EUYO.IS", "ETYAT.IS", "EUHOL.IS", "TEZOL.IS",
    "EUREN.IS", "EUPWR.IS", "EYGYO.IS", "FADE.IS", "FMIZP.IS", "FENER.IS", "FBB.IS", "FBBNK.IS", "FKPET.IS",
    "FLAP.IS", "FONET.IS", "FORMT.IS", "FRMPL.IS", "FORTE.IS", "FRIGO.IS", "FZLGY.IS", "GWIND.IS", "GSRAY.IS",
    "GARFA.IS", "GARFL.IS", "GRNYO.IS", "GATEG.IS", "GEDIK.IS", "GEDZA.IS", "GLCVY.IS", "GENIL.IS", "GENTS.IS",
    "GENKM.IS", "GEREL.IS", "GZNMI.IS", "GIPTA.IS", "GMTAS.IS", "GESAN.IS", "GLB.IS", "GLBMD.IS", "GLYHO.IS",
    "GGBVK.IS", "GSIPD.IS", "GOODY.IS", "GOKNR.IS", "GOLTS.IS", "GOZDE.IS", "GRTHO.IS", "GSDDE.IS", "GSDHO.IS",
    "GUBRF.IS", "GLRYH.IS", "GLRMK.IS", "GUNDG.IS", "GRSEL.IS", "HALKF.IS", "HLGYO.IS", "HLVKS.IS", "HALKI.IS",
    "HLY.IS", "HRKET.IS", "HATEK.IS", "HATSN.IS", "HAYVK.IS", "HDFFL.IS", "HDFGS.IS", "HEDEF.IS", "HDFVK.IS",
    "HDFYB.IS", "HYB.IS", "HEKTS.IS", "HKTM.IS", "HTTBT.IS", "HOROZ.IS", "HUBVC.IS", "HUNER.IS", "HUZFA.IS",
    "HURGZ.IS", "ENTRA.IS", "ICB.IS", "ICBCT.IS", "ICUGS.IS", "IAZ.IS", "INVAZ.IS", "INVES.IS", "ISKPL.IS",
    "IEYHO.IS", "IDGYO.IS", "IHEVA.IS", "IHLGM.IS", "IHGZT.IS", "IHAAS.IS", "IHLAS.IS", "IHYAY.IS", "IMASM.IS",
    "INDES.IS", "INFO.IS", "IYF.IS", "INTEK.IS", "INTEM.IS", "ISDMR.IS", "ISTFK.IS", "ISTVY.IS", "ISFAK.IS",
    "ISFIN.IS", "ISGYO.IS", "ISGSY.IS", "ISMEN.IS", "IYM.IS", "ISYAT.IS", "ISBIR.IS", "ISSEN.IS", "IZINV.IS",
    "IZENR.IS", "IZMDC.IS", "IZFAS.IS", "JANTS.IS", "KFEIN.IS", "KLKIM.IS", "KLSER.IS", "KLVKS.IS", "KLYPV.IS",
    "KTEST.IS", "KAPLM.IS", "KRDMA.IS", "KRDMB.IS", "KRDMD.IS", "KAREL.IS", "KARSN.IS", "KRTEK.IS", "KARTN.IS",
    "KATVK.IS", "KTLEV.IS", "KATMR.IS", "KFILO.IS", "KAYSE.IS", "KNTFA.IS", "KENT.IS", "KRVGD.IS", "KERVN.IS",
    "TCKRC.IS", "KZBGY.IS", "KLGYO.IS", "KLRHO.IS", "KMPUR.IS", "KLMSN.IS", "KCAER.IS", "KOCFN.IS", "KOCMT.IS",
    "KSFIN.IS", "KLSYN.IS", "KNFRT.IS", "KONTR.IS", "KONYA.IS", "KONKA.IS", "KGYO.IS", "KORDS.IS", "KRPLS.IS",
    "KORTS.IS", "KOTON.IS", "KOPOL.IS", "KRGYO.IS", "KRSTL.IS", "KRONT.IS", "KTKVK.IS", "KTSVK.IS", "KSTUR.IS",
    "KUVVA.IS", "KUYAS.IS", "KBORU.IS", "KZGYO.IS", "KUTPO.IS", "KTSKR.IS", "LIDER.IS", "LIDFA.IS", "LILAK.IS",
    "LMKDC.IS", "LINK.IS", "LOGO.IS", "LKMNH.IS", "LRSHO.IS", "LXGYO.IS", "LUKSK.IS", "LYDHO.IS", "LYDYE.IS",
    "MACKO.IS", "MAKIM.IS", "MAKTK.IS", "MANAS.IS", "MRBAS.IS", "MRS.IS", "MAGEN.IS", "MRMAG.IS", "MARKA.IS",
    "MARMR.IS", "MAALT.IS", "MRSHL.IS", "MRGYO.IS", "MARTI.IS", "MTRKS.IS", "MAVI.IS", "MZHLD.IS", "MEDTR.IS",
    "MEGMT.IS", "MEGAP.IS", "MEKAG.IS", "MEKMD.IS", "MSA.IS", "MNDRS.IS", "MEPET.IS", "MERCN.IS", "MRBKF.IS",
    "MBFTR.IS", "MERIT.IS", "MERKO.IS", "METRO.IS", "MTRYO.IS", "MCARD.IS", "MEYSU.IS", "MHRGY.IS", "MIATK.IS",
    "MDASM.IS", "MDS.IS", "MGROS.IS", "MINTF.IS", "MSGYO.IS", "MSY.IS", "MSYBN.IS", "MPARK.IS", "MMCAS.IS",
    "MNGFA.IS", "MOBTL.IS", "MOGAN.IS", "MNDTR.IS", "MOPAS.IS", "EGEPO.IS", "NATEN.IS", "NTGAZ.IS", "NTHOL.IS",
    "NETAS.IS", "NETCD.IS", "NIBAS.IS", "NUHCM.IS", "NUGYO.IS", "NURVK.IS", "NRBNK.IS", "NYB.IS", "OBAMS.IS",
    "OBASE.IS", "ODAS.IS", "ODINE.IS", "OFSYM.IS", "ONCSM.IS", "ONRYT.IS", "OPET.IS", "ORCAY.IS", "ORFIN.IS",
    "ORGE.IS", "ORMA.IS", "OSVKS.IS", "OMD.IS", "OSMEN.IS", "OSTIM.IS", "OTKAR.IS", "OTOKC.IS", "OTOSR.IS",
    "OTTO.IS", "OYAKC.IS", "OYA.IS", "OYYAT.IS", "OYAYO.IS", "OYLUM.IS", "OZKGY.IS", "OZATD.IS", "OZGYO.IS",
    "OZRDN.IS", "OZSUB.IS", "OZYSR.IS", "PAMEL.IS", "PNLSN.IS", "PAGYO.IS", "PAPIL.IS", "PRFFK.IS", "PRDGS.IS",
    "PRKME.IS", "PARSN.IS", "PBT.IS", "PBTR.IS", "PASEU.IS", "PSGYO.IS", "PAHOL.IS", "PATEK.IS", "PCILT.IS",
    "PEKGY.IS", "PENGD.IS", "PENTA.IS", "PSDTC.IS", "PETKM.IS", "PKENT.IS", "PETUN.IS", "PINSU.IS", "PNSUT.IS",
    "PKART.IS", "PLTUR.IS", "POLHO.IS", "POLTK.IS", "PRZMA.IS", "QFINF.IS", "QYATB.IS", "YBQ.IS", "QYHOL.IS",
    "FIN.IS", "QNBTR.IS", "QNBFF.IS", "QNBFK.IS", "QNBVK.IS", "QUAGR.IS", "QUFIN.IS", "RNPOL.IS", "RALYH.IS",
    "RAYSG.IS", "REEDR.IS", "RYGYO.IS", "RYSAS.IS", "RODRG.IS", "ROYAL.IS", "RGYAS.IS", "RTALB.IS", "RUBNS.IS",
    "RUZYE.IS", "SAFKR.IS", "SANEL.IS", "SNICA.IS", "SANFM.IS", "SANKO.IS", "SAMAT.IS", "SARKY.IS", "SASA.IS",
    "SVGYO.IS", "SAYAS.IS", "SDTTR.IS", "SEGMN.IS", "SEKUR.IS", "SELEC.IS", "SELVA.IS", "SERNT.IS", "SRVGY.IS",
    "SEYKM.IS", "SILVR.IS", "SNGYO.IS", "SKYLP.IS", "SMRTG.IS", "SMART.IS", "SODSN.IS", "SOKE.IS", "SKTAS.IS",
    "SONME.IS", "SNPAM.IS", "SUMAS.IS", "SUNTK.IS", "SURGY.IS", "SUWEN.IS", "SMRFA.IS", "SMRVA.IS", "SEKFK.IS",
    "SEGYO.IS", "SKY.IS", "SKYMD.IS", "SEK.IS", "SKBNK.IS", "SOKM.IS", "TABGD.IS", "TAC.IS", "TCRYT.IS",
    "TAMFA.IS", "TNZTP.IS", "TARKM.IS", "TATGD.IS", "TATEN.IS", "TAVHL.IS", "DRPHN.IS", "TEBFA.IS", "TEKTU.IS",
    "TKFEN.IS", "TKNSA.IS", "TMPOL.IS", "TRHOL.IS", "TEVKS.IS", "TAE.IS", "TRBNK.IS", "TERA.IS", "TRA.IS",
    "TEHOL.IS", "TFNVK.IS", "TGSAS.IS", "TIMUR.IS", "TRYKI.IS", "TRGYO.IS", "TRMET.IS", "TRENJ.IS", "TLMAN.IS",
    "TSPOR.IS", "TDGYO.IS", "TRMEN.IS", "TVM.IS", "TSGYO.IS", "TUCLK.IS", "TUKAS.IS", "TRCAS.IS", "TUREX.IS",
    "MARBL.IS", "TRKFN.IS", "TRILC.IS", "TCELL.IS", "TRKNT.IS", "TMSN.IS", "TRALT.IS", "PRKAB.IS", "TTKOM.IS",
    "TTRAK.IS", "TBORG.IS", "TURGG.IS", "TGB.IS", "THL.IS", "EXIMB.IS", "THR.IS", "ISATR.IS", "ISBTR.IS",
    "ISKUR.IS", "TIB.IS", "KLN.IS", "KLNMA.IS", "TSK.IS", "TSKB.IS", "TURSG.IS", "TVB.IS", "TV8TV.IS",
    "UFUK.IS", "ULAS.IS", "ULUFA.IS", "ULUSE.IS", "ULUUN.IS", "UMPAS.IS", "USAK.IS", "UCAYM.IS", "ULKER.IS",
    "UNLU.IS", "VAKFA.IS", "VAKFN.IS", "VKFYO.IS", "VAKVK.IS", "VAKKO.IS", "VANGD.IS", "VBTYZ.IS", "VDFLO.IS",
    "VRGYO.IS", "VERUS.IS", "VERTU.IS", "VESBE.IS", "VESTL.IS", "VKING.IS", "VSNMD.IS", "VDFAS.IS", "YKFKT.IS",
    "YKFIN.IS", "YKR.IS", "YKYAT.IS", "YKB.IS", "YAPRK.IS", "YATAS.IS", "YYLGD.IS", "YAYLA.IS", "YGGYO.IS",
    "YEOTK.IS", "YYAPI.IS", "YESIL.IS", "YBTAS.IS", "YIGIT.IS", "YONGA.IS", "YKSLN.IS", "YUNSA.IS", "ZGYO.IS",
    "ZEDUR.IS", "ZERGY.IS", "ZRGYO.IS", "ZKBVK.IS", "ZKBVR.IS", "ZOREN.IS", "BINHO.IS"
]

# N9: Exponential backoff parametreleri
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0   # saniye — 2, 4, 8 sn


def _download_with_retry(symbol: str, start_dt: str, end_dt: str) -> pd.DataFrame | None:
    """
    N9: yfinance.download + exponential backoff (3 deneme).
    Hata türü ne olursa olsun son denemede None döner.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            df = yf.download(
                symbol, start=start_dt, end=end_dt,
                auto_adjust=True, progress=False
            )
            if df is not None and not df.empty:
                return df
            return None
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_BASE ** (attempt + 1)
                time.sleep(delay)
    return None


def _fetch_shares_outstanding(tickers: list[str]) -> dict[str, float]:
    """
    N4: Her ticker için yfinance.Ticker.info["sharesOutstanding"] çek.

    Yavaş çağrıdır (ticker başına ~0.3s); build_database bir kez çağırır.
    Değer bulunamazsa veya hata alınırsa ticker atlanır (NaN → log(price) fallback).

    Returns
    -------
    dict[ticker_symbol, shares_outstanding_float]
    """
    result: dict[str, float] = {}
    n = len(tickers)
    for i, symbol in enumerate(tickers):
        print(f"[{i+1}/{n}] shares_outstanding çekiliyor: {symbol}", end="\r")
        for attempt in range(_MAX_RETRIES):
            try:
                info = yf.Ticker(symbol).info
                val = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
                if val and float(val) > 0:
                    result[symbol] = float(val)
                break
            except Exception:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_BASE ** (attempt + 1))
    print()  # satır sonu
    return result


def build_database(fetch_shares: bool = True) -> None:
    """
    BIST evrenini indir → market_db.parquet yaz.

    Parameters
    ----------
    fetch_shares : bool
        True → N4 shares_outstanding yfinance.info'dan çekilir ve
        her satıra "shares_outstanding" kolonu eklenir.
        False → sadece fiyat verisi (hızlı mod, mevcut davranış).
    """
    print(f"🚀 Toplam {len(TICKERS)} sembol için veritabanı inşası başlıyor...")
    start_time = datetime.now()

    # Zaman aralığı (Dinamik 5 Yıl)
    end_dt = datetime.today().strftime("%Y-%m-%d")
    start_dt = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    all_frames: list[pd.DataFrame] = []
    missing_tickers: list[dict] = []  # N9: missing-ticker raporu

    # ── Fiyat indirme ─────────────────────────────────────────────────
    for i, symbol in enumerate(TICKERS):
        print(f"[{i+1}/{len(TICKERS)}] ⬇️ İndiriliyor: {symbol}", end="\r")
        df = _download_with_retry(symbol, start_dt, end_dt)

        if df is None or len(df) < 50:
            missing_tickers.append({"symbol": symbol, "reason": "empty_or_short"})
            continue

        try:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            col_map = {
                "Open": "Popen", "High": "Phigh", "Low": "Plow",
                "Close": "Pclose", "Volume": "Vlot",
            }
            df = df.rename(columns=col_map)
            df["Ptyp"] = (df["Phigh"] + df["Plow"] + df["Pclose"]) / 3
            df = df[["Popen", "Phigh", "Plow", "Pclose", "Vlot", "Ptyp"]].copy()
            df["Ticker"] = symbol
            df.reset_index(inplace=True)
            all_frames.append(df)
        except Exception as exc:
            missing_tickers.append({"symbol": symbol, "reason": str(exc)})
            continue

    if not all_frames:
        print("\n❌ Hiç veri çekilemedi. Bağlantını kontrol et.")
        return

    print(f"\n\n📊 Birleştiriliyor...")
    final_db = pd.concat(all_frames, ignore_index=True)

    # Duplicate temizliği
    _n0 = len(final_db)
    final_db = final_db.drop_duplicates(subset=["Ticker", "Date"], keep="first")
    if len(final_db) < _n0:
        print(f"⚠️  {_n0 - len(final_db)} duplicate (Ticker,Date) satırı atıldı.")

    # ── N4: shares_outstanding ──────────────────────────────────────────
    if fetch_shares:
        print(f"\n📈 shares_outstanding çekiliyor ({len(TICKERS)} ticker)...")
        shares_map = _fetch_shares_outstanding(TICKERS)
        print(f"✅ {len(shares_map)} tickerda shares_outstanding bulundu.")

        final_db["shares_outstanding"] = final_db["Ticker"].map(shares_map)
        # NaN → float (factor_neutralize.py log(price) fallback'e düşer)
    else:
        final_db["shares_outstanding"] = float("nan")

    # ── Kaydet ────────────────────────────────────────────────────────
    print("💾 Parquet dosyasına yazılıyor...")
    db_path = os.path.join(DB_DIR, "market_db.parquet")
    final_db.to_parquet(db_path, index=False, engine="pyarrow")

    # N9: Missing-ticker raporu JSON
    if missing_tickers:
        report_path = os.path.join(DB_DIR, "missing_tickers.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(missing_tickers, f, ensure_ascii=False, indent=2)
        print(f"⚠️  {len(missing_tickers)} ticker eksik → {report_path}")

    elapsed = datetime.now() - start_time
    print(f"✅ İŞLEM TAMAMLANDI!")
    print(f"🔹 Toplam Satır: {len(final_db):,}")
    print(f"🔹 Başarılı Ticker: {len(all_frames)}")
    print(f"🔹 Eksik Ticker: {len(missing_tickers)}")
    print(f"🔹 Toplam Süre: {elapsed}")
    print(f"💾 Konum: {db_path}")


if __name__ == "__main__":
    build_database()
