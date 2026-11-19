"""Generate messy, realistic Indonesian sales files for testing column-role detection.

Run: python samples/make_messy.py   -> samples/messy/*.csv|xlsx (seeded, so the files are stable)
They imitate common layouts (POS export, marketplace orders, stock report, a hand-made shop Excel, supplier
purchases); they are synthetic and not any vendor's exact format. The correct role of every column is in
eval/column_roles.json.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

OUT = Path(__file__).parent / "messy"
DAYS = pd.date_range("2026-05-01", "2026-06-30", freq="D")
BULAN = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober",
         "November", "Desember"]


def pos_kasir_export(r) -> pd.DataFrame:
    """Cafe POS export: one row per item line, several lines per receipt."""
    menu = {"Minuman": [("Es Kopi Susu", 22_000), ("Kopi Hitam", 15_000), ("Teh Manis", 8_000), ("Matcha Latte", 28_000)],
            "Makanan": [("Roti Bakar", 18_000), ("Mie Goreng", 20_000), ("Nasi Goreng", 25_000)],
            "Snack": [("Kentang Goreng", 15_000), ("Pisang Goreng", 12_000)]}
    rows, n = [], 0
    for day in DAYS:
        for _ in range(int(r.integers(4, 9))):
            n += 1
            receipt = f"STR-{day:%Y%m%d}-{n:04d}"
            when = f"{int(r.integers(8, 22)):02d}:{int(r.integers(0, 60)):02d}"
            outlet = r.choice(["Cabang Kemang", "Cabang Depok"])
            pay = r.choice(["Tunai", "QRIS", "Debit BCA", "GoPay", "OVO"], p=[0.25, 0.35, 0.1, 0.2, 0.1])
            cashier = r.choice(["Kasir A", "Kasir B", "Kasir C"])
            for _ in range(int(r.integers(1, 4))):
                cat = r.choice(list(menu))
                item, price = menu[cat][int(r.integers(len(menu[cat])))]
                variant = r.choice(["Reguler", "Large", ""]) if cat == "Minuman" else ""
                price += 5_000 if variant == "Large" else 0
                qty = int(r.integers(1, 4))
                disc = int(r.choice([0, 0, 0, 2_000, 5_000]))
                rows.append({"Tanggal": f"{day:%d/%m/%Y}", "Jam": when, "No. Struk": receipt, "Outlet": outlet,
                             "Kategori": cat, "Nama Item": item, "Varian": variant, "Qty": qty,
                             "Harga Satuan": price, "Diskon": disc or None, "Subtotal": qty * price - disc,
                             "Metode Pembayaran": pay, "Kasir": cashier})
    return pd.DataFrame(rows)


def pesanan_marketplace(r) -> pd.DataFrame:
    """Online shop order export: order status, original vs discounted price, shipping, buyer city."""
    products = [("Kaos Polos Hitam", "KPH-01", 65_000), ("Kemeja Flanel", "KMF-02", 145_000),
                ("Celana Chino", "CCN-03", 175_000), ("Topi Baseball", "TPB-04", 45_000),
                ("Tas Selempang", "TSL-05", 120_000), ("Kaos Kaki 3 Pasang", "KKK-06", 35_000)]
    cities = [("Kota Jakarta Selatan", "DKI Jakarta"), ("Kota Bandung", "Jawa Barat"), ("Kota Surabaya", "Jawa Timur"),
              ("Kab. Sleman", "DI Yogyakarta"), ("Kota Medan", "Sumatera Utara"), ("Kota Makassar", "Sulawesi Selatan")]
    rows = []
    for _ in range(600):
        day = DAYS[int(r.integers(len(DAYS)))]
        name, sku, price = products[int(r.integers(len(products)))]
        qty = int(r.choice([1, 1, 1, 2, 2, 3]))
        disc_price = int(price * r.choice([1, 1, 0.9, 0.85]))
        ship = int(r.choice([0, 9_000, 12_000, 18_000, 24_000]))
        city, prov = cities[int(r.integers(len(cities)))]
        rows.append({"No. Pesanan": f"2605{int(r.integers(10**9, 10**10))}", "Status Pesanan": r.choice(
                         ["Selesai", "Selesai", "Selesai", "Dikirim", "Dibatalkan", "Belum Bayar"]),
                     "Waktu Pesanan Dibuat": f"{day:%Y-%m-%d} {int(r.integers(0, 24)):02d}:{int(r.integers(0, 60)):02d}",
                     "Nama Produk": name, "SKU": sku, "Jumlah": qty, "Harga Awal": price,
                     "Harga Setelah Diskon": disc_price, "Total Harga Produk": qty * disc_price, "Ongkos Kirim": ship,
                     "Total Pembayaran": qty * disc_price + ship, "Kota/Kabupaten": city, "Provinsi": prov,
                     "Metode Pembayaran": r.choice(["Transfer Bank", "COD", "Dompet Digital", "Kartu Kredit"]),
                     "Username Pembeli": f"pembeli_{int(r.integers(1, 400)):04d}"})
    return pd.DataFrame(rows)


def laporan_stok(r) -> pd.DataFrame:
    """Weekly stock movement per item: opening, in, out, closing."""
    items = [("BRG-001", "Beras 5kg", "karung"), ("BRG-002", "Minyak Goreng 2L", "pcs"), ("BRG-003", "Gula 1kg", "pcs"),
             ("BRG-004", "Telur", "kg"), ("BRG-005", "Mie Instan", "dus"), ("BRG-006", "Air Mineral", "dus"),
             ("BRG-007", "Sabun Mandi", "pcs"), ("BRG-008", "Kopi Sachet", "renceng")]
    rows = []
    for code, name, unit in items:
        stock = int(r.integers(20, 80))
        for week_end in pd.date_range("2026-05-03", "2026-06-28", freq="7D"):
            masuk = int(r.choice([0, 0, 24, 36, 48]))
            keluar = min(stock + masuk, int(r.integers(5, 40)))
            end = stock + masuk - keluar
            rows.append({"Tanggal": f"{week_end:%d-%m-%Y}", "Kode Barang": code, "Nama Barang": name, "Satuan": unit,
                         "Stok Awal": stock, "Masuk": masuk, "Keluar": keluar, "Stok Akhir": end,
                         "Keterangan": "Stok menipis" if end < 10 else ""})
            stock = end
    return pd.DataFrame(rows)


def pembelian_supplier(r) -> pd.DataFrame:
    """Purchases from suppliers: invoice, cost price, payment terms."""
    suppliers = ["PT Sumber Rejeki", "CV Maju Jaya", "UD Berkah Abadi", "Toko Grosir Sentosa"]
    goods = [("Beras 5kg", "karung", 68_000), ("Minyak Goreng 2L", "pcs", 34_000), ("Gula 1kg", "pcs", 15_500),
             ("Mie Instan", "dus", 110_000), ("Kopi Sachet", "renceng", 11_000), ("Sabun Mandi", "lusin", 40_000)]
    rows = []
    for i in range(160):
        day = DAYS[int(r.integers(len(DAYS)))]
        name, unit, cost = goods[int(r.integers(len(goods)))]
        qty = int(r.integers(2, 30))
        terms = r.choice(["Lunas", "Lunas", "Tempo 14 hari", "Tempo 30 hari"])
        due = "" if terms == "Lunas" else f"{day + pd.Timedelta(days=int(terms.split()[1])):%d/%m/%Y}"
        rows.append({"Tgl Beli": f"{day:%d/%m/%Y}", "No. Faktur": f"INV/{day:%y%m}/{i + 1:04d}",
                     "Supplier": r.choice(suppliers), "Nama Barang": name, "Qty": qty, "Satuan": unit,
                     "Harga Beli": cost, "Total": qty * cost, "Status Bayar": terms, "Jatuh Tempo": due})
    return pd.DataFrame(rows)


def buku_penjualan_manual(r, path: Path) -> None:
    """A shop's own Excel: title rows, merged cells, money as text, mixed date styles, daily TOTAL rows, blanks."""
    goods = [("Beras 5kg", 75_000), ("Minyak 2L", 38_000), ("Gula 1kg", 18_000), ("Telur 1kg", 30_000),
             ("Indomie", 3_500), ("Aqua 600ml", 3_500), ("Kopi sachet", 1_500), ("Rokok", 32_000)]
    wb = Workbook()
    ws = wb.active
    ws.title = "Mei 2026"
    ws["A1"] = "LAPORAN PENJUALAN TOKO BERKAH"
    ws["A2"] = "Bulan: Mei 2026"
    ws.merge_cells("A1:G1")
    ws.merge_cells("A2:G2")
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    ws.append(["No", "Tgl", "Nama Barang", "Jml", "Harga", "Total", "Ket"])
    no = 0
    for day in pd.date_range("2026-05-01", "2026-05-31", freq="D"):
        day_total = 0
        for _ in range(int(r.integers(3, 8))):
            no += 1
            name, price = goods[int(r.integers(len(goods)))]
            qty = int(r.integers(1, 6))
            style = int(r.integers(3))  # the same shop writes dates and money three different ways
            tgl = [f"{day.day} {BULAN[day.month]} {day.year}", f"{day:%d/%m/%Y}", f"{day:%d-%m-%y}"][style]
            harga = [f"Rp {price:,.0f}".replace(",", "."), price, f"{price // 1000}rb" if price % 1000 == 0 else price][style]
            total = qty * price
            day_total += total
            ws.append([no, tgl, name, qty, harga, f"Rp {total:,.0f}".replace(",", ".") if style == 0 else total,
                       r.choice(["", "", "", "hutang", "lunas"])])
        ws.append(["", "", "TOTAL", "", "", day_total, ""])
        if r.random() < 0.3:
            ws.append([])
    wb.save(path)


def heldout_kasir_v2(r) -> pd.DataFrame:
    """Held-out: another POS naming style. Used only for the final score, never while tuning."""
    goods = [("Bakso Urat", 25_000), ("Es Teh Jumbo", 7_000), ("Mie Ayam", 18_000), ("Es Jeruk", 9_000),
             ("Pangsit Goreng", 12_000)]
    rows = []
    for i in range(300):
        day = DAYS[int(r.integers(len(DAYS)))]
        name, price = goods[int(r.integers(len(goods)))]
        qty = int(r.integers(1, 5))
        rows.append({"Tgl Transaksi": f"{day:%d/%m/%Y}", "Kode Struk": f"TRX{i + 1:05d}", "Lokasi": r.choice(
                         ["Warung Pusat", "Warung Cabang"]), "Produk": name, "Jumlah Item": qty,
                     "Harga": price, "Total Bayar": qty * price, "Pembayaran": r.choice(["Cash", "QRIS", "Transfer"]),
                     "Pelanggan": f"Pelanggan {int(r.integers(1, 60))}"})
    return pd.DataFrame(rows)


def heldout_orders_en(r) -> pd.DataFrame:
    """Held-out: English CamelCase export. Used only for the final score, never while tuning."""
    goods = [("Desk Lamp", 24.5), ("USB Cable", 6.0), ("Notebook A5", 3.25), ("Backpack", 39.9)]
    rows = []
    for i in range(300):
        day = DAYS[int(r.integers(len(DAYS)))]
        name, price = goods[int(r.integers(len(goods)))]
        units = int(r.integers(1, 12))
        rows.append({"OrderID": f"SO-{50000 + i}", "OrderDate": f"{day:%m/%d/%Y}", "ProductName": name,
                     "Units": units, "UnitPrice": price, "LineTotal": round(units * price, 2),
                     "Region": r.choice(["North", "South", "East", "West"]),
                     "SalesRep": r.choice(["Rep 1", "Rep 2", "Rep 3"]), "ShipMode": r.choice(["Standard", "Express"])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    held = np.random.default_rng(99)
    (OUT / "heldout").mkdir(exist_ok=True)
    heldout_kasir_v2(held).to_csv(OUT / "heldout" / "kasir_v2.csv", index=False)
    heldout_orders_en(held).to_csv(OUT / "heldout" / "orders_en.csv", index=False)
    r = np.random.default_rng(2026)
    pos_kasir_export(r).to_csv(OUT / "pos_kasir_export.csv", index=False)
    pesanan_marketplace(r).to_csv(OUT / "pesanan_marketplace.csv", index=False, sep=";")
    laporan_stok(r).to_csv(OUT / "laporan_stok.csv", index=False)
    pembelian_supplier(r).to_csv(OUT / "pembelian_supplier.csv", index=False)
    buku_penjualan_manual(r, OUT / "buku_penjualan_manual.xlsx")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))
