"""Column names commonly seen in sales / POS / marketplace / HR spreadsheets (English + Indonesian), by role.

Names are normalised before lookup: lower case, anything not a letter or digit becomes "_", and a currency or
percent suffix is dropped ("Harga Jual (Rp)" -> "harga_jual"). These are common patterns, not any one vendor's
exact export format; extend the lists when a real file uses a name that isn't here.
"""
from __future__ import annotations

import re

ROLES: dict[str, dict] = {
    # ---- when ----
    "date": {"type": "date", "desc": "when the transaction happened", "names": [
        "date", "tanggal", "tgl", "order_date", "transaction_date", "trx_date", "sale_date", "sales_date",
        "invoice_date", "purchase_date", "created_at", "created_date", "datetime", "timestamp", "waktu",
        "tanggal_transaksi", "tgl_transaksi", "tanggal_order", "tanggal_pesanan", "tgl_pesanan", "tanggal_penjualan",
        "tgl_penjualan", "tanggal_invoice", "tanggal_faktur", "periode", "period"]},
    "month": {"type": "label", "desc": "month: a name (Mei) or a number (5)", "names": ["month", "bulan", "month_name", "nama_bulan"]},
    "year": {"type": "number", "desc": "year", "names": ["year", "tahun"]},

    # ---- what was sold ----
    "transaction_id": {"type": "id", "desc": "one sale / order / receipt", "names": [
        "order_id", "order_no", "order_number", "transaction_id", "trx_id", "invoice", "invoice_no", "invoice_number",
        "receipt_no", "bill_no", "no_invoice", "nomor_invoice", "no_faktur", "nomor_faktur", "faktur", "no_struk",
        "struk", "no_nota", "nota", "no_transaksi", "nomor_transaksi", "id_transaksi", "kode_transaksi",
        "no_pesanan", "nomor_pesanan", "id_pesanan"]},
    "product": {"type": "text", "desc": "product / item / menu name", "names": [
        "product", "product_name", "item", "item_name", "item_description", "menu", "menu_name", "produk",
        "nama_produk", "barang", "nama_barang", "nama_item", "nama_menu"]},
    "sku": {"type": "id", "desc": "product code", "names": [
        "sku", "product_code", "item_code", "product_id", "item_id", "barcode", "plu", "kode_produk", "kode_barang",
        "kode_item", "id_produk", "id_barang"]},
    "category": {"type": "text", "desc": "product group", "names": [
        "category", "product_category", "item_category", "sub_category", "subcategory", "kategori",
        "kategori_produk", "kategori_barang", "sub_kategori", "subkategori", "jenis", "jenis_barang", "jenis_produk",
        "golongan", "kelompok", "kelompok_barang", "product_line", "lini_produk"]},
    "brand": {"type": "text", "desc": "brand", "names": ["brand", "brand_name", "merek", "merk", "nama_merek"]},

    # ---- how many / how much ----
    "quantity": {"type": "number", "desc": "units sold in the row", "names": [
        "quantity", "qty", "qty_sold", "quantity_sold", "units", "units_sold", "sold", "item_sold", "items_sold",
        "pcs", "terjual", "qty_terjual", "jumlah_terjual", "jumlah_barang", "jumlah_item", "kuantitas", "jml",
        "jml_terjual", "qty_keluar", "jumlah_keluar", "barang_keluar"]},
    "unit": {"type": "text", "desc": "unit of measure", "names": ["uom", "unit_of_measure", "satuan"]},
    "price": {"type": "money", "desc": "selling price per unit", "names": [
        "price", "unit_price", "selling_price", "sale_price", "sales_price", "retail_price", "item_price",
        "price_per_unit", "harga", "harga_jual", "harga_satuan", "harga_per_unit", "harga_unit", "harga_eceran"]},
    "cost": {"type": "money", "desc": "cost per unit (what the shop paid)", "names": [
        "cost", "unit_cost", "cost_price", "purchase_price", "buy_price", "cogs", "harga_modal", "modal",
        "harga_beli", "harga_pokok", "hpp", "harga_pokok_penjualan", "biaya", "biaya_per_unit"]},
    "revenue": {"type": "money", "desc": "money from the sale (row total)", "names": [
        "revenue", "sales", "total_sales", "gross_sales", "net_sales", "sales_amount", "total_amount",
        "grand_total", "subtotal", "sub_total", "turnover", "income", "line_total", "pendapatan", "penjualan",
        "total_penjualan", "nilai_penjualan", "omzet", "omset", "total_harga", "total_bayar", "jumlah_bayar",
        "total_transaksi", "nominal", "pemasukan"]},
    "discount": {"type": "money", "desc": "discount", "names": [
        "discount", "disc", "discount_amount", "promo", "voucher", "diskon", "potongan", "potongan_harga",
        "nilai_diskon"]},
    "tax": {"type": "money", "desc": "tax", "names": ["tax", "vat", "tax_amount", "pajak", "ppn", "pb1"]},
    "service_charge": {"type": "money", "desc": "service charge", "names": [
        "service", "service_charge", "biaya_layanan", "service_fee"]},
    "shipping": {"type": "money", "desc": "delivery cost", "names": [
        "shipping", "shipping_cost", "shipping_fee", "delivery_fee", "ongkir", "ongkos_kirim", "biaya_kirim"]},
    "profit": {"type": "money", "desc": "profit", "names": [
        "profit", "gross_profit", "net_profit", "margin", "gross_margin", "laba", "laba_kotor", "laba_bersih",
        "untung", "keuntungan", "margin_kotor"]},

    # ---- stock ----
    "stock": {"type": "number", "desc": "stock on hand after the row", "names": [
        "stock", "stock_on_hand", "on_hand", "qty_on_hand", "inventory", "ending_stock", "closing_stock",
        "available_stock", "available_qty", "stok", "stok_akhir", "sisa_stok", "stok_tersedia", "persediaan",
        "saldo_stok", "stok_saat_ini"]},
    "stock_in": {"type": "number", "desc": "units received / restocked", "names": [
        "stock_in", "qty_in", "received", "qty_received", "restock", "restock_qty", "stok_masuk", "qty_masuk",
        "jumlah_masuk", "barang_masuk", "pembelian"]},
    "opening_stock": {"type": "number", "desc": "stock at the start of the period", "names": [
        "opening_stock", "beginning_stock", "stok_awal", "saldo_awal"]},

    # ---- who / where / how ----
    "customer": {"type": "text", "desc": "customer", "names": [
        "customer", "customer_name", "customer_id", "client", "buyer", "member", "member_name", "pelanggan",
        "nama_pelanggan", "id_pelanggan", "pembeli", "nama_pembeli", "konsumen"]},
    "city": {"type": "text", "desc": "city / area (earlier names win when two columns match)", "names": [
        "city", "kota", "kota_kabupaten", "kabupaten", "kecamatan", "region", "area", "wilayah", "daerah",
        "province", "provinsi", "state"]},
    "country": {"type": "text", "desc": "country", "names": ["country", "negara", "country_name"]},
    "store": {"type": "text", "desc": "outlet / branch", "names": [
        "store", "store_name", "outlet", "outlet_name", "branch", "branch_name", "location", "shop", "cabang",
        "nama_outlet", "nama_toko", "nama_cabang", "lokasi", "gerai"]},
    "channel": {"type": "text", "desc": "where the sale came from", "names": [
        "channel", "sales_channel", "platform", "marketplace", "source", "order_source", "order_type",
        "saluran", "saluran_penjualan", "sumber", "sumber_pesanan", "tipe_order", "jenis_pesanan"]},
    "payment": {"type": "text", "desc": "how it was paid", "names": [
        "payment", "payment_method", "payment_type", "payment_mode", "pembayaran", "metode_pembayaran",
        "cara_bayar", "jenis_pembayaran", "tipe_pembayaran", "metode_bayar"]},
    "supplier": {"type": "text", "desc": "who the goods were bought from", "names": [
        "supplier", "supplier_name", "vendor", "vendor_name", "pemasok", "nama_supplier", "nama_pemasok",
        "distributor"]},
    "cashier": {"type": "text", "desc": "who made the sale", "names": [
        "cashier", "cashier_name", "salesperson", "sales_person", "sales_rep", "served_by", "kasir", "nama_kasir",
        "sales_name", "nama_sales"]},
    "status": {"type": "text", "desc": "order / payment status", "names": [
        "status", "order_status", "payment_status", "transaction_status", "status_pesanan", "status_pembayaran",
        "status_transaksi"]},

    # ---- people (HR) ----
    "employee_id": {"type": "id", "desc": "employee", "names": [
        "employee_id", "emp_id", "staff_id", "nik", "nip", "id_karyawan", "no_karyawan", "nomor_induk"]},
    "employee_name": {"type": "text", "desc": "employee name", "names": [
        "employee", "employee_name", "staff_name", "nama_karyawan", "nama_pegawai"]},
    "department": {"type": "text", "desc": "department / division", "names": [
        "department", "dept", "division", "team", "unit_kerja", "departemen", "divisi", "bagian"]},
    "position": {"type": "text", "desc": "job title", "names": [
        "position", "job_title", "title", "role", "level", "grade", "jabatan", "posisi", "golongan_jabatan"]},
    "salary": {"type": "money", "desc": "pay", "names": [
        "salary", "monthly_salary", "base_salary", "basic_salary", "wage", "pay", "gaji", "gaji_pokok",
        "gaji_bulanan", "upah", "take_home_pay", "thp"]},
    "hire_date": {"type": "date", "desc": "start date", "names": [
        "hire_date", "join_date", "start_date", "joined", "tanggal_masuk", "tgl_masuk", "tanggal_bergabung",
        "mulai_kerja"]},
    "age": {"type": "number", "desc": "age", "names": ["age", "customer_age", "umur", "usia"]},
    "gender": {"type": "text", "desc": "gender", "names": ["gender", "sex", "jenis_kelamin", "jk", "kelamin"]},
    "experience": {"type": "number", "desc": "years of experience / tenure", "names": [
        "years_experience", "experience", "years_of_experience", "tenure", "masa_kerja", "lama_bekerja",
        "pengalaman"]},
    "performance": {"type": "number", "desc": "performance score", "names": [
        "performance", "performance_score", "rating", "score", "kpi", "kpi_score", "nilai_kinerja", "kinerja"]},
    "training": {"type": "number", "desc": "training hours", "names": [
        "training_hours", "training", "hours_of_training", "jam_pelatihan", "pelatihan", "jam_training"]},
    "attrition": {"type": "text", "desc": "left the company (yes/no)", "names": [
        "attrition", "left", "left_company", "resigned", "terminated", "churn", "resign", "status_keluar"]},
}

# Names that really do mean different things: decide from the values (e.g. small whole numbers = quantity,
# large numbers = money) or ask the user.
AMBIGUOUS: dict[str, list[str]] = {
    "jumlah": ["quantity", "revenue"],  # "jumlah" is both "count" and "amount" in Indonesian invoices
    "total": ["revenue", "quantity"],
    "amount": ["revenue", "discount"],
    "unit": ["unit", "quantity"],
    "value": ["revenue", "price"],
    "nilai": ["revenue", "performance"],
    "type": ["category", "channel", "payment"],
    "tipe": ["category", "channel", "payment"],
    "keluar": ["quantity", "attrition"],
    "masuk": ["stock_in", "hire_date"],
    "id": ["transaction_id", "sku", "customer", "employee_id"],
    "name": ["product", "customer", "employee_name"],
    "nama": ["product", "customer", "employee_name"],
    "group": ["category", "department"],
    "kode": ["sku", "transaction_id"],
}

# The frequent question types, what they need ("a|b" = either role), and how people ask them.
INTENTS: dict[str, dict] = {
    "total": {"needs": ["revenue|profit|quantity"],
              "examples": ["What is the total revenue?", "Berapa total penjualan bulan ini?"]},
    "count": {"needs": [], "examples": ["How many orders are there?", "Berapa jumlah transaksi di Bandung?"]},
    "average": {"needs": ["revenue|price|salary|age"],
                "examples": ["What is the average order value?", "Berapa rata-rata gaji karyawan?"]},
    "top_group": {"needs": ["product|category|city|store|payment|channel|department", "revenue|quantity|profit"],
                  "examples": ["Which product sells the most?", "Produk apa yang paling laris?"]},
    "bottom_group": {"needs": ["product|category|city|store", "revenue|quantity|profit"],
                     "examples": ["Which product sells the least?", "Produk mana yang paling jarang terjual?"]},
    "filter_value": {"needs": ["city|store|category|channel|payment"],
                     "examples": ["Total revenue in Jakarta?", "Berapa pendapatan dari outlet Kemang?"]},
    "filter_period": {"needs": ["date"],
                      "examples": ["Revenue in March 2026?", "Laba bulan Maret 2026 berapa?"]},
    "best_period": {"needs": ["date", "revenue|quantity|profit"],
                    "examples": ["Which month had the highest revenue?", "Bulan apa penjualan tertinggi?"]},
    "trend_chart": {"needs": ["date", "revenue|quantity|profit"],
                    "examples": ["Show the monthly revenue trend", "Grafik penjualan mingguan"]},
    "group_chart": {"needs": ["category|city|store|product", "revenue|quantity|profit"],
                    "examples": ["Chart revenue by city", "Grafik laba per kategori"]},
    "weekday": {"needs": ["date", "revenue|quantity"],
                "examples": ["Which day is the busiest?", "Hari apa penjualan paling sepi?"]},
    "restock": {"needs": ["date", "product", "quantity", "stock"],
                "examples": ["What should I restock?", "Produk apa yang harus segera di-restock?"]},
    "stock_of_item": {"needs": ["date", "product", "stock"],
                      "examples": ["How much Minyak Goreng is left?", "Berapa sisa stok Telur?"]},
    "margin": {"needs": ["product|category", "profit"],
               "examples": ["Which product has the best margin?", "Produk mana yang untungnya paling besar?"]},
    "share": {"needs": ["category|payment|channel|city"],
              "examples": ["What percentage of orders were paid with QRIS?", "Berapa persen penjualan online?"]},
    "clean": {"needs": [], "examples": ["Clean the data", "Bersihkan data yang kosong dan duplikat"]},
    "overview": {"needs": [], "examples": ["Give me an overview of this data", "Ringkasan data ini"]},
}

_SUFFIX = re.compile(r"_(rp|idr|usd|pct|persen|percent)$")  # not "unit": "harga_per_unit" must stay whole


def normalise(column) -> str:
    """'Harga Jual (Rp)' -> 'harga_jual'; 'Order-Date' -> 'order_date'."""
    name = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
    return _SUFFIX.sub("", name) or name


_LOOKUP = {name: role for role, info in ROLES.items() for name in info["names"]}


def role_of(column) -> str | None:
    """The role a column name points to, or None if unknown or ambiguous (then decide from the values)."""
    name = normalise(column)
    return None if name in AMBIGUOUS else _LOOKUP.get(name)


def candidates(column) -> list[str]:
    """Every role a name could mean: one for a known name, several for an ambiguous one, none if unknown."""
    name = normalise(column)
    if name in AMBIGUOUS:
        return list(AMBIGUOUS[name])
    return [_LOOKUP[name]] if name in _LOOKUP else []


def available_intents(roles: set[str]) -> list[str]:
    """Which frequent question types a dataset with these roles can answer."""
    def met(need):  # "a|b" = any of them
        return any(r in roles for r in need.split("|"))
    return [intent for intent, info in INTENTS.items() if all(met(n) for n in info["needs"])]
