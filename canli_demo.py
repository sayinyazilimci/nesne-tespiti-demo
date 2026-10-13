# -*- coding: utf-8 -*-
"""
Canlı Nesne Tespiti Demosu
SKY LAB Akademi · Gökyüzünden Nesne Bulmak: Nesne Tespitine Giriş

  python canli_demo.py --hazirla           Sunumdan önce, internet varken BİR KEZ çalıştır.
  python canli_demo.py                     Kamerayla canlı demo.
  python canli_demo.py --kamera 1          Başka bir kamera.
  python canli_demo.py --kaynak video.mp4  Kamera yerine video ya da resim.
  python canli_demo.py --drone kare.jpg    Drone modunda kendi fotoğrafın.

Kısayollar: yaz + Enter = ara · ↑/↓ = güven eşiği · F1 yardım · F2 dondur · F3 NMS
            F4 eşik altı adaylar · F5 drone modu · F6 ayna · F11 tam ekran · Esc geri
"""
from __future__ import annotations

import argparse
import functools
import os
import re
import sys
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

KLASOR = Path(__file__).resolve().parent
# ultralytics, metin modelini (CLIP) çalışma klasöründeki weights/ altında arar.
# Programı nereden çalıştırırsan çalıştır aynı dosyaları bulsun diye kendi klasörüne geçiyoruz.
os.chdir(KLASOR)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

AGIRLIK = KLASOR / "weights" / "yolov8s-worldv2.pt"
AGIRLIK_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt"
DRONE_KLASORU = KLASOR / "ornekler" / "drone"
VIDEO_KLASORU = KLASOR / "ornekler" / "video"
VIDEO_FPS = 7.5        # yarışma videolarından alınan karelerin hızı
RESIM_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VARSAYILAN_ARAMA = "insan, telefon, şişe, bardak"
DRONE_SINIFLAR = [("insan", "person"), ("araç", "car")]
ALT_ESIK = 0.05        # model bunun altındaki adayları hiç döndürmez
NMS_IOU = 0.7          # NMS açıkken: bu kadar örtüşen kutulardan zayıf olan silinir
BIRLESTIRME_IOU = 0.5  # dilimlerden gelen kutular birleştirilirken
ICICE_ORAN = 0.8       # videoda: küçük kutunun bu kadarı daha emin bir kutunun içindeyse aynı insan sayılır
ORTUSME = 0.2          # komşu dilimler %20 üst üste biner

ARKA_HEX, PANEL_HEX, KUTU_HEX = "#0B1118", "#121B26", "#1C2734"
ARKA = (11, 17, 24)
YAZI = (232, 237, 242)
SOLUK = (150, 162, 178)
VURGU = (255, 176, 32)
KIRMIZI = (248, 113, 113)
KOYU = (12, 16, 22)
RENKLER = [(255, 176, 32), (34, 211, 238), (244, 114, 182), (163, 230, 53),
           (167, 139, 250), (251, 146, 60), (96, 165, 250), (248, 113, 113)]
BOS = np.zeros((0, 6), np.float32)

# --------------------------------------------------------------------------- Türkçe → İngilizce
# Model İngilizce kelimelerle çalışıyor. Salondan gelecek kelimeler için küçük bir sözlük;
# sözlükte olmayan kelimeler olduğu gibi gönderilir (İngilizce yazmak da çalışır).
SOZLUK = {
    "insan": "person", "kisi": "person", "adam": "man", "kadin": "woman", "cocuk": "child",
    "bebek": "baby", "yuz": "face", "el": "hand", "goz": "eye", "sac": "hair", "sakal": "beard",
    "biyik": "mustache", "telefon": "cell phone", "cep telefonu": "cell phone",
    "akilli telefon": "smartphone", "sise": "bottle", "su sisesi": "water bottle", "bardak": "cup",
    "kupa": "mug", "fincan": "cup", "termos": "thermos", "gozluk": "glasses",
    "gunes gozlugu": "sunglasses", "sapka": "hat", "bere": "beanie", "kep": "cap", "canta": "bag",
    "sirt cantasi": "backpack", "el cantasi": "handbag", "bilgisayar": "laptop",
    "dizustu": "laptop", "laptop": "laptop", "tablet": "tablet", "klavye": "keyboard",
    "fare": "computer mouse", "mouse": "computer mouse", "ekran": "monitor", "monitor": "monitor",
    "televizyon": "television", "kumanda": "remote control", "sarj aleti": "charger",
    "kablo": "cable", "kulaklik": "headphones", "kitap": "book", "defter": "notebook",
    "kalem": "pen", "kursun kalem": "pencil", "silgi": "eraser", "makas": "scissors",
    "kagit": "paper", "saat": "watch", "kol saati": "wristwatch", "duvar saati": "clock",
    "anahtar": "keys", "cuzdan": "wallet", "kart": "card", "kimlik": "id card", "para": "money",
    "yuzuk": "ring", "kolye": "necklace", "kupe": "earring", "ayakkabi": "shoe", "mont": "jacket",
    "ceket": "jacket", "kazak": "sweater", "tisort": "t-shirt", "gomlek": "shirt", "kravat": "tie",
    "atki": "scarf", "maske": "face mask", "eldiven": "glove", "semsiye": "umbrella",
    "sandalye": "chair", "masa": "table", "kapi": "door", "pencere": "window", "lamba": "lamp",
    "perde": "curtain", "tahta": "whiteboard", "yazi tahtasi": "whiteboard",
    "projeksiyon": "projector", "bitki": "potted plant", "saksi": "potted plant",
    "cop kutusu": "trash can", "dolap": "cabinet", "araba": "car", "arac": "vehicle",
    "otomobil": "car", "kamyon": "truck", "otobus": "bus", "bisiklet": "bicycle",
    "motosiklet": "motorcycle", "motor": "motorcycle", "ucak": "airplane",
    "helikopter": "helicopter", "drone": "drone", "dron": "drone", "iha": "drone", "gemi": "ship",
    "tekne": "boat", "tren": "train", "traktor": "tractor", "bicerdover": "combine harvester",
    "is makinesi": "construction vehicle", "kepce": "excavator", "kopek": "dog", "kedi": "cat",
    "kus": "bird", "at": "horse", "inek": "cow", "koyun": "sheep", "top": "ball",
    "futbol topu": "soccer ball", "oyuncak": "toy", "ayicik": "teddy bear", "gitar": "guitar",
    "elma": "apple", "muz": "banana", "portakal": "orange", "ekmek": "bread", "simit": "bagel",
    "cikolata": "chocolate bar", "kahve": "coffee cup", "cay bardagi": "tea glass",
    "kirmizi": "red", "mavi": "blue", "yesil": "green", "sari": "yellow", "siyah": "black",
    "beyaz": "white", "turuncu": "orange", "mor": "purple", "pembe": "pink", "gri": "gray",
    "kahverengi": "brown", "lacivert": "navy blue", "buyuk": "big", "kucuk": "small",
    "pist": "runway", "inis alani": "landing pad", "yol": "road", "ev": "house",
    "bina": "building", "agac": "tree", "cati": "roof",
}
_TR_KUCUK = str.maketrans({"I": "ı", "İ": "i"})
_TR_KATLA = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def katla(metin: str) -> str:
    """'Şişe' -> 'sise': Türkçe küçük harf + ASCII'ye katlama."""
    return re.sub(r"\s+", " ", metin.translate(_TR_KUCUK).lower().translate(_TR_KATLA)).strip()


def cevir(ifade: str) -> str:
    anahtar = katla(ifade)
    if anahtar in SOZLUK:
        return SOZLUK[anahtar]
    kelimeler = []
    for kelime in anahtar.split():
        kok = re.sub(r"(lar|ler)$", "", kelime) if len(kelime) > 4 else kelime
        kelimeler.append(SOZLUK.get(kelime) or SOZLUK.get(kok) or kelime)
    return " ".join(kelimeler)


def arama_coz(metin: str) -> list[tuple[str, str]]:
    """'insan, kırmızı telefon' -> [('insan', 'person'), ('kırmızı telefon', 'red cell phone')]"""
    sonuc, gorulen = [], set()
    for parca in re.split(r"[,;/+]|\s+ve\s+", metin, flags=re.IGNORECASE):
        parca = parca.strip()
        if not parca:
            continue
        ingilizce = cevir(parca)
        if ingilizce and ingilizce not in gorulen:
            gorulen.add(ingilizce)
            sonuc.append((parca, ingilizce))
    return sonuc[:8]


# --------------------------------------------------------------------------- model
def agirlik_hazirla(yaz=print) -> None:
    """Model dosyası yoksa ya da bozuksa indirir."""
    if AGIRLIK.exists() and zipfile.is_zipfile(AGIRLIK):
        return
    AGIRLIK.parent.mkdir(parents=True, exist_ok=True)
    gecici = AGIRLIK.with_suffix(".part")
    yaz("Model indiriliyor (yaklaşık 25 MB)…")
    urllib.request.urlretrieve(AGIRLIK_URL, gecici)
    if not zipfile.is_zipfile(gecici):
        gecici.unlink(missing_ok=True)
        raise RuntimeError("İndirilen model dosyası bozuk çıktı. İnternet bağlantısını kontrol edip tekrar dene.")
    gecici.replace(AGIRLIK)


def cihaz_sec(istek: str) -> str:
    import torch
    if istek == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return istek


class Dedektor:
    """YOLO-World: aranacak sınıfları metinle aldığımız (açık sözlüklü) nesne dedektörü."""

    def __init__(self, cihaz: str):
        from ultralytics import YOLO
        self.model = YOLO(str(AGIRLIK))
        self.cihaz = cihaz
        self.kilit = threading.Lock()
        self.siniflar: tuple[str, ...] = ()

    def _siniflari_ayarla(self, siniflar) -> None:
        siniflar = tuple(siniflar)
        if siniflar != self.siniflar:
            self.model.set_classes(list(siniflar))
            self.siniflar = siniflar

    def hazirla(self, siniflar) -> None:
        with self.kilit:
            self._siniflari_ayarla(siniflar)

    @staticmethod
    def _diziye(sonuc) -> np.ndarray:
        kutular = sonuc.boxes
        if kutular is None or len(kutular) == 0:
            return BOS.copy()
        return np.concatenate([kutular.xyxy.cpu().numpy(), kutular.conf.cpu().numpy()[:, None],
                               kutular.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32)

    def tespit(self, bgr: np.ndarray, siniflar, nms: bool = True, imgsz: int = 640) -> np.ndarray:
        """Her satır bir aday kutu: x1, y1, x2, y2, güven, sınıf."""
        if bgr is None or bgr.size == 0:
            return BOS.copy()
        with self.kilit:
            self._siniflari_ayarla(siniflar)
            sonuc = self.model.predict(np.ascontiguousarray(bgr), conf=ALT_ESIK,
                                       iou=NMS_IOU if nms else 1.0, imgsz=imgsz,
                                       device=self.cihaz, max_det=300, verbose=False)[0]
        return self._diziye(sonuc)

    def tespit_toplu(self, goruntuler, siniflar, imgsz: int = 640, parti: int = 16) -> list[np.ndarray]:
        """Birçok görüntüyü (ör. dilimleri) partiler halinde tek seferde işler; ekran kartında çok daha hızlı."""
        sonuclar = []
        with self.kilit:
            self._siniflari_ayarla(siniflar)
            for i in range(0, len(goruntuler), parti):
                parca = [np.ascontiguousarray(g) for g in goruntuler[i:i + parti]]
                for r in self.model.predict(parca, conf=ALT_ESIK, iou=NMS_IOU, imgsz=imgsz,
                                            device=self.cihaz, max_det=300, verbose=False):
                    sonuclar.append(self._diziye(r))
        return sonuclar


def dilimler(w: int, h: int, boyut: int, ortusme: float = ORTUSME) -> list[tuple[int, int, int, int]]:
    adim = max(1, int(boyut * (1 - ortusme)))

    def eksen(n):
        son = max(n - boyut, 0)
        degerler = list(range(0, son + 1, adim))
        if degerler[-1] != son:
            degerler.append(son)
        return degerler

    return [(x, y, min(x + boyut, w), min(y + boyut, h)) for y in eksen(h) for x in eksen(w)]


def birlestir(tespit: np.ndarray, iou: float = BIRLESTIRME_IOU, ic_ice: float | None = None) -> np.ndarray:
    """Dilimlerden gelen kutuları sınıf bazında NMS ile birleştirir.

    ic_ice verilirse: dilim sınırına denk gelen bir insan iki dilimde ayrı kutu alabilir (biri tam,
    biri yarım). IoU'ları düşük kaldığı için NMS ikisini de tutar. Küçük kutunun bu orandan fazlası
    daha emin bir kutunun içindeyse (SAHI'deki IOS ölçütü) onu da sileriz.
    """
    if len(tespit) == 0:
        return tespit
    import torch
    from torchvision.ops import batched_nms
    t = torch.from_numpy(np.ascontiguousarray(tespit))
    tut = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), iou).numpy()
    tespit = tespit[tut]                      # güvene göre azalan sırada
    if ic_ice is None:
        return tespit
    alan = (tespit[:, 2] - tespit[:, 0]) * (tespit[:, 3] - tespit[:, 1])
    kalan = np.ones(len(tespit), bool)
    for i in range(len(tespit)):
        if not kalan[i]:
            continue
        j = np.arange(i + 1, len(tespit))
        j = j[kalan[j] & (tespit[j, 5] == tespit[i, 5])]
        if len(j) == 0:
            continue
        gx = np.clip(np.minimum(tespit[i, 2], tespit[j, 2]) - np.maximum(tespit[i, 0], tespit[j, 0]), 0, None)
        gy = np.clip(np.minimum(tespit[i, 3], tespit[j, 3]) - np.maximum(tespit[i, 1], tespit[j, 1]), 0, None)
        kalan[j[gx * gy / np.minimum(alan[i], alan[j]) > ic_ice]] = False
    return tespit[kalan]


def sahi_dilimleri(dedektor: "Dedektor", bgr: np.ndarray, siniflar, boyut: int):
    """SAHI: kareyi üst üste binen dilimlere böler, hepsini tek seferde tarar, kutuları kareye taşır."""
    h, w = bgr.shape[:2]
    liste = dilimler(w, h, boyut)
    parcalar = dedektor.tespit_toplu([bgr[y1:y2, x1:x2] for x1, y1, x2, y2 in liste], siniflar)
    for (x1, y1, _, _), p in zip(liste, parcalar):
        if len(p):
            p[:, [0, 2]] += x1
            p[:, [1, 3]] += y1
    return liste, parcalar


def resimler(klasor: Path) -> list[Path]:
    if not klasor.is_dir():
        return []
    return sorted(p for p in klasor.iterdir() if p.suffix.lower() in RESIM_UZANTILARI)


def drone_ornegi() -> Path | None:
    liste = resimler(DRONE_KLASORU)
    return liste[0] if liste else None


# --------------------------------------------------------------------------- görüntü kaynağı
def resim_oku(yol) -> np.ndarray | None:
    try:  # cv2.imread Türkçe karakterli yollarda çalışmıyor; bu yol çalışıyor
        return cv2.imdecode(np.fromfile(str(yol), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def kamera_ac(indeks: int):
    arka_uclar = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for arka_uc in arka_uclar:
        cap = cv2.VideoCapture(indeks, arka_uc)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


class Kaynak:
    """Kamera, video ya da resimden arka planda kare okur; her zaman en yeni kareyi tutar."""

    def __init__(self, kaynak, ayna: bool = False):
        self.kare: np.ndarray | None = None
        self.sayac = 0
        self.hata: str | None = None
        self.ayna = ayna
        self.calisiyor = True
        threading.Thread(target=self._calis, args=(kaynak,), daemon=True).start()

    def _yayinla(self, kare: np.ndarray) -> None:
        self.kare = cv2.flip(kare, 1) if self.ayna else kare
        self.sayac += 1

    def _calis(self, kaynak) -> None:
        if isinstance(kaynak, str) and Path(kaynak).suffix.lower() in RESIM_UZANTILARI:
            resim = resim_oku(kaynak)
            if resim is None:
                self.hata = f"Resim okunamadı: {kaynak}"
                return
            son_ayna = None
            while self.calisiyor:
                if self.ayna != son_ayna:
                    son_ayna = self.ayna
                    self._yayinla(resim)
                time.sleep(0.1)
            return

        dosya = not isinstance(kaynak, int)
        cap = cv2.VideoCapture(str(kaynak)) if dosya else kamera_ac(kaynak)
        if cap is None or not cap.isOpened():
            self.hata = (f"Video açılamadı: {kaynak}" if dosya else
                         f"Kamera {kaynak} açılamadı. Başka kamera için: python canli_demo.py --kamera 1")
            return
        bekleme = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30) if dosya else 0.0
        hatali = 0
        while self.calisiyor:
            t0 = time.perf_counter()
            ok, kare = cap.read()
            if not ok:
                if dosya:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                hatali += 1
                if hatali > 60:
                    self.hata = "Kameradan görüntü gelmiyor. Başka bir program kamerayı kullanıyor olabilir."
                    break
                time.sleep(0.05)
                continue
            hatali = 0
            self._yayinla(kare)
            if dosya:
                time.sleep(max(0.0, bekleme - (time.perf_counter() - t0)))
        cap.release()


# --------------------------------------------------------------------------- çizim
@dataclass
class Gorunum:
    esik: float = 0.25
    nms: bool = True
    hayalet: bool = False
    donmus: bool = False
    fps: float = 0.0
    model_ms: float = 0.0
    cihaz: str = "cpu"
    bildirim: str | None = None
    yardim: bool = False


@dataclass
class DroneDurumu:
    resim: np.ndarray | None = None
    ad: str = ""
    adim: int = 0                 # 0 ham kare · 1 tam kare · 2 dilimleme · 3 karşılaştırma
    mesgul: bool = False
    tam: np.ndarray | None = None
    tam_sure: float = 0.0
    dilim_listesi: list = field(default_factory=list)
    aktif_dilim: int = -1
    dilim_ham: list = field(default_factory=list)
    dilimli: np.ndarray | None = None
    dilim_sure: float = 0.0


@dataclass
class VideoDurumu:
    kareler: list = field(default_factory=list)
    aktif: bool = False
    sahi: bool = True
    i: int = 0
    dilim_sayisi: int = 0
    onbellek: dict = field(default_factory=dict)
    son: tuple | None = None   # (kare, tespit, süre, sıra, önbellekten mi)


@functools.lru_cache(maxsize=None)
def yazi_tipi(boyut: int, kalin: bool = False):
    adlar = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if kalin else \
            ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    fontlar = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for ad in adlar:
        for yol in (fontlar / ad, Path(ad)):
            try:
                return ImageFont.truetype(str(yol), boyut)
            except OSError:
                continue
    return ImageFont.load_default(boyut)


def sayi(x: float, basamak: int = 2) -> str:
    return f"{x:.{basamak}f}".replace(".", ",")


def sigdir(bgr: np.ndarray, w: int, h: int, arka=ARKA):
    """Kareyi en-boy oranını bozmadan w×h tuvale yerleştirir. Ölçek ve kaydırmayı da döndürür."""
    fh, fw = bgr.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
    kucuk = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    tuval = Image.new("RGB", (w, h), arka)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    tuval.paste(Image.fromarray(cv2.cvtColor(kucuk, cv2.COLOR_BGR2RGB)), (ox, oy))
    return tuval, s, ox, oy


def say(tespit: np.ndarray, esik: float, n: int) -> list[int]:
    sayim = [0] * n
    for guven, sinif in zip(tespit[:, 4], tespit[:, 5]):
        if guven >= esik and int(sinif) < n:
            sayim[int(sinif)] += 1
    return sayim


def kesikli(ciz, kutu, renk, kalinlik=1, parca=7, bosluk=5):
    x1, y1, x2, y2 = kutu
    for ax, ay, bx, by in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        uzunluk = max(abs(bx - ax), abs(by - ay))
        if uzunluk < 1:
            continue
        dx, dy = (bx - ax) / uzunluk, (by - ay) / uzunluk
        t = 0.0
        while t < uzunluk:
            s = min(t + parca, uzunluk)
            ciz.line((ax + dx * t, ay + dy * t, ax + dx * s, ay + dy * s), fill=renk, width=kalinlik)
            t += parca + bosluk


def kutulari_ciz(ciz, tespit, adlar, esik, s, ox, oy, hayalet=False, kalinlik=3, boy=20):
    if len(tespit) == 0:
        return
    f = yazi_tipi(boy, True)
    fk = yazi_tipi(max(12, boy - 6))
    hayalet_sayisi = 0
    for i in np.argsort(tespit[:, 4]):  # zayıflar önce, güçlüler üstte kalsın
        x1, y1, x2, y2, guven, sinif = tespit[i]
        sinif = int(sinif)
        if sinif >= len(adlar):
            continue
        X1, Y1, X2, Y2 = ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s
        if guven < esik:
            if hayalet and hayalet_sayisi < 150:
                kesikli(ciz, (X1, Y1, X2, Y2), (215, 225, 235, 160))
                if X2 - X1 > 44:
                    ciz.text((X1 + 4, Y1 + 3), sayi(guven), font=fk, fill=(215, 225, 235, 190))
                hayalet_sayisi += 1
            continue
        renk = RENKLER[sinif % len(RENKLER)]
        ciz.rectangle((X1, Y1, X2, Y2), outline=renk + (255,), width=kalinlik)
        etiket = f"{adlar[sinif]} {sayi(guven)}"
        yukseklik = boy + 10
        genislik = ciz.textlength(etiket, font=f) + 14
        ey = Y1 - yukseklik if Y1 - yukseklik >= 0 else Y1
        ciz.rectangle((X1 - 1, ey, X1 + genislik, ey + yukseklik), fill=renk + (235,))
        ciz.text((X1 + 7, ey + yukseklik / 2), etiket, font=f, fill=KOYU, anchor="lm")


def panel(ciz, x, y, satirlar, bosluk=14, ara=6, sag=False, alt=False, w=None, h=None):
    """satirlar: [(metin, font, renk)]. Yuvarlak köşeli koyu bir kutunun içine alt alta yazar."""
    genislik = max(ciz.textlength(m, font=f) for m, f, _ in satirlar) + 2 * bosluk
    boylar = [f.size * 1.25 for _, f, _ in satirlar]
    yukseklik = sum(boylar) + ara * (len(satirlar) - 1) + 2 * bosluk
    if sag:
        x = w - x - genislik
    if alt:
        y = h - y - yukseklik
    ciz.rounded_rectangle((x, y, x + genislik, y + yukseklik), radius=14, fill=(9, 14, 20, 215))
    cy = y + bosluk
    for (metin, f, renk), boy in zip(satirlar, boylar):
        ciz.text((x + bosluk, cy + boy / 2), metin, font=f, fill=renk, anchor="lm")
        cy += boy + ara
    return x, y, genislik, yukseklik


def esik_gostergesi(ciz, w, h, g: Gorunum, o: float):
    m = int(20 * o)
    f, fb = yazi_tipi(int(18 * o)), yazi_tipi(int(30 * o), True)
    gen, yuk = int(300 * o), int(104 * o)
    x, y = m, h - m - yuk
    ciz.rounded_rectangle((x, y, x + gen, y + yuk), radius=14, fill=(9, 14, 20, 215))
    ciz.text((x + 16 * o, y + 24 * o), "Güven eşiği", font=f, fill=SOLUK, anchor="lm")
    ciz.text((x + gen - 16 * o, y + 26 * o), sayi(g.esik), font=fb, fill=VURGU, anchor="rm")
    by = y + 54 * o
    ciz.rounded_rectangle((x + 16 * o, by, x + gen - 16 * o, by + 8 * o), radius=4, fill=(40, 52, 66, 255))
    dolu = x + 16 * o + (gen - 32 * o) * g.esik
    ciz.rounded_rectangle((x + 16 * o, by, dolu, by + 8 * o), radius=4, fill=VURGU + (255,))
    ciz.ellipse((dolu - 8 * o, by - 4 * o, dolu + 8 * o, by + 12 * o), fill=YAZI)
    nms = "NMS açık" if g.nms else "NMS KAPALI"
    ciz.text((x + 16 * o, y + 84 * o), nms, font=yazi_tipi(int(17 * o), True),
             fill=YAZI if g.nms else KIRMIZI, anchor="lm")
    if g.hayalet:
        ciz.text((x + gen - 16 * o, y + 84 * o), "- - eşik altı", font=yazi_tipi(int(17 * o)),
                 fill=SOLUK, anchor="rm")


def bildirim_ciz(ciz, w, h, metin, o):
    f = yazi_tipi(int(22 * o), True)
    gen = ciz.textlength(metin, font=f) + 40 * o
    x, y = (w - gen) / 2, h - 150 * o
    ciz.rounded_rectangle((x, y, x + gen, y + 50 * o), radius=25 * o, fill=VURGU + (240,))
    ciz.text((w / 2, y + 25 * o), metin, font=f, fill=KOYU, anchor="mm")


YARDIM = [
    ("Yaz + Enter", "Aranacak nesneler (virgülle ayır)"),
    ("↑ / ↓", "Güven eşiğini değiştir"),
    ("F2", "Görüntüyü dondur / devam et"),
    ("F3", "NMS aç / kapat"),
    ("F4", "Eşik altındaki adayları göster"),
    ("F5", "Drone modu: tam kare ve SAHI dilimleme"),
    ("Enter / ← →", "Drone: sonraki adım / önceki-sonraki fotoğraf"),
    ("V · S", "Drone: video oynat · SAHI aç / kapat"),
    ("F6", "Aynalı görüntü"),
    ("F11", "Tam ekran"),
    ("Esc", "Geri / tam ekrandan çık"),
]


def yardim_ciz(img: Image.Image, o: float) -> None:
    w, h = img.size
    ciz = ImageDraw.Draw(img, "RGBA")
    ciz.rectangle((0, 0, w, h), fill=(0, 0, 0, 150))
    f, fb, fbas = yazi_tipi(int(22 * o)), yazi_tipi(int(22 * o), True), yazi_tipi(int(30 * o), True)
    gen, satir = int(640 * o), int(42 * o)
    yuk = int(90 * o) + satir * len(YARDIM)
    x, y = (w - gen) / 2, (h - yuk) / 2
    ciz.rounded_rectangle((x, y, x + gen, y + yuk), radius=18, fill=(12, 18, 26, 245))
    ciz.text((x + 32 * o, y + 42 * o), "Kısayollar", font=fbas, fill=YAZI, anchor="lm")
    for i, (tus, aciklama) in enumerate(YARDIM):
        cy = y + 90 * o + i * satir
        ciz.text((x + 32 * o, cy), tus, font=fb, fill=VURGU, anchor="lm")
        ciz.text((x + 220 * o, cy), aciklama, font=f, fill=YAZI, anchor="lm")


def bos_ekran(w: int, h: int, metin: str, alt: str | None = None) -> Image.Image:
    img = Image.new("RGB", (w, h), ARKA)
    ciz = ImageDraw.Draw(img)
    o = h / 720
    ciz.text((w / 2, h / 2 - 16 * o), metin, font=yazi_tipi(int(28 * o), True), fill=YAZI, anchor="mm")
    if alt:
        ciz.text((w / 2, h / 2 + 26 * o), alt, font=yazi_tipi(int(20 * o)), fill=SOLUK, anchor="mm")
    return img


def kamera_ciz(kare, tespit, aktif, g: Gorunum, w, h) -> Image.Image:
    img, s, ox, oy = sigdir(kare, w, h)
    ciz = ImageDraw.Draw(img, "RGBA")
    o = h / 720
    adlar = [tr for tr, _ in aktif]
    kutulari_ciz(ciz, tespit, adlar, g.esik, s, ox, oy, hayalet=g.hayalet, boy=int(20 * o))
    m = int(20 * o)

    # sol üst: durum ve aranan nesneler
    f, fb, fk = yazi_tipi(int(20 * o)), yazi_tipi(int(20 * o), True), yazi_tipi(int(16 * o))
    durum = "DONDURULDU" if g.donmus else "CANLI"
    cihaz = "GPU" if g.cihaz.startswith("cuda") else "CPU"
    hiz = f"{g.fps:.0f} kare/sn  ·  " if g.fps >= 0.5 and not g.donmus else ""
    ust = f"{durum}  ·  {hiz}{cihaz}  ·  model {g.model_ms:.0f} ms"
    gen = max(ciz.textlength(ust, font=fb) + 52 * o,
              sum(ciz.textlength(tr, font=fb) + 34 * o for tr in adlar) + 24 * o,
              ciz.textlength("Modele giden: " + ", ".join(en for _, en in aktif), font=fk) + 32 * o)
    yuk = 128 * o
    ciz.rounded_rectangle((m, m, m + gen, m + yuk), radius=14, fill=(9, 14, 20, 215))
    nokta = (34, 211, 238) if g.donmus else KIRMIZI
    ciz.ellipse((m + 16 * o, m + 21 * o, m + 30 * o, m + 35 * o), fill=nokta)
    ciz.text((m + 40 * o, m + 28 * o), ust, font=fb, fill=YAZI, anchor="lm")
    cx = m + 16 * o
    for i, tr in enumerate(adlar):
        renk = RENKLER[i % len(RENKLER)]
        tg = ciz.textlength(tr, font=fb) + 24 * o
        ciz.rounded_rectangle((cx, m + 52 * o, cx + tg, m + 86 * o), radius=17 * o, fill=renk + (235,))
        ciz.text((cx + tg / 2, m + 69 * o), tr, font=fb, fill=KOYU, anchor="mm")
        cx += tg + 10 * o
    ciz.text((m + 16 * o, m + 108 * o), "Modele giden: " + ", ".join(en for _, en in aktif),
             font=fk, fill=SOLUK, anchor="lm")

    # sağ üst: kaç tane bulundu
    sayim = say(tespit, g.esik, len(adlar))
    fs, fsb = yazi_tipi(int(22 * o)), yazi_tipi(int(40 * o), True)
    satir = 52 * o
    gen = max([ciz.textlength(tr, font=fs) for tr in adlar] + [80 * o]) + 130 * o
    yuk = 54 * o + satir * len(adlar)
    x = w - m - gen
    ciz.rounded_rectangle((x, m, x + gen, m + yuk), radius=14, fill=(9, 14, 20, 215))
    ciz.text((x + 18 * o, m + 28 * o), "Bulunan", font=yazi_tipi(int(18 * o)), fill=SOLUK, anchor="lm")
    for i, (tr, n) in enumerate(zip(adlar, sayim)):
        cy = m + 54 * o + i * satir + satir / 2 - 6 * o
        renk = RENKLER[i % len(RENKLER)]
        ciz.rounded_rectangle((x + 18 * o, cy - 7 * o, x + 32 * o, cy + 7 * o), radius=3, fill=renk)
        ciz.text((x + 44 * o, cy), tr, font=fs, fill=YAZI, anchor="lm")
        ciz.text((x + gen - 18 * o, cy), str(n), font=fsb, fill=renk if n else SOLUK, anchor="rm")

    if not g.nms:
        ham = int((tespit[:, 4] >= g.esik).sum()) if len(tespit) else 0
        bas = f"NMS KAPALI  ·  aynı nesneye birden çok kutu  ·  {ham} kutu"
        fbn = yazi_tipi(int(22 * o), True)
        gn = ciz.textlength(bas, font=fbn) + 40 * o
        ciz.rounded_rectangle(((w - gn) / 2, m + 150 * o, (w + gn) / 2, m + 196 * o), radius=12,
                              fill=KIRMIZI + (235,))
        ciz.text((w / 2, m + 173 * o), bas, font=fbn, fill=KOYU, anchor="mm")

    esik_gostergesi(ciz, w, h, g, o)
    fy = yazi_tipi(int(17 * o))
    gy = ciz.textlength("F1 yardım", font=fy) + 24 * o
    ciz.rounded_rectangle((w - m - gy, h - m - 34 * o, w - m, h - m), radius=17 * o, fill=(9, 14, 20, 200))
    ciz.text((w - m - gy / 2, h - m - 17 * o), "F1 yardım", font=fy, fill=SOLUK, anchor="mm")
    if g.bildirim:
        bildirim_ciz(ciz, w, h, g.bildirim, o)
    if g.yardim:
        yardim_ciz(img, o)
    return img


def drone_ciz(d: DroneDurumu, g: Gorunum, w, h) -> Image.Image:
    if d.resim is None:
        return bos_ekran(w, h, "Drone fotoğrafı bulunamadı", "Resim seç… düğmesiyle bir fotoğraf aç")
    if d.adim == 3:
        return karsilastirma_ciz(d, g, w, h)
    o = h / 720
    adlar = [tr for tr, _ in DRONE_SINIFLAR]
    img, s, ox, oy = sigdir(d.resim, w, h)
    ciz = ImageDraw.Draw(img, "RGBA")
    rh, rw = d.resim.shape[:2]
    fbas, f, fb = yazi_tipi(int(18 * o), True), yazi_tipi(int(22 * o)), yazi_tipi(int(24 * o), True)

    if d.adim == 1 and d.tam is not None:
        kutulari_ciz(ciz, d.tam, adlar, g.esik, s, ox, oy, hayalet=g.hayalet, boy=int(17 * o))
    if d.adim == 2:
        bitti = d.dilimli is not None
        for i, (x1, y1, x2, y2) in enumerate(d.dilim_listesi):
            kutu = (ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s)
            if i == d.aktif_dilim:
                ciz.rectangle(kutu, fill=VURGU + (55,), outline=VURGU + (255,), width=4)
            else:
                ciz.rectangle(kutu, outline=(255, 255, 255, 70 if bitti else 110), width=1)
        parcalar = list(d.dilim_ham)
        tespit = d.dilimli if bitti else (np.concatenate(parcalar) if parcalar else BOS)
        kutulari_ciz(ciz, tespit, adlar, g.esik, s, ox, oy, hayalet=g.hayalet, boy=int(17 * o))

    satirlar = []
    if d.adim == 0:
        satirlar = [(f"DRONE KARESİ  ·  {rw}×{rh}", fbas, VURGU),
                    ("Bu fotoğrafta kaç insan var?", fb, YAZI),
                    ("Enter: tek seferde tara", f, SOLUK)]
    elif d.adim == 1:
        n = say(d.tam, g.esik, 2) if d.tam is not None else [0, 0]
        satirlar = [("1  ·  TAM KARE", fbas, VURGU),
                    ("Model kareyi 640 piksele küçültüp bir kez bakıyor.", f, YAZI),
                    (f"insan {n[0]}  ·  araç {n[1]}  ·  {sayi(d.tam_sure)} sn", fb, YAZI),
                    ("Enter: dilimleyerek tara", f, SOLUK)]
    elif d.adim == 2:
        toplam = len(d.dilim_listesi)
        if d.dilimli is None:
            satirlar = [("2  ·  SAHI DİLİMLEME", fbas, VURGU),
                        (f"Kare {toplam} parçaya bölündü, her parça tam çözünürlükte taranıyor.", f, YAZI),
                        (f"Dilim {max(d.aktif_dilim, 0) + 1} / {toplam}", fb, YAZI)]
        else:
            n = say(d.dilimli, g.esik, 2)
            satirlar = [("2  ·  SAHI DİLİMLEME", fbas, VURGU),
                        (f"{toplam} dilim + tam kare, kutular NMS ile birleştirildi.", f, YAZI),
                        (f"insan {n[0]}  ·  araç {n[1]}  ·  {sayi(d.dilim_sure)} sn", fb, YAZI),
                        ("Enter: karşılaştır", f, SOLUK)]
    m = int(20 * o)
    if satirlar:
        panel(ciz, m, m, satirlar, bosluk=int(16 * o))
    esik_gostergesi(ciz, w, h, g, o)
    if g.bildirim:
        bildirim_ciz(ciz, w, h, g.bildirim, o)
    if g.yardim:
        yardim_ciz(img, o)
    return img


def yakin_bolge(d: DroneDurumu, esik: float):
    """Dilimlemenin bulduğu insanların çevresi: karşılaştırmada yakından göstermek için."""
    if d.dilimli is None or d.resim is None:
        return None
    insanlar = d.dilimli[(d.dilimli[:, 5] == 0) & (d.dilimli[:, 4] >= esik)]
    if len(insanlar) == 0:
        return None
    rh, rw = d.resim.shape[:2]
    x1, y1, x2, y2 = insanlar[:, 0].min(), insanlar[:, 1].min(), insanlar[:, 2].max(), insanlar[:, 3].max()
    if x2 - x1 > rw * 0.2 or y2 - y1 > rh * 0.2:  # insanlar dağınıksa en eminine odaklan
        x1, y1, x2, y2 = insanlar[np.argmax(insanlar[:, 4]), :4]
    bw = min(rw, max(rw * 0.09, (x2 - x1) * 2.5, (y2 - y1) * 2.5 * 16 / 9))
    bh = min(rh, bw * 9 / 16)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bx1 = int(min(max(cx - bw / 2, 0), rw - bw))
    by1 = int(min(max(cy - bh / 2, 0), rh - bh))
    return bx1, by1, int(bx1 + bw), int(by1 + bh)


def karsilastirma_ciz(d: DroneDurumu, g: Gorunum, w, h) -> Image.Image:
    o = h / 720
    img = Image.new("RGB", (w, h), ARKA)
    ciz = ImageDraw.Draw(img, "RGBA")
    adlar = [tr for tr, _ in DRONE_SINIFLAR]
    rh, rw = d.resim.shape[:2]
    m, ust = int(20 * o), int(64 * o)
    yarim = (w - 3 * m) // 2
    ih = int(min(yarim * rh / rw, h * 0.52))
    bolge = yakin_bolge(d, g.esik)
    fbas, fb = yazi_tipi(int(26 * o), True), yazi_tipi(int(28 * o), True)
    f, fv = yazi_tipi(int(20 * o)), yazi_tipi(int(20 * o), True)
    kat = d.dilim_sure / d.tam_sure if d.tam_sure > 0 else 0
    taraflar = [("Tam kare", d.tam, d.tam_sure, "1 tarama", ""),
                ("SAHI: tam kare + dilimler", d.dilimli, d.dilim_sure, f"{len(d.dilim_listesi) + 1} tarama",
                 f"yaklaşık {kat:.0f} kat süre" if kat else "")]
    for i, (baslik, tespit, sure, aciklama, ek) in enumerate(taraflar):
        x = m + i * (yarim + m)
        parca, s, ox, oy = sigdir(d.resim, yarim, ih)
        pciz = ImageDraw.Draw(parca, "RGBA")
        if tespit is not None:
            kutulari_ciz(pciz, tespit, adlar, g.esik, s, ox, oy, kalinlik=2, boy=int(14 * o))
        if bolge:
            bx1, by1, bx2, by2 = bolge
            pciz.rectangle((ox + bx1 * s, oy + by1 * s, ox + bx2 * s, oy + by2 * s),
                           outline=(255, 255, 255, 230), width=2)
        img.paste(parca, (x, ust))
        ciz.text((x, ust - 18 * o), baslik, font=fbas, fill=VURGU if i else YAZI, anchor="ls")
        n = say(tespit, g.esik, 2) if tespit is not None else [0, 0]
        sy = ust + ih + 40 * o
        ciz.text((x, sy), f"insan {n[0]}   ·   araç {n[1]}", font=fb, fill=YAZI, anchor="ls")
        alt_satir = f"{sayi(sure)} sn  ·  {aciklama}"
        ciz.text((x, sy + 30 * o), alt_satir, font=f, fill=SOLUK, anchor="ls")
        if ek:
            ciz.text((x + ciz.textlength(alt_satir + "  ·  ", font=f), sy + 30 * o), ek, font=fv,
                     fill=VURGU, anchor="ls")
        if bolge:
            zy = int(sy + 50 * o)
            zh = int(h - zy - m)
            zw = int(min(yarim, zh * 16 / 9))
            if zh > 60 * o:
                bx1, by1, bx2, by2 = bolge
                yakin, s2, ox2, oy2 = sigdir(d.resim[by1:by2, bx1:bx2], zw, zh)
                if tespit is not None and len(tespit):
                    kayik = tespit.copy()
                    kayik[:, [0, 2]] -= bx1
                    kayik[:, [1, 3]] -= by1
                    kutulari_ciz(ImageDraw.Draw(yakin, "RGBA"), kayik, adlar, g.esik, s2, ox2, oy2,
                                 kalinlik=3, boy=int(15 * o))
                img.paste(yakin, (x, zy))
                ciz.rectangle((x, zy, x + zw, zy + zh), outline=(255, 255, 255, 230), width=2)
                ciz.text((x + zw + 16 * o, zy + 14 * o), "Yakından", font=f, fill=SOLUK, anchor="lm")
    if g.bildirim:
        bildirim_ciz(ciz, w, h, g.bildirim, o)
    if g.yardim:
        yardim_ciz(img, o)
    return img


def video_ciz(v: VideoDurumu, g: Gorunum, w, h) -> Image.Image:
    if v.son is None:
        return bos_ekran(w, h, "Video hazırlanıyor…", "İlk kare işleniyor")
    kare, tespit, sure, sira, onbellek = v.son
    o = h / 720
    adlar = [tr for tr, _ in DRONE_SINIFLAR]
    img, s, ox, oy = sigdir(kare, w, h)
    ciz = ImageDraw.Draw(img, "RGBA")
    kutulari_ciz(ciz, tespit, adlar, g.esik, s, ox, oy, hayalet=g.hayalet, kalinlik=2, boy=int(15 * o))
    n = say(tespit, g.esik, 2)
    fbas, f, fb = yazi_tipi(int(18 * o), True), yazi_tipi(int(22 * o)), yazi_tipi(int(24 * o), True)
    mod = f"SAHI açık: {v.dilim_sayisi} dilim + tam kare" if v.sahi else "SAHI kapalı: yalnızca tam kare"
    kaynak = "önbellekten" if onbellek else "canlı hesaplandı"
    satirlar = [(f"VİDEO  ·  kare {sira + 1} / {len(v.kareler)}", fbas, VURGU),
                (mod, f, YAZI),
                (f"{sayi(sure)} sn/kare  ·  {kaynak}", f, SOLUK),
                ("S: SAHI aç / kapat  ·  V: durdur", f, SOLUK)]
    m = int(20 * o)
    panel(ciz, m, m, satirlar, bosluk=int(16 * o))
    # sağ üst: büyük sayaç
    fs, fsb = yazi_tipi(int(22 * o)), yazi_tipi(int(56 * o), True)
    gen, yuk = int(230 * o), int(150 * o)
    x = w - m - gen
    ciz.rounded_rectangle((x, m, x + gen, m + yuk), radius=14, fill=(9, 14, 20, 215))
    ciz.text((x + 20 * o, m + 30 * o), "insan", font=fs, fill=SOLUK, anchor="lm")
    ciz.text((x + gen - 20 * o, m + 62 * o), str(n[0]), font=fsb, fill=RENKLER[0], anchor="rm")
    ciz.text((x + 20 * o, m + 112 * o), f"araç  {n[1]}", font=fs, fill=RENKLER[1], anchor="lm")
    esik_gostergesi(ciz, w, h, g, o)
    if g.bildirim:
        bildirim_ciz(ciz, w, h, g.bildirim, o)
    if g.yardim:
        yardim_ciz(img, o)
    return img


# --------------------------------------------------------------------------- uygulama
class Uygulama:
    def __init__(self, root, args):
        import tkinter as tk
        self.tk = tk
        self.root = root
        self.args = args
        self.cihaz = "cpu"
        self.dedektor: Dedektor | None = None
        self.durum = "Model yükleniyor…"
        self.alt_durum = "İlk açılış 10-20 saniye sürebilir."
        self.kaynak: Kaynak | None = None
        self.mod = "kamera"
        self.aktif = arama_coz(args.arama or VARSAYILAN_ARAMA) or arama_coz(VARSAYILAN_ARAMA)
        self.g = Gorunum(cihaz="cpu")
        self._son_img = None
        self.donmus: np.ndarray | None = None
        self.sonuc = None
        self.bildirim_bitis = 0.0
        self.kilit = threading.Lock()
        self.calisiyor = True
        self.tam_ekran = False
        self.d = DroneDurumu()
        self.galeri: list[Path] = []
        self.galeri_i = 0
        self.v = VideoDurumu(kareler=resimler(VIDEO_KLASORU))
        self.model_ms = 0.0
        self._son_bitis = 0.0
        self._foto = None
        self._arayuz_kur()
        threading.Thread(target=self._baslat, daemon=True).start()
        self.root.after(30, self._dongu)

    # ----- arayüz
    def _dugme(self, ust, metin, komut, vurgu=False):
        tk = self.tk
        return tk.Button(ust, text=metin, command=komut, font=("Segoe UI", 12, "bold" if vurgu else "normal"),
                         bg="#FFB020" if vurgu else KUTU_HEX, fg="#0C1016" if vurgu else "#E8EDF2",
                         activebackground="#FFC550" if vurgu else "#26354A",
                         activeforeground="#0C1016" if vurgu else "#FFFFFF",
                         relief="flat", bd=0, padx=14, pady=7, cursor="hand2", takefocus=0)

    def _arayuz_kur(self):
        tk, r = self.tk, self.root
        r.title("Canlı Nesne Tespiti · SKY LAB Akademi")
        r.configure(bg=ARKA_HEX)
        r.geometry("1280x820")
        r.minsize(900, 560)
        try:
            r.state("zoomed")  # Windows'ta ekranı kaplasın
        except tk.TclError:
            pass

        self.cubuk = tk.Frame(r, bg=PANEL_HEX, padx=16, pady=12)
        tk.Label(self.cubuk, text="Ne arayalım?", bg=PANEL_HEX, fg="#96A2B2",
                 font=("Segoe UI", 13)).pack(side="left", padx=(0, 10))
        self.giris = tk.Entry(self.cubuk, font=("Segoe UI", 18), bg=KUTU_HEX, fg="#E8EDF2",
                              insertbackground="#FFB020", relief="flat", width=16)
        self.giris.insert(0, ", ".join(tr for tr, _ in self.aktif))
        self.giris.pack(side="left", fill="x", expand=True, ipady=6)
        self._dugme(self.cubuk, "Ara", self.ara, vurgu=True).pack(side="left", padx=(8, 18))
        tk.Label(self.cubuk, text="Eşik", bg=PANEL_HEX, fg="#96A2B2", font=("Segoe UI", 12)).pack(side="left")
        self.esik_cubugu = tk.Scale(self.cubuk, from_=0.05, to=0.95, resolution=0.05, orient="horizontal",
                                    length=150, width=18, sliderlength=26, showvalue=False, takefocus=0,
                                    bg="#FFB020", troughcolor=KUTU_HEX, highlightthickness=0, bd=0,
                                    sliderrelief="flat", activebackground="#FFC550",
                                    command=lambda v: self.esik_ayarla(float(v)))
        self.esik_cubugu.set(self.g.esik)
        self.esik_cubugu.pack(side="left", padx=(6, 14))
        self.dondur_dugmesi = self._dugme(self.cubuk, "Dondur F2", self.dondur)
        self.dondur_dugmesi.pack(side="left", padx=3)
        self.nms_dugmesi = self._dugme(self.cubuk, "NMS açık F3", self.nms_degistir)
        self.nms_dugmesi.pack(side="left", padx=3)
        self.hayalet_dugmesi = self._dugme(self.cubuk, "Eşik altı F4", self.hayalet_degistir)
        self.hayalet_dugmesi.pack(side="left", padx=3)
        self._dugme(self.cubuk, "Drone F5", self.drone_degistir).pack(side="left", padx=3)

        self.drone_cubugu = tk.Frame(r, bg=PANEL_HEX, padx=16, pady=12)
        self._dugme(self.drone_cubugu, "◀  Kamera  Esc", self.drone_degistir).pack(side="left", padx=3)
        self._dugme(self.drone_cubugu, "‹", lambda: self.galeri_git(-1)).pack(side="left", padx=(12, 3))
        self._dugme(self.drone_cubugu, "›", lambda: self.galeri_git(1)).pack(side="left", padx=3)
        self._dugme(self.drone_cubugu, "Resim seç…", self.drone_resim_sec).pack(side="left", padx=3)
        self.drone_etiketi = tk.Label(self.drone_cubugu, text="", bg=PANEL_HEX, fg="#96A2B2",
                                      font=("Segoe UI", 12))
        self.drone_etiketi.pack(side="left", padx=16)
        self._dugme(self.drone_cubugu, "İleri  ▶  Enter", self.drone_ileri, vurgu=True).pack(side="right", padx=3)
        self.video_dugmesi = self._dugme(self.drone_cubugu, "Video  V", self.video_degistir)
        self.video_dugmesi.pack(side="right", padx=3)
        self.sahi_dugmesi = self._dugme(self.drone_cubugu, "SAHI açık  S", self.sahi_degistir)
        self.sahi_dugmesi.pack(side="right", padx=3)

        self.cubuk.pack(side="bottom", fill="x")
        self.tuval = tk.Canvas(r, bg=ARKA_HEX, highlightthickness=0, bd=0)
        self.tuval.pack(side="top", fill="both", expand=True)
        self.resim_id = self.tuval.create_image(0, 0, anchor="nw")

        r.bind("<Return>", lambda e: self.drone_ileri() if self.mod == "drone" else self.ara())
        r.bind("<KP_Enter>", lambda e: self.drone_ileri() if self.mod == "drone" else self.ara())
        r.bind("<space>", lambda e: self.drone_ileri() if self.mod == "drone" else None)
        r.bind("<Left>", lambda e: self.galeri_git(-1))
        r.bind("<Right>", lambda e: self.galeri_git(1))
        for tus in ("v", "V"):
            r.bind(f"<KeyPress-{tus}>", lambda e: self.video_degistir() if self.mod == "drone" else None)
        for tus in ("s", "S"):
            r.bind(f"<KeyPress-{tus}>", lambda e: self.sahi_degistir() if self.mod == "drone" else None)
        r.bind("<Up>", lambda e: self.esik_ayarla(self.g.esik + 0.05))
        r.bind("<Down>", lambda e: self.esik_ayarla(self.g.esik - 0.05))
        r.bind("<F1>", lambda e: self.yardim_degistir())
        r.bind("<F2>", lambda e: self.dondur())
        r.bind("<F3>", lambda e: self.nms_degistir())
        r.bind("<F4>", lambda e: self.hayalet_degistir())
        r.bind("<F5>", lambda e: self.drone_degistir())
        r.bind("<F6>", lambda e: self.ayna_degistir())
        r.bind("<F11>", lambda e: self.tam_ekran_degistir())
        r.bind("<Escape>", lambda e: self.geri())
        r.protocol("WM_DELETE_WINDOW", self.kapat)
        self.giris.focus_set()

    # ----- arka plan işleri
    def _baslat(self):
        try:
            import torch
            torch.set_num_threads(max(2, (os.cpu_count() or 4) - 2))  # arayüze de işlemci kalsın
            self.cihaz = cihaz_sec(self.args.cihaz)
            self.g.cihaz = self.cihaz
            self.kaynak = Kaynak(self.args.kaynak if self.args.kaynak else self.args.kamera, self.args.ayna)
            agirlik_hazirla(lambda m: setattr(self, "durum", m))
            self.durum = "Model yükleniyor…"
            dedektor = Dedektor(self.cihaz)
            self.durum = "Metin modeli hazırlanıyor…"
            dedektor.tespit(np.zeros((480, 640, 3), np.uint8), [en for _, en in self.aktif])
            self.dedektor = dedektor
            self.durum = None
        except ImportError as e:
            self.durum, self.alt_durum = "Gerekli paket eksik", f"{e}  →  pip install -r requirements.txt"
            return
        except Exception as e:
            self.durum = "Model yüklenemedi"
            self.alt_durum = f"{type(e).__name__}: {e}"[:150] + "  →  internet varken: python canli_demo.py --hazirla"
            return
        self._calis()

    def _calis(self):
        son_anahtar = None
        while self.calisiyor:
            if self.mod != "kamera" or self.kaynak is None:
                time.sleep(0.05)
                continue
            donmus = self.donmus
            kare = donmus if donmus is not None else self.kaynak.kare
            if kare is None:
                time.sleep(0.02)
                continue
            aktif, nms = tuple(self.aktif), self.g.nms
            anahtar = (id(donmus) if donmus is not None else self.kaynak.sayac, aktif, nms)
            if anahtar == son_anahtar:
                time.sleep(0.01)
                continue
            son_anahtar = anahtar
            t0 = time.perf_counter()
            try:
                tespit = self.dedektor.tespit(kare, [en for _, en in aktif], nms=nms)
            except Exception as e:
                self.bildir(f"Hata: {e}"[:90])
                time.sleep(0.5)
                continue
            simdi = time.perf_counter()
            ms = (simdi - t0) * 1000
            self.model_ms = ms if self.model_ms == 0 else 0.85 * self.model_ms + 0.15 * ms
            self.g.model_ms = self.model_ms
            if self._son_bitis:  # gerçekte saniyede kaç kare işlendi (kamera hızıyla sınırlı)
                anlik = 1.0 / max(simdi - self._son_bitis, 1e-3)
                self.g.fps = anlik if self.g.fps == 0 else 0.85 * self.g.fps + 0.15 * anlik
            self._son_bitis = simdi
            with self.kilit:
                self.sonuc = (tespit, aktif)

    # ----- ekran döngüsü
    def _dongu(self):
        if not self.calisiyor:
            return
        try:
            from PIL import ImageTk
            w, h = max(self.tuval.winfo_width(), 320), max(self.tuval.winfo_height(), 240)
            self.g.bildirim = self._bildirim_metni if time.time() < self.bildirim_bitis else None
            if self.mod == "drone":
                img = video_ciz(self.v, self.g, w, h) if self.v.aktif else drone_ciz(self.d, self.g, w, h)
                self._drone_etiketi_guncelle()
            elif self.durum or self.dedektor is None:
                img = bos_ekran(w, h, self.durum or "Hazırlanıyor…", self.alt_durum)
            else:
                kare = self.donmus if self.donmus is not None else (self.kaynak.kare if self.kaynak else None)
                if kare is None:
                    hata = self.kaynak.hata if self.kaynak else None
                    img = bos_ekran(w, h, hata or "Kamera açılıyor…",
                                    "F5 ile drone moduna geçebilirsin" if hata else None)
                else:
                    with self.kilit:
                        sonuc = self.sonuc
                    tespit, aktif = sonuc if sonuc else (BOS, tuple(self.aktif))
                    img = kamera_ciz(kare, tespit, aktif, self.g, w, h)
            self._son_img = img
            self._foto = ImageTk.PhotoImage(img)
            self.tuval.itemconfigure(self.resim_id, image=self._foto)
        except Exception as e:  # ekran döngüsü hiçbir hatada durmasın
            print("Çizim hatası:", e)
        self.root.after(15, self._dongu)

    # ----- komutlar
    def bildir(self, metin, sure=2.5):
        self._bildirim_metni = metin
        self.bildirim_bitis = time.time() + sure

    def ara(self):
        aktif = arama_coz(self.giris.get())
        if not aktif:
            self.bildir("Aranacak bir nesne yaz: ör. insan, telefon")
            return
        self.aktif = aktif
        self.bildir("Aranıyor: " + ", ".join(f"{tr} → {en}" if katla(tr) != en else tr for tr, en in aktif)[:80])

    def esik_ayarla(self, deger):
        deger = round(min(0.95, max(0.05, deger)), 2)
        if abs(deger - self.g.esik) < 1e-6:
            return
        self.g.esik = deger
        self.esik_cubugu.set(deger)

    def dondur(self):
        if self.mod != "kamera":
            return
        if self.donmus is None:
            kare = self.kaynak.kare if self.kaynak else None
            if kare is None:
                return
            self.donmus = kare
            self.g.donmus = True
            self.dondur_dugmesi.configure(text="Devam F2")
        else:
            self.donmus = None
            self.g.donmus = False
            self.dondur_dugmesi.configure(text="Dondur F2")

    def nms_degistir(self):
        self.g.nms = not self.g.nms
        self.nms_dugmesi.configure(text="NMS açık F3" if self.g.nms else "NMS kapalı F3",
                                   bg=KUTU_HEX if self.g.nms else "#7F1D1D")
        self.bildir("NMS açık: her nesneye tek kutu" if self.g.nms else "NMS kapalı: ham adayların hepsi")

    def hayalet_degistir(self):
        self.g.hayalet = not self.g.hayalet
        self.hayalet_dugmesi.configure(bg="#26354A" if self.g.hayalet else KUTU_HEX)
        self.bildir("Eşiğin altında kalan adaylar kesik çizgiyle gösteriliyor" if self.g.hayalet
                    else "Eşik altı adaylar gizlendi")

    def ayna_degistir(self):
        if self.kaynak:
            self.kaynak.ayna = not self.kaynak.ayna

    def yardim_degistir(self):
        self.g.yardim = not self.g.yardim

    def tam_ekran_degistir(self):
        self.tam_ekran = not self.tam_ekran
        self.root.attributes("-fullscreen", self.tam_ekran)

    def geri(self):
        if self.g.yardim:
            self.g.yardim = False
        elif self.v.aktif:
            self.video_degistir()
        elif self.tam_ekran:
            self.tam_ekran_degistir()
        elif self.mod == "drone":
            self.drone_degistir()

    def kapat(self):
        self.calisiyor = False
        if self.kaynak:
            self.kaynak.calisiyor = False
        self.root.after(100, self.root.destroy)

    # ----- drone modu
    def drone_degistir(self):
        if self.mod == "kamera":
            if not self.galeri:
                secilen = Path(self.args.drone) if self.args.drone else None
                if secilen and secilen.is_dir():
                    self.galeri = resimler(secilen)
                elif secilen:
                    self.galeri = [secilen]
                else:
                    self.galeri = resimler(DRONE_KLASORU)
            if self.d.resim is None and self.galeri:
                self._drone_resmi_yukle(self.galeri[self.galeri_i])
            self.mod = "drone"
            self.d.adim = 0
            self.cubuk.pack_forget()
            self.drone_cubugu.pack(side="bottom", fill="x", before=self.tuval)
            self.tuval.focus_set()
        else:
            self.v.aktif = False
            self.mod = "kamera"
            self.drone_cubugu.pack_forget()
            self.cubuk.pack(side="bottom", fill="x", before=self.tuval)
            self.giris.focus_set()

    def galeri_git(self, adim: int):
        if self.mod != "drone" or self.v.aktif or self.d.mesgul or not self.galeri:
            return
        self.galeri_i = (self.galeri_i + adim) % len(self.galeri)
        self._drone_resmi_yukle(self.galeri[self.galeri_i])

    def _drone_resmi_yukle(self, yol: Path):
        resim = resim_oku(yol)
        if resim is None:
            self.bildir(f"Resim açılamadı: {yol.name}")
            return
        self.d = DroneDurumu(resim=resim, ad=yol.name)

    def drone_resim_sec(self):
        from tkinter import filedialog
        yol = filedialog.askopenfilename(title="Drone fotoğrafı seç",
                                         filetypes=[("Resimler", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if yol:
            self._drone_resmi_yukle(Path(yol))

    def _drone_etiketi_guncelle(self):
        if self.v.aktif:
            metin = f"Video  ·  {len(self.v.kareler)} kare  ·  {'SAHI açık' if self.v.sahi else 'SAHI kapalı'}"
        else:
            metin = {0: "Adım 0 / 3  ·  ham kare", 1: "Adım 1 / 3  ·  tam kare",
                     2: "Adım 2 / 3  ·  SAHI", 3: "Adım 3 / 3  ·  karşılaştırma"}.get(self.d.adim, "")
            if self.galeri:
                metin += f"  ·  fotoğraf {self.galeri_i + 1} / {len(self.galeri)}"
        if self.drone_etiketi.cget("text") != metin:
            self.drone_etiketi.configure(text=metin)

    def drone_ileri(self):
        d = self.d
        if self.v.aktif or d.mesgul or d.resim is None:
            return
        if self.dedektor is None:
            self.bildir("Model henüz hazır değil")
            return
        if d.adim == 0:
            threading.Thread(target=self._drone_tam, daemon=True).start()
        elif d.adim == 1:
            threading.Thread(target=self._drone_dilimle, daemon=True).start()
        elif d.adim == 2 and d.dilimli is not None:
            d.adim = 3
        elif d.adim == 3:
            d.adim = 0

    def _drone_tam(self):
        d = self.d
        d.mesgul = True
        try:
            siniflar = [en for _, en in DRONE_SINIFLAR]
            self.dedektor.hazirla(siniflar)
            t0 = time.perf_counter()
            d.tam = self.dedektor.tespit(d.resim, siniflar)
            d.tam_sure = time.perf_counter() - t0
            d.adim = 1
        except Exception as e:
            self.bildir(f"Hata: {e}"[:90])
        finally:
            d.mesgul = False

    def _drone_dilimle(self):
        d = self.d
        d.mesgul = True
        try:
            siniflar = [en for _, en in DRONE_SINIFLAR]
            t0 = time.perf_counter()
            liste, parcalar = sahi_dilimleri(self.dedektor, d.resim, siniflar, self.args.dilim)
            birlesik = birlestir(np.concatenate([d.tam if d.tam is not None else BOS] + parcalar))
            d.dilim_sure = d.tam_sure + (time.perf_counter() - t0)  # gerçek hesap süresi
            # Hesap bitti; izleyenler görebilsin diye dilimleri tek tek gösteriyoruz.
            d.dilim_listesi, d.dilim_ham, d.dilimli, d.aktif_dilim = liste, [], None, 0
            d.adim = 2
            bekleme = max(0.06, 2.4 / len(liste))
            for i, parca in enumerate(parcalar):
                if not self.calisiyor or self.mod != "drone" or self.d is not d:
                    return
                d.aktif_dilim = i
                d.dilim_ham.append(parca)
                time.sleep(bekleme)
            d.dilimli = birlesik
            d.aktif_dilim = -1
        except Exception as e:
            self.bildir(f"Hata: {e}"[:90])
        finally:
            d.mesgul = False

    # ----- video: ardışık kareler, SAHI açık ya da kapalı
    def video_degistir(self):
        v = self.v
        if v.aktif:
            v.aktif = False
            self.video_dugmesi.configure(text="Video  V")
            return
        if self.mod != "drone" or self.d.mesgul:
            return
        if not v.kareler:
            self.bildir("ornekler/video klasöründe kare yok")
            return
        if self.dedektor is None:
            self.bildir("Model henüz hazır değil")
            return
        v.aktif, v.son = True, None
        self.video_dugmesi.configure(text="Durdur  V")
        threading.Thread(target=self._video_calis, daemon=True).start()

    def sahi_degistir(self):
        self.v.sahi = not self.v.sahi
        self.sahi_dugmesi.configure(text="SAHI açık  S" if self.v.sahi else "SAHI kapalı  S",
                                    bg=KUTU_HEX if self.v.sahi else "#7F1D1D")
        self.bildir("SAHI açık: tam kare + dilimler" if self.v.sahi else "SAHI kapalı: yalnızca tam kare")

    def _video_calis(self):
        v = self.v
        siniflar = [en for _, en in DRONE_SINIFLAR]
        aralik = 1.0 / VIDEO_FPS
        try:
            while v.aktif and self.calisiyor and self.mod == "drone":
                t0 = time.perf_counter()
                sira = v.i % len(v.kareler)
                kare = resim_oku(v.kareler[sira])
                if kare is None:
                    v.i = sira + 1
                    continue
                anahtar = (sira, v.sahi, self.args.dilim)
                onbellekten = anahtar in v.onbellek
                if onbellekten:
                    tespit, sure = v.onbellek[anahtar]
                else:
                    t1 = time.perf_counter()
                    tespit = self.dedektor.tespit(kare, siniflar)
                    if v.sahi:
                        liste, parcalar = sahi_dilimleri(self.dedektor, kare, siniflar, self.args.dilim)
                        tespit = birlestir(np.concatenate([tespit] + parcalar), ic_ice=ICICE_ORAN)
                        v.dilim_sayisi = len(liste)
                    sure = time.perf_counter() - t1
                    v.onbellek[anahtar] = (tespit, sure)
                v.son = (kare, tespit, sure, sira, onbellekten)
                v.i = sira + 1
                time.sleep(max(0.0, aralik - (time.perf_counter() - t0)))
        except Exception as e:
            self.bildir(f"Hata: {e}"[:90])
        finally:
            v.aktif = False
            try:
                self.root.after(0, lambda: self.video_dugmesi.configure(text="Video  V"))
            except Exception:
                pass


# --------------------------------------------------------------------------- komut satırı
def hazirla(args):
    print("1/4  Model dosyası kontrol ediliyor…")
    agirlik_hazirla()
    cihaz = cihaz_sec(args.cihaz)
    print(f"2/4  Model yükleniyor ({'GPU' if cihaz != 'cpu' else 'CPU'}). "
          "İlk seferde metin modeli (CLIP, ~350 MB) iner…")
    d = Dedektor(cihaz)
    d.tespit(np.zeros((480, 640, 3), np.uint8), [en for _, en in arama_coz(VARSAYILAN_ARAMA)])
    if cihaz != "cpu":
        import torch
        print(f"     Ekran kartı: {torch.cuda.get_device_name(0)}")
    print("3/4  Drone örneği deneniyor…")
    ornek = Path(args.drone) if args.drone else drone_ornegi()
    resim = resim_oku(ornek) if ornek else None
    if resim is None:
        print("     Drone resmi bulunamadı (ornekler/drone), bu adım atlandı.")
    else:
        siniflar = [en for _, en in DRONE_SINIFLAR]
        d.tespit(resim, siniflar)  # ısınma
        t0 = time.perf_counter()
        tam = d.tespit(resim, siniflar)
        t_tam = time.perf_counter() - t0
        t0 = time.perf_counter()
        liste, parcalar = sahi_dilimleri(d, resim, siniflar, args.dilim)
        dilimli = birlestir(np.concatenate([tam] + parcalar))
        t_dilim = t_tam + time.perf_counter() - t0
        a, b = say(tam, 0.25, 2), say(dilimli, 0.25, 2)
        print(f"     {ornek.name} ({resim.shape[1]}x{resim.shape[0]})")
        print(f"     Tam kare : insan {a[0]}, araç {a[1]}  ({sayi(t_tam)} sn)")
        print(f"     SAHI     : insan {b[0]}, araç {b[1]}  ({len(liste)} dilim, {sayi(t_dilim)} sn)")
    print(f"     Video için {len(resimler(VIDEO_KLASORU))} kare var (ornekler/video).")
    print(f"4/4  Kamera {args.kamera} kontrol ediliyor…")
    cap = kamera_ac(args.kamera)
    if cap is None:
        print("     Kamera açılamadı. Sunumda başka kamera için: --kamera 1")
    else:
        cap.release()
        print("     Kamera çalışıyor.")
    print("\nHazır. Bundan sonra internet olmadan da çalışır:  python canli_demo.py")


def ornek_cikti(args):
    """Pencere açmadan örnek ekran görüntüleri üretir (kontrol için)."""
    klasor = Path(args.ornek_cikti)
    klasor.mkdir(parents=True, exist_ok=True)
    d = Dedektor(cihaz_sec(args.cihaz))
    ornek = Path(args.drone) if args.drone else drone_ornegi()
    resim = resim_oku(args.kaynak or ornek)
    w, h = 1280, 720
    aktif = arama_coz(args.arama or VARSAYILAN_ARAMA)
    g = Gorunum(cihaz=d.cihaz, fps=11)
    tespit = d.tespit(resim, [en for _, en in aktif])
    kamera_ciz(resim, tespit, aktif, g, w, h).save(klasor / "1_kamera.png")
    g.hayalet = True
    kamera_ciz(resim, tespit, aktif, g, w, h).save(klasor / "2_esik_alti.png")
    g.hayalet, g.nms = False, False
    ham = d.tespit(resim, [en for _, en in aktif], nms=False)
    kamera_ciz(resim, ham, aktif, g, w, h).save(klasor / "3_nms_kapali.png")
    g.nms, g.yardim = True, True
    kamera_ciz(resim, tespit, aktif, g, w, h).save(klasor / "4_yardim.png")
    g.yardim = False

    dr = DroneDurumu(resim=resim_oku(ornek), ad="drone")
    siniflar = [en for _, en in DRONE_SINIFLAR]
    drone_ciz(dr, g, w, h).save(klasor / "5_drone_0.png")
    t0 = time.perf_counter()
    dr.tam = d.tespit(dr.resim, siniflar)
    dr.tam_sure = time.perf_counter() - t0
    dr.adim = 1
    drone_ciz(dr, g, w, h).save(klasor / "6_drone_1.png")
    rh, rw = dr.resim.shape[:2]
    dr.dilim_listesi = dilimler(rw, rh, args.dilim)
    dr.adim = 2
    t0 = time.perf_counter()
    for i, (x1, y1, x2, y2) in enumerate(dr.dilim_listesi):
        p = d.tespit(dr.resim[y1:y2, x1:x2], siniflar).copy()
        p[:, [0, 2]] += x1
        p[:, [1, 3]] += y1
        dr.dilim_ham.append(p)
        if i == len(dr.dilim_listesi) // 2:
            dr.aktif_dilim = i
            drone_ciz(dr, g, w, h).save(klasor / "7_drone_2_tarama.png")
    dr.dilimli = birlestir(np.concatenate([dr.tam] + dr.dilim_ham))
    dr.dilim_sure = dr.tam_sure + time.perf_counter() - t0
    dr.aktif_dilim = -1
    drone_ciz(dr, g, w, h).save(klasor / "8_drone_2_bitti.png")
    dr.adim = 3
    drone_ciz(dr, g, w, h).save(klasor / "9_drone_3.png")
    print("Kaydedildi:", klasor)


def main():
    ap = argparse.ArgumentParser(description="Canlı nesne tespiti demosu (SKY LAB Akademi)")
    ap.add_argument("--hazirla", action="store_true", help="modelleri indir ve her şeyi test et")
    ap.add_argument("--kamera", type=int, default=0, help="kamera numarası (varsayılan 0)")
    ap.add_argument("--kaynak", help="kamera yerine video ya da resim dosyası")
    ap.add_argument("--drone", help="drone modunda açılacak fotoğraf")
    ap.add_argument("--dilim", type=int, default=640, help="dilim boyutu, piksel (varsayılan 640)")
    ap.add_argument("--cihaz", default="auto", help="auto, cpu ya da cuda:0")
    ap.add_argument("--ayna", action="store_true", help="görüntüyü aynala")
    ap.add_argument("--arama", help="açılışta aranacak nesneler, ör. \"insan, telefon\"")
    ap.add_argument("--ornek-cikti", help=argparse.SUPPRESS)
    ap.add_argument("--test-kayit", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.hazirla:
        hazirla(args)
        return
    if args.ornek_cikti:
        ornek_cikti(args)
        return

    if sys.platform == "win32":
        try:  # yüksek çözünürlüklü ekranlarda bulanık görünmesin
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    import tkinter as tk
    root = tk.Tk()
    app = Uygulama(root, args)
    if args.test_kayit:
        test_senaryosu(root, app, Path(args.test_kayit))
    root.mainloop()


def test_senaryosu(root, app: Uygulama, klasor: Path):
    """Pencereyi kendi kendine gezdirip ekranları kaydeder (geliştirme sırasında kontrol için)."""
    klasor.mkdir(parents=True, exist_ok=True)

    def bekle(kosul, sonra, sinir=600):
        if kosul() or sinir <= 0:
            sonra()
        else:
            root.after(100, lambda: bekle(kosul, sonra, sinir - 1))

    def kaydet(ad):
        if app._son_img is not None:
            app._son_img.save(klasor / ad)

    def pencere(ad):
        try:  # düğme çubuğu dahil bütün pencere
            from PIL import ImageGrab
            x, y = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height())).save(klasor / ad)
        except Exception as e:
            print("Pencere görüntüsü alınamadı:", e)

    def a1():
        kaydet("tk_1_kamera.png")
        pencere("tk_pencere_kamera.png")
        app.esik_ayarla(0.10)
        app.hayalet_degistir()
        root.after(1200, a2)

    def a2():
        kaydet("tk_2_esik.png")
        app.esik_ayarla(0.25)
        app.hayalet_degistir()
        app.drone_degistir()
        root.after(800, lambda: (app.drone_ileri(), bekle(lambda: app.d.adim == 1 and not app.d.mesgul, a3)))

    def a3():
        root.after(500, lambda: (kaydet("tk_3_tam.png"), app.drone_ileri(), root.after(1100, a4)))

    def a4():
        kaydet("tk_4_tarama.png")
        bekle(lambda: app.d.dilimli is not None, lambda: root.after(400, a5))

    def a5():
        kaydet("tk_5_dilimli.png")
        app.drone_ileri()
        root.after(700, a6)

    def a6():
        kaydet("tk_6_karsilastirma.png")
        pencere("tk_pencere_drone.png")
        app.video_degistir()
        root.after(7000, a7)

    def a7():
        kaydet("tk_7_video_sahi.png")
        pencere("tk_pencere_video.png")
        app.sahi_degistir()
        root.after(3000, a8)

    def a8():
        kaydet("tk_8_video_tamkare.png")
        app.video_degistir()
        root.after(600, lambda: (app.galeri_git(1), root.after(800, a9)))

    def a9():
        kaydet("tk_9_galeri.png")
        app.drone_degistir()
        root.after(1500, lambda: (kaydet("tk_10_geri.png"), app.kapat()))

    bekle(lambda: app.dedektor is not None and app.sonuc is not None, lambda: root.after(1500, a1))


if __name__ == "__main__":
    main()
