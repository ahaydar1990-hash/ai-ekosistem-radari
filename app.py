import os
import re
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ══════════════════════════════════════════════════════════════
# SAYFA AYARLARI VE ÖZEL CSS
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Ekosistem Radarı",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary   : #0a0e17;
    --bg-secondary : #0f1520;
    --bg-card      : #131b2e;
    --bg-hover     : #1a2540;
    --accent-cyan  : #00d4ff;
    --accent-green : #00ff88;
    --accent-amber : #ffb830;
    --accent-red   : #ff4d6d;
    --text-primary : #e2e8f0;
    --text-muted   : #64748b;
    --text-dim     : #334155;
    --border       : #1e2d45;
    --border-bright: #2a3f5f;
    --glow-cyan    : 0 0 20px rgba(0,212,255,0.15);
    --glow-green   : 0 0 20px rgba(0,255,136,0.15);
}

.stApp { background-color: var(--bg-primary); font-family: 'DM Sans', sans-serif; color: var(--text-primary); }
.main .block-container { background-color: var(--bg-primary); padding: 1.5rem 2rem 3rem 2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; border-right: 1px solid var(--border) !important; }
.radar-header { background: linear-gradient(135deg, #0f1520 0%, #131b2e 50%, #0f1520 100%); border: 1px solid var(--border-bright); border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; position: relative; overflow: hidden; }
.radar-header::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(ellipse at 30% 40%, rgba(0,212,255,0.05) 0%, transparent 60%), radial-gradient(ellipse at 70% 60%, rgba(0,255,136,0.04) 0%, transparent 60%); pointer-events: none; }
.radar-title { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: var(--accent-cyan); letter-spacing: -0.5px; margin: 0; line-height: 1.1; }
.radar-subtitle { font-size: 0.95rem; color: var(--text-muted); margin-top: 0.4rem; letter-spacing: 0.5px; }
.radar-badge { display: inline-block; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); color: var(--accent-cyan); font-family: 'Space Mono', monospace; font-size: 0.7rem; padding: 3px 10px; border-radius: 20px; margin-right: 8px; margin-top: 0.8rem; }
.radar-badge.green { background: rgba(0,255,136,0.1); border-color: rgba(0,255,136,0.3); color: var(--accent-green); }
.radar-badge.amber { background: rgba(255,184,48,0.1);  border-color: rgba(255,184,48,0.3);  color: var(--accent-amber); }
.metric-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.1rem 1.2rem; position: relative; overflow: hidden; transition: border-color 0.2s, box-shadow 0.2s; }
.metric-card:hover { border-color: var(--border-bright); box-shadow: var(--glow-cyan); }
.metric-card::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; border-radius: 12px 12px 0 0; }
.metric-card.cyan::after  { background: var(--accent-cyan); }
.metric-card.green::after { background: var(--accent-green); }
.metric-card.amber::after { background: var(--accent-amber); }
.metric-card.red::after   { background: var(--accent-red); }
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: var(--text-primary); line-height: 1; }
.metric-label { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.8px; }
.metric-icon  { font-size: 1.4rem; float: right; opacity: 0.6; }
.chat-message-user { background: rgba(0,212,255,0.06); border: 1px solid rgba(0,212,255,0.15); border-radius: 12px 12px 4px 12px; padding: 0.9rem 1.2rem; margin-bottom: 1rem; font-size: 0.95rem; color: var(--text-primary); }
.chat-message-bot { background: rgba(0,255,136,0.04); border: 1px solid rgba(0,255,136,0.12); border-radius: 12px 12px 12px 4px; padding: 0.9rem 1.2rem; margin-bottom: 1rem; font-size: 0.95rem; color: var(--text-primary); line-height: 1.65; }
.chat-label-user { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--accent-cyan); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }
.chat-label-bot  { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--accent-green); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }
.fallback-warning { background: rgba(255,184,48,0.07); border-left: 3px solid var(--accent-amber); padding: 0.5rem 0.8rem; border-radius: 0 6px 6px 0; font-size: 0.82rem; color: var(--accent-amber); margin-bottom: 0.6rem; }
.source-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px; margin-top: 0.8rem; }
.source-card { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem 1rem; font-size: 0.82rem; transition: border-color 0.2s; }
.source-card:hover { border-color: var(--border-bright); }
.source-type { font-family: 'Space Mono', monospace; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
.source-type.haber { color: var(--accent-cyan); }
.source-type.arac  { color: var(--accent-green); }
.source-title   { color: var(--text-primary); font-weight: 500; }
.source-preview { color: var(--text-muted); margin-top: 0.3rem; font-size: 0.78rem; line-height: 1.4; }
.source-link a  { color: var(--accent-cyan); text-decoration: none; font-size: 0.75rem; font-family: 'Space Mono', monospace; }
.data-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 10px; transition: border-color 0.2s, box-shadow 0.2s; }
.data-card:hover { border-color: var(--border-bright); box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.data-card-title { font-weight: 600; font-size: 0.92rem; color: var(--text-primary); margin-bottom: 0.35rem; }
.data-card-meta  { font-family: 'Space Mono', monospace; font-size: 0.68rem; color: var(--text-muted); margin-bottom: 0.4rem; }
.data-card-desc  { font-size: 0.83rem; color: #94a3b8; line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.data-card-link a { color: var(--accent-cyan); text-decoration: none; font-size: 0.75rem; font-family: 'Space Mono', monospace; }
.section-header { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green); box-shadow: 0 0 8px rgba(0,255,136,0.6); margin-right: 6px; animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.sidebar-logo    { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: var(--accent-cyan); font-weight: 700; margin-bottom: 0.2rem; }
.sidebar-version { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--text-dim); margin-bottom: 1.5rem; }
.sidebar-stat    { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid var(--border); font-size: 0.83rem; }
.sidebar-stat-label { color: var(--text-muted); }
.sidebar-stat-value { font-family: 'Space Mono', monospace; color: var(--text-primary); font-size: 0.8rem; }
.stTextInput > div > div > input { background-color: var(--bg-card) !important; border: 1px solid var(--border-bright) !important; border-radius: 10px !important; color: var(--text-primary) !important; font-family: 'DM Sans', sans-serif !important; padding: 0.7rem 1rem !important; font-size: 0.95rem !important; }
.stTextInput > div > div > input:focus { border-color: var(--accent-cyan) !important; box-shadow: 0 0 0 2px rgba(0,212,255,0.1) !important; }
.stButton > button { background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,212,255,0.05)) !important; border: 1px solid rgba(0,212,255,0.4) !important; color: var(--accent-cyan) !important; font-family: 'Space Mono', monospace !important; font-size: 0.82rem !important; border-radius: 8px !important; padding: 0.5rem 1.2rem !important; transition: all 0.2s ease !important; }
.stButton > button:hover { background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(0,212,255,0.12)) !important; border-color: var(--accent-cyan) !important; box-shadow: var(--glow-cyan) !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-muted) !important; font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important; padding: 0.6rem 1.2rem !important; border-radius: 0 !important; border-bottom: 2px solid transparent !important; text-transform: uppercase; letter-spacing: 1px; }
.stTabs [aria-selected="true"] { color: var(--accent-cyan) !important; border-bottom: 2px solid var(--accent-cyan) !important; }
h1, h2, h3, h4 { color: var(--text-primary) !important; }
#MainMenu{display:none!important;}
footer{display:none!important;}
[data-testid="stToolbar"]{display:none!important;}
[data-testid="stHeader"]{display:none!important;}
header{display:none!important;}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# YAPILANDIRMA
# ══════════════════════════════════════════════════════════════

VERI_DOSYASI   = "ai_ekosistem_verisi.csv"
FAISS_DIZINI   = "faiss_index"
GOMME_MODELI   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_BOYUTU   = 500
CHUNK_ORTUSME  = 50
GETIRME_SAYISI = 5
GEMINI_MODELI  = "gemini-2.5-flash"

def _api_anahtari_al() -> str:
    # Önce Streamlit Secrets'tan oku
    try:
        k = st.secrets.get("GEMINI_API_KEY", "")
        if k: return k
    except Exception:
        pass
    # Sonra environment variable
    k = os.environ.get("GEMINI_API_KEY", "")
    if k: return k
    # Son olarak Colab
    try:
        from google.colab import userdata
        k = userdata.get("GEMINI_API_KEY") or ""
        if k: return k
    except Exception:
        pass
    return ""

GEMINI_API_ANAHTARI = _api_anahtari_al()

RAG_PROMPT_SABLONU = """Sen "AI Ekosistem Radarı" adlı samimi ve yardımsever bir yapay zeka asistanısın.
Kullanıcıyla doğal bir sohbet şeklinde konuş. Robotik veya liste halinde değil, akıcı paragraflar halinde yanıt ver.

ZORUNLU KURALLAR:
1. DAIMA Türkçe yanıt ver.
2. Yanıtını önce BAĞLAM'daki bilgilere dayandır.
3. Bağlamda hem haber hem araç varsa, SORUYA UYGUN olanı seç.
4. Bağlamda cevap YOKSA, şunu yaz: "⚠️ Güncel veritabanımda bu bilgi yok, genel bilgime göre:"
5. Yanıtın doğal, samimi ve TAM olsun.
6. Sohbet geçmişini dikkate al ve bağlantılı yanıt ver.

BAĞLAM:
{context}

SORU: {question}

DOĞAL TÜRKÇE YANIT (tam ve akıcı):"""

ONERILEN_SORULAR = [
    "LangChain nedir ve ne işe yarar?",
    "GitHub'da en çok yıldız alan AI projeleri?",
    "FlowiseAI ile neler yapılabilir?",
    "Google Gemini son haberleri neler?",
    "Açık kaynak LLM projeleri hangileri?",
    "AI ajan geliştirmek için hangi araçlar kullanılır?",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ══════════════════════════════════════════════════════════════
# VERİ TOPLAMA FONKSİYONLARI — 24 SAATTE BİR OTOMATİK GÜNCELLEME
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def veri_topla_ve_isle() -> pd.DataFrame:
    eski_df = pd.DataFrame()
    if os.path.exists(VERI_DOSYASI):
        try:
            eski_df = pd.read_csv(VERI_DOSYASI)
        except: pass

    haberler = _haberleri_cek()
    araclar = _arac_depolarini_cek(max_sayfa=10)

    if not haberler and not araclar and not eski_df.empty:
        return eski_df

    df_yeni = pd.DataFrame(haberler + araclar)
    df_final = pd.concat([eski_df, df_yeni]).drop_duplicates(subset=['link']).reset_index(drop=True)
    df_final.to_csv(VERI_DOSYASI, index=False)
    return df_final


def _haberleri_cek(max_haber: int = 100) -> List[Dict]:
    from bs4 import BeautifulSoup
    kaynaklar = [
    "https://the-decoder.com/feed/",
    "https://venturebeat.com/category/ai/feed/",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.artificialintelligence-news.com/feed/",
    "https://towardsdatascience.com/feed",
    ]
    tum_haberler = []
    gorulmus_linkler = set()

    for kaynak_url in kaynaklar:
        try:
            yanit = requests.get(kaynak_url, headers=HEADERS, timeout=15)
            yanit.raise_for_status()
            soup = BeautifulSoup(yanit.content, "xml")
            makaleler = soup.find_all("item")[:max_haber]

            for makale in makaleler:
                baslik_tag = makale.find("title")
                baslik = baslik_tag.text.strip() if baslik_tag else ""
                if not baslik: continue

                link_tag = makale.find("link")
                link = link_tag.text.strip() if link_tag else ""

                if link in gorulmus_linkler: continue
                gorulmus_linkler.add(link)

                ozet_tag = makale.find("description")
                ozet = BeautifulSoup(ozet_tag.text, "html.parser").get_text(" ").strip() if ozet_tag else ""
                ozet = re.sub(r"\s+", " ", ozet)[:500]

                tarih_tag = makale.find("pubDate")
                tarih = tarih_tag.text.strip() if tarih_tag else ""

                kaynak_adi = kaynak_url.split("/")[2].replace("www.", "").replace("feeds.feedburner.com/", "")

                tum_haberler.append({
                    "baslik"   : baslik,
                    "ozet"     : ozet,
                    "tarih"    : tarih,
                    "link"     : link,
                    "kaynak"   : kaynak_adi,
                    "tur"      : "haber",
                    "rag_metni": f"HABER: {baslik} — {ozet}",
                })
            time.sleep(1)
        except Exception:
            continue
    return tum_haberler


def _arac_depolarini_cek(max_sayfa=10) -> List[Dict]:
    from bs4 import BeautifulSoup
    konular = [
    'ai-tools', 'llm', 'artificial-intelligence',
    'machine-learning', 'generative-ai',
    'chatgpt', 'stable-diffusion', 'image-generation',
]
    tum_araclar = []
    gorulmus = set()

    for konu in konular:
        for sayfa_no in range(1, 6):
            url = f"https://github.com/topics/{konu}?page={sayfa_no}"
            try:
                yanit = requests.get(url, headers=HEADERS, timeout=15)
                if yanit.status_code != 200: break

                soup = BeautifulSoup(yanit.text, 'html.parser')
                kartlar = soup.find_all('article', class_='border')
                if not kartlar: break

                for kart in kartlar:
                    linkler = [lk for lk in kart.find_all('a', href=True) if lk.get('href', '').startswith('/') and 'login' not in lk.get('href', '')]
                    if len(linkler) < 2: continue

                    repo_link = 'https://github.com' + linkler[1].get('href', '')
                    if repo_link in gorulmus: continue
                    gorulmus.add(repo_link)

                    repo_sahibi = linkler[0].text.strip()
                    repo_adi = linkler[1].text.strip()
                    aciklama = kart.find('p').text.strip() if kart.find('p') else ""
                    yildiz = kart.find('a', href=lambda h: h and 'stargazers' in h).text.strip() if kart.find('a', href=lambda h: h and 'stargazers' in h) else "0"
                    dil = kart.find('span', itemprop='programmingLanguage').text.strip() if kart.find('span', itemprop='programmingLanguage') else "Bilinmiyor"

                    tum_araclar.append({
                        'arac_adi': f"{repo_sahibi}/{repo_adi}",
                        'aciklama': re.sub(r'\s+', ' ', aciklama),
                        'yildiz_sayisi': re.sub(r'\s+', '', yildiz).replace('Star', ''),
                        'programlama_dili': dil,
                        'konu': konu,
                        'link': repo_link,
                        'kaynak': 'github.com/topics',
                        'tur': 'arac',
                        'rag_metni': f"AI ARACI: {repo_sahibi}/{repo_adi} (Dil: {dil}, Yıldız: {yildiz}) — {aciklama}"
                    })
                time.sleep(2)
            except Exception:
                continue
    return tum_araclar

# ══════════════════════════════════════════════════════════════
# RAG FONKSİYONLARI
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def rag_sistemini_yukle(veri_karmasik: str) -> Dict:
    try:
        df = pd.read_csv(VERI_DOSYASI, encoding="utf-8")
        df = df.dropna(subset=["rag_metni"])
        df = df[df["rag_metni"].str.strip() != ""].reset_index(drop=True)

        belgeler = []
        for _, satir in df.iterrows():
            tur    = str(satir.get("tur", "bilinmiyor"))
            baslik = str(satir.get("baslik", "") if tur == "haber" else satir.get("arac_adi", ""))
            belgeler.append(Document(
                page_content=str(satir["rag_metni"]),
                metadata={"tur": tur, "kaynak": str(satir.get("kaynak", "")),
                          "baslik": baslik, "link": str(satir.get("link", ""))},
            ))

        parcalayici = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_BOYUTU, chunk_overlap=CHUNK_ORTUSME,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        parcalar = parcalayici.split_documents(belgeler)

        gomme = HuggingFaceEmbeddings(
            model_name=GOMME_MODELI,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        faiss_yolu = os.path.join(FAISS_DIZINI, "index.faiss")
        if os.path.exists(faiss_yolu):
            vektor = FAISS.load_local(FAISS_DIZINI, gomme, allow_dangerous_deserialization=True)
        else:
            vektor = FAISS.from_documents(parcalar, gomme)
            os.makedirs(FAISS_DIZINI, exist_ok=True)
            vektor.save_local(FAISS_DIZINI)

        retriever = vektor.as_retriever(search_type="similarity", search_kwargs={"k": GETIRME_SAYISI})
        prompt    = PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT_SABLONU)
        return {"retriever": retriever, "prompt": prompt, "hazir": True}

    except Exception as hata:
        return {"retriever": None, "prompt": None, "hazir": False, "hata": str(hata)}


def gemini_cagir(prompt_metni: str) -> str:
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODELI}:generateContent?key={GEMINI_API_ANAHTARI}"
    govde   = {
        "contents": [{"parts": [{"text": prompt_metni}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 3000, "topP": 0.9},
    }
    try:
        yanit = requests.post(api_url, json=govde, headers={"Content-Type": "application/json"}, timeout=30)
        yanit.raise_for_status()
        return (yanit.json().get("candidates", [{}])[0].get("content", {})
                .get("parts", [{}])[0].get("text", "⚠️ Boş yanıt.").strip())
    except requests.exceptions.Timeout:
        return "❌ Zaman aşımı. Tekrar deneyin."
    except requests.exceptions.HTTPError as e:
        kod = str(e)
        if "429" in kod: return "❌ API kotası doldu. Birkaç dakika bekleyin."
        if "401" in kod or "403" in kod: return "❌ Geçersiz API anahtarı."
        return f"❌ HTTP Hatası: {e}"
    except Exception as e:
        return f"❌ Hata: {e}"


def rag_sorgula(soru: str, rag: Dict) -> Dict:
    soru_kucuk = soru.lower()
    haber_kelimeleri = ["haber", "gelişme", "son", "bugün", "bu hafta", "duyuru", "çıktı", "yayınlandı", "news", "gündem"]
    arac_kelimeleri  = ["araç", "tool", "proje", "repo", "github", "ücretsiz", "kullan", "öneri", "öner", "hangi"]

    haber_skoru = sum(1 for k in haber_kelimeleri if k in soru_kucuk)
    arac_skoru  = sum(1 for k in arac_kelimeleri  if k in soru_kucuk)

    ilgili = rag["retriever"].invoke(soru)

    if haber_skoru > arac_skoru:
        haberler = [b for b in ilgili if b.metadata.get("tur") == "haber"]
        araclar  = [b for b in ilgili if b.metadata.get("tur") == "arac"]
        ilgili   = haberler + araclar
    elif arac_skoru > haber_skoru:
        araclar  = [b for b in ilgili if b.metadata.get("tur") == "arac"]
        haberler = [b for b in ilgili if b.metadata.get("tur") == "haber"]
        ilgili   = araclar + haberler

    baglam_listesi = []
    for b in ilgili:
        tur = str(b.metadata.get("tur", "?")).upper()
        kaynak = str(b.metadata.get("kaynak", "?"))
        metin = str(b.page_content)
        # تم الاستغناء عن f-string هنا بالكامل لمنع أخطاء النسخ واللصق
        satir = "[" + tur + " - " + kaynak + "]:\n" + metin
        baglam_listesi.append(satir)

    baglam = "\n\n".join(baglam_listesi)

    prompt = rag["prompt"].format(context=baglam, question=soru)
    yanit = gemini_cagir(prompt)

    kaynaklar = [
        {
            "tur"      : b.metadata.get("tur", ""),
            "baslik"   : b.metadata.get("baslik", ""),
            "kaynak"   : b.metadata.get("kaynak", ""),
            "link"     : b.metadata.get("link", ""),
            "on_izleme": b.page_content[:150] + "...",
        }
        for b in ilgili
    ]
    return {"yanit": yanit, "kaynaklar": kaynaklar}

# ══════════════════════════════════════════════════════════════
# UI YARDIMCI FONKSİYONLARI
# ══════════════════════════════════════════════════════════════

def metrik_goster(deger, etiket, ikon, renk):
    st.markdown(f"""<div class="metric-card {renk}">
        <span class="metric-icon">{ikon}</span>
        <div class="metric-value">{deger}</div>
        <div class="metric-label">{etiket}</div>
    </div>""", unsafe_allow_html=True)

def haber_karti(satir):
    baslik = satir.get("baslik","")
    ozet   = str(satir.get("ozet",""))[:220]
    tarih  = str(satir.get("tarih",""))[:30]
    link   = satir.get("link","#")
    link_h = f'<div class="data-card-link"><a href="{link}" target="_blank">→ Habere git</a></div>' if link and link != "nan" else ""
    st.markdown(f"""<div class="data-card">
        <div class="data-card-meta">📰 {tarih}</div>
        <div class="data-card-title">{baslik}</div>
        <div class="data-card-desc">{ozet}</div>
        {link_h}</div>""", unsafe_allow_html=True)

def arac_karti(satir):
    ad       = satir.get("arac_adi","")
    aciklama = str(satir.get("aciklama",""))[:200]
    yildiz   = satir.get("yildiz_sayisi","?")
    dil      = satir.get("programlama_dili","")
    konu     = satir.get("konu","")
    link     = satir.get("link","#")
    meta     = f"⭐ {yildiz}"
    if dil and dil != "nan":  meta += f" · {dil}"
    if konu: meta += f" · #{konu}"
    link_h = f'<div class="data-card-link"><a href="{link}" target="_blank">→ GitHub\'a git</a></div>' if link and link != "nan" else ""
    st.markdown(f"""<div class="data-card">
        <div class="data-card-meta">🛠️ {meta}</div>
        <div class="data-card-title">{ad}</div>
        <div class="data-card-desc">{aciklama}</div>
        {link_h}</div>""", unsafe_allow_html=True)

def kaynaklar_goster(kaynaklar):
    if not kaynaklar: return
    st.markdown('<div class="section-header">Kullanılan Kaynaklar</div>', unsafe_allow_html=True)
    st.markdown('<div class="source-grid">', unsafe_allow_html=True)
    for k in kaynaklar:
        tur    = k.get("tur","")
        baslik = k.get("baslik","")[:55]
        onizl  = k.get("on_izleme","")[:120]
        kay    = k.get("kaynak","")
        link   = k.get("link","")
        tcls   = "haber" if tur == "haber" else "arac"
        tlbl   = "📰 HABER" if tur == "haber" else "🛠️ ARAÇ"
        lh     = f'<div class="source-link"><a href="{link}" target="_blank">↗ {kay}</a></div>' if link and link != "nan" else ""
        st.markdown(f"""<div class="source-card">
            <div class="source-type {tcls}">{tlbl}</div>
            <div class="source-title">{baslik}</div>
            <div class="source-preview">{onizl}...</div>
            {lh}</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# OTURUM DURUMU
# ══════════════════════════════════════════════════════════════

for anahtar, varsayilan in [
    ("sohbet_gecmisi", []),
    ("df_veri", None),
    ("rag_sistemi", None),
    ("veri_yuklendi", False),
]:
    if anahtar not in st.session_state:
        st.session_state[anahtar] = varsayilan

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ AI RADAR</div><div class="sidebar-version">v1.0.0 — Günlük Güncelleme</div>', unsafe_allow_html=True)
    veri_yukle_btn = st.button("⟳  Veriyi Güncelle", use_container_width=True)
    st.markdown("---")

    if st.session_state.df_veri is not None:
        df_sb    = st.session_state.df_veri
        haber_sb = len(df_sb[df_sb["tur"] == "haber"])
        arac_sb  = len(df_sb[df_sb["tur"] == "arac"])
        st.markdown(f"""
        <div class="sidebar-stat"><span class="sidebar-stat-label">📰 Haber</span><span class="sidebar-stat-value">{haber_sb}</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-label">🛠️ Araç</span><span class="sidebar-stat-value">{arac_sb}</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-label">📦 Toplam</span><span class="sidebar-stat-value">{haber_sb+arac_sb}</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-label">🔄 Güncelleme</span><span class="sidebar-stat-value">Her 24s</span></div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    rag_hazir = st.session_state.rag_sistemi is not None and st.session_state.rag_sistemi.get("hazir", False)
    if rag_hazir:
        st.markdown('<span class="status-dot"></span> **RAG Sistemi Aktif**', unsafe_allow_html=True)
    else:
        st.markdown("🔴 RAG Sistemi Pasif")
    st.markdown("---")
    st.markdown("**Hakkında**\n\nYapay zeka araçları ve haberleri\notomatik toplanır, RAG ile sorgulanır.\n\n**Kaynaklar:** the-decoder.com · github.com/topics\n\n**Stack:** LangChain · FAISS · Gemini · Streamlit")

# ══════════════════════════════════════════════════════════════
# VERİ YÜKLEME VE RAG BAŞLATMA
# ══════════════════════════════════════════════════════════════

if not st.session_state.veri_yuklendi or veri_yukle_btn:
    if os.path.exists(VERI_DOSYASI) and not veri_yukle_btn:
        with st.spinner("📂 Mevcut veri yükleniyor..."):
            st.session_state.df_veri = pd.read_csv(VERI_DOSYASI, encoding="utf-8")
    else:
        with st.spinner("🕷️ Veriler toplanıyor (the-decoder + GitHub)... ~30-60 saniye"):
            st.session_state.df_veri = veri_topla_ve_isle()

    if st.session_state.df_veri is not None and len(st.session_state.df_veri) > 0:
        with st.spinner("🧠 AI motoru başlatılıyor..."):
            karma = str(len(st.session_state.df_veri))
            st.session_state.rag_sistemi = rag_sistemini_yukle(karma)

    st.session_state.veri_yuklendi = True
    if veri_yukle_btn:
        st.rerun()

# ══════════════════════════════════════════════════════════════
# ANA İÇERİK
# ══════════════════════════════════════════════════════════════

df  = st.session_state.df_veri
rag = st.session_state.rag_sistemi

haber_n = len(df[df["tur"] == "haber"]) if df is not None else 0
arac_n  = len(df[df["tur"] == "arac"])  if df is not None else 0
guncelleme = datetime.now().strftime("%d.%m.%Y %H:%M")

st.markdown(f"""<div class="radar-header">
    <div class="radar-title">⚡ AI Ekosistem Radarı</div>
    <div class="radar-subtitle">Yapay zeka araçları ve haberlerini anlık izle, sorgula, keşfet.</div>
    <div>
        <span class="radar-badge">RAG</span>
        <span class="radar-badge green">CANLI</span>
        <span class="radar-badge amber">Güncelleme: {guncelleme}</span>
    </div>
</div>""", unsafe_allow_html=True)

# Metrikler
col1, col2, col3, col4 = st.columns(4)
with col1: metrik_goster(haber_n,            "Haber Kaydı",  "📰", "cyan")
with col2: metrik_goster(arac_n,             "AI Aracı",     "🛠️", "green")
with col3: metrik_goster(haber_n + arac_n,   "Toplam Kayıt", "📦", "amber")
with col4: metrik_goster("24s",              "Güncelleme",   "🔄", "red")

# Sekmeler
sekme1, sekme2, sekme3 = st.tabs(["💬  AI Asistan", "📰  Haberler", "🛠️  Araçlar"])

# ── SEKME 1: SOHBET ────────────────────────────────────────
with sekme1:
    st.markdown('<div class="section-header">Önerilen Sorular</div>', unsafe_allow_html=True)
    oneri_cols = st.columns(3)
    for i, oneri in enumerate(ONERILEN_SORULAR):
        with oneri_cols[i % 3]:
            if st.button(oneri, key=f"oneri_{i}", use_container_width=True):
                st.session_state["bekleyen_soru"] = oneri
                st.rerun()

    st.markdown("---")

    for mesaj in st.session_state.sohbet_gecmisi:
        if mesaj["rol"] == "kullanici":
            st.markdown(f'<div class="chat-message-user"><div class="chat-label-user">👤 Sen</div>{mesaj["icerik"]}</div>', unsafe_allow_html=True)
        else:
            yanit_metni = mesaj["icerik"]
            if "⚠️" in yanit_metni:
                satirlar = yanit_metni.split("\n", 1)
                uyari = satirlar[0]
                kalan = satirlar[1] if len(satirlar) > 1 else ""
                st.markdown(f'<div class="chat-message-bot"><div class="chat-label-bot">⚡ AI Radar</div><div class="fallback-warning">{uyari}</div>{kalan}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-bot"><div class="chat-label-bot">⚡ AI Radar</div>{yanit_metni}</div>', unsafe_allow_html=True)
            if mesaj.get("kaynaklar"):
                kaynaklar_goster(mesaj["kaynaklar"])

    st.markdown("---")
    col_inp, col_btn, col_tmp = st.columns([5, 1, 1])
    with col_inp:
        if "bekleyen_soru" not in st.session_state:
            st.session_state["bekleyen_soru"] = ""
        soru_girisi = st.text_input("Soru:", value=st.session_state["bekleyen_soru"], placeholder="Ör: Hangi AI araçları ücretsiz?", label_visibility="collapsed")
    with col_btn:
        sorgula_btn = st.button("SORGULA", use_container_width=True)
    with col_tmp:
        if st.button("TEMİZLE", use_container_width=True):
            st.session_state.sohbet_gecmisi = []
            st.rerun()

    if sorgula_btn and soru_girisi.strip():
        st.session_state["bekleyen_soru"] = ""

        # Rate limit koruması — sorgular arası 3 saniye bekleme
        simdi = time.time()
        son_sorgu = st.session_state.get("son_sorgu_zamani", 0)
        gecen = simdi - son_sorgu
        if gecen < 3:
            st.warning(f"⏳ Lütfen {int(3 - gecen) + 1} saniye bekleyin...")
            time.sleep(int(3 - gecen) + 1)
        st.session_state["son_sorgu_zamani"] = time.time()

        if not rag or not rag.get("hazir"):
            st.error("⚠️ RAG sistemi hazır değil. Veriyi Güncelle butonuna tıklayın.")
        elif not GEMINI_API_ANAHTARI:
            st.error("⚠️ Gemini API anahtarı bulunamadı. Secrets'a GEMINI_API_KEY ekleyin.")
        else:
            st.session_state.sohbet_gecmisi.append({"rol": "kullanici", "icerik": soru_girisi.strip()})
            with st.spinner("🔍 Veri tabanında aranıyor..."):
                sonuc = rag_sorgula(soru_girisi.strip(), rag)
            st.session_state.sohbet_gecmisi.append({"rol": "bot", "icerik": sonuc["yanit"], "kaynaklar": sonuc["kaynaklar"]})
            st.rerun()

# ── SEKME 2: HABERLER ──────────────────────────────────────
with sekme2:
    if df is None or len(df) == 0:
        st.info("Veri yüklenmedi. Sol panelden güncelleyin.")
    else:
        df_h = df[df["tur"] == "haber"].copy()
        col_a2, col_s2 = st.columns([3, 1])
        with col_a2:
            ara2 = st.text_input("Ara:", placeholder="Haberlerde ara...", label_visibility="collapsed")
        with col_s2:
            sir2 = st.selectbox("Sırala", ["En Yeni", "A–Z"], label_visibility="collapsed")

        df_h2 = df_h[df_h["baslik"].str.contains(ara2, case=False, na=False) | df_h["ozet"].str.contains(ara2, case=False, na=False)] if ara2 else df_h
        if sir2 == "A–Z": df_h2 = df_h2.sort_values("baslik")

        st.markdown(f'<div class="section-header">{len(df_h2)} Haber Bulundu</div>', unsafe_allow_html=True)
        for i in range(0, len(df_h2), 2):
            c1, c2 = st.columns(2)
            with c1: haber_karti(df_h2.iloc[i])
            if i + 1 < len(df_h2):
                with c2: haber_karti(df_h2.iloc[i + 1])

# ── SEKME 3: ARAÇLAR ───────────────────────────────────────
with sekme3:
    if df is None or len(df) == 0:
        st.info("Veri yüklenmedi. Sol panelden güncelleyin.")
    else:
        df_a = df[df["tur"] == "arac"].copy()
        ca1, ca2, ca3 = st.columns([3, 1, 1])
        with ca1:
            ara3 = st.text_input("Ara:", placeholder="Araçlarda ara...", label_visibility="collapsed", key="ara_arac")

        with ca2:
            if "konu" in df_a.columns:
                konular3  = ["Tümü"] + sorted(df_a["konu"].dropna().unique().tolist())
            else:
                konular3 = ["Tümü"]
            sec_konu3 = st.selectbox("Konu", konular3, label_visibility="collapsed")

        with ca3:
            if "programlama_dili" in df_a.columns:
                diller3  = ["Tüm Diller"] + sorted(df_a["programlama_dili"].dropna().replace("", pd.NA).dropna().unique().tolist())
            else:
                diller3 = ["Tüm Diller"]
            sec_dil3 = st.selectbox("Dil", diller3, label_visibility="collapsed")

        df_a2 = df_a.copy()
        if ara3:       df_a2 = df_a2[df_a2["arac_adi"].str.contains(ara3, case=False, na=False) | df_a2["aciklama"].str.contains(ara3, case=False, na=False)]
        if sec_konu3 != "Tümü":      df_a2 = df_a2[df_a2["konu"] == sec_konu3]
        if sec_dil3  != "Tüm Diller": df_a2 = df_a2[df_a2["programlama_dili"] == sec_dil3]

        st.markdown(f'<div class="section-header">{len(df_a2)} Araç Bulundu</div>', unsafe_allow_html=True)
        for i in range(0, len(df_a2), 3):
            cols3 = st.columns(3)
            for j, col in enumerate(cols3):
                if i + j < len(df_a2):
                    with col: arac_karti(df_a2.iloc[i + j])
