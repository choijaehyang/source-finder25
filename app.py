# app.py
import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="자료 소스 큐레이터 MyGPT (Streamlit)",
    page_icon="🔎",
    layout="wide"
)

# -----------------------------
# 0) 카테고리 동의어 사전 (이상의 로직 반영)
# -----------------------------
CATEGORY_SYNONYMS: Dict[str, List[str]] = {
    "정부·공공 데이터": ["정부","공공","통계","상권","행정","정책","KDI","통계청","공공데이터","KOSIS","PRISM","NKIS"],
    "해외 동향": ["해외","글로벌","국제","OECD","Deloitte","PwC","Accenture","Ipsos","Consumer Reports","국외"],
    "회사 동향": ["기업","경쟁사","IR","스타트업","증권","애널리스트","컨센서스","한경","THE VC","DART","재무"],
    "산업 동향": ["산업","시장","트렌드","콘텐츠","광고","IT","무역","핀테크","프랜차이즈","식품","미디어"],
    "마케팅 조사": ["조사","서베이","여론","컨슈머","갤럽","오픈서베이","칸타","TrendWatching","패널"],
    "학술": ["학술","논문","스칼라","아카데믹","국회도서관","학위논문"],
    "뉴스레터": ["뉴스레터","어피티","뉴닉","캐릿","스타트업 위클리","콘텐타","어거스트","요약"]
}

DEFAULT_TOP_K = 10

# -----------------------------
# 1) 데이터 로딩
# -----------------------------
@st.cache_data(show_spinner=False)
def load_sources(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 필수 컬럼 검사
    need_cols = {"category","site_name","url","short_desc","tags"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {sorted(miss)}")
    # 결측치 방어
    for c in ["category","site_name","url","short_desc","tags"]:
        df[c] = df[c].fillna("")
    return df

def ensure_sources() -> Tuple[pd.DataFrame, str]:
    # 기본 파일명
    default_path = "sources.csv"
    if os.path.exists(default_path):
        return load_sources(default_path), default_path
    # 안내용 샘플(최소 셋)
    sample = pd.DataFrame([
        ["산업 동향","KOTRA 해외시장뉴스","https://news.kotra.or.kr","국가별 산업 동향/수출 트렌드","해외,산업,무역,시장"],
        ["정부·공공 데이터","공공데이터 포털","https://www.data.go.kr","공공데이터·API 제공 포털","공공데이터,API,자료"],
        ["회사 동향","DART 전자공시","https://dart.fss.or.kr","상장사 공시/IR 자료","IR,공시,재무"],
        ["학술","구글 스칼라","https://scholar.google.com","학술 논문 검색","논문,검색,학술"],
        ["뉴스레터","뉴닉","https://newneek.co","시사/용어 요약형 뉴스레터","뉴스,요약,트렌드"]
    ], columns=["category","site_name","url","short_desc","tags"])
    return sample, "(임시 샘플 메모리 로딩)"

# -----------------------------
# 2) 텍스트 전처리 & 벡터화
# -----------------------------
@st.cache_data(show_spinner=False)
def build_vectorizer(df: pd.DataFrame):
    corpus = (df["site_name"].astype(str) + " " +
              df["short_desc"].astype(str) + " " +
              df["tags"].astype(str) + " " +
              df["category"].astype(str)
             ).str.lower().tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def expand_query_with_synonyms(q: str) -> Tuple[str, List[str]]:
    q_low = q.lower()
    boost_categories = []
    for cat, syns in CATEGORY_SYNONYMS.items():
        if any(s.lower() in q_low for s in syns) or cat.lower() in q_low:
            boost_categories.append(cat)
    return q, boost_categories

# -----------------------------
# 3) 유사도 + 카테고리 가중치
# -----------------------------
def score_and_rank(
    df: pd.DataFrame,
    vectorizer,
    X,
    query: str,
    manual_boost_cats: List[str] = None,
    auto_boost: bool = True,
    boost_weight: float = 0.25
) -> pd.DataFrame:
    manual_boost_cats = manual_boost_cats or []
    q_expanded, auto_boost_cats = expand_query_with_synonyms(query) if auto_boost else (query, [])
    q_vec = vectorizer.transform([q_expanded.lower()])
    sim = cosine_similarity(q_vec, X).ravel()

    # 카테고리 가중치: 수동 + 자동
    boost_set = set(manual_boost_cats) | set(auto_boost_cats)
    if boost_set:
        cat_arr = df["category"].values
        mul = np.ones_like(sim)
        for cat in boost_set:
            mul = mul * np.where(cat_arr == cat, 1.0 + boost_weight, 1.0)
        sim = sim * mul

    out = df.copy()
    out["score"] = sim
    # tie-breaker: score desc, site_name asc
    out = out.sort_values(by=["score","site_name"], ascending=[False, True], kind="mergesort")
    return out

# -----------------------------
# 4) URL 체크(옵션)
# -----------------------------
def check_url_status(url: str, timeout=5) -> str:
    if not url or not url.startswith(("http://","https://")):
        return "N/A"
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        return str(r.status_code)
    except Exception:
        return "ERR"

# -----------------------------
# 5) UI
# -----------------------------
def main():
    st.title("🔎 자료 소스 큐레이터 MyGPT (Streamlit 버전)")
    st.caption("키워드를 입력하면, 문서( CSV )의 카테고리별 자료 출처를 연관도 순으로 정렬해 보여줍니다.")

    with st.sidebar:
        st.header("⚙️ 옵션")
        uploaded = st.file_uploader("CSV 업로드 (sources.csv 컬럼 고정)", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.success("CSV 업로드 완료")
        else:
            df, path = ensure_sources()
            if path.endswith(".csv"):
                st.info(f"CSV 사용: {path}")
            else:
                st.warning("로컬 sources.csv가 없어 샘플로 동작합니다. 실제 사용시 sources.csv를 추가하세요.")

        # 벡터화 준비
        try:
            vectorizer, X = build_vectorizer(df)
        except Exception as e:
            st.error(f"CSV 형식 오류: {e}")
            st.stop()

        # 사용자 옵션
        group_by_cat = st.toggle("카테고리별 그룹 표시", value=True)
        top_k = st.slider("출력 개수(전체 기준)", min_value=5, max_value=50, value=DEFAULT_TOP_K, step=5)
        top_per_cat = st.slider("카테고리별 상위 N (그룹 표시 시 적용)", 1, 10, 5) if group_by_cat else None

        manual_cats = st.multiselect(
            "가중치를 줄 카테고리 선택(선택)",
            sorted(df["category"].dropna().unique().tolist())
        )
        boost_weight = st.slider("카테고리 가중치(%)", 0, 100, 25, step=5) / 100.0

        do_auto_boost = st.toggle("동의어 기반 자동 카테고리 감지", value=True)
        do_url_check = st.toggle("URL 유효성 검사(느릴 수 있음)", value=False)
        show_table = st.toggle("원본 테이블 보기", value=False)

    # 메인 입력
    query = st.text_input("🔍 주제/키워드 입력", placeholder="예: 핀테크 결제 데이터 트렌드, 프랜차이즈 법률, 상권 분석 ...")
    search_btn = st.button("검색 실행", type="primary", use_container_width=True)

    if show_table:
        with st.expander("원본 데이터 미리보기"):
            st.dataframe(df, use_container_width=True, height=300)

    if not search_btn and not query:
        st.info("좌측에서 옵션을 조정하고, 위 입력창에 키워드를 입력한 뒤 **검색 실행**을 눌러주세요.")
        return

    if not query:
        st.warning("키워드를 입력해주세요.")
        return

    # 점수 계산
    ranked = score_and_rank(
        df, vectorizer, X,
        query=query,
        manual_boost_cats=manual_cats,
        auto_boost=do_auto_boost,
        boost_weight=boost_weight
    )

    # 상위 N 필터
    ranked_top = ranked.head(top_k)

    # (옵션) URL 체크
    if do_url_check:
        with st.status("URL 상태 확인 중...", expanded=False):
            ranked_top["url_status"] = ranked_top["url"].apply(check_url_status)
            time.sleep(0.3)

    # 표준 출력 포맷
    st.subheader("🔎 검색 결과")
    st.caption(f"키워드: **{query}** | 카테고리 가중치: {manual_cats or '없음'} | 자동 감지: {do_auto_boost} | 가중치계수: {boost_weight:.2f}")

    if group_by_cat:
        blocks = []
        for cat, sub in ranked_top.groupby("category", sort=False):
            sub = sub.copy()
            if top_per_cat:
                sub = sub.head(top_per_cat)
            blocks.append((cat, sub))

        for cat, sub in blocks:
            st.markdown(f"### 📂 {cat}")
            for i, row in sub.iterrows():
                score_pct = f"{row['score']*100:,.1f}%"
                url = row["url"] if row["url"] else "(URL 없음)"
                desc = row["short_desc"] if row["short_desc"] else ""
                line = f"**• {row['site_name']}** — {url} — {desc}  \n(연관도: {score_pct})"
                if do_url_check:
                    line += f"  \nURL 상태: `{row.get('url_status','N/A')}`"
                st.markdown(line)
            st.divider()
    else:
        for i, row in ranked_top.iterrows():
            score_pct = f"{row['score']*100:,.1f}%"
            url = row["url"] if row["url"] else "(URL 없음)"
            desc = row["short_desc"] if row["short_desc"] else ""
            line = f"**• [{row['category']}] {row['site_name']}** — {url} — {desc}  \n(연관도: {score_pct})"
            if do_url_check:
                line += f"  \nURL 상태: `{row.get('url_status','N/A')}`"
            st.markdown(line)

    # 다운로드
    csv_bytes = ranked_top.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 결과 CSV 다운로드",
        data=csv_bytes,
        file_name="search_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success("완료! 옵션을 바꿔 다시 검색해 보세요.")

if __name__ == "__main__":
    main()
