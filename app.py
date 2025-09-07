import pandas as pd

rows = [
    # 1. 산업 동향
    ["산업 동향","대한민국 정책브리핑","https://www.korea.kr","국내 경제·정책 동향과 지표 제공","정책,경제,통계,자료"],
    ["산업 동향","KOTRA 해외시장뉴스","https://news.kotra.or.kr","국가별 산업 동향 및 수출 트렌드","해외,산업,무역,시장"],
    ["산업 동향","ITFIND (IT 지식포털)","https://www.itfind.or.kr","IT 업계 동향/자료 모음","IT,동향,리포트"],
    ["산업 동향","한국광고총연합회 광고정보센터","https://www.adic.or.kr","광고/마케팅 자료와 크리에이티브 사례","광고,마케팅,사례"],
    ["산업 동향","한국콘텐츠진흥원(KOCCA)","https://www.kocca.kr","게임/드라마/영화 등 콘텐츠 산업 동향","콘텐츠,정책,동향"],
    ["산업 동향","네이버 트렌드","https://datalab.naver.com/keyword/trendSearch.naver","국내 검색어 트렌드 분석","검색트렌드,국내"],
    ["산업 동향","구글 트렌드","https://trends.google.com","전 세계 검색어 트렌드 분석","검색트렌드,해외"],
    ["산업 동향","빅카인즈(BigKinds)","https://www.bigkinds.or.kr","뉴스 빅데이터 분석","뉴스,빅데이터,분석"],
    ["산업 동향","한국무역협회(KITA)","https://www.kita.net","무역 동향 및 통계","무역,통계,보고서"],
    ["산업 동향","DMC리포트","https://www.dmcreport.co.kr","디지털 미디어/광고/마케팅 자료","미디어,광고,인사이트"],
    ["산업 동향","한국예탁결제원 크라우드넷","https://www.crowdnet.or.kr","크라우드펀딩 통계/현황","크라우드펀딩,통계"],
    ["산업 동향","한국핀테크산업협회","https://www.fintechkorea.org","핀테크 업계 동향/자료","핀테크,산업"],
    ["산업 동향","한국식품산업협회","https://www.kfia.or.kr","식품산업 동향/통계","식품,산업,통계"],
    ["산업 동향","한국프랜차이즈산업협회","https://www.kfa.or.kr","프랜차이즈 산업 동향/자료","프랜차이즈,산업"],
    ["산업 동향","LG경영연구원","https://www.lgeri.com","경제/산업 분석 리포트","경제,산업,리서치"],

    # 2. 회사 동향
    ["회사 동향","DART 전자공시","https://dart.fss.or.kr","상장사 공시/IR 자료","IR,공시,재무"],
    ["회사 동향","CRETOP(비상장 기업)","","비상장 기업 정보","비상장,기업정보"],
    ["회사 동향","THE VC","https://thevc.kr","스타트업/투자 정보","스타트업,투자"],
    ["회사 동향","혁신의 숲","","기업 규모/매출/사용자 지표 등","기업정보,지표"],
    ["회사 동향","네이버 증권","https://finance.naver.com","리서치/산업/기업 리포트","증권,리포트"],
    ["회사 동향","한경컨센서스","https://consensus.hankyung.com","애널리스트 리포트/전망","리포트,애널리스트"],
    ["회사 동향","MK증권","https://vip.mk.co.kr","기업/산업 리포트","리포트,산업"],
    ["회사 동향","한국은행 기업경영분석","https://www.bok.or.kr","업종별 재무지표","재무지표,통계"],
    ["회사 동향","삼정KPMG Insight","https://home.kpmg/kr","산업/경영 전략 인사이트","컨설팅,전략"],
    ["회사 동향","삼일PwC Insight","https://www.pwc.com/kr","산업/경영 전략 인사이트","컨설팅,전략"],

    # 3. 해외 동향
    ["해외 동향","OECD","https://www.oecd.org","국가별 경제/사회 데이터","정책,데이터"],
    ["해외 동향","Ipsos","https://www.ipsos.com","글로벌 마케팅 리서치 리포트","리서치,설문"],
    ["해외 동향","Accenture","https://www.accenture.com","디지털/기술 트렌드 리포트","디지털,트렌드"],
    ["해외 동향","Deloitte Insights","https://www2.deloitte.com","산업/트렌드 리포트","트렌드,컨설팅"],
    ["해외 동향","PwC Global","https://www.pwc.com","글로벌 트렌드/인사이트","컨설팅,트렌드"],
    ["해외 동향","Consumer Reports","https://www.consumerreports.org","소비자 제품 리뷰/비교","소비자,제품리뷰"],

    # 4. 정부·공공 데이터
    ["정부·공공 데이터","통계청 MDIS","https://mdis.kostat.go.kr","국가승인통계 데이터베이스","통계,MDIS"],
    ["정부·공공 데이터","공공데이터 포털","https://www.data.go.kr","공공데이터 개방/다운로드/API","공공데이터,API"],
    ["정부·공공 데이터","KDI 한국개발연구원","https://www.kdi.re.kr","경제 연구 보고서","경제,연구"],
    ["정부·공공 데이터","PRISM 정책연구관리시스템","https://www.prism.go.kr","정부 정책연구 보고서","정책,연구"],
    ["정부·공공 데이터","국가정책연구포털(NKIS)","https://www.nkis.re.kr","정책 연구 보고서 통합 포털","정책,포털"],
    ["정부·공공 데이터","소상공인 상권분석 시스템","https://sg.sbiz.or.kr","상권/유동인구/예상매출 분석","상권,소상공인"],
    ["정부·공공 데이터","KOSIS 국가통계포털","https://kosis.kr","국가 통계 포털","통계,국가"],

    # 5. 마케팅 조사 회사
    ["마케팅 조사","컨슈머 인사이트","https://www.consumerinsight.co.kr","국내 소비자 조사 데이터","소비자,조사"],
    ["마케팅 조사","한국갤럽조사연구소","https://www.gallup.co.kr","여론/소비자 조사","여론,조사"],
    ["마케팅 조사","오픈서베이","https://www.opensurvey.co.kr","모바일 패널 기반 조사 리포트","모바일,서베이"],
    ["마케팅 조사","칸타 코리아","https://www.kantar.com/kr","미디어/광고/이커머스 리포트","광고,미디어"],
    ["마케팅 조사","TrendWatching","https://www.trendwatching.com","글로벌 트렌드 분석","트렌드,글로벌"],

    # 6. 학술
    ["학술","구글 스칼라","https://scholar.google.com","학술 논문 검색","논문,검색"],
    ["학술","네이버 아카데믹","https://academic.naver.com","국내 학술/논문 검색","학술,검색"],
    ["학술","전자국회도서관","https://dl.nanet.go.kr","국내 학술지/학위논문/발간자료","국회도서관,논문"],

    # 7. 뉴스레터
    ["뉴스레터","어피티(머니레터)","https://uppity.co.kr","이슈별 경제 뉴스레터","경제,뉴스레터"],
    ["뉴스레터","뉴닉","https://newneek.co","시사 이슈 요약 뉴스레터","뉴스,요약"],
    ["뉴스레터","콘텐타","https://www.contenta.co.kr","마케팅/콘텐츠 뉴스레터","마케팅,콘텐츠"],
    ["뉴스레터","어거스트(미디어)","","미디어 이슈 큐레이션","미디어,뉴스레터"],
    ["뉴스레터","안전가옥","https://safehouse.kr","문화/스토리 산업 이슈","문화,스토리"],
    ["뉴스레터","캐릿","https://www.careet.net","Z세대 트렌드 리포트/뉴스레터","Z세대,트렌드"],
    ["뉴스레터","스타트업 위클리","","스타트업 동향 뉴스레터","스타트업,뉴스레터"],
]

df = pd.DataFrame(rows, columns=["category","site_name","url","short_desc","tags"])
csv_path = "/mnt/data/mygpt_sources_seed.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

import textwrap, json, os, io

# Also prepare a short README for how to update the CSV
readme = """
My GPT 자료 소스 CSV 사용법 (요약)

1) 'mygpt_sources_seed.csv'를 엽니다.
2) url이 빈 항목은 공식 홈페이지/리포트 센터 URL을 채웁니다.
3) short_desc와 tags는 검색 정확도를 높이는 역할을 합니다. 한국어, 쉼표 구분.
4) 저장 후 GPT 빌더의 Knowledge에 업로드하세요.
"""
with open("/mnt/data/README_MyGPT_Sources.txt","w",encoding="utf-8") as f:
    f.write(readme.strip())

csv_path, "/mnt/data/README_MyGPT_Sources.txt"
