"""행정안전부 2026-09 수원시 연령별 인구현황 XLSX 검증 및 시각화.

입력: 행정안전부에서 내려받은 XLSX (65~69세 ~ 100세 이상, 남/여 구분)
출력: 구별/동별 고령인구 CSV, 고령인구 비율 그래프, 분석 요약문.
주의: 주민등록 고령인구는 전체 교통약자 수나 이동 불편 관측치가 아닙니다.

실행: python src/analyze_202609_population.py --input data/raw/202609_202609_연령별인구현황_월간.xlsx --out results
의존성: matplotlib (그래프 제작). 원본 XLSX 파싱은 파이썬 표준 라이브러리만 사용.
"""
from __future__ import annotations
import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

S = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
CITY_CODE = '4111000000'
GU_NAMES = ['장안구', '권선구', '팔달구', '영통구']
SOURCE_URL = 'https://jumin.mois.go.kr/ageStatMonth.do'

def load_cells(path: Path) -> dict[int, dict[str, str]]:
    """xlsx XML에서 첫 번째 워크시트의 셀 문자열 읽기; 공유문자열/인라인 지원."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall(f'{S}si'):
                shared.append(''.join(n.text or '' for n in si.iter(f'{S}t')))
        sheet_path = 'xl/worksheets/sheet1.xml'
        if sheet_path not in z.namelist():
            raise ValueError('sheet1.xml을 찾을 수 없습니다. 행안부 원본 형식인지 확인하세요.')
        root = ET.fromstring(z.read(sheet_path))
    result = {}
    for row in root.findall(f'.//{S}sheetData/{S}row'):
        row_number = int(row.attrib['r'])
        cells = {}
        for c in row.findall(f'{S}c'):
            key = re.match(r'[A-Z]+', c.attrib['r']).group()
            kind = c.attrib.get('t')
            v = c.find(f'{S}v')
            if kind == 'inlineStr':
                inline = c.find(f'{S}is')
                cells[key] = ''.join(n.text or '' for n in inline.iter(f'{S}t')) if inline is not None else ''
            elif v is not None:
                cells[key] = shared[int(v.text)] if kind == 's' else (v.text or '')
            else:
                cells[key] = ''
        result[row_number] = cells
    return result

def n(value: str) -> int:
    return int(str(value).replace(',', '').strip())

def prepare(path: Path):
    rows = load_cells(path)
    assert '2026년09월' in ''.join(rows[2].values()).replace(' ', ''), '원본 기준월이 2026-09가 아닙니다.'
    header = rows[4]
    assert header.get('E') == '65~69세' and header.get('L') == '100세 이상', '65세 이상 범위를 확인하세요.'
    output = []
    for idx in sorted(rows):
        if idx < 5 or not rows[idx].get('A', '').isdigit():
            continue
        r = rows[idx]
        code = str(r['A']).strip()
        place = r['B'].strip().replace('경기도 수원시', '').strip()
        gu = next((g for g in GU_NAMES if g in place), None)
        dong = place.split()[-1] if place else '수원시'
        level = '시' if code == CITY_CODE else ('구' if code.endswith('00000') else '동')
        total, senior = n(r['C']), n(r['D'])
        male, female = n(r['N']), n(r['X'])
        bands = [n(r[k]) for k in ['E','F','G','H','I','J','K','L']]
        aged80 = sum(bands[3:])
        if sum(bands) != senior or male + female != senior:
            raise ValueError(f'{idx}행 {place}: 연령별 또는 성별 합계 불일치')
        assert 0 <= senior <= total
        output.append(dict(code=code, level=level, gu=gu or '', dong=dong,
                           total_population=total, elderly_65_plus=senior,
                           elderly_pct=round(100*senior/total, 4),
                           elderly_male=male, elderly_female=female,
                           elderly_80_plus=aged80))
    city = [r for r in output if r['level'] == '시']
    gu_rows = [r for r in output if r['level'] == '구']
    dongs = [r for r in output if r['level'] == '동']
    if len(city) != 1 or len(gu_rows) != 4 or len(dongs) != 44:
        raise ValueError(f'기대하는 수원시 1행/구 4행/동 44행과 다릅니다: {len(city)},{len(gu_rows)},{len(dongs)}')
    for metric in ('total_population', 'elderly_65_plus', 'elderly_male', 'elderly_female', 'elderly_80_plus'):
        if sum(r[metric] for r in gu_rows) != city[0][metric] or sum(r[metric] for r in dongs) != city[0][metric]:
            raise ValueError(f'수원시/구/동 합계가 불일치: {metric}')
        for g in gu_rows:
            if sum(r[metric] for r in dongs if r['gu'] == g['gu']) != g[metric]:
                raise ValueError(f'{g["gu"]} 동별 합계 불일치: {metric}')
    return city[0], gu_rows, dongs

def save_csv(path: Path, records):
    columns = ['code','level','gu','dong','total_population','elderly_65_plus','elderly_pct',
               'elderly_male','elderly_female','elderly_80_plus']
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader(); w.writerows(records)

def plot_records(gu_rows, dongs, out: Path, city_pct: float):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    for family in ['NanumGothic','NanumSquare','Noto Sans CJK KR']:
        if family in fonts:
            plt.rcParams['font.family'] = family
            break
    plt.rcParams['axes.unicode_minus'] = False

    groups = sorted(gu_rows, key=lambda r:r['elderly_pct'], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    b = ax.bar([r['gu'] for r in groups], [r['elderly_pct'] for r in groups])
    ax.axhline(city_pct, linestyle='--', linewidth=1, label=f'수원시 전체 {city_pct:.2f}%')
    ax.set_ylabel('65세 이상 인구 비율 (%)')
    ax.set_ylim(0, max(r['elderly_pct'] for r in groups) * 1.23)
    ax.set_title('수원시 4개 구별 고령인구 비율 | 2026년 9월')
    ax.legend()
    for bar, row in zip(b, groups):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.3,
                f'{row["elderly_pct"]:.2f}%\n({row["elderly_65_plus"]:,}명)', ha='center', fontsize=9)
    fig.text(.5,.015,'자료: 행정안전부 주민등록 인구통계 · 2026-09-30 · 외국인 제외',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1])
    fig.savefig(out/'suwon_gu_elderly_pct_202609.png',dpi=180)
    plt.close(fig)

    for key, filename, label, title, fmt in [
        ('elderly_pct','suwon_dong_elderly_pct_top10_202609.png','65세 이상 인구 비율 (%)',
         '수원시 고령인구 비율 상위 10개 동 | 2026년 9월', lambda x:f'{x:.2f}%'),
        ('elderly_65_plus','suwon_dong_elderly_count_top10_202609.png','65세 이상 인구 (명)',
         '수원시 고령인구 수 상위 10개 동 | 2026년 9월', lambda x:f'{x:,}명')
    ]:
        top = sorted(dongs, key=lambda r:r[key], reverse=True)[:10]
        names = [r['gu']+' '+r['dong'] for r in top][::-1]
        values = [r[key] for r in top][::-1]
        fig, ax = plt.subplots(figsize=(9, 6.3))
        ax.barh(names, values)
        ax.set_xlim(0,max(values)*1.21)
        ax.set_xlabel(label); ax.set_title(title)
        for y, value in enumerate(values):
            ax.text(value+max(values)*.015,y,fmt(value), va='center',fontsize=9)
        fig.text(.5,.01,'자료: 행정안전부 주민등록 인구통계 · 2026-09-30 · 외국인 제외',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.035,1,1]); fig.savefig(out/filename,dpi=180); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path,required=True)
    parser.add_argument('--out', type=Path,default=Path('results'))
    args=parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    city, gu, dongs = prepare(args.input)
    save_csv(args.out/'suwon_dong_65plus_202609.csv', sorted(dongs,key=lambda r:(r['gu'],r['dong'])))
    save_csv(args.out/'suwon_gu_65plus_202609.csv', gu)
    plot_records(gu, dongs, args.out, city['elderly_pct'])
    highest=sorted(dongs,key=lambda r:r['elderly_pct'],reverse=True)[0]
    most=sorted(dongs,key=lambda r:r['elderly_65_plus'],reverse=True)[0]
    brief=(f'수원시 2026년 9월말 주민등록 인구 분석\n'
           f'전체 인구 {city["total_population"]:,}명, 65세 이상 {city["elderly_65_plus"]:,}명, '
           f'고령인구 비율 {city["elderly_pct"]:.2f}%.\n'
           f'동 44곳, 구 4곳; 구·동 합계 및 성별/연령별 합계 일치 확인.\n'
           f'고령인구 비율 최고 동: {highest["gu"]} {highest["dong"]} '
           f'{highest["elderly_pct"]:.2f}%.\n'
           f'고령인구 수 최고 동: {most["gu"]} {most["dong"]} '
           f'{most["elderly_65_plus"]:,}명.\n'
           f'출처: {SOURCE_URL}\n'
           '주의: 주민등록인구는 외국인을 제외하며, 65세 이상 인구는 교통약자 전체 인구도 이동불편 실측 결과도 아닙니다.\n')
    (args.out/'analysis_summary_202609.txt').write_text(brief,encoding='utf-8')
    print(brief)

if __name__ == '__main__': main()
