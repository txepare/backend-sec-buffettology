import requests
import re
import json

headers = {'User-Agent': 'BuffettologyBot jaime@example.com'}
r = requests.get('https://www.sec.gov/files/company_tickers.json', headers=headers)
data = r.json()

def normalize_name(name):
    name = name.lower()
    name = re.sub(r'[\.,/\\#\$\%\^\&\*\;\:\{\}\=\_\`\~\(\)\-\[\]]', ' ', name)
    suffixes = {
        'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 
        'ltd', 'limited', 'plc', 'llc', 'lp', 'group', 'holdings', 
        'holding', 'class a', 'class b', 'class c', 'com', 'new', 'de', 'the'
    }
    tokens = [w for w in name.split() if w not in suffixes]
    return ' '.join(tokens).strip()

companies = []
ticker_map = {}
norm_title_map = {}

for k, v in data.items():
    ticker = v['ticker'].upper().strip()
    # Normalize ticker variations (e.g. BRK-B vs BRK.B)
    ticker_std = ticker.replace('.', '-')
    title = v['title'].strip()
    cik = str(v['cik_str']).zfill(10)
    norm_title = normalize_name(title)
    
    item = {'ticker': ticker, 'ticker_std': ticker_std, 'title': title, 'cik': cik, 'norm_title': norm_title}
    companies.append(item)
    ticker_map[ticker] = item
    ticker_map[ticker_std] = item
    ticker_map[ticker.replace('-', '.')] = item
    if norm_title and norm_title not in norm_title_map:
        norm_title_map[norm_title] = item

def resolve(query):
    clean_q = query.strip()
    upper_q = clean_q.upper()
    std_q = upper_q.replace('.', '-')
    
    # 1. Direct ticker
    if upper_q in ticker_map:
        return ticker_map[upper_q], "direct_ticker"
    if std_q in ticker_map:
        return ticker_map[std_q], "std_ticker"
        
    # 2. Normalized title exact
    norm_q = normalize_name(clean_q)
    if norm_q in norm_title_map:
        return norm_title_map[norm_q], "exact_norm_title"
        
    # 3. Starts with normalized title
    for c in companies:
        if c['norm_title'].startswith(norm_q) and len(c['norm_title']) <= len(norm_q) + 15:
            return c, "prefix_norm_title"
            
    # 4. Words match / Substring match
    for c in companies:
        if norm_q in c['norm_title']:
            return c, "contains_norm_title"
            
    # 5. Yahoo Finance search fallback (for colloquial names like Google -> GOOGL, Facebook -> META)
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={clean_q}&quotesCount=5&newsCount=0"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        if resp.status_code == 200:
            quotes = resp.json().get('quotes', [])
            for q in quotes:
                sym = q.get('symbol', '').upper().replace('.', '-')
                if sym in ticker_map:
                    return ticker_map[sym], "yahoo_search_fallback"
    except Exception as e:
        pass
        
    return None, "not_found"

test_queries = [
    'AAPL', 'aapl', 'Apple', 'apple', 'Microsoft', 'coca cola', 'Coca-Cola', 
    'Alphabet', 'Google', 'Amazon', 'Tesla', 'Nvidia', 'Berkshire', 'Meta', 
    'Facebook', 'Nike', 'Visa', 'Johnson & Johnson', 'Walt Disney', 'Disney',
    'BRK.B', 'BRK-B', 'brk.b', 'Costco', 'Walmart', 'McDonalds', 'Netflix'
]

for q in test_queries:
    res, method = resolve(q)
    if res:
        print(f"'{q}' -> {res['ticker']} ({res['title']}) [method: {method}]")
    else:
        print(f"'{q}' -> NOT FOUND")
