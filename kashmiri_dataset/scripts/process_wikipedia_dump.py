import bz2
import xml.etree.ElementTree as ET
import re
import json
import unicodedata
from pathlib import Path

BASE_DIR = Path("/home/musa/Kashmiri Project")
WIKI_BZ2 = BASE_DIR / "kashmiri_dataset" / "raw" / "external" / "kashmiri_wikipedia" / "kswiki-latest-pages-articles.xml.bz2"
OUT_DIR = BASE_DIR / "kashmiri_dataset" / "processed" / "text" / "wikipedia"
OUT_FILE = OUT_DIR / "kashmiri_wikipedia_cleaned.jsonl"
OUT_TXT = OUT_DIR / "kashmiri_wikipedia_plain.txt"

print(f"Opening Kashmiri Wikipedia XML dump: {WIKI_BZ2}")

def clean_wikitext(text):
    # Remove comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove templates {{...}}
    text = re.sub(r'\{\{[^\{\}]*\}\}', '', text)
    text = re.sub(r'\{\{[^\{\}]*\}\}', '', text) # nested pass
    # Remove files/images [[File:...]]
    text = re.sub(r'\[\[(?:File|Image|فائل):[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    # Convert internal links [[Target|Label]] -> Label, [[Target]] -> Target
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Remove external links [http... Label]
    text = re.sub(r'\[http[^\s\]]+(?:\s+([^\]]+))?\]', r'\1', text)
    # Remove references <ref...>...</ref> and <ref.../>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<ref[^/>]*/>', '', text, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove wiki headers == Header ==
    text = re.sub(r'={2,}[^=]+={2,}', '', text)
    # Remove bullet points and indents
    text = re.sub(r'^[*#;:]+\s*', '', text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

article_count = 0
sentence_count = 0
word_count = 0
char_count = 0

with bz2.open(WIKI_BZ2, 'rt', encoding='utf-8') as f_in, \
     open(OUT_FILE, 'w', encoding='utf-8') as f_out, \
     open(OUT_TXT, 'w', encoding='utf-8') as f_txt:
    
    # Fast iterative XML parse
    title = None
    text_buffer = []
    in_text = False
    in_page = False
    
    for event, elem in ET.iterparse(f_in, events=('start', 'end')):
        tag = elem.tag.split('}')[-1] # strip namespace
        
        if event == 'start' and tag == 'page':
            in_page = True
            title = None
        elif event == 'end' and tag == 'title' and in_page:
            title = elem.text
        elif event == 'end' and tag == 'text' and in_page:
            raw_text = elem.text or ""
            # Filter non-articles (redirects, templates, categories)
            if title and not any(title.startswith(ns) for ns in ['Category:', 'Template:', 'Wikipedia:', 'Help:', 'File:', 'MediaWiki:', 'Module:']):
                if not raw_text.strip().lower().startswith('#redirect'):
                    cleaned = clean_wikitext(raw_text)
                    if len(cleaned) > 50: # min length filter
                        article_count += 1
                        lines = [line.strip() for line in cleaned.split('\n') if len(line.strip()) > 10]
                        doc_sentences = []
                        for l in lines:
                            # split into sentences on common delimiters (۔, ।, ?, !, \n)
                            sents = re.split(r'[۔।?!]+', l)
                            for s in sents:
                                s = s.strip()
                                if len(s) > 5:
                                    doc_sentences.append(s)
                                    sentence_count += 1
                                    words = s.split()
                                    word_count += len(words)
                                    char_count += len(s)
                                    f_txt.write(s + "\n")
                        
                        doc_record = {
                            "id": f"kswiki_{article_count}",
                            "title": title,
                            "sentences": doc_sentences,
                            "text": "\n".join(doc_sentences),
                            "source": "kashmiri_wikipedia_dump_2026",
                            "license": "CC-BY-SA 3.0"
                        }
                        f_out.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
            elem.clear()
        elif event == 'end' and tag == 'page':
            in_page = False
            elem.clear()

print(f"Wikipedia extraction complete!")
print(f"  Articles extracted: {article_count:,}")
print(f"  Sentences: {sentence_count:,}")
print(f"  Words: {word_count:,}")
print(f"  Characters: {char_count:,}")
print(f"  Saved JSONL to: {OUT_FILE}")
print(f"  Saved Plain text to: {OUT_TXT}")
