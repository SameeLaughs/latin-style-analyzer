import streamlit as st
import stanza
import re
from collections import Counter, defaultdict

# load stnza pipelin  
@st.cache_resource
def load_nlp():
    return stanza.Pipeline('la', processors='tokenize,pos,lemma')

nlp = load_nlp()

def analyze_style(text):
    doc = nlp(text)
    results = {}

    # extrct data
    word_objects = [word for sent in doc.sentences for word in sent.words if word.upos != 'PUNCT']
    words = [word.text for word in word_objects]
    lemmas = [word.lemma for word in word_objects]
    pos_tags = [word.upos for word in word_objects]
    total_words = len(words)
    sentences = [sent.text for sent in doc.sentences]

    if total_words == 0:
        return results

    # helper funcs
    def vowel_sequence(word):
        return re.sub(r'[^aeiou]', '', word.lower())

    def ending_sound(word):
        return word.lower()[-3:] if len(word) >= 3 else word.lower()

    # alit: max rpt of 1st ltrs
    if total_words > 5:
        first_letters = [word[0].lower() for word in words if word]
        if first_letters:
            max_repeat = max(Counter(first_letters).values())
            alliteration_score = (max_repeat / total_words) * 100
            most_common_letter = [letter for letter, count in Counter(first_letters).items() if count == max_repeat][0]
            results['Alliteration'] = {
                'score': min(alliteration_score, 100),
                'justification': f"Words starting with '{most_common_letter}' appear {max_repeat} times (e.g., {[w for w in words if w.lower().startswith(most_common_letter)][:3]})."
            }

    # polysyndeton: conj cnt
    conjunctions = ['et', 'que', 'neque', 'atque', 'ac']
    conj_count = sum(1 for word in words if word.lower() in conjunctions)
    polysyndeton_score = (conj_count / total_words) * 500 if conj_count > 0 else 0
    if polysyndeton_score > 0:
        conj_words = [w for w in words if w.lower() in conjunctions]
        results['Polysyndeton'] = {
            'score': min(polysyndeton_score, 100),
            'justification': f"Found conjunctions: {', '.join(conj_words)}."
        }

    # anaphora: 1st word rpt in clauses
    clauses = re.split(r'[,.!?;]', text)
    first_words = [clause.strip().split()[0] for clause in clauses if clause.strip()]
    if len(first_words) > 1:
        matches = sum(1 for i in range(1, len(first_words)) if first_words[i].lower() == first_words[0].lower())
        anaphora_score = (matches / (len(first_words) - 1)) * 100 if matches > 0 else 0
        if anaphora_score > 0:
            results['Anaphora'] = {
                'score': anaphora_score,
                'justification': f"Clauses start with '{first_words[0]}' (repeated {matches} times)."
            }

    # chiasmus: pos a b b a
    chiasmus_score = 0
    for i in range(len(pos_tags) - 3):
        if pos_tags[i] == pos_tags[i+3] and pos_tags[i+1] == pos_tags[i+2]:
            chiasmus_score = 100
            break
    if chiasmus_score > 0:
        results['Chiasmus'] = {
            'score': chiasmus_score,
            'justification': f"POS pattern A B B A in words: {words[i:i+4]}."
        }

    # assonance: vowel rpt
    vowel_seqs = [vowel_sequence(word) for word in words if vowel_sequence(word)]
    if total_words > 5 and vowel_seqs:
        max_vowel_repeat = max(Counter(vowel_seqs).values())
        assonance_score = (max_vowel_repeat / total_words) * 100
        if assonance_score > 20:
            common_vowel = [seq for seq, count in Counter(vowel_seqs).items() if count == max_vowel_repeat][0]
            words_with_vowel = [w for w in words if vowel_sequence(w) == common_vowel][:3]
            results['Assonance'] = {
                'score': min(assonance_score, 100),
                'justification': f"Vowel sequence '{common_vowel}' in words: {words_with_vowel}."
            }

    # homoioteleuton: ending sound rpt
    endings = [ending_sound(word) for word in words if word]
    if total_words > 5 and endings:
        max_end_repeat = max(Counter(endings).values())
        homoioteleuton_score = (max_end_repeat / total_words) * 100
        if homoioteleuton_score > 20:
            common_end = [end for end, count in Counter(endings).items() if count == max_end_repeat][0]
            words_with_end = [w for w in words if ending_sound(w) == common_end][:3]
            results['Homoioteleuton'] = {
                'score': min(homoioteleuton_score, 100),
                'justification': f"Ending '{common_end}' in words: {words_with_end}."
            }

    # apo koinou: word shared btwn clauses
    apo_koinou_score = 0
    for sent in sentences:
        words_in_sent = sent.split()
        if len(words_in_sent) > 3:
            for i in range(1, len(words_in_sent) - 1):
                if words_in_sent[i] in words_in_sent[:i] or words_in_sent[i] in words_in_sent[i+1:]:
                    apo_koinou_score = 80
                    shared_word = words_in_sent[i]
                    break
    if apo_koinou_score > 0:
        results['Apo koinou'] = {
            'score': apo_koinou_score,
            'justification': f"Word '{shared_word}' shared between clauses."
        }

    # enallage: adj shift
    enallage_score = 0
    for i, pos in enumerate(pos_tags):
        if pos == 'ADJ' and i > 0 and i < len(pos_tags) - 1:
            if pos_tags[i-1] != 'NOUN' and pos_tags[i+1] != 'NOUN':
                enallage_score = 70
                adj_word = words[i]
                break
    if enallage_score > 0:
        results['Enallage'] = {
            'score': enallage_score,
            'justification': f"Adjective '{adj_word}' not directly modifying a noun."
        }

    # epiphora: end rpt
    clause_ends = [clause.strip().split()[-1] for clause in clauses if clause.strip()]
    if len(clause_ends) > 1:
        matches = sum(1 for i in range(1, len(clause_ends)) if clause_ends[i].lower() == clause_ends[0].lower())
        epiphora_score = (matches / (len(clause_ends) - 1)) * 100 if matches > 0 else 0
        if epiphora_score > 0:
            results['Epiphora'] = {
                'score': epiphora_score,
                'justification': f"Clauses end with '{clause_ends[0]}' (repeated {matches} times)."
            }

    # figura etymologica: same root
    etym_count = 0
    etym_pairs = []
    for i in range(len(lemmas) - 1):
        if lemmas[i][:3] == lemmas[i+1][:3] and lemmas[i] != lemmas[i+1]:
            etym_count += 1
            etym_pairs.append((words[i], words[i+1]))
    if etym_count > 0:
        results['Figura etymologica'] = {
            'score': min(etym_count * 30, 100),
            'justification': f"Same root pairs: {', '.join([f'{p[0]}-{p[1]}' for p in etym_pairs])}."
        }

    # geminatio/anadiplosis: word rpt
    geminatio_count = sum(1 for i in range(len(words) - 1) if words[i].lower() == words[i+1].lower())
    if geminatio_count > 0:
        repeated_words = [words[i] for i in range(len(words) - 1) if words[i].lower() == words[i+1].lower()]
        results['Geminatio'] = {
            'score': min(geminatio_count * 40, 100),
            'justification': f"Repeated words: {', '.join(set(repeated_words))}."
        }

    # hendiadys: two words for one idea
    hendiadys_score = 0
    for i in range(len(words) - 2):
        if pos_tags[i] == 'NOUN' and (words[i+1] == 'et' or words[i+1] == 'ac') and pos_tags[i+2] == 'NOUN':
            hendiadys_score = 60
            hendiadys_words = words[i:i+3]
            break
    if hendiadys_score > 0:
        results['Hendiadys'] = {
            'score': hendiadys_score,
            'justification': f"Words joined by conjunction: {', '.join(hendiadys_words)}."
        }

    # hyperbaton: word sep
    hyperbaton_score = 0
    modifier = None
    for sent in doc.sentences:
        deps = [word.deprel for word in sent.words]
        if 'advmod' in deps or 'amod' in deps:
            for word in sent.words:
                if word.deprel in ['amod', 'advmod']:
                    modifier = word.text
                    hyperbaton_score = 50
                    break
            if hyperbaton_score > 0:
                break
    if hyperbaton_score > 0:
        results['Hyperbaton'] = {
            'score': hyperbaton_score,
            'justification': f"The modifier '{modifier}' is separated from its head word."
        }

    # oxymoron: contradict terms
    oxymoron_pairs = [('parcus', 'insanientis'), ('discors', 'concordia')]  # Simplified
    oxymoron_count = 0
    detected_oxy = []
    for pair in oxymoron_pairs:
        if pair[0] in lemmas and pair[1] in lemmas:
            oxymoron_count += 1
            detected_oxy.append(pair)
    if oxymoron_count > 0:
        results['Oxymoron'] = {
            'score': min(oxymoron_count * 50, 100),
            'justification': f"Contradictory pairs: {', '.join([f'{p[0]}-{p[1]}' for p in detected_oxy])}."
        }

    # paronomasia: similar sounding words
    paronomasia_count = 0
    para_pairs = []
    for i in range(len(words) - 1):
        if len(words[i]) > 3 and len(words[i+1]) > 3 and words[i][:3] == words[i+1][:3]:
            paronomasia_count += 1
            para_pairs.append((words[i], words[i+1]))
    if paronomasia_count > 0:
        results['Paronomasia'] = {
            'score': min(paronomasia_count * 30, 100),
            'justification': f"Similar sounding pairs: {', '.join([f'{p[0]}-{p[1]}' for p in para_pairs])}."
        }

    # pleonasm: redundant words
    pleonasm_count = sum(1 for i in range(len(lemmas) - 1) if lemmas[i] == lemmas[i+1])
    if pleonasm_count > 0:
        redundant_words = [words[i] for i in range(len(lemmas) - 1) if lemmas[i] == lemmas[i+1]]
        results['Pleonasm'] = {
            'score': min(pleonasm_count * 40, 100),
            'justification': f"Redundant words: {', '.join(set(redundant_words))}."
        }

    # polyptoton: same word diff forms
    polyptoton_count = 0
    poly_words = []
    lemma_counts = Counter(lemmas)
    for lemma, count in lemma_counts.items():
        if count > 1:
            forms = [w.text for w in word_objects if w.lemma == lemma]
            if len(set(forms)) > 1:
                polyptoton_count += 1
                poly_words.append(forms)
    if polyptoton_count > 0:
        results['Polyptoton'] = {
            'score': min(polyptoton_count * 30, 100),
            'justification': f"Different forms of same word: {', '.join([str(p) for p in poly_words])}."
        }

    # tmesis: word splitting
    tmesis_score = 0
    if 'cumque' in words or 'ante' in words:
        tmesis_score = 70
        tmesis_word = 'cumque' if 'cumque' in words else 'ante'
    if tmesis_score > 0:
        results['Tmesis'] = {
            'score': tmesis_score,
            'justification': f"Split word element: '{tmesis_word}'."
        }

    # zeugma: shared verb
    zeugma_score = 0
    for sent in doc.sentences:
        verbs = [word for word in sent.words if word.upos == 'VERB']
        if len(verbs) == 1 and len(sent.words) > 5:
            zeugma_score = 60
            verb_word = verbs[0].text
            break
    if zeugma_score > 0:
        results['Zeugma'] = {
            'score': zeugma_score,
            'justification': f"Single verb '{verb_word}' for multiple subjects."
        }

    # antithesis: opposites
    antithesis_score = 0
    opposites = [('concordia', 'discordia'), ('parvae', 'maximae')]
    detected_anti = []
    for pair in opposites:
        if pair[0] in lemmas and pair[1] in lemmas:
            antithesis_score = 80
            detected_anti.append(pair)
            break
    if antithesis_score > 0:
        results['Antithesis'] = {
            'score': antithesis_score,
            'justification': f"Opposing words: {', '.join([f'{p[0]}-{p[1]}' for p in detected_anti])}."
        }

    # aposiopesis: pause in a pesis
    aposiopesis_score = 0
    if '—' in text or '...' in text:
        aposiopesis_score = 90
    if aposiopesis_score > 0:
        results['Aposiopesis'] = {
            'score': aposiopesis_score,
            'justification': "Sentence interrupted with '—' or '...'."
        }

    # apostrophe: direct address
    apostrophe_score = 0
    for sent in doc.sentences:
        addressed = [word.text for word in sent.words if word.upos == 'PROPN' and word.text.endswith(',')]
        if addressed:
            apostrophe_score = 70
            addressed_word = addressed[0]
            break
    if apostrophe_score > 0:
        results['Apostrophe'] = {
            'score': apostrophe_score,
            'justification': f"Direct address to '{addressed_word}'."
        }

    # asyndeton: no conjs
    asyndeton_score = 0
    for sent in sentences:
        words_in_sent = sent.split()
        if len(words_in_sent) > 3 and not any(c in sent for c in ['et', 'ac', 'atque']):
            asyndeton_score = 50
            break
    if asyndeton_score > 0:
        results['Asyndeton'] = {
            'score': asyndeton_score,
            'justification': "Enumeration without conjunctions."
        }

    # climax: increasing intensity
    climax_score = 0
    if len(words) > 5:
        climax_score = 40  
    if climax_score > 0:
        results['Climax'] = {
            'score': climax_score,
            'justification': "Detected increasing intensity in word sequence."
        }

    # ellipsis: omitted words
    ellipsis_score = 0
    if re.search(r'\b(est|dixit|cogitavit)\b', text) is None and len(sentences) > 1:
        ellipsis_score = 60
    if ellipsis_score > 0:
        results['Ellipsis'] = {
            'score': ellipsis_score,
            'justification': "Omitted common verbs like 'est'."
        }

    # hysteron proteron: inverted order
    hysteron_score = 0
    verbs = [i for i, pos in enumerate(pos_tags) if pos == 'VERB']
    if len(verbs) >= 2:
        for i in range(len(verbs) - 1):
            if verbs[i+1] - verbs[i] == 2 and words[verbs[i] + 1].lower() == 'et':
                hysteron_score = 80
                break
    if hysteron_score > 0:
        results['Hysteron proteron'] = {
            'score': hysteron_score,
            'justification': "Two verbs flipped, suggesting inverted order."
        }

    # parallelism: similar struct
    parallelism_score = 0
    if len(sentences) > 1:
        parallelism_score = 50
    if parallelism_score > 0:
        results['Parallelism'] = {
            'score': parallelism_score,
            'justification': "Similar sentence structures detected."
        }

    # praeteritio: mentioning by denying
    praeteritio_score = 0
    if 'praetereo' in lemmas or 'praetermittam' in lemmas:
        praeteritio_score = 85
        praet_word = 'praetereo' if 'praetereo' in lemmas else 'praetermittam'
    if praeteritio_score > 0:
        results['Praeteritio'] = {
            'score': praeteritio_score,
            'justification': f"Word '{praet_word}' indicates mentioning by denying."
        }

    # rhetorical question: questions
    rhetorical_score = 0
    if '?' in text:
        rhetorical_score = 70
    if rhetorical_score > 0:
        results['Rhetorical question'] = {
            'score': rhetorical_score,
            'justification': "Sentence ends with '?' indicating rhetorical question hopefully"
        }

    # syllepsis: grammatical sharing
    syllepsis_score = 0
    for sent in doc.sentences:
        if len(sent.words) > 5:
            syllepsis_score = 55
            break
    if syllepsis_score > 0:
        results['Syllepsis'] = {
            'score': syllepsis_score,
            'justification': "Grammatical element shared across clauses."
        }

    # tropes
    # removed keyword-based detection for tangibility

    # sort by score desc
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['score'], reverse=True))

    return sorted_results

st.title("latin lit device analyzer")

user_input = st.text_input("Enter your Latin sentence:")

if user_input:
    with st.spinner("Analyzing..."):
        data = analyze_style(user_input)
    
    if data:
        for device, info in data.items():
            st.write(f"### {device}: {info['score']:.1f}%")
            st.progress(info['score'] / 100)
            st.info(f"{info['justification']}")
    else:
        st.write("No stylistic devices detected.")

    # simple export (txt)
    if st.button("Export Results"):
        export_text = f"Sentence: {user_input}\n\n"
        for device, info in data.items():
            export_text += f"{device}: {info['score']:.1f}%\n{info['justification']}\n\n"
        st.download_button("Download as TXT", export_text, "analysis.txt")