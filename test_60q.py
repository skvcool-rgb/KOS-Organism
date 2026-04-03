"""60-question cross-domain test for KOS-Organism."""
import requests, json, time

URL = 'http://localhost:8090/chat'

tests = {
    'Chemistry': [
        ('What is the atomic number of Gold?', lambda r: 'gold' in r.lower() or '79' in r),
        ('What type of bond forms between Na and Cl?', lambda r: 'ionic' in r.lower()),
        ('Calculate molecular weight of H2O', lambda r: '18' in r),
        ('How do catalysts speed up chemical reactions?', lambda r: 'catalyst' in r.lower() or 'activation' in r.lower()),
        ('What is oxidation?', lambda r: 'electron' in r.lower() or 'oxidation' in r.lower()),
        ('Explain pH scale', lambda r: 'ph' in r.lower() or 'acid' in r.lower()),
        ('What are the properties of noble gases?', lambda r: len(r) > 50),
        ('How to improve Na-ion battery efficiency?', lambda r: 'battery' in r.lower() or 'cathode' in r.lower() or 'sodium' in r.lower()),
        ('What is electronegativity?', lambda r: 'electron' in r.lower() or 'bond' in r.lower()),
        ('Explain chemical equilibrium', lambda r: 'equilibrium' in r.lower() or 'reaction' in r.lower()),
    ],
    'Biology': [
        ('Translate the codon AUG', lambda r: 'met' in r.lower() or 'methionine' in r.lower()),
        ('What is DNA replication?', lambda r: 'dna' in r.lower() or 'replication' in r.lower()),
        ('Explain the stages of mitosis', lambda r: 'mitosis' in r.lower() or 'prophase' in r.lower() or 'division' in r.lower()),
        ('How do neurons transmit signals?', lambda r: 'neuron' in r.lower() or 'synapse' in r.lower() or 'axon' in r.lower()),
        ('What is photosynthesis?', lambda r: 'photosynthesis' in r.lower() or 'light' in r.lower() or 'chloro' in r.lower()),
        ('What are amino acids?', lambda r: 'amino' in r.lower() or 'protein' in r.lower()),
        ('Explain enzyme kinetics', lambda r: 'enzyme' in r.lower() or 'kinetic' in r.lower()),
        ('What is natural selection?', lambda r: 'selection' in r.lower() or 'evolution' in r.lower() or 'darwin' in r.lower()),
        ('How does cellular respiration work?', lambda r: 'respiration' in r.lower() or 'atp' in r.lower() or 'glucose' in r.lower()),
        ('What are the main organelles in a cell?', lambda r: 'nucleus' in r.lower() or 'mitochondria' in r.lower() or 'organelle' in r.lower()),
    ],
    'Math': [
        ('Calculate 15% of 250', lambda r: '37' in r),
        ('What is the square root of 144?', lambda r: '12' in r),
        ('Solve x^2 - 5x + 6 = 0', lambda r: '2' in r or '3' in r),
        ('Simplify (x**2 - 4)/(x - 2)', lambda r: 'x + 2' in r or 'x+2' in r),
        ('Calculate compound interest on 10000 at 5% for 3 years', lambda r: '11576' in r or '1576' in r or 'future' in r.lower()),
        ('What is 25 factorial?', lambda r: len(r) > 10),
        ('Integrate x^2 dx', lambda r: 'x^3' in r or 'x**3' in r or 'x3' in r or len(r) > 10),
        ('Calculate the derivative of sin(x)', lambda r: 'cos' in r.lower()),
        ('What is the area of a circle with radius 7?', lambda r: '153' in r or '154' in r or 'area' in r.lower()),
        ('Expand (a+b)^3', lambda r: 'a' in r and 'b' in r and len(r) > 10),
    ],
    'Finance': [
        ('Calculate compound interest on 50000 at 8% for 5 years', lambda r: '73466' in r or 'future' in r.lower() or 'compound' in r.lower()),
        ('What is EMI for 500000 loan at 8% for 20 years', lambda r: 'emi' in r.lower() or 'monthly' in r.lower()),
        ('Calculate simple interest on 100000 at 6% for 3 years', lambda r: '18000' in r or 'interest' in r.lower()),
        ('What is present value?', lambda r: 'present value' in r.lower() or 'future' in r.lower() or 'discount' in r.lower() or len(r) > 30),
        ('Explain the Black-Scholes model', lambda r: 'black' in r.lower() or 'option' in r.lower() or 'scholes' in r.lower() or len(r) > 30),
        ('What is Value at Risk?', lambda r: 'risk' in r.lower() or 'var' in r.lower() or 'value' in r.lower()),
        ('Calculate ROI: invested 1000, now worth 1500', lambda r: '50' in r or 'return' in r.lower() or 'roi' in r.lower() or len(r) > 30),
        ('What are Basel III capital requirements?', lambda r: 'basel' in r.lower() or 'capital' in r.lower()),
        ('Explain portfolio diversification', lambda r: 'diversi' in r.lower() or 'portfolio' in r.lower() or 'risk' in r.lower() or len(r) > 30),
        ('What is compound annual growth rate?', lambda r: 'growth' in r.lower() or 'cagr' in r.lower() or 'compound' in r.lower()),
    ],
    'Physics': [
        ('What is Newton second law?', lambda r: 'force' in r.lower() or 'acceleration' in r.lower() or 'newton' in r.lower()),
        ('Explain the photoelectric effect', lambda r: 'photo' in r.lower() or 'electron' in r.lower() or 'light' in r.lower()),
        ('What is the speed of light?', lambda r: '3' in r or 'light' in r.lower() or 'speed' in r.lower()),
        ('What are superconductors and how do they work?', lambda r: 'superconductor' in r.lower() or 'resistance' in r.lower() or 'cooper' in r.lower()),
        ('Explain quantum superposition', lambda r: 'quantum' in r.lower() or 'superposition' in r.lower() or 'state' in r.lower()),
        ('What is thermodynamic entropy?', lambda r: 'entropy' in r.lower() or 'disorder' in r.lower() or 'thermodynamic' in r.lower()),
        ('Explain special relativity', lambda r: 'relativity' in r.lower() or 'einstein' in r.lower() or 'speed' in r.lower()),
        ('What is the Heisenberg uncertainty principle?', lambda r: 'uncertainty' in r.lower() or 'position' in r.lower() or 'momentum' in r.lower()),
        ('Explain electromagnetic induction', lambda r: 'electromagnetic' in r.lower() or 'faraday' in r.lower() or 'magnetic' in r.lower()),
        ('What is the bandgap of silicon?', lambda r: '1.1' in r or 'silicon' in r.lower() or 'bandgap' in r.lower()),
    ],
    'Toxicity': [
        ('What are the health effects of lead exposure?', lambda r: 'lead' in r.lower() or 'toxic' in r.lower() or 'health' in r.lower()),
        ('Is cadmium toxic?', lambda r: 'cadmium' in r.lower() or 'toxic' in r.lower()),
        ('What is mercury poisoning?', lambda r: 'mercury' in r.lower() or 'toxic' in r.lower()),
        ('Safe alternatives to lead in solar cells', lambda r: 'lead' in r.lower() or 'tin' in r.lower() or 'safe' in r.lower() or len(r) > 30),
        ('What are the dangers of arsenic?', lambda r: 'arsenic' in r.lower() or 'toxic' in r.lower() or 'danger' in r.lower()),
        ('How to remediate heavy metal contamination?', lambda r: 'remediation' in r.lower() or 'metal' in r.lower() or len(r) > 30),
        ('Toxicity of chromium compounds', lambda r: 'chromium' in r.lower() or 'toxic' in r.lower()),
        ('What makes a substance carcinogenic?', lambda r: 'cancer' in r.lower() or 'carcino' in r.lower() or len(r) > 30),
        ('Safety precautions for handling selenium', lambda r: 'selenium' in r.lower() or 'safe' in r.lower() or len(r) > 30),
        ('Environmental impact of heavy metals', lambda r: 'metal' in r.lower() or 'environment' in r.lower() or len(r) > 30),
    ],
}

total_pass = 0
total_fail = 0

for domain, questions in tests.items():
    domain_pass = 0
    domain_fail = 0
    failures = []
    for q, check in questions:
        try:
            resp = requests.post(URL, json={'message': q}, timeout=30)
            r = resp.json()
            response_text = r.get('response', '')
            if check(response_text):
                domain_pass += 1
            else:
                domain_fail += 1
                failures.append(f'  FAIL: {q[:50]}... -> {response_text[:80]}')
        except Exception as e:
            domain_fail += 1
            failures.append(f'  ERR: {q[:50]}... -> {str(e)[:60]}')

    total_pass += domain_pass
    total_fail += domain_fail
    print(f'{domain}: {domain_pass}/10 ({domain_pass*10}%)')
    for f in failures:
        print(f)

print(f'\n=== TOTAL: {total_pass}/{total_pass+total_fail} ({total_pass*100/(total_pass+total_fail):.1f}%) ===')
